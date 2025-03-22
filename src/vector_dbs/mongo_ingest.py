import ollama
import pymongo
import numpy as np
import os
import fitz
import time
import psutil
import tracemalloc
import csv
from bson.binary import Binary

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["embedding_db"]
collection = db["embeddings"]

# Embedding model and vector dimension
VECTOR_DIM = 768


# Clear MongoDB collection
def clear_mongo_collection():
    print("Clearing existing MongoDB collection...")
    collection.delete_many({})
    print("MongoDB collection cleared.")


# Get text embedding
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store embedding in MongoDB
def store_embedding(file: str, page: int, chunk: str, embedding: list):
    document = {
        "file": file,
        "page": page,
        "chunk": chunk,
        "embedding": Binary(np.array(embedding, dtype=np.float32).tobytes())
    }
    collection.insert_one(document)
    print(f"Stored embedding for page {page}, chunk: {chunk[:30]}...")


# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [(page.number, page.get_text()) for page in doc]


# Split text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


# Track memory usage
def track_memory_usage():
    return psutil.virtual_memory().percent


# Process PDFs in a directory
def process_pdfs(data_dir):
    start_time = time.time()
    tracemalloc.start()

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    for file_name in pdf_files:
        pdf_path = os.path.join(data_dir, file_name)
        text_by_page = extract_text_from_pdf(pdf_path)

        for page_num, text in text_by_page:
            chunks = split_text_into_chunks(text)
            for chunk_index, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                store_embedding(file_name, page_num, chunk, embedding)

        print(f"Processed {file_name}")

    elapsed_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ensure stats directory exists
    stats_dir = "stats"
    os.makedirs(stats_dir, exist_ok=True)

    stats_path = os.path.join(stats_dir, "mongo_processing.csv")

    with open(stats_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Vector DB", "Embedding Model", "Peak Memory (MB)", "Total Processing Time (s)"])
        writer.writerow(["MongoDB", "nomic-embed-text", peak_memory / 1024 / 1024, elapsed_time])

    print(f"Total processing time: {elapsed_time:.2f}s")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"Exported stats to {stats_path}")


# Main function
def main():
    clear_mongo_collection()
    process_pdfs("../../data/")
    print("\n--- Done processing PDFs ---\n")


if __name__ == "__main__":
    main()
