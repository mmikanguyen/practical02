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

    document_count = 0
    chunk_count = 0

    for file_name in pdf_files:
        pdf_path = os.path.join(data_dir, file_name)
        text_by_page = extract_text_from_pdf(pdf_path)
        document_count += 1

        for page_num, text in text_by_page:
            chunks = split_text_into_chunks(text)
            chunk_count += len(chunks)
            for chunk_index, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                store_embedding(file_name, page_num, chunk, embedding)

        print(f"Processed {file_name}")

    elapsed_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ensure stats directory exists at the expected location
    stats_dir = os.path.abspath(os.path.join(os.getcwd(), "stats"))
    if not os.path.exists(stats_dir):
        print(f"Error: Stats directory not found at {stats_dir}. Please create it manually.")
        return  # Exit the function if the folder doesn't exist

    stats_path = os.path.join(stats_dir, "mongo_processing.csv")

    with open(stats_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if the file is empty
        if file.tell() == 0:
            writer.writerow(["Vector DB", "Embedding Model", "Peak Memory (MB)", "Total Processing Time (s)", "Documents Processed", "Chunks Processed"])

        # Append the new stats to the CSV
        writer.writerow(["mongo", "nomic-embed-text", peak_memory / 1024 / 1024, elapsed_time, document_count, chunk_count])

    print(f"Total processing time: {elapsed_time:.2f}s")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"Documents processed: {document_count}")
    print(f"Chunks processed: {chunk_count}")
    print(f"Exported stats to {stats_path}")


# Main function
def main():
    clear_mongo_collection()
    process_pdfs("../data/")
    print("\n--- Done processing PDFs ---\n")


if __name__ == "__main__":
    main()
