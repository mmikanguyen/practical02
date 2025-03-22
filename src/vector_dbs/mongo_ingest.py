import ollama
import pymongo
import numpy as np
import os
import fitz
from bson.binary import Binary
import time
import psutil

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["embedding_db"]
collection = db["embeddings"]
VECTOR_DIM = 768

def clear_mongo_collection():
    print("Clearing existing MongoDB collection...")
    collection.delete_many({})
    print("MongoDB collection cleared.")

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    document = {
        "file": file,
        "page": page,
        "chunk": chunk,
        "embedding": Binary(np.array(embedding, dtype=np.float32).tobytes())  # Store as Binary
    }
    collection.insert_one(document)
    print(f"Stored embedding for: {chunk}")


# Extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks


# Track memory usage
def track_memory_usage():
    memory_info = psutil.virtual_memory()
    return memory_info.percent  # Returns memory usage as a percentage

def process_pdfs(data_dir):
    start_time = time.time()  # Start timing for the entire process
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

    elapsed_time = time.time() - start_time  # End timing for the entire process
    print(f"Total time taken for processing PDFs: {elapsed_time:.2f} seconds.")

    # Track memory usage after processing all PDFs
    memory_usage = track_memory_usage()
    print(f"Memory usage after processing all PDFs: {memory_usage}%")


def main():
    clear_mongo_collection()
    process_pdfs("../../data/")
    print("\n---Done processing PDFs---\n")
if __name__ == "__main__":
    main()