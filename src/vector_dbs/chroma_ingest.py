import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import chromadb
import ollama
import numpy as np
import fitz
import os
import time
import tracemalloc
import psutil


chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="embeddings")
VECTOR_DIM = 768


# Clear ChromaDB collection
def clear_chroma_store():
    print("Clearing existing Chroma store...")
    collection.delete(where={"$exists": True})
    print("Chroma store cleared.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store embedding in ChromaDB
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{chunk[:30]}"
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
    )
    print(f"Stored embedding for: {chunk[:30]}...")


# Extract text from a PDF by page
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
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

    print(f"Total processing time: {elapsed_time:.2f}s")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")




"""
def query_chroma(query_text: str):
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=5)

    for doc_id, metadata in zip(results["ids"], results["metadatas"]):
        print(f"{doc_id} \n ----> {metadata}\n")
"""

def main():
    clear_chroma_store()
    process_pdfs("../../data/")
    print("\n---Done processing PDFs---\n")
    #query_chroma("What is the capital of France?")


if __name__ == "__main__":
    main()
