import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import time
import tracemalloc
import psutil
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import multiprocessing
import gc

_model_cache = {}

EMBEDDING_MODEL = "all-mpnet-base-v2"
# EMBEDDING_MODEL = 'hkunlp/instructor-xl'
# EMBEDDING_MODEL = "nomic-embed-text"

# Anna's port
chroma_client = chromadb.HttpClient(host="localhost", port=6381)

# Mika's port
# chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="embeddings")
VECTOR_DIM = 768

def clear_chroma_store():
    print("Clearing existing Chroma store...")
    collection.delete(where={"$exists": True})
    print("Chroma store cleared.")

# nomic-embed
#def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
  #  response = ollama.embeddings(model=model, prompt=text)
  #  return response["embedding"]

# sentence_transformers
# def get_embedding(text: str, model: str = SentenceTransformer("all-mpnet-base-v2")) -> list:
#     return model.encode(text).tolist()

#instructor-xl
# def get_embedding(text: str, model: SentenceTransformer = SentenceTransformer("hkunlp/instructor-xl")) -> list:
#     # Generate and return the embedding for the input text
#     return model.encode([text])[0]

def get_embedding(text: str, model_name: str = EMBEDDING_MODEL) -> list:
    global _model_cache

    if model_name == "nomic-embed-text" or model_name.startswith("llama"):
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]

    elif model_name in ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]:
        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name)
        model = _model_cache[model_name]
        return model.encode(text).tolist()

    elif model_name == "hkunlp/instructor-xl":
        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name)
        model = _model_cache[model_name]
        return model.encode([text])[0].tolist()

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Please use 'nomic-embed-text', 'all-mpnet-base-v2', or 'hkunlp/instructor-xl'")

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{chunk[:30]}"
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
    )
    print(f"Stored embedding for: {chunk[:30]}...")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def split_text_into_chunks(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks


def track_memory_usage():
    memory_info = psutil.virtual_memory()
    return memory_info.percent


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
                embedding = get_embedding(chunk, EMBEDDING_MODEL)
                store_embedding(file_name, page_num, chunk, embedding)

        print(f"Processed {file_name}")

    elapsed_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ensure stats directory exists at the expected location
    stats_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "stats"))
    if not os.path.exists(stats_dir):
        print(f"Error: Stats directory not found at {stats_dir}. Please create it manually.")
        return  # Exit the function if the folder doesn't exist

    stats_path = os.path.join(stats_dir, "chroma_processing.csv")

    with open(stats_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if the file is empty
        if file.tell() == 0:
            writer.writerow(["Vector DB", "Embedding Model", "Peak Memory (MB)", "Total Processing Time (s)", "Documents Processed", "Chunks Processed"])

        # Append the new stats to the CSV
        writer.writerow(["chroma", EMBEDDING_MODEL, peak_memory / 1024 / 1024, elapsed_time, document_count, chunk_count])

    print(f"Total processing time: {elapsed_time:.2f}s")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"Documents processed: {document_count}")
    print(f"Chunks processed: {chunk_count}")
    print(f"Exported stats to {stats_path}")



def main():
    # Set the start method to 'spawn' which is less prone to resource leaks
    if __name__ == "__main__" and multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    clear_chroma_store()
    process_pdfs("../../data/")

    # Add explicit cleanup
    gc.collect()

    # If you're using a SentenceTransformer model, try to clean it up explicitly
    if 'model' in globals():
        del model
        gc.collect()

    print("\n--- Done processing PDFs ---\n")