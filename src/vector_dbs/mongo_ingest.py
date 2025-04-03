import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threading
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threading

import ollama
import pymongo
import numpy as np
import fitz
import time
import psutil
import tracemalloc
import csv
import gc
from bson.binary import Binary
from sentence_transformers import SentenceTransformer
import threading

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["embedding_db"]
collection = db["embeddings"]

# Embedding model and vector dimension
# EMBEDDING_MODEL = "hkunlp/instructor-xl"
EMBEDDING_MODEL = "nomic-embed-text"
# EMBEDDING_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

VECTOR_DIM = 768

# Thread-local storage for model instances
thread_local = threading.local()


# Clear MongoDB collection
def clear_mongo_collection():
    print("Clearing existing MongoDB collection...")
    collection.delete_many({})
    print("MongoDB collection cleared.")


def get_embedding_model(model_name=EMBEDDING_MODEL):
    """Get a thread-local model instance"""
    if not hasattr(thread_local, 'model') or thread_local.model_name != model_name:
        if model_name in ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "hkunlp/instructor-xl"]:
            print(f"Initializing model {model_name}...")
            thread_local.model = SentenceTransformer(model_name)
            thread_local.model_name = model_name
    return thread_local.model


def get_embeddings_batch(texts, model_name=EMBEDDING_MODEL, batch_size=32):
    """Process embeddings in batches to be more efficient"""
    if model_name == "nomic-embed-text" or model_name.startswith("llama"):
        # Ollama doesn't support batch processing, so process one by one
        return [ollama.embeddings(model=model_name, prompt=text)["embedding"] for text in texts]

    # Use SentenceTransformer's batch capability
    model = get_embedding_model(model_name)

    all_embeddings = []
    # Process in smaller batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if model_name == "hkunlp/instructor-xl":
            # Special handling for instructor models
            embeddings = model.encode(batch).tolist()
        else:
            embeddings = model.encode(batch).tolist()
        all_embeddings.extend(embeddings)

    return all_embeddings


def get_embedding(text, model_name=EMBEDDING_MODEL):
    """Get embedding for a single text"""
    if model_name == "nomic-embed-text" or model_name.startswith("llama"):
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]

    model = get_embedding_model(model_name)

    if model_name == "hkunlp/instructor-xl":
        return model.encode([text])[0].tolist()
    else:
        return model.encode(text).tolist()


# Store embedding in MongoDB
def store_embedding(file, page, chunk, embedding):
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
    text_by_page = [(page.number, page.get_text()) for page in doc]
    doc.close()  # Explicitly close the document
    return text_by_page


# Split text into chunks with overlap
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


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

    # Pre-initialize the model to avoid concurrent initialization
    if EMBEDDING_MODEL in ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "hkunlp/instructor-xl"]:
        get_embedding_model(EMBEDDING_MODEL)

    for file_name in pdf_files:
        pdf_path = os.path.join(data_dir, file_name)
        text_by_page = extract_text_from_pdf(pdf_path)
        document_count += 1

        for page_num, text in text_by_page:
            chunks = split_text_into_chunks(text)
            chunk_count += len(chunks)

            # Process chunks in batches of 16 to reduce model calls
            batch_size = 16
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Get embeddings for the batch
                embeddings = get_embeddings_batch(batch_chunks, EMBEDDING_MODEL, batch_size)

                # Store each embedding
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    store_embedding(file_name, page_num, chunk, embedding)

                # Force garbage collection after each batch
                gc.collect()

        print(f"Processed {file_name}")
        # Force garbage collection after each file
        gc.collect()

    elapsed_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ensure stats directory exists at the expected location
    stats_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "stats"))
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)  # Create directory if it doesn't exist
        print(f"Created stats directory at {stats_dir}")

    stats_path = os.path.join(stats_dir, "mongo_processing.csv")

    with open(stats_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if the file is empty
        if file.tell() == 0:
            writer.writerow(["vector_db",
                             "embedding_model",
                             "peak_memory_mb",
                             "total_processing_time",
                             "docs_processed",
                             "chunks_processed",
                             "chunk_size",
                             "chunk_overlap"])

        # Append the new stats to the CSV
        writer.writerow(["mongo",
                         EMBEDDING_MODEL,
                         peak_memory / 1024 / 1024,
                         elapsed_time,
                         document_count,
                         chunk_count,
                         CHUNK_SIZE,
                         CHUNK_OVERLAP])

    print(f"Total processing time: {elapsed_time:.2f}s")
    print(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"Documents processed: {document_count}")
    print(f"Chunks processed: {chunk_count}")
    print(f"Exported stats to {stats_path}")


# Main function
def main():
    clear_mongo_collection()
    process_pdfs("../../data/")
    print("\n--- Done processing PDFs ---\n")

    # Clean up resources
    thread_local.__dict__.clear()
    gc.collect()


if __name__ == "__main__":
    main()