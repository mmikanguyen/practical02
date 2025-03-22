#!/usr/bin/env python3

import ollama
import redis
import numpy as np
import os
import fitz
import argparse
import csv
import time
import tracemalloc
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_DB = 0

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

CHUNK_SIZE = 300 # should this be a list? go through different chunk sizes ?
CHUNK_OVERLAP = 50 # is this necessary

EMBEDDING_MODEL = "nomic-embed-text" # same Q as above - list of 3 diff embedding models

def get_redis_connection():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB
    )

def clear_redis_db(redis_client):
    print("Clearing Redis database...")
    redis_client.flushdb()
    print("Database cleared")

def create_vector_index(redis_client):
    """Create the vector search index in Redis"""
    # Try to drop existing index
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
        print("Dropped existing index")
    except redis.exceptions.ResponseError:
        print("No existing index to drop")

    # Create new index
    index_cmd = f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA 
            file TEXT SORTABLE
            page TEXT SORTABLE
            chunk_id TEXT SORTABLE
            text TEXT
            embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """

    redis_client.execute_command(index_cmd)
    print(f"Created vector index '{INDEX_NAME}'")

def store_document_chunk(redis_client, document_id, page_num, chunk_id, text, embedding):
        key = f"{DOC_PREFIX}{document_id}:p{page_num}:c{chunk_id}"

        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        redis_client.hset(
            key,
            mapping={
                "file": document_id,
                "page": str(page_num),
                "chunk_id": str(chunk_id),
                "text": text,
                "embedding": embedding_bytes
            }
        )
def get_embedding(text, model=EMBEDDING_MODEL):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def extract_text_from_pdf(pdf_path):
    # strip stop words and punctuation??

    try:
        doc = fitz.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            if text.strip():
                pages.append((page_num, text))

        return pages
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()

    if not words:
        return [""]

    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def process_document(redis_client, file_path):
    """Process a single PDF file"""
    file_name = Path(file_path).name
    print(f"Processing: {file_name}")

    # Extract text by page
    pages = extract_text_from_pdf(file_path)

    total_chunks = 0
    total_chars = 0

    # Process each page sequentially (no threading)
    for page_num, text in pages:
        print(f"  Processing page {page_num}...")

        # Split text into chunks
        chunks = split_text_into_chunks(text)

        # Process each chunk
        for chunk_id, chunk_text in enumerate(chunks):
            # Skip empty chunks
            if not chunk_text.strip():
                continue

            # Generate embedding
            embedding = get_embedding(chunk_text)

            # Store in vector database
            store_document_chunk(
                redis_client,
                document_id=file_name,
                page_num=page_num,
                chunk_id=chunk_id,
                text=chunk_text,
                embedding=embedding
            )

            total_chars += len(chunk_text)
            total_chunks += 1

    print(f"Completed {file_name}: {len(pages)} pages, {total_chunks} chunks, {total_chars} characters")
    return total_chunks, len(pages)  # Return chunk and page count for this document


def process_directory(redis_client, directory_path):
    pdf_files = list(Path(directory_path).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return 0, 0  # Return 0 if no files found

    print(f"Found {len(pdf_files)} PDF files to process")

    total_documents = 0
    total_chunks = 0

    # Process each file
    for pdf_file in pdf_files:
        file_chunks, file_pages = process_document(redis_client, pdf_file)
        total_documents += 1
        total_chunks += file_chunks

    return total_documents, total_chunks



def run_query(redis_client, query_text, k=3):
    """Run a test query against the vector store"""
    from redis.commands.search.query import Query

    # Get embedding for query
    embedding = get_embedding(query_text)
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

    # Create search query
    q = (
        Query(f"*=>[KNN {k} @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("file", "page", "chunk_id", "text", "score")
        .dialect(2)
    )

    # Execute search
    results = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": embedding_bytes}
    )

    print(f"\nResults for query: '{query_text}'")
    for doc in results.docs:
        print(f"Document: {doc.file}, Page: {doc.page}, Chunk: {doc.chunk_id}")
        print(f"Score: {doc.score}")
        print(f"Text: {doc.text[:150]}...")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Document ingestion system for vector search")
    parser.add_argument("--data", type=str, default="../data", help="Directory containing PDF files")
    parser.add_argument("--clear", action="store_true", help="Clear existing database before ingestion")
    parser.add_argument("--test", type=str, help="Run a test query after ingestion")
    args = parser.parse_args()

    redis_client = get_redis_connection()

    start_time = time.time()
    tracemalloc.start()  # Start tracking memory

    if args.clear:
        clear_redis_db(redis_client)
        create_vector_index(redis_client)

    print(f"Processing documents from: {args.data}")
    total_documents, total_chunks = process_directory(redis_client, args.data)

    elapsed_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Peak Memory Usage: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"Total documents processed: {total_documents}")
    print(f"Total chunks processed: {total_chunks}")

    stats_dir = "stats"
    os.makedirs(stats_dir, exist_ok=True)

    stats_path = os.path.join(stats_dir, "redis_processing.csv")

    # Open the CSV file in append mode
    with open(stats_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file is empty (i.e., no header row), write the header
        if file.tell() == 0:  # Check if file is empty
            writer.writerow(
                ["Vector DB", "Embedding Model", "Peak Memory (MB)", "Total Processing Time (s)", "Total Documents",
                 "Total Chunks"])

        # Append the new stats
        writer.writerow(
            ["redis", "nomic-embed-text", peak_memory / 1024 / 1024, elapsed_time, total_documents, total_chunks])

main()


