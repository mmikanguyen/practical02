#!/usr/bin/env python3

import ollama
import redis
import numpy as np
import os
import fitz
import argparse
import time

REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_DB = 0

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

CHUNK_SIZE = 300 # should this be a list? go through different chunk sizes ?

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

def create_vector_index():
    pass

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
    if not text or not text.strip():
        return [0.0] * VECTOR_DIM

    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0.0] * VECTOR_DIM


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


def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()

    if not words:
        return [""]

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def main():

    parser = argparse.ArgumentParser(description="Document ingestion system for vector search")
    parser.add_argument("--data", type=str, default="../data", help="Directory containing PDF files")
    parser.add_argument("--clear", action="store_true", help="Clear existing database before ingestion")
    parser.add_argument("--test", type=str, help="Run a test query after ingestion")
    args = parser.parse_args()

    redis_client = get_redis_connection()

    # time how long it takes to query
    start_time = time.time()

    # clear db
    if args.clear:
        clear_redis_db(redis_client)
        create_vector_index(redis_client)

    data_dir = args.data
    print(f"Processing documents from: {data_dir}")

    elapsed = time.time() - start_time


if __name__ == "__main__":
    main()