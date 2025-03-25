# driver.py

import argparse
import time
from src.vector_dbs.redis_ingest import process_directory, clear_redis_db, create_vector_index
from src.vector_dbs.redis_ingest import get_redis_connection, get_embedding
from src.vector_dbs.redis_ingest import CHUNK_SIZE, CHUNK_OVERLAP
import redis

# List of embedding models to test
embedding_models = [
    "all-mpnet-base-v2",
    "hkunlp/instructor-xl",
    "nomic-embed-text",
    # Add more models here as needed
]

# LLMs and Response Models
response_models = [
    "mistral:latest",
    "llama2:7b",
    # Add more LLMs here if needed
]

# Vector databases to test
vector_databases = [
    "redis",
    # Add other vector database options here if needed
]


def test_combo(redis_client, embedding_model, vector_db):
    """
    Test different combinations of embeddings and vector database.
    """
    print(f"Testing with embedding model: {embedding_model}, Vector DB: {vector_db}")

    if vector_db == "redis":
        # Clear Redis and create index if using Redis
        clear_redis_db(redis_client)
        create_vector_index(redis_client)

    # Start the processing of documents (assuming you have a directory path set)
    start_time = time.time()
    directory_path = "/path/to/your/pdf/directory"  # Change this path as necessary
    total_documents, total_chunks = process_directory(redis_client, directory_path)

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Total documents processed: {total_documents}, Total chunks processed: {total_chunks}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test different combos of embedding models and vector DBs")
    parser.add_argument("--clear", action="store_true", help="Clear the Redis database before processing")
    parser.add_argument("--test_query", type=str, help="Run a test query after ingestion")
    args = parser.parse_args()

    # Create Redis client
    redis_client = get_redis_connection()

    # Loop over all combinations of embedding models, vector databases, and response models
    for embedding_model in embedding_models:
        for vector_db in vector_databases:
            test_combo(redis_client, embedding_model, vector_db)

    # Optionally, run a test query after processing (if provided in args)
    if args.test_query:
        print("Running test query...")
        # Add your query logic here, e.g., querying Redis or Mongo for vector search results


if __name__ == "__main__":
    main()
