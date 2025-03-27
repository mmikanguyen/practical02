import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import redis
import chromadb
import time
import json
import numpy as np
import pymongo
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
import datetime
import csv
from redis.commands.search.field import VectorField, TextField
from src.vector_dbs.redis_ingest import CHUNK_SIZE
from src.vector_dbs.redis_ingest import CHUNK_OVERLAP
from src.vector_dbs.redis_ingest import EMBEDDING_MODEL

# Embedding models
embedding_model = EMBEDDING_MODEL
# chunks
CHUNK_SIZE = CHUNK_SIZE
CHUNK_OVERLAP=CHUNK_OVERLAP
# LLM
response_model = "mistral:latest"
#response_model = 'llama2:7b'

# redis connection
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text: str, model_name: str = embedding_model) -> list:
    # Handle Ollama embeddings
    if model_name == "nomic-embed-text" or model_name.startswith("llama"):
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]

    # Handle SentenceTransformer models
    elif model_name in ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]:
        model = SentenceTransformer(model_name)
        return model.encode(text).tolist()

    # Handle instructor models which require special formatting
    elif model_name == "hkunlp/instructor-xl":
        model = SentenceTransformer(model_name)
        return model.encode([text])[0].tolist()

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Please use 'nomic-embed-text', 'all-mpnet-base-v2', or 'hkunlp/instructor-xl'")


def search_embeddings_redis(query, top_k=3, db="redis"):
    stats = {
        "query_time": 0,
        "database_used": db
    }

    start_time = time.time()

    # Make sure to use the same model that was used during ingestion
    query_embedding = get_embedding(query, embedding_model)

    # Add debug info
    print(f"Query embedding length: {len(query_embedding)}")

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    # Use the same query structure as before
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk_id", "vector_distance")
        .dialect(2)
    )

    # Perform the search
    results = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": query_vector}
    )
#
#     stats = {
#         "query_time": 0,
#         "database_used": db
#     }
#
#     start_time = time.time()
#
#     query_embedding = get_embedding(query)
#
#     # Convert embedding to bytes for Redis search
#     query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
#
#     # Construct the vector similarity search query
#     # Use a more standard RediSearch vector search syntax
#     # q = Query("*").sort_by("embedding", query_vector)
#
#     q = (
#         Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
#         .sort_by("vector_distance")
#         .return_fields("id", "file", "page", "chunk", "vector_distance")
#         .dialect(2)
#     )
#
#     # Perform the search
#     results = redis_client.ft(INDEX_NAME).search(
#         q, query_params={"vec": query_vector}
#     )

    top_results = []
    unique_docs = set()

    # Check if we have results
    if results and hasattr(results, 'docs') and results.docs:
        for result in results.docs:
            # Get file name for tracking unique documents
            file_name = getattr(result, 'file', 'Unknown file')
            unique_docs.add(file_name)

            # Add to results
            top_results.append({
                "file": file_name,
                "page": getattr(result, 'page', 'Unknown page'),
                "chunk": getattr(result, 'chunk', 'Unknown chunk'),
                "similarity": getattr(result, 'vector_distance', 0),
            })

    top_results = top_results[:top_k]

    stats["query_time"] = time.time() - start_time

    # Print results for debugging
    for result in top_results:
        print(
            f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
        )

    return top_results, stats

def generate_rag_response(query, context_results, stats=None):

    gen_start_time = time.time()

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print("Generating response...")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model=response_model, messages=[{"role": "user", "content": prompt}]
    )

    if stats:
        stats["generation_time"] = time.time() - gen_start_time
        stats["total_time"] = stats["query_time"] + stats["generation_time"]

    return response["message"]["content"], stats

def print_statistics(stats):
    if stats is None:
        print("\n--- Query Statistics ---")
        print("No statistics available")
        print("------------------------")
        return
    print("\n--- Query Statistics ---")
    print(f"Query time: {stats.get('query_time', 0):.4f} seconds")
    if "generation_time" in stats:
        print(f"Generation time: {stats['generation_time']:.4f} seconds")
    if "total_time" in stats:
        print(f"Total time: {stats['total_time']:.4f} seconds")
    print("Database used:", stats.get("database_used", "chroma"))
    print("------------------------")


def log_stats_to_csv(stats, query, file_path):
    # need to align with ingest files !!

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    fieldnames = [
        'file',
        'timestamp',
        'query',
        'database',
        'embedding_model',
        'llm',
        'query_time',
        'generation_time',
        'total_time',
        "chunk_size",
        "chunk_overlap"
    ]

    file_exists = os.path.isfile(file_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        'file': "redis_search.py",
        'timestamp': timestamp,
        'query': query,
        'database': stats.get('database_used', 'unknown'),
        'embedding_model': embedding_model,
        'llm': response_model,
        'query_time': stats.get('query_time', 0),
        'generation_time': stats.get('generation_time', 0),
        'total_time': stats.get('total_time', 0),
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP
    }

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Stats logged to {file_path}")

def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    # while True:
    #     query = input("\nEnter your search query: ")
    #
    #     if query.lower() == "exit":
    #         break
    #
    #     # Search for relevant embeddings from the chosen database
    #     context_results, stats = search_embeddings_redis(query)
    #
    #     print("Stats after search:", stats)
    #
    #     response, updated_stats = generate_rag_response(query, context_results, stats)
    #     print_statistics(updated_stats)
    #
    #     file_path = "stats/redis_search.csv"
    #     log_stats_to_csv(updated_stats, query, file_path)
    #
    #
    #     print("\n--- Response ---")
    #     print(response)

    while True:
        try:
            query = input("\nEnter your search query: ").strip()

            if query.lower() == "exit":
                break

            # check for empty queries
            if not query:
                print("Query cannot be empty. Please try again.")
                continue

            print(f"Processing query: '{query}'")
            query_embedding = get_embedding(query, embedding_model)

            if query_embedding is None or len(query_embedding) == 0:
                print("Error: Unable to generate embedding for this query. Please try a different query.")
                continue

            print(f"Query embedding length: {len(query_embedding)}")

            # Search for relevant embeddings
            context_results, stats = search_embeddings_redis(query)

            print("Stats after search:", stats)

            # Generate response
            response, updated_stats = generate_rag_response(query, context_results, stats)
            print_statistics(updated_stats)

            # Log stats
            file_path = "stats/redis_search.csv"
            log_stats_to_csv(updated_stats, query, file_path)

            print("\n--- Response ---")
            print(response)

            # small delay before accepting next input to ensure terminal is ready
            time.sleep(0.5)

        except EOFError:
            print("\nInput error detected. Resetting input.")
            time.sleep(1)
            continue

        except KeyboardInterrupt:
            print("\nExiting search interface...")
            break

        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try a different query or check your system configuration.")
            time.sleep(0.5)



if __name__ == "__main__":

    interactive_search()
