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


# Embedding models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.HttpClient(host="localhost", port=6381)
chroma_collection = chroma_client.get_or_create_collection(name="embeddings")


VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    #other embedding SentenceTransformer("all-MiniLM-L6-v2") or SentenceTransformer("all-mpnet-base-v2")
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings_chroma(query, top_k=3, db="chroma"):
    stats = {
        "query_time": 0,
        "database_used": db
    }

    start_time = time.time()

    query_embedding = get_embedding(query)

    # Convert embedding to numpy array for ChromaDB search
    query_vector = np.array(query_embedding, dtype=np.float32)
    # Perform the search on ChromaDB
    results = chroma_collection.query(
        query_embeddings=[query_vector.tolist()],  # Convert to list for ChromaDB compatibility
        n_results=top_k,
    )

    top_results = []
    unique_docs = set()

    # Check if we have results
    if results and 'metadatas' in results and results['metadatas']:
        for i in range(len(results['metadatas'][0])):  # Iterate using index
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if 'distances' in results else 0

            top_results.append({
                "file": metadata.get("file", "Unknown file"),
                "page": metadata.get("page", "Unknown page"),
                "chunk": metadata.get("chunk", "Unknown chunk"),
                "similarity": 1 - distance,  # Convert distance to similarity
            })

    stats["query_time"] = time.time() - start_time

    # Print results for debugging
    for result in top_results:
        print(
            f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']:.2f}"
        )

    # Print results for debugging
    for result in top_results:
        print(
            f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']:.2f}"
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
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
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
        'query_time',
        'generation_time',
        'total_time'
    ]

    file_exists = os.path.isfile(file_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        'file': "chroma_search.py",
        'timestamp': timestamp,
        'query': query,
        'database': stats.get('database_used', 'unknown'),
        'query_time': stats.get('query_time', 0),
        'generation_time': stats.get('generation_time', 0),
        'total_time': stats.get('total_time', 0)
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

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings from the chosen database
        context_results, stats = search_embeddings_chroma(query)  # or switch to MongoDB or Redis here

        print("Stats after search:", stats)

        response, updated_stats = generate_rag_response(query, context_results, stats)
        print_statistics(updated_stats)

        file_path = "stats/chroma_search.csv"

        log_stats_to_csv(updated_stats, query, file_path)


        print("\n--- Response ---")
        print(response)




if __name__ == "__main__":

    interactive_search()
