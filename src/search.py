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
from redis.commands.search.field import VectorField, TextField


# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_collection = chroma_client.get_or_create_collection(name="embeddings")
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["embedding_db"]
collection = db["embeddings"]

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def search_embeddings_mongo(query, top_k=3):
    pass

def search_embeddings_chroma(query, top_k=3):
    stats = {
        "documents_searched": 0,
        "chunks_used": 0,
        "query_time": 0,
        "database_used": "chroma"  # Assuming ChromaDB is the default
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

    stats["documents_searched"] = len(unique_docs)
    stats["chunks_used"] = len(top_results)
    stats["query_time"] = time.time() - start_time

    # Print results for debugging
    for result in top_results:
        print(
            f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']:.2f}"
        )

    return top_results, stats


'''

def search_embeddings_redis(query, top_k=3):

    query_embedding = get_embedding(query)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

'''
def generate_rag_response(query, context_results, stats=None):

    gen_start_time = time.time()

    if stats is None:
        stats = {
            "documents_searched": 0,
            "chunks_used": len(context_results),
            "query_time": 0,
            "database_used": "chroma"
        }

    original_query_time = stats.get("query_time", 0)
    print(f"DEBUG - Inside generate_rag_response - Original query time: {original_query_time}")

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

    stats["generation_time"] = time.time() - gen_start_time
    stats["total_time"] = stats.get("query_time", 0) + stats["generation_time"]
    stats["query_time"] = original_query_time

    print(f"DEBUG - After response generation - Final query time: {stats['query_time']}")


    return response["message"]["content"], stats

def print_statistics(stats):
    if stats is None:
        print("\n--- Query Statistics ---")
        print("No statistics available")
        print("------------------------")
        return

    print("\n--- Query Statistics ---")
    print(f"Documents searched: {stats.get('documents_searched', 0)}")
    print(f"Chunks used: {stats.get('chunks_used', 0)}")
    print(f"Query time: {stats.get('query_time', 0):.4f} seconds")
    if "generation_time" in stats:
        print(f"Generation time: {stats['generation_time']:.4f} seconds")
    if "total_time" in stats:
        print(f"Total time: {stats['total_time']:.4f} seconds")
    print("Database used:", stats.get("database_used", "chroma"))
    print("------------------------")

def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results, stats = search_embeddings_chroma(query)

        print("DEBUG - Stats after search:", stats)


        # Generate RAG response
        response, updated_stats = generate_rag_response(query, context_results, stats=stats)
        print_statistics(updated_stats)

        print("\n--- Response ---")
        print(response)



if __name__ == "__main__":
    interactive_search()
