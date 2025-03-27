import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import redis
import chromadb
import time
import json
import numpy as np
import pymongo
import ollama
from redis.commands.search.query import Query
import datetime
import csv
from redis.commands.search.field import VectorField, TextField
from sentence_transformers import SentenceTransformer
from src.vector_dbs.chroma_ingest import CHUNK_SIZE
from src.vector_dbs.chroma_ingest import CHUNK_OVERLAP
from src.vector_dbs.chroma_ingest import EMBEDDING_MODEL

Anna = 6381
Mika = 8000
chroma_client = chromadb.HttpClient(host="localhost", port=Mika)
chroma_collection = chroma_client.get_or_create_collection(name="embeddings")

#RESPONSE_MODEL = 'llama2:7b'
RESPONSE_MODEL = 'mistral:latest'

EMBEDDING_MODEL = EMBEDDING_MODEL
CHUNK_SIZE = CHUNK_SIZE
CHUNK_OVERLAP = CHUNK_OVERLAP

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
#    #other embedding SentenceTransformer("all-MiniLM-L6-v2") or SentenceTransformer("all-mpnet-base-v2")
#    response = ollama.embeddings(model=model, prompt=text)
#    return response["embedding"]


# def get_embedding(text: str, model: str = SentenceTransformer("all-mpnet-base-v2")) -> list:
#     return model.encode(text).tolist()

# def get_embedding(text: str, model: SentenceTransformer = SentenceTransformer("hkunlp/instructor-xl")) -> list:
#     # Generate and return the embedding for the input text
#     return model.encode([text])[0]


def get_embedding(text: str, model_name: str = EMBEDDING_MODEL) -> list:
    """Get embedding vector for input text using specified model."""
    # Add validation to prevent empty queries
    if not text or text.strip() == "":
        raise ValueError("Empty query provided. Please enter a valid query.")

    try:
        if model_name == "nomic-embed-text" or model_name.startswith("llama"):
            print(f"Generating embedding with {model_name}...")
            response = ollama.embeddings(model=model_name, prompt=text)
            if "embedding" not in response or not response["embedding"]:
                raise ValueError(f"Failed to get embedding from {model_name} model")
            return response["embedding"]  # Already returns a list

        elif model_name in ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]:
            model = SentenceTransformer(model_name)
            embedding = model.encode(text)
            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            return embedding  # If it's already a list

        elif model_name == "hkunlp/instructor-xl":
            model = SentenceTransformer(model_name)
            embedding = model.encode([text])[0]
            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            return embedding  # If it's already a list

        else:
            raise ValueError(
                f"Unsupported model: {model_name}. Please use 'nomic-embed-text', 'all-mpnet-base-v2', or 'hkunlp/instructor-xl'")

    except Exception as e:
        print(f"Error generating embedding with {model_name}: {str(e)}")
        # Rethrow with more context
        raise ValueError(f"Failed to generate embedding: {str(e)}")
def search_embeddings_chroma(query, top_k=3, db="chroma"):
    stats = {
        "query_time": 0,
        "database_used": db
    }

    start_time = time.time()

    # Get and validate embedding
    query_embedding = get_embedding(query, EMBEDDING_MODEL)
    if query_embedding is None or len(query_embedding) == 0:
        raise ValueError("Received empty embedding for query")

    query_vector = np.array(query_embedding, dtype=np.float32)

    # Perform the search on ChromaDB
    results = chroma_collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
    )

    # Debugging: print the results structure
    print("Chroma query results:", results)

    top_results = []
    if 'metadatas' in results and results['metadatas']:
        for i in range(len(results['metadatas'][0])):  # Iterate using index
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if 'distances' in results else 0

            top_results.append({
                "file": metadata.get("file", "Unknown file"),
                "page": metadata.get("page", "Unknown page"),
                "chunk": metadata.get("chunk", "Unknown chunk"),
                "similarity": 1 - distance,  # Convert distance to similarity
            })
    else:
        print("No valid metadata found in results")

    stats["query_time"] = time.time() - start_time

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
        model=RESPONSE_MODEL, messages=[{"role": "user", "content": prompt}]
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
        'file': "chroma_search.py",
        'timestamp': timestamp,
        'query': query,
        'database': stats.get('database_used', 'unknown'),
        'embedding_model': EMBEDDING_MODEL,
        'llm': RESPONSE_MODEL,
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

    while True:
        try:
            # Use input() with a clear prompt and flush stdout to ensure proper behavior
            print("\nEnter your search query: ", end="", flush=True)
            query = input().strip()

            if not query:
                print("Empty query provided. Please enter a valid query.")
                continue

            if query.lower() == "exit":
                break

            # Search for relevant embeddings
            try:
                print(f"Processing query: '{query}'")  # Debug line to confirm actual query

                # Add error checking around embedding generation
                query_embedding = get_embedding(query, EMBEDDING_MODEL)
                if not query_embedding or len(query_embedding) == 0:
                    print("Warning: Generated an empty embedding. Check the embedding model connection.")
                    continue

                context_results, stats = search_embeddings_chroma(query)
                print("Stats after search:", stats)

                response, updated_stats = generate_rag_response(query, context_results, stats)
                print_statistics(updated_stats)

                file_path = "stats/chroma_search.csv"
                log_stats_to_csv(updated_stats, query, file_path)

                print("\n--- Response ---")
                print(response)

                # Give the user time to read the response
                #print("\nPress Enter to continue...", end="", flush=True)
                #input()

            except ValueError as e:
                print(f"Error during search: {str(e)}")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                import traceback
                traceback.print_exc()  # Print the full error trace for debugging

        except KeyboardInterrupt:
            print("\nExiting due to keyboard interrupt.")
            break
        except EOFError:
            print("\nEnd of input detected. Exiting.")
            break
        except Exception as e:
            print(f"Input error: {str(e)}")
            print("Please try again or type 'exit' to quit.")



if __name__ == "__main__":

    interactive_search()
