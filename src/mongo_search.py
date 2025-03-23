import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import pymongo
import ollama
import datetime
import csv
from sentence_transformers import SentenceTransformer



# Embedding models
# embedding_model = SentenceTransformer("all-mpnet-base-v2")
embedding_model = 'nomic-embed-text'
response_model = 'mistral:latest'
#response_model = 'llama2:7b'

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
    #other embedding SentenceTransformer("all-MiniLM-L6-v2") or SentenceTransformer("all-mpnet-base-v2")
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# def get_embedding(text: str, model: str = SentenceTransformer("all-mpnet-base-v2")) -> list:
#     return model.encode(text).tolist()

# def get_embedding(text: str, model: SentenceTransformer = SentenceTransformer("hkunlp/instructor-xl")) -> list:
#     # Generate and return the embedding for the input text
#     return model.encode([text])[0]

def search_embeddings_mongo(query, top_k=3, db="mongo"):

    stats = {
        "query_time": 0,
        "database_used": db
    }

    start_time = time.time()

    query_embedding = get_embedding(query)

    # Convert the query embedding to a numpy array
    query_vector = np.array(query_embedding, dtype=np.float32)

    # Retrieve all documents from MongoDB collection
    all_docs = collection.find()

    # List to hold documents with their cosine similarity scores
    scored_docs = []

    for doc in all_docs:
        if isinstance(doc.get('embedding'), bytes):
            # Convert binary data to numpy array - adjust the shape as needed
            doc_embedding = np.frombuffer(doc['embedding'], dtype=np.float32)
        elif isinstance(doc.get('embedding'), list):
            doc_embedding = np.array(doc['embedding'], dtype=np.float32)
        else:
            print(f"Skipping document with unknown embedding format: {type(doc.get('embedding'))}")
            continue

        # # Extract embedding from the document
        # doc_embedding = np.array(doc['embedding'], dtype=np.float32)
        if len(doc_embedding) != len(query_vector):
            print(f"Skipping document with mismatched embedding dimension: {len(doc_embedding)} vs {len(query_vector)}")
            continue


        # Compute the cosine similarity between the query and the document embedding
        similarity = cosine_similarity(query_vector, doc_embedding)

        # Append the document and its similarity score to the list
        scored_docs.append({
            'file': doc.get('file', 'Unknown file'),
            'page': doc.get('page', 'Unknown page'),
            'chunk': doc.get('chunk', 'Unknown chunk'),
            'similarity': similarity
        })

    # Sort documents by similarity in descending order and get the top_k
    top_results = sorted(scored_docs, key=lambda x: x['similarity'], reverse=True)[:top_k]

    stats["query_time"] = time.time() - start_time

    # Print results for debugging
    for result in top_results:
        print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']:.2f}")

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

    # # Generate response using Ollama
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
        'query_time',
        'generation_time',
        'total_time'
    ]

    file_exists = os.path.isfile(file_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        'file': "mongo_search.py",
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
        try:
            query = input("\nEnter your search query: ").strip()

            if query.lower() == "exit":
                break

            # check for empty queries
            if not query:
                print("Query cannot be empty. Please try again.")
                continue

            print(f"Processing query: '{query}'")
            query_embedding = get_embedding(query)

            # dont change this or else wont work for instructor
            if query_embedding is None or len(query_embedding) == 0:
                print("Error: Unable to generate embedding for this query. Please try a different query.")
                continue

            print(f"Query embedding length: {len(query_embedding)}")

            # Search for relevant embeddings
            context_results, stats = search_embeddings_mongo(query)

            print("Stats after search:", stats)

            # Generate response
            response, updated_stats = generate_rag_response(query, context_results, stats)
            print_statistics(updated_stats)

            # Log stats
            file_path = "stats/mongo_search.csv"
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
