import os
import fitz
import numpy as np
import ollama
import chromadb
import pymongo
import redis
from bson.binary import Binary
from redis.commands.search.query import Query

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Initialize databases
db_chroma = chromadb.HttpClient(host="localhost", port=8000)
collection_chroma = db_chroma.get_or_create_collection(name="pdf_embeddings")

client_mongo = pymongo.MongoClient("mongodb://localhost:27017/")
db_mongo = client_mongo["embedding_db"]
collection_mongo = db_mongo["embeddings"]

redis_client = redis.Redis(host="localhost", port=6380, db=0)


def clear_all_stores():
    collection_chroma.delete(where={"$exists": True})
    print("ChromaDB store cleared.")

    collection_mongo.delete_many({})
    print("MongoDB collection cleared.")

    redis_client.flushdb()
    print("Redis store cleared.")


def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass
    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Redis HNSW index created.")


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def store_embedding(file, page, chunk, embedding):
    doc_id = f"{file}_page_{page}_chunk_{chunk[:30]}"
    collection_chroma.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
    )

    document = {
        "file": file,
        "page": page,
        "chunk": chunk,
        "embedding": Binary(np.array(embedding, dtype=np.float32).tobytes()),
    }
    collection_mongo.insert_one(document)

    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )
    print(f"Stored embedding for: {chunk[:30]}...")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(file_name, str(page_num), chunk, embedding)
            print(f" -----> Processed {file_name}")


def main():
    clear_all_stores()
    create_hnsw_index()
    process_pdfs("../../data/")
    print("\n---Done processing PDFs---\n")


if __name__ == "__main__":
    main()
