import ollama
import pymongo
import numpy as np
import os
import fitz
from bson.binary import Binary

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["embedding_db"]
collection = db["embeddings"]
VECTOR_DIM = 768

def clear_mongo_collection():
    print("Clearing existing MongoDB collection...")
    collection.delete_many({})
    print("MongoDB collection cleared.")

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    document = {
        "file": file,
        "page": page,
        "chunk": chunk,
        "embedding": Binary(np.array(embedding, dtype=np.float32).tobytes())  # Store as Binary
    }
    collection.insert_one(document)
    print(f"Stored embedding for: {chunk}")


# Extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


# Example MongoDB query to retrieve documents with similar embeddings
import faiss

def query_mongo(query_text: str):
    # Get the embedding for the query (assuming it's a high-dimensional vector)
    query_embedding = get_embedding(query_text)

    # Fetch all embeddings from the collection
    embeddings_data = collection.find({})  # You may want to limit this to just the necessary fields for performance

    # Convert MongoDB embeddings to a NumPy array
    embeddings_list = [doc['embedding'] for doc in embeddings_data]
    embeddings_array = np.array(embeddings_list).astype('float32')

    # Build the FAISS index (L2 distance)
    index = faiss.IndexFlatL2(embeddings_array.shape[1])  # Assuming 2D vectors
    index.add(embeddings_array)  # Add embeddings to the index

    # Search for the nearest neighbor(s)
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)  # k = 5 nearest neighbors

    # Retrieve and display results
    for i in range(5):
        print(f"Document {I[0][i]} - Distance: {D[0][i]}")
        doc = collection.find_one({"_id": I[0][i]})  # Find the document by its ID
        print(f"{doc['file']} - Page: {doc['page']}, Chunk: {doc['chunk']}")
        print(f"Embedding: {doc['embedding']}")

def main():
    clear_mongo_collection()
    process_pdfs("../../data/")
    print("\n---Done processing PDFs---\n")
    # Example query:
    query_mongo("Efficient search in vector databases")


if __name__ == "__main__":
    main()