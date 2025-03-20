import chromadb
import ollama
import numpy as np
import fitz
import os


db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="pdf_embeddings")

VECTOR_DIM = 768


# Clear ChromaDB collection
def clear_chroma_store():
    print("Clearing existing Chroma store...")
    collection.delete(where={"$exists": True})  # Clear all documents
    print("Chroma store cleared.")



# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store embedding in ChromaDB
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{chunk[:30]}"
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
    )
    print(f"Stored embedding for: {chunk[:30]}...")


# Extract text from a PDF by page
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
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


# Query ChromaDB
def query_chroma(query_text: str):
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=5)

    for doc_id, metadata in zip(results["ids"], results["metadatas"]):
        print(f"{doc_id} \n ----> {metadata}\n")



def main():
    clear_chroma_store()
    process_pdfs("../data/")
    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?")


if __name__ == "__main__":
    main()
