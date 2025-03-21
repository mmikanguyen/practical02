from sentence_transformers import SentenceTransformer

def get_embedding(text: str, model: str = SentenceTransformer("all-mpnet-base-v2")) -> list:
    return model.encode(text).tolist()
