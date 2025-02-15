# computes cosine similarity between stored embeddings and query embedding
# returns top-k most similar documents
import numpy as np

def cosine_similarity(store_embeddings, query_embedding, top_k=5):
    query_embedding = query_embedding.flatten()  
    # Compute cosine similarity
    dot_product = np.dot(store_embeddings, query_embedding)
    magnitudes = np.linalg.norm(store_embeddings, axis=1)
    query_magnitude = np.linalg.norm(query_embedding)
    similarities = dot_product / np.clip(magnitudes * query_magnitude, a_min=1e-10, a_max=None)
    # Get indices of top-k most similar vectors (ensure valid range)
    top_k = min(top_k, len(store_embeddings))
    return np.argsort(similarities)[::-1][:top_k]

