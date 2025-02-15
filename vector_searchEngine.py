# verctor-based storage and search engine
# uses embeddings for similarity search
import numpy as np
from cosine_dotproduct import cosine_similarity

class VectorSearchEngine:
    def __init__(self, embedder):
        self.embedder = embedder
        self.docs = []
        # Initialize embeddings as an empty 2D array
        self.embeddings = np.empty((0, self.embedder.get_sentence_embedding_dimension()))

    def from_docs(self, docs):
        self.docs = docs
        self.embeddings = np.array(self.embedder.encode(docs))  # Convert docs to embeddings

    def similarity_search(self, query, top_k=5):
        if len(self.docs) == 0:
            print("No documents available.")
            return []

        query_embedding = self.embedder.encode([query]).flatten()  # Ensure query is 1D
        top_indices = cosine_similarity(self.embeddings, query_embedding, top_k)

        return [self.docs[i] for i in top_indices]