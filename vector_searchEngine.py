import numpy as np
from scipy.spatial import KDTree
from datasketch import MinHash, MinHashLSH
from cosine_dotproduct import cosine_similarity

class VectorSearchEngine:
    def __init__(self, embedder, method="cosine"):
        self.embedder = embedder
        self.docs = []
        self.embeddings = np.empty((0, self.embedder.get_sentence_embedding_dimension()))
        self.method = method
        self.kd_tree = None
        self.lsh = None
        self.minhashes = []

    def from_docs(self, docs):
        """Encodes documents into embeddings and initializes the selected search method."""
        self.docs = docs
        self.embeddings = np.array(self.embedder.encode(docs))  
        
        if self.method == "kd_tree":
            print("Building KD-Tree index...")
            self.kd_tree = KDTree(self.embeddings)  
            print("KD-Tree built successfully.")

        elif self.method == "lsh":
            print("Building LSH index...")
            self.lsh = MinHashLSH(threshold=0.5, num_perm=256)  
            for i, embedding in enumerate(self.embeddings):
                minhash = self._get_minhash(embedding)
                self.lsh.insert(str(i), minhash) 
                self.minhashes.append(minhash)
            print(f"LSH index built with {len(self.docs)} documents.")

    def similarity_search(self, query, top_k=5):
        """Performs similarity search using the selected method."""
        if len(self.docs) == 0:
            print("No documents available.")
            return []

        query_embedding = self.embedder.encode([query]).flatten()  
        
        if self.method == "kd_tree":
            _, top_indices = self.kd_tree.query(query_embedding.reshape(1, -1), k=min(top_k, len(self.docs)))
            top_indices = top_indices.flatten()
            print(f"KD-Tree matches: {top_indices}")
            return [self.docs[i] for i in top_indices]

        elif self.method == "lsh":
            minhash_query = self._get_minhash(query_embedding)
            print(f"Query MinHash: {minhash_query.digest()[:5]}")
            lsh_matches = list(self.lsh.query(minhash_query))  
            print(f"LSH Matches: {lsh_matches}")

            if not lsh_matches:
                return []

            top_indices = [int(match) for match in lsh_matches if match.isdigit() and int(match) < len(self.docs)]
            return [self.docs[i] for i in top_indices[:top_k]]

        else:
            top_indices = cosine_similarity(self.embeddings, query_embedding, top_k)
            print(f"Cosine Similarity matches: {top_indices}")
            return [self.docs[i] for i in top_indices]

    def _get_minhash(self, vector):
        """Convert a dense embedding vector into a MinHash signature."""
        minhash = MinHash(num_perm=256)
        
        # Convert continuous vector into binary representation (sign bit hashing)
        binary_vector = vector > np.median(vector)  # Convert to 0s and 1s

        for bit in binary_vector:
            minhash.update(str(int(bit)).encode("utf8"))

        return minhash
