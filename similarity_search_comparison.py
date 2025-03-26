import numpy as np
import time
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = [
    "Machine learning is amazing.",
    "Artificial intelligence is the future.",
    "Deep learning powers many AI applications.",
    "Python is a great programming language for AI.",
    "Natural language processing helps computers understand text.",
    "The stock market fluctuates daily based on multiple factors.",
    "Investing in cryptocurrency can be highly volatile.",
    "Tesla and Apple stocks have performed well in recent years.",
    "Interest rates impact loan affordability and housing markets.",
    "Economic inflation affects purchasing power worldwide."
]

num_docs = len(docs)
dim = model.get_sentence_embedding_dimension() 
top_k = 3  
num_perm = 128  


embeddings = np.array(model.encode(docs))


query = "How does AI work in finance?"
query_embedding = np.array(model.encode([query]))


kd_tree = KDTree(embeddings)

start_time = time.time()
_, kd_indices = kd_tree.query(query_embedding, k=top_k)
kd_time = time.time() - start_time

kd_matches = [docs[i] for i in kd_indices.flatten()]


start_time = time.time()
cos_similarities = cosine_similarity(embeddings, query_embedding).flatten()
cos_indices = np.argsort(cos_similarities)[-top_k:][::-1]  # Get top-k indices in descending order
cos_time = time.time() - start_time

cos_matches = [docs[i] for i in cos_indices]


lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
minhashes = []

# Build LSH Index
for i, embedding in enumerate(embeddings):
    minhash = MinHash(num_perm=num_perm)
    for value in embedding:
        minhash.update(str(value).encode('utf8'))
    lsh.insert(str(i), minhash)
    minhashes.append(minhash)

# Query LSH
query_minhash = MinHash(num_perm=num_perm)
for value in query_embedding.flatten():
    query_minhash.update(str(value).encode('utf8'))

start_time = time.time()
lsh_matches = list(lsh.query(query_minhash))
lsh_time = time.time() - start_time

lsh_matches = [docs[int(i)] for i in lsh_matches if i.isdigit() and int(i) < len(docs)][:top_k]

#Print Results 
print("Similarity Search Comparison")
print(f"Query: '{query}'\n")

print(f"⏳ KD-Tree Search Time: {kd_time:.6f} seconds")
print(f"🔹 KD-Tree Matches: {kd_matches}\n")

print(f"⏳ Cosine Similarity Search Time: {cos_time:.6f} seconds")
print(f"🔹 Cosine Similarity Matches: {cos_matches}\n")

print(f"⏳ LSH (MinHash) Search Time: {lsh_time:.6f} seconds")
print(f"🔹 LSH Matches: {lsh_matches}\n")
