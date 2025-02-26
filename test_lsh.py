import numpy as np
from datasketch import MinHash, MinHashLSH

query = np.array([0.0, 0.3, 0.5])

vectors = np.array([
    [0.8, 0.2, 0.1],
    [0.3, 0.7, 0.5],
    [0.6, 0.4, 0.3],
    [0.2, 0.9, 0.8],
    [0.5, 0.5, 0.5]
])

# Number of permutations for MinHash
num_perm = 128

# Create MinHash objects for each vector
minhashes = []
for vector in vectors:
    m = MinHash(num_perm=num_perm)
    for i, value in enumerate(vector):
        # Use the index and value as a feature
        m.update(f"{i}:{value}".encode('utf-8'))
    minhashes.append(m)

# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
for i, minhash in enumerate(minhashes):
    lsh.insert(f"vec{i}", minhash)

# Create MinHash for the query vector
query_minhash = MinHash(num_perm=num_perm)
for i, value in enumerate(query):
    query_minhash.update(f"{i}:{value}".encode('utf-8'))

# Query the LSH index
result = lsh.query(query_minhash)
print(f"Approximate Nearest Neighbor Indices: {result}")

# Compute exact distances for comparison
distances = np.linalg.norm(vectors - query, axis=1)
closest_index = np.argmin(distances)
print(f"Exact Nearest Neighbor Index: {closest_index}")
print(f"Exact Nearest Neighbor: {vectors[closest_index]}")