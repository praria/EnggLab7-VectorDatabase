from scipy.spatial import KDTree
import numpy as np

query = np.array([0.0, 0.3, 0.5])
vectors = np.array([[0.8, 0.2, 0.1],
                    [0.3, 0.7, 0.5],
                    [0.6, 0.4, 0.3],
                    [0.2, 0.9, 0.8],
                    [0.5, 0.5, 0.5]])

# Build KD-Tree
kd_tree = KDTree(vectors)

# Query KD-Tree for the nearest neighbor
distance, index = kd_tree.query(query)

print(f"KD-Tree Nearest Neighbor Index: {index}")
print(f"KD-Tree Nearest Neighbor: {vectors[index]}")
