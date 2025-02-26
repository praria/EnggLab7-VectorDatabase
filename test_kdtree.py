import numpy as np

query = np.array([0.0, 0.3, 0.5])
vectors = np.array([[0.8, 0.2, 0.1],
                    [0.3, 0.7, 0.5],
                    [0.6, 0.4, 0.3],
                    [0.2, 0.9, 0.8],
                    [0.5, 0.5, 0.5]])

distances = np.linalg.norm(vectors - query, axis=1)
closest_index = np.argmin(distances)

print(f"Exact Nearest Neighbor Index: {closest_index}")
print(f"Exact Nearest Neighbor: {vectors[closest_index]}")
