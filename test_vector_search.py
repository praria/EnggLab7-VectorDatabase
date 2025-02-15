import unittest
import numpy as np
from sentence_transformers import SentenceTransformer
from cosine_dotproduct import cosine_similarity
from vector_searchEngine import VectorSearchEngine

class TestCosineSimilarity(unittest.TestCase):
    def setUp(self):
        # Initialize test data for cosine similarity.
        self.store_embeddings = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        self.query_embedding = np.array([1, 2, 3])

    def test_cosine_similarity_correctness(self):
        #Test if cosine similarity returns the correct indices.
        top_k = 2
        result = cosine_similarity(self.store_embeddings, self.query_embedding, top_k)
        
        # Expect index 0 (exact match) and index 1 (next most similar)
        self.assertIn(0, result)
        self.assertIn(1, result)

    def test_cosine_similarity_empty_store(self):
        #Test behavior with an empty store_embeddings array.
        result = cosine_similarity(np.array([]).reshape(0, 3), self.query_embedding, top_k=3)
        self.assertEqual(len(result), 0)  # No documents available

    def test_cosine_similarity_zero_query(self):
        #Test behavior when query embedding is a zero vector.
        zero_query = np.array([0, 0, 0])
        result = cosine_similarity(self.store_embeddings, zero_query, top_k=3)
        self.assertEqual(len(result), 3)  # Should return all indices but with very low similarity

    def test_cosine_similarity_identical_vectors(self):
        #Test case where stored vectors and query are identical.
        identical_vectors = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        query = np.array([3, 3, 3])
        result = cosine_similarity(identical_vectors, query, top_k=3)
        self.assertEqual(result[0], 2)  # Expect exact match to be ranked first

    def test_cosine_similarity_negative_values(self):
        #Test similarity computation with negative values.
        neg_store = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
        neg_query = np.array([-1, -2, -3])
        result = cosine_similarity(neg_store, neg_query, top_k=2)
        self.assertIn(0, result)  # Expect the most similar vector at index 0

    def test_cosine_similarity_high_dimensional_vectors(self):
        #Test performance with high-dimensional vectors (e.g., 512D).
        np.random.seed(42)
        high_dim_store = np.random.rand(1000, 512)
        high_dim_query = np.random.rand(512)
        result = cosine_similarity(high_dim_store, high_dim_query, top_k=5)
        self.assertEqual(len(result), 5)  # Should return top-5 results

class TestVectorSearchEngine(unittest.TestCase):
    def setUp(self):
        """Initialize the Vector Search Engine with a real embedder."""
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.search_engine = VectorSearchEngine(self.embedder)
        self.docs = ["Apple is a fruit", "The sky is blue", "Dogs are loyal animals"]
        self.search_engine.from_docs(self.docs)

    def test_document_storage(self):
        """Ensure documents are stored correctly."""
        self.assertEqual(len(self.search_engine.docs), len(self.docs))

    def test_embedding_generation(self):
        """Test if embeddings are generated correctly."""
        self.assertEqual(self.search_engine.embeddings.shape[0], len(self.docs))

    def test_similarity_search(self):
        """Test if similarity search returns expected results."""
        query = "What color is the sky?"
        results = self.search_engine.similarity_search(query, top_k=2)
        self.assertGreaterEqual(len(results), 1)  # At least one result should be returned
        self.assertIn("The sky is blue", results)  # Expected match

    def test_empty_database(self):
        """Test behavior when no documents are loaded."""
        empty_engine = VectorSearchEngine(self.embedder)
        results = empty_engine.similarity_search("test query", top_k=3)
        self.assertEqual(len(results), 0)  # Should return empty list

    def test_unrelated_query(self):
        """Test a query that has no relevant results."""
        query = "How do airplanes fly?"
        results = self.search_engine.similarity_search(query, top_k=2)
        self.assertEqual(len(results), 2)  # It should still return results, but they may be weakly related

    def test_short_query(self):
        """Test searching with a one-word query."""
        query = "fruit"
        results = self.search_engine.similarity_search(query, top_k=2)
        self.assertGreater(len(results), 0)  # Expect at least one match

    def test_large_corpus_search(self):
        """Stress test with a large corpus of documents."""
        large_docs = ["Doc " + str(i) for i in range(10000)]  # 10,000 documents
        large_engine = VectorSearchEngine(self.embedder)
        large_engine.from_docs(large_docs)

        query = "Find me something"
        results = large_engine.similarity_search(query, top_k=10)
        self.assertEqual(len(results), 10)  # Expect exactly 10 results

if __name__ == "__main__":
    unittest.main()
