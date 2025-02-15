from vector_searchEngine import VectorSearchEngine
from sentence_transformers import SentenceTransformer

# Create a sentence embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize VectorSearchEngine
search_engine = VectorSearchEngine(embedder)

# Sample documents
docs = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
]

# Test document ingestion
print(search_engine.from_docs(docs))

# Test similarity search
query = "how does a journey begin?"
matches = search_engine.similarity_search(query, top_k=2)
print("\nQuery:", query)
print("Top Matches:")
for match in matches:
    print(f"- {match}")