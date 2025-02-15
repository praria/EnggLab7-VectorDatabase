# main application runner
from ollama_helper import ask_ollama
from vector_searchEngine import VectorSearchEngine
from sentence_transformers import SentenceTransformer

def load_documents(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().split("\n\n")  # Split by paragraphs
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def main():
    # Load the embedding model and initialize VectorSearchEngine
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    search_engine = VectorSearchEngine(embedder)

    # Load documents from file
    docs = load_documents("example/opportunityCost.txt")
    if not docs:
        print("No documents found. Exiting.")
        return

    search_engine.from_docs(docs)

    print("\nWelcome to the lesson about Opportunity Cost Q&A Bot!")
    print("Ask anything about the document. Type '/bye' to exit.")

    while True:
        query = input("\nAsk anything: ").strip()
        if query.lower() == "/bye":
            print("Goodbye!")
            break

        # Find relevant document chunks
        matches = search_engine.similarity_search(query, top_k=5)
        if not matches:
            print("No relevant information found.")
            continue

        context = " ".join(matches)
        prompt = f"Context: {context}\n\nUser Question: {query}\n\nAnswer:"
        
        # Get AI-generated response
        answer = ask_ollama(prompt)
        print("\nAI Response:")
        print(answer)

if __name__ == "__main__":
    main()
