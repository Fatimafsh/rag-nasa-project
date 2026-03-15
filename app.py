import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persist_directory = "db"


# Build or load vector DB
if not os.path.exists(persist_directory):
    print("Creating database...")

    loader = PyPDFLoader("data/nasa.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = splitter.split_documents(docs)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory
    )

    db.persist()

else:
    print("Loading existing database...")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

# Metric function: Context Relevance
def keyword_feedback(results):
    
    keywords = ["NASA", "space", "Mars", "astronaut", "rocket", "satellite", "ISS", "Apollo", "mission"]
    score = 0
    for chunk in results:
        if any(k.lower() in chunk.page_content.lower() for k in keywords):
            score += 1
    return score / max(len(results), 1)

# Query loop
print("\nRAG ready 🚀")

while True:
    query = input("\nAsk (type exit to quit): ").strip()

    if query.lower() == "exit":
        print("Program stopped.")
        break

    if not query:
        print("Please enter a question.")
        continue

    results = db.similarity_search(query, k=2)

    print("\nRetrieved Chunks:\n")
    for i, r in enumerate(results):
        print(f"Result {i+1}:")
        print(r.page_content)
        print("-" * 50)

    # Evaluate Context Relevance
    score = keyword_feedback(results)
    print(f"Context Relevance Score: {score:.2f}")