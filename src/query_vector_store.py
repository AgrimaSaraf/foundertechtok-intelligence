import chromadb
from sentence_transformers import SentenceTransformer


# 1. Connect to the existing Chroma database
client = chromadb.PersistentClient(
    path="data/chroma_db"
)


# 2. Open our FounderTechTok collection
collection = client.get_collection(
    name="foundertechtok"
)


# 3. Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Ask the user a question
question = input("\nAsk FounderTechTok Intelligence a question: ")


# 5. Convert ONLY the question into an embedding
question_embedding = model.encode(question)


# 6. Search Chroma for the 3 closest chunks
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=3
)


# 7. Display the results
print("\nQUESTION:")
print(question)


for i in range(len(results["documents"][0])):

    print(f"\nRESULT {i + 1}")

    print(
        "Distance:",
        results["distances"][0][i]
    )

    print(
        "Guest:",
        results["metadatas"][0][i]["guest"]
    )

    print(
        "Timestamp:",
        results["metadatas"][0][i]["start_time"]
    )

    print("Text:")
    print(results["documents"][0][i])