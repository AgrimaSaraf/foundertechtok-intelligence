import chromadb

from google import genai
from sentence_transformers import SentenceTransformer


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="foundertechtok"
)


# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Connect to Gemini
gemini_client = genai.Client()

# Ask the user a question
question = input("\nAsk FounderTechTok Intelligence a question: ")


# Convert the question into an embedding
question_embedding = embedding_model.encode(question)


# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=3
)


# Show retrieved chunks
for i in range(len(results["documents"][0])):

    print(f"\nRESULT {i + 1}")

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

    # Build context from retrieved chunks
context_parts = []

for i in range(len(results["documents"][0])):

    guest = results["metadatas"][0][i]["guest"]
    timestamp = results["metadatas"][0][i]["start_time"]
    text = results["documents"][0][i]

    context_parts.append(
        f"""
Source {i + 1}
Guest: {guest}
Timestamp: {timestamp}
Transcript:
{text}
"""
    )


context = "\n".join(context_parts)

prompt = f"""
You are FounderTechTok Intelligence.

Answer the user's question using ONLY the podcast transcript context below.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information, say:
   "I don't have enough evidence in the podcast archive to answer that."
3. Keep the answer concise and clear.
4. Cite supporting sources using the guest name and timestamp.

Question:
{question}

Podcast Context:
{context}
"""

response = gemini_client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\nFOUNDERTECHTOK INTELLIGENCE ANSWER:\n")
print(response.text)

