from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer


# 1. Load our processed transcript chunks
chunks_path = Path("data/processed/episode_01_chunks.json")

with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Loaded chunks:", len(chunks))


# 2. Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Extract text from every chunk
texts = [chunk["text"] for chunk in chunks]


# 4. Turn the chunks into embeddings
embeddings = model.encode(texts)


# 5. Create a persistent Chroma database
client = chromadb.PersistentClient(
    path="data/chroma_db"
)


# 6. Create our FounderTechTok collection
collection = client.get_or_create_collection(
    name="foundertechtok"
)


# 7. Create IDs for every chunk
ids = [
    f"episode_{chunk['episode']}_chunk_{chunk['chunk_id']}"
    for chunk in chunks
]


# 8. Create metadata
metadatas = [
    {
        "episode": chunk["episode"],
        "guest": chunk["guest"],
        "start_time": chunk["start_time"]
    }
    for chunk in chunks
]


# 9. Store everything in Chroma
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


print("Vector database created successfully!")
print("Vectors stored:", collection.count())