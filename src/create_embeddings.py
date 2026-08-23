from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


# 1. Load processed transcript chunks
chunks_path = Path("data/processed/episode_01_chunks.json")

with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Take the text from every chunk
texts = []

for chunk in chunks:
    texts.append(chunk["text"])


# 4. Convert text into embeddings
embeddings = model.encode(texts)


# 5. Inspect the result
print("Number of chunks:", len(chunks))
print("Number of embeddings:", len(embeddings))

print("\nSize of one embedding:")
print(len(embeddings[0]))

print("\nFirst 10 numbers of first embedding:")
print(embeddings[0][:10])