from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# 1. Load transcript chunks
chunks_path = Path("data/processed/episode_01_chunks.json")

with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Create embeddings for all transcript chunks
texts = [chunk["text"] for chunk in chunks]

chunk_embeddings = model.encode(texts)


# 4. Ask a question
question = "What did Khushy say about helping customers?"


# 5. Convert question into an embedding
question_embedding = model.encode(question)


# 6. Compare question with every transcript chunk
scores = cos_sim(question_embedding, chunk_embeddings)[0]


# 7. Get the 3 best matching chunks
top_results = scores.argsort(descending=True)[:3]


# 8. Show results
print("QUESTION:")
print(question)

for rank, index in enumerate(top_results, start=1):

    chunk = chunks[index]

    print(f"\nRESULT {rank}")
    print("Similarity:", float(scores[index]))
    print("Guest:", chunk["guest"])
    print("Timestamp:", chunk["start_time"])
    print("Text:")
    print(chunk["text"])
