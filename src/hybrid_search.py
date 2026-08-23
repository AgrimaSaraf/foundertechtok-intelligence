import json
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

chunks_path = Path("data/processed/episode_01_chunks.json")

with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]

# BM25 setup
tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)


# Vector search setup
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = chroma_client.get_collection(
    name="foundertechtok"
)

# Ask a question
question = input("\nAsk FounderTechTok Intelligence a question: ")

# -------------------------
# BM25 SEARCH
# -------------------------

tokenized_question = question.lower().split()

bm25_scores = bm25.get_scores(tokenized_question)

bm25_top_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:5]

# -------------------------
# VECTOR SEARCH
# -------------------------

question_embedding = embedding_model.encode(question)

vector_results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=5
)

vector_ids = vector_results["ids"][0]

vector_top_indices = []

for vector_id in vector_ids:

    chunk_number = int(
        vector_id.split("_")[-1]
    )

    index = chunk_number - 1

    vector_top_indices.append(index)

    # -------------------------
# RECIPROCAL RANK FUSION
# -------------------------

rrf_scores = {}

k = 60 ###a common RRF constant. Don't overthink it for now.

for rank, index in enumerate(
    bm25_top_indices,
    start=1
):

    rrf_scores[index] = (
        rrf_scores.get(index, 0)
        + 1 / (k + rank)
    )

for rank, index in enumerate(
    vector_top_indices,
    start=1
):
    rrf_scores[index] = (
        rrf_scores.get(index, 0)
        + 1 / (k + rank)
    )

    hybrid_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)[:5]

print("\nHYBRID SEARCH RESULTS:")

for rank, (index, score) in enumerate(
    hybrid_results,
    start=1
):

    chunk = chunks[index]

    print(f"\nRESULT {rank}")
    print("RRF Score:", score)
    print("Guest:", chunk["guest"])
    print("Timestamp:", chunk["start_time"])
    print("Text:")
    print(chunk["text"])



