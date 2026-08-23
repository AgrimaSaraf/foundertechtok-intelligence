import json
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------
# 1. Load transcript chunks
# --------------------------------

chunks_path = Path("data/processed/episode_01_chunks.json")

with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]


# --------------------------------
# 2. Set up BM25
# --------------------------------

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)


# --------------------------------
# 3. Set up vector search
# --------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = chroma_client.get_collection(
    name="foundertechtok"
)


# --------------------------------
# 4. Load reranker
# --------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------
# 5. Ask question
# --------------------------------

question = input(
    "\nAsk FounderTechTok Intelligence a question: "
)


# --------------------------------
# 6. BM25 search
# --------------------------------

tokenized_question = question.lower().split()

bm25_scores = bm25.get_scores(
    tokenized_question
)

bm25_top_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:5]


# --------------------------------
# 7. Vector search
# --------------------------------

question_embedding = embedding_model.encode(
    question
)

vector_results = collection.query(
    query_embeddings=[
        question_embedding.tolist()
    ],
    n_results=5
)

vector_ids = vector_results["ids"][0]

vector_top_indices = []

for vector_id in vector_ids:

    chunk_number = int(
        vector_id.split("_")[-1]
    )

    vector_top_indices.append(
        chunk_number - 1
    )


# --------------------------------
# 8. Reciprocal Rank Fusion
# --------------------------------

rrf_scores = {}

k = 60


# BM25 rankings
for rank, index in enumerate(
    bm25_top_indices,
    start=1
):

    rrf_scores[index] = (
        rrf_scores.get(index, 0)
        + 1 / (k + rank)
    )


# Vector rankings
for rank, index in enumerate(
    vector_top_indices,
    start=1
):

    rrf_scores[index] = (
        rrf_scores.get(index, 0)
        + 1 / (k + rank)
    )


# Take top 5 hybrid candidates
hybrid_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)[:5]


# --------------------------------
# 9. Build question-document pairs
# --------------------------------

pairs = []

for index, _ in hybrid_results:

    pairs.append(
        [
            question,
            chunks[index]["text"]
        ]
    )


# --------------------------------
# 10. Rerank
# --------------------------------

rerank_scores = reranker.predict(
    pairs
)


reranked = []

for i, (index, _) in enumerate(
    hybrid_results
):

    reranked.append(
        (
            index,
            float(rerank_scores[i])
        )
    )


# Best results first
reranked = sorted(
    reranked,
    key=lambda item: item[1],
    reverse=True
)[:3]


# --------------------------------
# 11. Display results
# --------------------------------

print("\nRERANKED RESULTS:")

for rank, (index, score) in enumerate(
    reranked,
    start=1
):

    chunk = chunks[index]

    print(f"\nRESULT {rank}")

    print(
        "Reranker Score:",
        score
    )

    print(
        "Guest:",
        chunk["guest"]
    )

    print(
        "Timestamp:",
        chunk["start_time"]
    )

    print("Text:")

    print(
        chunk["text"]
    )