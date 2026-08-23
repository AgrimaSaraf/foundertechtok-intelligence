import json
from pathlib import Path

import chromadb
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load transcript chunks
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

# Reranker setup
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Gemini setup
gemini_client = genai.Client()

# Ask the user a question
question = input(
    "\nAsk FounderTechTok Intelligence a question: "
)

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

    vector_top_indices.append(
        chunk_number - 1
    )

print("\nBM25 TOP INDICES:")
print(bm25_top_indices)

print("\nVECTOR TOP INDICES:")
print(vector_top_indices)

# -------------------------
# RRF FUSION
# -------------------------

rrf_scores = {}
k = 60

for rank, index in enumerate(bm25_top_indices, start=1):
    rrf_scores[index] = (
        rrf_scores.get(index, 0)
        + 1 / (k + rank)
    )

for rank, index in enumerate(vector_top_indices, start=1):
    rrf_scores[index] = (
        rrf_scores.get(index, 0)
        + 1 / (k + rank)
    )

hybrid_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)[:5]

# -------------------------
# RERANKING
# -------------------------

pairs = []

for index, _ in hybrid_results:
    pairs.append(
        [
            question,
            chunks[index]["text"]
        ]
    )

rerank_scores = reranker.predict(pairs)

reranked = []

for i, (index, _) in enumerate(hybrid_results):
    reranked.append(
        (
            index,
            float(rerank_scores[i])
        )
    )

reranked = sorted(
    reranked,
    key=lambda item: item[1],
    reverse=True
)[:3]

print("\nFINAL EVIDENCE CHUNKS:")

for rank, (index, score) in enumerate(
    reranked,
    start=1
):

    chunk = chunks[index]

    print(f"\nRESULT {rank}")
    print("Reranker Score:", score)
    print("Guest:", chunk["guest"])
    print("Timestamp:", chunk["start_time"])
    print("Text:")
    print(chunk["text"])

# -------------------------
# BUILD CONTEXT
# -------------------------

context_parts = []

for rank, (index, score) in enumerate(
    reranked,
    start=1
):
    chunk = chunks[index]

    context_parts.append(
        f"""
Source {rank}
Guest: {chunk["guest"]}
Timestamp: {chunk["start_time"]}
Transcript:
{chunk["text"]}
"""
    )

context = "\n".join(context_parts)

# -------------------------
# BUILD PROMPT
# -------------------------

prompt = f"""
You are FounderTechTok Intelligence.

Answer the user's question using ONLY the podcast excerpts provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the excerpts do not contain enough information to answer the question,
   respond exactly:

   I don't have enough evidence in the FounderTechTok archive to answer that.

4. If the answer is supported, answer clearly and cite the relevant
   guest name and timestamp.

QUESTION:
{question}

PODCAST EXCERPTS:
{context}
"""
# -------------------------
# GENERATE ANSWER
# -------------------------

response = gemini_client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\nFOUNDERTECHTOK INTELLIGENCE ANSWER:\n")
print(response.text)

