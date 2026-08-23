from pathlib import Path
import json

from rank_bm25 import BM25Okapi


# Load processed chunks
chunks_path = Path("data/processed/episode_01_chunks.json")

with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# Extract the text
texts = [chunk["text"] for chunk in chunks]

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)

question = input("\nAsk a keyword-style question: ")

tokenized_question = question.lower().split()

scores = bm25.get_scores(tokenized_question)

top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:3]

for rank, index in enumerate(top_indices, start=1):

    chunk = chunks[index]

    print(f"\nRESULT {rank}")
    print("BM25 Score:", scores[index])
    print("Guest:", chunk["guest"])
    print("Timestamp:", chunk["start_time"])
    print("Text:")
    print(chunk["text"])

    