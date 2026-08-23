# FounderTechTok Intelligence

An AI knowledge system for exploring insights from FounderTechTok podcast conversations.

## Goal

FounderTechTok Intelligence will allow users to ask questions across podcast episodes and receive answers grounded in the original conversations.

Example:

> What have FounderTechTok guests said about finding product-market fit?

The system will retrieve relevant podcast segments and generate an answer with:

- episode title
- guest name
- timestamp
- supporting quote

## Planned Architecture

Podcast Episodes
→ Transcription
→ Semantic Chunking
→ Embeddings
→ Hybrid Retrieval
→ Reranking
→ LLM Generation
→ Grounded Citations
→ Evaluation

🎙️ FounderTechTok transcript
          ↓
     Chunking + overlap
          ↓
 Metadata + timestamps
          ↓
       Embeddings
          ↓
        ChromaDB
          ↓
    ┌─────┴─────┐
    ↓           ↓
  BM25        Vector
 keywords     semantics
    ↓           ↓
    └─────┬─────┘
          ↓
      RRF fusion
          ↓
    Hybrid Top 5
          ↓
 Cross-Encoder Reranker
          ↓
 Best 3 evidence chunks
          ↓
        Gemini
          ↓
 Grounded answer + citations


## Status

🚧 Currently under development.

# FounderTechTok Intelligence

FounderTechTok Intelligence is a retrieval-augmented generation (RAG) system built on top of FounderTechTok podcast transcripts.

Instead of treating podcast episodes as passive long-form content, the system converts transcripts into a searchable knowledge base that can retrieve relevant podcast evidence and generate grounded answers to user questions.

The project implements a hybrid retrieval pipeline combining lexical search, semantic search, reciprocal rank fusion, cross-encoder reranking, grounded LLM generation, citation handling, abstention, and automated evaluation.

---

## What It Does

A user can ask a natural-language question about a FounderTechTok episode, such as:

> How do salespeople build trust with customers?

FounderTechTok Intelligence:

1. searches transcript chunks using BM25,
2. performs semantic vector retrieval,
3. combines both rankings using Reciprocal Rank Fusion,
4. reranks the strongest candidates with a cross-encoder,
5. selects the best transcript evidence,
6. sends only that evidence to Gemini,
7. generates an answer grounded in the podcast,
8. cites relevant timestamps,
9. abstains when the retrieved evidence is insufficient.

The goal is not simply to generate plausible answers.

The system is designed to answer questions from the FounderTechTok archive while remaining traceable to the original podcast evidence.

---

# Architecture

The high-level pipeline is:

```text
Podcast Episode
      ↓
Transcript
      ↓
Timestamp-Aware Chunking
      ↓
────────────────────────────
        INDEXING
────────────────────────────
      ↓
Sentence Transformer Embeddings
      ↓
Chroma Vector Store


User Question
      ↓
┌───────────────────────────────┐
│                               │
↓                               ↓
BM25 Search              Vector Search
Lexical Retrieval        Semantic Retrieval
│                               │
└──────────────┬────────────────┘
               ↓
      Reciprocal Rank Fusion
               ↓
        Hybrid Candidates
               ↓
     Cross-Encoder Reranking
               ↓
         Top-3 Evidence
               ↓
      Grounded Generation
            Gemini
               ↓
      Answer + Timestamps
               ↓
     Abstain if unsupported
```

The system therefore separates:

- retrieval,
- ranking,
- generation,
- evaluation.

This makes each component independently testable.

---

# Chunking

Podcast transcripts are divided into smaller timestamp-aware chunks before retrieval.

Each chunk contains metadata such as:

```json
{
    "guest": "Khushy Aggarwal",
    "start_time": "31:05",
    "text": "..."
}
```

Timestamp preservation is important because the system should not only retrieve relevant information but also trace generated answers back to the corresponding point in the podcast.

The processed chunks for Episode 01 are stored in:

```text
data/processed/episode_01_chunks.json
```

---

# Embeddings

Each transcript chunk is converted into a dense vector representation using:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

Conceptually:

```text
Transcript chunk
      ↓
Embedding model
      ↓
Dense vector
```

These vectors allow semantically similar passages to be retrieved even when the user's question does not contain exactly the same words as the transcript.

---

# BM25

FounderTechTok Intelligence also implements BM25 lexical retrieval using:

```text
rank_bm25
```

BM25 searches for transcript chunks based on token and keyword relevance.

For example, a query containing:

```text
salespeople incentives
```

may retrieve chunks containing those terms directly.

BM25 is useful because exact terminology, names, products, and phrases can sometimes be captured better by lexical retrieval than embeddings alone.

---

# Vector Retrieval

Semantic retrieval is performed against a persistent Chroma vector database.

The query is embedded using the same Sentence Transformer model:

```text
Question
    ↓
all-MiniLM-L6-v2
    ↓
Query embedding
    ↓
Chroma similarity search
    ↓
Top semantic chunks
```

This allows the system to retrieve conceptually relevant passages even when there is limited lexical overlap between the question and transcript.

---

# Reciprocal Rank Fusion

BM25 and vector retrieval produce two different ranked result sets.

FounderTechTok Intelligence combines them using Reciprocal Rank Fusion (RRF).

For a document ranked at position `r`, its contribution is approximately:

```text
1 / (k + r)
```

where this implementation uses:

```text
k = 60
```

Scores from the lexical and semantic rankings are added together.

Conceptually:

```text
BM25 Ranking ─────┐
                  ├──→ RRF → Hybrid Ranking
Vector Ranking ───┘
```

This gives the system the benefits of both exact lexical matching and semantic similarity.

---

# Cross-Encoder Reranking

The strongest hybrid candidates are then reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike the embedding model, which independently embeds the query and documents, the cross-encoder evaluates the question and transcript chunk together.

Each pair takes the form:

```python
[
    question,
    transcript_chunk
]
```

The cross-encoder produces a relevance score for each candidate.

The candidates are sorted by this score, and the top three become the final evidence supplied to the generation layer.

The retrieval pipeline is therefore:

```text
BM25 Top-K
       +
Vector Top-K
       ↓
      RRF
       ↓
Hybrid Candidates
       ↓
Cross-Encoder
       ↓
Top-3 Evidence
```

---

# Gemini Grounded Generation

After retrieval and reranking, only the selected podcast evidence is provided to Gemini.

The model is explicitly instructed to:

- use only the retrieved podcast evidence,
- avoid outside knowledge,
- avoid inventing information,
- cite timestamps,
- abstain when evidence is insufficient.

This creates a separation between what the language model may know generally and what FounderTechTok Intelligence is actually allowed to claim.

Conceptually:

```text
Question
+
Top-3 Transcript Chunks
+
Grounding Instructions
        ↓
      Gemini
        ↓
Grounded Answer
```

A generated answer may look like:

```text
Salespeople build trust by understanding what the customer
needs and helping in a way that feels authentic rather than
pushy [31:05].
```

The timestamp allows the user to trace the answer back to the podcast.

---

# Abstention

A RAG system should not answer every question.

For questions that are unsupported by the podcast archive, FounderTechTok Intelligence is instructed to return:

```text
I don't have enough evidence in the FounderTechTok archive to answer that.
```

For example:

```text
What did Khushy say about nuclear fusion?
```

should not cause the system to use Gemini's general knowledge about nuclear fusion.

Instead, it should recognize that the retrieved FounderTechTok evidence does not support an answer.

Abstention helps reduce hallucination and keeps the system bounded by its source material.

---

# Evaluation

FounderTechTok Intelligence includes an automated RAG evaluation harness:

```text
src/evaluate_rag.py
```

The current evaluation framework measures four properties.

### 1. Retrieval Hit@3

Tests whether at least one expected transcript timestamp appears in the final top-three retrieved chunks.

```text
Hit@3 =
successful retrieval questions
/
answerable retrieval questions
```

### 2. Abstention Accuracy

Tests whether the system correctly refuses questions that cannot be answered from the podcast evidence.

```text
Abstention Accuracy =
correct abstentions
/
unanswerable questions
```

### 3. Citation Correctness

Generated timestamps are extracted from answers and checked against the timestamps actually retrieved by the RAG pipeline.

This detects cases where the model generates a citation that was not present in its supplied evidence.

### 4. Groundedness

A separate evaluation step checks whether factual claims in a generated answer are supported by the retrieved podcast evidence.

This is intended to detect answers that sound plausible but introduce unsupported claims.

---

# How to Run

Clone the repository and enter the project directory.

Create and activate a Python environment if desired.

Install dependencies:

```bash
pip install -r requirements.txt
```

Build or populate the transcript/vector data as required by the project pipeline.

Then run the retrieval/generation system:

```bash
python3 src/hybrid_rag.py
```

Run the evaluation harness with:

```bash
python3 src/evaluate_rag.py
```

Gemini generation requires a valid Gemini API configuration.

Generation-dependent evaluations may be skipped if the API is unavailable or quota has been exhausted.

---

# Current v1 Results

The current development evaluation dataset contains:

```text
6 answerable questions
4 unanswerable questions
10 total questions
```

The retrieval pipeline currently achieves:

```text
Retrieval Hit@3

Hits: 6
Positive questions: 6
Hit@3: 1.0
```

Therefore, on the current Episode 01 development evaluation set:

```text
Hit@3 = 100%
```

This means the expected supporting transcript region appeared within the final three retrieved chunks for all six answerable development questions.

Generation-dependent metrics are not reported as final until a complete Gemini evaluation run succeeds.

The project evaluates those separately as:

```text
Abstention Accuracy
Citation Correctness
Groundedness
```

The current dataset is small and based on a single podcast episode, so these results should be interpreted as development-set performance rather than evidence of general performance across the full FounderTechTok archive.

---

# Limitations

FounderTechTok Intelligence v1 has several important limitations.

### Small evaluation dataset

The current evaluation contains only a small number of manually constructed questions.

A 100% Hit@3 result on six positive questions should therefore not be interpreted as 100% retrieval accuracy in general.

### Single-episode evaluation

The current evaluation primarily tests Episode 01.

Performance may change as additional guests, topics, speaking styles, and episodes are added.

### Timestamp-level relevance

Evaluation currently relies on expected transcript timestamps.

Some questions may legitimately have supporting evidence across multiple neighboring chunks.

### Model-dependent generation

Answer generation and some evaluation stages depend on an external language model API.

API availability, quota, and model changes can affect generation experiments.

### LLM-as-judge evaluation

Groundedness evaluation uses a language model as a judge.

This is useful for automated testing but should not be treated as a perfect substitute for human evaluation.

### Basic tokenization

The current BM25 implementation uses relatively simple text tokenization.

More advanced normalization or tokenization could improve lexical retrieval.

---

# Future Work

Future versions can expand FounderTechTok Intelligence from a single-episode RAG prototype into a larger podcast intelligence system.

Potential directions include:

- indexing the complete FounderTechTok archive,
- episode-level and guest-level metadata filtering,
- improved transcript chunking,
- larger evaluation datasets,
- held-out evaluation questions,
- Recall@K and MRR retrieval metrics,
- answer relevance evaluation,
- human evaluation,
- confidence scoring,
- improved abstention thresholds,
- query rewriting,
- metadata-aware retrieval,
- conversational search across episodes,
- guest discovery,
- topic clustering,
- cross-episode synthesis,
- source-linked podcast playback,
- web/API interface,
- production deployment.

A future query could therefore move beyond a single episode:

```text
What have FounderTechTok guests said about
how AI is changing software engineering?
```

The system could retrieve evidence across multiple conversations, identify recurring ideas, compare perspectives, and link every claim back to the original podcast evidence.

---

# Project Structure

```text
foundertechtok-intelligence/
│
├── data/
│   ├── chroma_db/
│   ├── evaluation/
│   ├── processed/
│   │   └── episode_01_chunks.json
│   └── transcripts/
│
├── src/
│   ├── bm25_search.py
│   ├── build_vector_store.py
│   ├── chunk_transcript.py
│   ├── chunk_with_timestamps.py
│   ├── create_embeddings.py
│   ├── evaluate_rag.py
│   ├── hybrid_rag.py
│   ├── hybrid_search.py
│   ├── load_transcript.py
│   ├── query_vector_store.py
│   ├── rag_answer.py
│   ├── rerank_results.py
│   ├── save_chunks.py
│   ├── semantic_search.py
│   └── test_gemini.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Status

**FounderTechTok Intelligence v1 — evaluation and reproducibility stage.**

Current verified retrieval result:

```text
Hit@3: 1.0
```

Final generation metrics will be recorded after completing the cached generation, abstention, citation, and groundedness evaluation run.