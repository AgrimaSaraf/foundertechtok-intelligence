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
