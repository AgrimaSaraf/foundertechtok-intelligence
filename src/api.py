from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.multi_episode_rag import ask


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATIC_DIR = PROJECT_ROOT / "static"

INDEX_FILE = STATIC_DIR / "index.html"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="FounderTechTok Intelligence API",
    description=(
        "Grounded multi-episode RAG over "
        "FounderTechTok podcast transcripts"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def root():

    if not INDEX_FILE.exists():

        return {
            "status": "ERROR",
            "message": "Frontend file not found.",
            "expected_path": str(INDEX_FILE),
        }

    return FileResponse(
        INDEX_FILE
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "FounderTechTok Intelligence",
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(
    request: AskRequest
):

    question = request.question.strip()

    if not question:

        return {
            "status": "ERROR",
            "answer": "",
            "citations": [],
            "error": "Question cannot be empty.",
        }


    result = ask(
        question
    )


    return {
        "question":
            question,

        "status":
            result.get(
                "status"
            ),

        "answer":
            result.get(
                "answer"
            ),

        "citations":
            result.get(
                "citations",
                []
            ),

        "retrieval_latency_sec":
            result.get(
                "retrieval_latency_sec"
            ),

        "generation_latency_sec":
            result.get(
                "generation_latency_sec"
            ),

        "total_latency_sec":
            result.get(
                "total_latency_sec"
            ),

        "retrieved_chunk_count":
            result.get(
                "retrieved_chunk_count"
            ),

        "expanded_context_count":
            result.get(
                "expanded_context_count"
            ),

        "error":
            result.get(
                "error"
            ),
    }