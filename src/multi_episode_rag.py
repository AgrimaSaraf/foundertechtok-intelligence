# ============================================================
# FounderTechTok Intelligence
# Multi-Episode Grounded RAG
#
# Retrieval:
# BM25 + dense retrieval
# -> Reciprocal Rank Fusion
# -> Cross-encoder reranking
# -> Neighbor context expansion
# -> Short-gap transcript span filling
#
# Generation:
# Gemini grounded ONLY in expanded podcast evidence
#
# Reliability:
# Explicit abstention
# Citation validation
# Quote validation
# Retry/backoff for Gemini 429 errors
# Distinguish API failures from model failures
# ============================================================

import json
import os
import re
import time
import random

from google import genai

from src.multi_episode_search import (
    chunks,
    search,
)


# ============================================================
# CONFIG
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


ABSTENTION_MESSAGE = (
    "I don't have enough evidence in the "
    "FounderTechTok archive to answer that."
)


MAX_GEMINI_RETRIES = 5

BASE_RETRY_DELAY_SEC = 2


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_api_key = os.getenv(
    "GEMINI_API_KEY"
)


if not gemini_api_key:

    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Export it before running this script."
    )


gemini_client = genai.Client(
    api_key=gemini_api_key
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize whitespace and casing
    for quote validation.
    """

    if not text:

        return ""


    text = text.lower()


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================
# BUILD EVIDENCE FROM EXPANDED CONTEXT
# ============================================================

def build_expanded_evidence(
    expanded_context_indices
):
    """
    Convert expanded transcript-context indices into
    deterministic evidence objects.

    IMPORTANT:

    Retrieval ranking and generation evidence are now
    intentionally separate.

    Retrieval metrics continue to use:

        retrieval_results["reranked"]

    Gemini receives:

        retrieval_results["expanded_context"]

    This allows neighboring transcript chunks and
    short-gap span filling to improve answer grounding
    without artificially changing Hit@K / MRR.
    """

    evidence = []


    seen = set()


    for index in expanded_context_indices:

        if index in seen:

            continue


        seen.add(
            index
        )


        chunk = chunks[
            index
        ]


        rank = (
            len(
                evidence
            )
            + 1
        )


        source = {
            "source_id":
                f"S{rank}",

            "chunk_index":
                index,

            "chunk_id":
                chunk[
                    "chunk_id"
                ],

            "episode_id":
                chunk[
                    "episode_id"
                ],

            "episode_title":
                chunk.get(
                    "episode_title",
                    ""
                ),

            "guest":
                chunk.get(
                    "guest",
                    ""
                ),

            "start_time":
                chunk[
                    "start_time"
                ],

            "end_time":
                chunk[
                    "end_time"
                ],

            "start_seconds":
                chunk[
                    "start_seconds"
                ],

            "end_seconds":
                chunk.get(
                    "end_seconds",
                    chunk[
                        "start_seconds"
                    ]
                ),

            "text":
                chunk[
                    "text"
                ],
        }


        evidence.append(
            source
        )


    return evidence


# ============================================================
# FORMAT EVIDENCE FOR GEMINI
# ============================================================

def evidence_to_prompt_text(
    evidence
):
    """
    Convert expanded evidence into prompt-readable text.
    """

    parts = []


    for source in evidence:

        parts.append(
            f"""
SOURCE_ID: {source["source_id"]}
EPISODE: {source["episode_title"]}
GUEST: {source["guest"]}
TIMESTAMP: {source["start_time"]} - {source["end_time"]}
TRANSCRIPT:
{source["text"]}
""".strip()
        )


    return "\n\n".join(
        parts
    )


# ============================================================
# BUILD GROUNDED PROMPT
# ============================================================

def build_prompt(
    question,
    evidence
):
    """
    Gemini must answer only from supplied
    FounderTechTok transcript evidence.
    """

    evidence_text = (
        evidence_to_prompt_text(
            evidence
        )
    )


    prompt = f"""
You are FounderTechTok Intelligence.

You answer questions ONLY using the supplied evidence
retrieved from the FounderTechTok podcast archive.

==================================================
STRICT RULES
==================================================

1. Use ONLY the podcast evidence supplied below.

2. Do NOT use outside knowledge.

3. Do NOT infer facts that are not clearly supported
   by the supplied transcript evidence.

4. The supplied evidence may contain neighboring transcript
   chunks included to preserve conversational context.

5. Ignore any supplied passages that are irrelevant to
   the user's question.

6. If the supplied evidence is insufficient to answer
   the question confidently, return:

   "status": "INSUFFICIENT_EVIDENCE"

7. If the evidence supports an answer, return:

   "status": "ANSWER"

8. Every substantive answer must cite one or more
   SOURCE_ID values from the supplied evidence.

9. For every citation, include a SHORT VERBATIM QUOTE
   copied directly from the cited transcript.

10. Never invent a quote.

11. Never invent a SOURCE_ID.

12. If multiple guests provide relevant evidence,
    synthesize them carefully and preserve differences
    in perspective.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

If answerable:

{{
  "status": "ANSWER",
  "answer": "A concise evidence-grounded answer.",
  "citations": [
    {{
      "source_id": "S1",
      "quote": "exact short quote copied from transcript"
    }}
  ]
}}

If unanswerable:

{{
  "status": "INSUFFICIENT_EVIDENCE",
  "answer": "",
  "citations": []
}}

==================================================
USER QUESTION
==================================================

{question}

==================================================
PODCAST EVIDENCE
==================================================

{evidence_text}
"""


    return prompt.strip()


# ============================================================
# PARSE GEMINI JSON
# ============================================================

def parse_json_response(
    response_text
):
    """
    Parse JSON while tolerating accidental
    Markdown code fences.
    """

    text = response_text.strip()


    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r"^```\s*",
        "",
        text
    )


    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    return json.loads(
        text.strip()
    )


# ============================================================
# GEMINI RETRY WRAPPER
# ============================================================

def generate_with_retry(
    generate_fn,
    max_retries=MAX_GEMINI_RETRIES,
    base_delay=BASE_RETRY_DELAY_SEC
):
    """
    Retry transient Gemini 429 / RESOURCE_EXHAUSTED
    errors using exponential backoff.

    Non-rate-limit errors are raised immediately.
    """

    for attempt in range(
        max_retries
    ):

        try:

            return generate_fn()


        except Exception as error:

            error_text = str(
                error
            )


            is_rate_limit = (
                "429" in error_text
                or
                "RESOURCE_EXHAUSTED"
                in error_text
            )


            if not is_rate_limit:

                raise


            if (
                attempt
                == max_retries - 1
            ):

                raise


            wait_seconds = (
                base_delay
                * (
                    2 ** attempt
                )
                + random.uniform(
                    0,
                    1
                )
            )


            print(
                f"Gemini rate limited. "
                f"Retry "
                f"{attempt + 1}/"
                f"{max_retries - 1} "
                f"in "
                f"{wait_seconds:.1f}s..."
            )


            time.sleep(
                wait_seconds
            )


# ============================================================
# VALIDATE CITATIONS
# ============================================================

def validate_citations(
    generation,
    evidence
):
    """
    Validate:

    1. cited SOURCE_ID exists
    2. supporting quote literally appears
       in the corresponding transcript chunk
    """

    evidence_map = {
        source[
            "source_id"
        ]:
            source
        for source in evidence
    }


    valid_citations = []

    invalid_citations = []


    for citation in generation.get(
        "citations",
        []
    ):

        source_id = citation.get(
            "source_id"
        )


        quote = citation.get(
            "quote",
            ""
        ).strip()


        # ----------------------------------------------------
        # Validate source ID
        # ----------------------------------------------------

        if source_id not in evidence_map:

            invalid_citations.append(
                {
                    "reason":
                        "unknown_source_id",

                    "source_id":
                        source_id,

                    "quote":
                        quote,
                }
            )


            continue


        source = evidence_map[
            source_id
        ]


        # ----------------------------------------------------
        # Validate quote
        # ----------------------------------------------------

        normalized_quote = normalize_text(
            quote
        )


        normalized_source = normalize_text(
            source[
                "text"
            ]
        )


        if (
            not normalized_quote
            or
            normalized_quote
            not in normalized_source
        ):

            invalid_citations.append(
                {
                    "reason":
                        "quote_not_found",

                    "source_id":
                        source_id,

                    "quote":
                        quote,
                }
            )


            continue


        valid_citations.append(
            {
                "source_id":
                    source_id,

                "episode_id":
                    source[
                        "episode_id"
                    ],

                "episode_title":
                    source[
                        "episode_title"
                    ],

                "guest":
                    source[
                        "guest"
                    ],

                "start_time":
                    source[
                        "start_time"
                    ],

                "end_time":
                    source[
                        "end_time"
                    ],

                "chunk_id":
                    source[
                        "chunk_id"
                    ],

                "quote":
                    quote,
            }
        )


    return (
        valid_citations,
        invalid_citations
    )


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(
    question,
    evidence
):
    """
    Call Gemini using ONLY expanded
    FounderTechTok evidence.

    Retries temporary rate-limit failures.
    """

    prompt = build_prompt(
        question,
        evidence
    )


    generation_start = (
        time.perf_counter()
    )


    def call_gemini():

        return (
            gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        )


    response = generate_with_retry(
        call_gemini,
        max_retries=MAX_GEMINI_RETRIES,
        base_delay=BASE_RETRY_DELAY_SEC
    )


    generation_latency = (
        time.perf_counter()
        - generation_start
    )


    raw_text = (
        response.text
        if response.text
        else ""
    )


    generation = parse_json_response(
        raw_text
    )


    return (
        generation,
        generation_latency,
        response
    )


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def ask(
    question
):
    """
    Complete multi-episode RAG pipeline:

        question
          ↓
        BM25 Top-20
          +
        dense Top-20
          ↓
        RRF Top-20
          ↓
        cross-encoder Top-5
          ↓
        neighbor expansion
          ↓
        short-gap span filling
          ↓
        expanded generation evidence
          ↓
        Gemini
          ↓
        citation validation
          ↓
        answer / abstention / API error

    Retrieval ranking remains separate from
    generation-context expansion.
    """

    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieval_start = (
        time.perf_counter()
    )


    retrieval_results = search(
        question
    )


    retrieval_latency = (
        time.perf_counter()
        - retrieval_start
    )


    # --------------------------------------------------------
    # Keep raw reranked retrieval results available.
    #
    # These remain the correct results for Hit@K / MRR.
    # --------------------------------------------------------

    reranked = retrieval_results.get(
        "reranked",
        []
    )


    # --------------------------------------------------------
    # Expanded context is for generation only.
    # --------------------------------------------------------

    expanded_context = (
        retrieval_results.get(
            "expanded_context",
            []
        )
    )


    # --------------------------------------------------------
    # Defensive fallback.
    #
    # If expanded_context is unexpectedly unavailable,
    # use the raw reranked chunk indices instead of
    # crashing.
    # --------------------------------------------------------

    if not expanded_context:

        expanded_context = [
            index
            for index, _
            in reranked
        ]


    evidence = build_expanded_evidence(
        expanded_context
    )


    # ========================================================
    # NO RETRIEVAL EVIDENCE
    # ========================================================

    if not evidence:

        total_latency = (
            time.perf_counter()
            - total_start
        )


        return {
            "status":
                "INSUFFICIENT_EVIDENCE",

            "answer":
                ABSTENTION_MESSAGE,

            "citations":
                [],

            "invalid_citations":
                [],

            "error":
                None,

            "retrieval_latency_sec":
                round(
                    retrieval_latency,
                    3
                ),

            "generation_latency_sec":
                0.0,

            "total_latency_sec":
                round(
                    total_latency,
                    3
                ),

            "retrieved_chunk_count":
                len(
                    reranked
                ),

            "expanded_context_count":
                0,

            "evidence":
                [],
        }


    # ========================================================
    # GENERATION
    # ========================================================

    try:

        (
            generation,
            generation_latency,
            raw_response
        ) = generate_answer(
            question,
            evidence
        )


    except Exception as error:

        total_latency = (
            time.perf_counter()
            - total_start
        )


        error_text = str(
            error
        )


        is_rate_limit = (
            "429" in error_text
            or
            "RESOURCE_EXHAUSTED"
            in error_text
        )


        status = (
            "API_ERROR"
            if is_rate_limit
            else "GENERATION_ERROR"
        )


        return {
            "status":
                status,

            "answer":
                "",

            "citations":
                [],

            "invalid_citations":
                [],

            "error":
                error_text,

            "retrieval_latency_sec":
                round(
                    retrieval_latency,
                    3
                ),

            "generation_latency_sec":
                None,

            "total_latency_sec":
                round(
                    total_latency,
                    3
                ),

            "retrieved_chunk_count":
                len(
                    reranked
                ),

            "expanded_context_count":
                len(
                    evidence
                ),

            "evidence":
                evidence,
        }


    # ========================================================
    # MODEL STATUS
    # ========================================================

    status = (
        generation.get(
            "status",
            ""
        )
        .strip()
        .upper()
    )


    # ========================================================
    # MODEL ABSTENTION
    # ========================================================

    if (
        status
        == "INSUFFICIENT_EVIDENCE"
    ):

        total_latency = (
            time.perf_counter()
            - total_start
        )


        return {
            "status":
                "INSUFFICIENT_EVIDENCE",

            "answer":
                ABSTENTION_MESSAGE,

            "citations":
                [],

            "invalid_citations":
                [],

            "error":
                None,

            "retrieval_latency_sec":
                round(
                    retrieval_latency,
                    3
                ),

            "generation_latency_sec":
                round(
                    generation_latency,
                    3
                ),

            "total_latency_sec":
                round(
                    total_latency,
                    3
                ),

            "retrieved_chunk_count":
                len(
                    reranked
                ),

            "expanded_context_count":
                len(
                    evidence
                ),

            "evidence":
                evidence,
        }


    # ========================================================
    # UNEXPECTED MODEL STATUS
    # ========================================================

    if status != "ANSWER":

        total_latency = (
            time.perf_counter()
            - total_start
        )


        return {
            "status":
                "GENERATION_ERROR",

            "answer":
                "",

            "citations":
                [],

            "invalid_citations":
                [],

            "error":
                (
                    "Unexpected model status: "
                    f"{status}"
                ),

            "retrieval_latency_sec":
                round(
                    retrieval_latency,
                    3
                ),

            "generation_latency_sec":
                round(
                    generation_latency,
                    3
                ),

            "total_latency_sec":
                round(
                    total_latency,
                    3
                ),

            "retrieved_chunk_count":
                len(
                    reranked
                ),

            "expanded_context_count":
                len(
                    evidence
                ),

            "evidence":
                evidence,
        }


    # ========================================================
    # VALIDATE CITATIONS
    # ========================================================

    (
        valid_citations,
        invalid_citations
    ) = validate_citations(
        generation,
        evidence
    )


    # ========================================================
    # FAIL CLOSED IF NO VALID CITATION
    # ========================================================

    if not valid_citations:

        total_latency = (
            time.perf_counter()
            - total_start
        )


        return {
            "status":
                "INSUFFICIENT_VALIDATED_EVIDENCE",

            "answer":
                ABSTENTION_MESSAGE,

            "citations":
                [],

            "invalid_citations":
                invalid_citations,

            "error":
                None,

            "retrieval_latency_sec":
                round(
                    retrieval_latency,
                    3
                ),

            "generation_latency_sec":
                round(
                    generation_latency,
                    3
                ),

            "total_latency_sec":
                round(
                    total_latency,
                    3
                ),

            "retrieved_chunk_count":
                len(
                    reranked
                ),

            "expanded_context_count":
                len(
                    evidence
                ),

            "evidence":
                evidence,
        }


    # ========================================================
    # SUCCESS
    # ========================================================

    total_latency = (
        time.perf_counter()
        - total_start
    )


    return {
        "status":
            "ANSWER",

        "answer":
            generation.get(
                "answer",
                ""
            ).strip(),

        "citations":
            valid_citations,

        "invalid_citations":
            invalid_citations,

        "error":
            None,

        "retrieval_latency_sec":
            round(
                retrieval_latency,
                3
            ),

        "generation_latency_sec":
            round(
                generation_latency,
                3
            ),

        "total_latency_sec":
            round(
                total_latency,
                3
            ),

        "retrieved_chunk_count":
            len(
                reranked
            ),

        "expanded_context_count":
            len(
                evidence
            ),

        "evidence":
            evidence,
    }


# ============================================================
# PRETTY OUTPUT
# ============================================================

def print_answer(
    question,
    result
):

    print(
        "\n================================"
    )


    print(
        "FOUNDERTECHTOK INTELLIGENCE"
    )


    print(
        "================================"
    )


    print(
        "\nQuestion:"
    )


    print(
        question
    )


    print(
        "\nStatus:"
    )


    print(
        result[
            "status"
        ]
    )


    # ========================================================
    # ERROR
    # ========================================================

    if result[
        "status"
    ] in {
        "API_ERROR",
        "GENERATION_ERROR"
    }:

        print(
            "\nGeneration failed:"
        )


        print(
            result.get(
                "error"
            )
        )


        print(
            "\nRetrieval still completed successfully."
        )


        print(
            "Retrieved chunks:",
            result.get(
                "retrieved_chunk_count"
            )
        )


        print(
            "Expanded context chunks:",
            result.get(
                "expanded_context_count"
            )
        )


        return


    # ========================================================
    # ANSWER
    # ========================================================

    print(
        "\nAnswer:"
    )


    print(
        result[
            "answer"
        ]
    )


    # ========================================================
    # CONTEXT INFO
    # ========================================================

    print(
        "\nRetrieved chunks:",
        result.get(
            "retrieved_chunk_count"
        )
    )


    print(
        "Expanded context chunks:",
        result.get(
            "expanded_context_count"
        )
    )


    # ========================================================
    # CITATIONS
    # ========================================================

    citations = result.get(
        "citations",
        []
    )


    if citations:

        print(
            "\n================================"
        )


        print(
            "VALIDATED SOURCES"
        )


        print(
            "================================"
        )


        for number, citation in enumerate(
            citations,
            start=1
        ):

            print(
                f"\n[{number}]"
            )


            print(
                "Episode:",
                citation[
                    "episode_title"
                ]
            )


            print(
                "Guest:",
                citation[
                    "guest"
                ]
            )


            print(
                "Timestamp:",
                citation[
                    "start_time"
                ],
                "-",
                citation[
                    "end_time"
                ]
            )


            print(
                "Chunk:",
                citation[
                    "chunk_id"
                ]
            )


            print(
                "Supporting quote:"
            )


            print(
                f'"{citation["quote"]}"'
            )


    # ========================================================
    # INVALID CITATIONS
    # ========================================================

    invalid = result.get(
        "invalid_citations",
        []
    )


    if invalid:

        print(
            "\nInvalid citations rejected:",
            len(
                invalid
            )
        )


    # ========================================================
    # LATENCY
    # ========================================================

    print(
        "\n================================"
    )


    print(
        "LATENCY"
    )


    print(
        "================================"
    )


    print(
        "Retrieval:",
        result.get(
            "retrieval_latency_sec"
        ),
        "sec"
    )


    print(
        "Generation:",
        result.get(
            "generation_latency_sec"
        ),
        "sec"
    )


    print(
        "Total:",
        result.get(
            "total_latency_sec"
        ),
        "sec"
    )


# ============================================================
# CLI
# ============================================================

def main():

    print(
        "\nFounderTechTok Intelligence"
    )


    print(
        "Multi-Episode Grounded RAG"
    )


    print(
        "\nRetrieval:"
    )


    print(
        "BM25 + dense"
        " -> RRF"
        " -> cross-encoder"
        " -> context expansion"
    )


    print(
        "\nType 'exit' to quit."
    )


    while True:

        question = input(
            "\nAsk FounderTechTok Intelligence: "
        ).strip()


        if question.lower() in {
            "exit",
            "quit"
        }:

            break


        if not question:

            continue


        result = ask(
            question
        )


        print_answer(
            question,
            result
        )


if __name__ == "__main__":

    main()