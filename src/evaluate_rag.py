# FounderTechTok Intelligence
# Complete RAG Evaluation Harness
#
# Evaluates:
# 1. Retrieval Hit@3
# 2. Abstention accuracy
# 3. Citation correctness
# 4. Groundedness / faithfulness
#
# Retrieval pipeline:
# BM25 + Dense Vector Search
# -> Reciprocal Rank Fusion
# -> Cross-Encoder Reranking
# -> Top-3 Evidence
#
# Generation:
# Gemini
#
# Includes:
# - Answer caching
# - Groundedness-judge caching
# - Gemini quota/error handling


from pathlib import Path
import hashlib
import json
import re

import chromadb

from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHUNKS_PATH = Path(
    "data/processed/episode_01_chunks.json"
)

CACHE_PATH = Path(
    "data/evaluation/generation_cache.json"
)

CHROMA_PATH = "data/chroma_db"

COLLECTION_NAME = "foundertechtok"

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

GENERATION_MODEL_NAME = (
    "gemini-3.6-flash"
)

BM25_TOP_K = 5

VECTOR_TOP_K = 5

HYBRID_TOP_K = 5

FINAL_TOP_K = 3

RRF_K = 60


# ============================================================
# 2. EVALUATION DATASET
# ============================================================

test_cases = [
    {
        "question":
            "How do salespeople build trust with customers?",

        "expected_timestamps":
            ["31:05"],

        "should_answer":
            True,
    },

    {
        "question":
            "What did Khushy say about incentive payments?",

        "expected_timestamps":
            ["3:18", "4:20"],

        "should_answer":
            True,
    },

    {
        "question":
            "How does the software help retail salespeople?",

        "expected_timestamps":
            ["3:18", "4:20"],

        "should_answer":
            True,
    },

    {
        "question":
            "Why is authentic selling important?",

        "expected_timestamps":
            ["2:17", "3:18"],

        "should_answer":
            True,
    },

    {
        "question":
            "How does technology make incentives more flexible?",

        "expected_timestamps":
            ["25:02", "26:07"],

        "should_answer":
            True,
    },

    {
        "question":
            "What does Khushy say about understanding the customer?",

        "expected_timestamps":
            ["2:17", "31:05"],

        "should_answer":
            True,
    },

    {
        "question":
            "What did Khushy say about nuclear fusion?",

        "expected_timestamps":
            [],

        "should_answer":
            False,
    },

    {
        "question":
            "What is Khushy's opinion on quantum computing?",

        "expected_timestamps":
            [],

        "should_answer":
            False,
    },

    {
        "question":
            "What did Khushy say about cryptocurrency regulation?",

        "expected_timestamps":
            [],

        "should_answer":
            False,
    },

    {
        "question":
            "What is Khushy's view on Mars colonization?",

        "expected_timestamps":
            [],

        "should_answer":
            False,
    },
]


# ============================================================
# 3. LOAD TRANSCRIPT CHUNKS
# ============================================================

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


texts = [
    chunk["text"]
    for chunk in chunks
]


# ============================================================
# 4. BM25 SETUP
# ============================================================

tokenized_corpus = [
    text.lower().split()
    for text in texts
]


bm25 = BM25Okapi(
    tokenized_corpus
)


# ============================================================
# 5. VECTOR SEARCH SETUP
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# 6. RERANKER SETUP
# ============================================================

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


# ============================================================
# 7. GEMINI SETUP
# ============================================================

try:

    gemini_client = genai.Client()

except Exception as error:

    gemini_client = None

    print(
        "\nWARNING: Gemini could not be initialized."
    )

    print(
        "Generation evaluations will be skipped."
    )

    print(
        "Reason:",
        error
    )


# ============================================================
# 8. LOAD CACHE
# ============================================================

CACHE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


if CACHE_PATH.exists():

    try:

        with open(
            CACHE_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            generation_cache = json.load(file)

    except Exception:

        generation_cache = {}

else:

    generation_cache = {}


# ============================================================
# 9. SAVE CACHE
# ============================================================

def save_cache():

    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            generation_cache,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# 10. CREATE CACHE KEY
# ============================================================

def create_cache_key(
    question,
    context
):

    cache_input = (
        GENERATION_MODEL_NAME
        + "\n"
        + question
        + "\n"
        + context
    )

    return hashlib.sha256(
        cache_input.encode("utf-8")
    ).hexdigest()


# ============================================================
# 11. BUILD CONTEXT
# ============================================================

def build_context(
    reranked_results
):

    context_parts = []


    for rank, (
        index,
        score
    ) in enumerate(
        reranked_results,
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


    return "\n".join(
        context_parts
    )


# ============================================================
# 12. GENERATE GROUNDED ANSWER
# ============================================================

def generate_grounded_answer(
    question,
    context
):

    cache_key = create_cache_key(
        question,
        context
    )


    # --------------------------------------------------------
    # USE SAVED ANSWER IF AVAILABLE
    # --------------------------------------------------------

    if (
        cache_key in generation_cache
        and generation_cache[
            cache_key
        ].get("answer")
    ):

        print(
            "\n[CACHE] Using saved Gemini answer."
        )

        return (
            generation_cache[
                cache_key
            ]["answer"]
        )


    # --------------------------------------------------------
    # GEMINI UNAVAILABLE
    # --------------------------------------------------------

    if gemini_client is None:

        return None


    # --------------------------------------------------------
    # GENERATION PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are FounderTechTok Intelligence.

Answer the user's question using ONLY
the podcast evidence below.

Rules:

1. Do not use outside knowledge.

2. Do not invent information.

3. If the evidence supports an answer,
answer clearly and concisely.

4. Cite important claims using timestamps
in square brackets.

Example:

Salespeople build trust by understanding
the customer [31:05].

5. Use ONLY timestamps that appear
in the Podcast Evidence.

6. If the podcast evidence does not contain
enough information to answer the question,
respond EXACTLY with:

I don't have enough evidence in the FounderTechTok archive to answer that.

Question:
{question}

Podcast Evidence:
{context}
"""


    try:

        response = (
            gemini_client
            .models
            .generate_content(
                model=GENERATION_MODEL_NAME,
                contents=prompt
            )
        )


        answer = response.text.strip()


        if cache_key not in generation_cache:

            generation_cache[
                cache_key
            ] = {}


        generation_cache[
            cache_key
        ]["question"] = question

        generation_cache[
            cache_key
        ]["model"] = GENERATION_MODEL_NAME

        generation_cache[
            cache_key
        ]["answer"] = answer


        save_cache()


        return answer


    except Exception as error:

        print(
            "\n[GENERATION ERROR]"
        )

        print(
            "Question:",
            question
        )

        print(
            "Reason:",
            error
        )


        error_text = str(
            error
        )


        if (
            "RESOURCE_EXHAUSTED"
            in error_text
            or "429"
            in error_text
        ):

            print(
                "\nGemini quota exhausted."
            )

            print(
                "Retrieval evaluation will continue."
            )


        return None


# ============================================================
# 13. GROUNDEDNESS EVALUATOR
# ============================================================

def evaluate_groundedness(
    question,
    context,
    answer
):

    cache_key = create_cache_key(
        question,
        context
    )


    # --------------------------------------------------------
    # USE SAVED JUDGMENT
    # --------------------------------------------------------

    if (
        cache_key in generation_cache
        and "grounded" in generation_cache[
            cache_key
        ]
    ):

        print(
            "[CACHE] Using saved groundedness judgment."
        )

        return generation_cache[
            cache_key
        ]["grounded"]


    # --------------------------------------------------------
    # GEMINI UNAVAILABLE
    # --------------------------------------------------------

    if gemini_client is None:

        return None


    # --------------------------------------------------------
    # GROUNDEDNESS PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Determine whether the generated answer is fully supported
by the supplied podcast evidence.

Rules:

1. Use ONLY the podcast evidence.

2. Every factual claim in the answer must be supported
by the evidence.

3. Do not reward an answer merely because it sounds plausible.

4. If the answer adds information not supported by
the podcast evidence, return FAIL.

5. If the answer contradicts the evidence, return FAIL.

6. If the answer is fully supported by the evidence,
return PASS.

Return ONLY one word:

PASS

or

FAIL

Question:
{question}

Podcast Evidence:
{context}

Generated Answer:
{answer}
"""


    try:

        response = (
            gemini_client
            .models
            .generate_content(
                model=GENERATION_MODEL_NAME,
                contents=prompt
            )
        )


        result = (
            response.text
            .strip()
            .upper()
        )


        if result == "PASS":

            grounded = True


        elif result == "FAIL":

            grounded = False


        else:

            print(
                "\n[GROUNDING WARNING]"
            )

            print(
                "Unexpected evaluator output:",
                result
            )

            return None


        if cache_key not in generation_cache:

            generation_cache[
                cache_key
            ] = {}


        generation_cache[
            cache_key
        ]["grounded"] = grounded


        save_cache()


        return grounded


    except Exception as error:

        print(
            "\n[GROUNDING ERROR]"
        )

        print(
            "Question:",
            question
        )

        print(
            "Reason:",
            error
        )


        return None


# ============================================================
# 14. METRIC COUNTERS
# ============================================================

total_positive = 0

retrieval_hits = 0


total_negative_tests = 0

correct_abstentions = 0


total_citation_tests = 0

correct_citation_tests = 0


total_groundedness_tests = 0

correct_groundedness_tests = 0


generation_skipped = 0

groundedness_skipped = 0


# ============================================================
# 15. START EVALUATION
# ============================================================

print(
    "\n================================"
)

print(
    "FOUNDERTECHTOK INTELLIGENCE"
)

print(
    "RAG EVALUATION"
)

print(
    "================================"
)

print(
    "Evaluation questions:",
    len(test_cases)
)


# ============================================================
# 16. RUN EACH TEST
# ============================================================

for test in test_cases:

    question = test[
        "question"
    ]


    print(
        "\n================================"
    )

    print(
        "Question:",
        question
    )

    print(
        "Should answer:",
        test["should_answer"]
    )

    print(
        "================================"
    )


    # ========================================================
    # A. BM25 RETRIEVAL
    # ========================================================

    tokenized_question = (
        question
        .lower()
        .split()
    )


    bm25_scores = bm25.get_scores(
        tokenized_question
    )


    bm25_top_indices = sorted(
        range(
            len(bm25_scores)
        ),

        key=lambda i:
            bm25_scores[i],

        reverse=True
    )[
        :BM25_TOP_K
    ]


    # ========================================================
    # B. VECTOR RETRIEVAL
    # ========================================================

    question_embedding = (
        embedding_model
        .encode(
            question
        )
    )


    vector_results = collection.query(

        query_embeddings=[
            question_embedding.tolist()
        ],

        n_results=
            VECTOR_TOP_K
    )


    vector_ids = (
        vector_results[
            "ids"
        ][0]
    )


    vector_top_indices = []


    for vector_id in vector_ids:

        chunk_number = int(
            vector_id
            .split("_")[-1]
        )


        vector_top_indices.append(
            chunk_number - 1
        )


    # ========================================================
    # C. RECIPROCAL RANK FUSION
    # ========================================================

    rrf_scores = {}


    for rank, index in enumerate(
        bm25_top_indices,
        start=1
    ):

        rrf_scores[index] = (
            rrf_scores.get(
                index,
                0
            )

            + 1
            / (
                RRF_K
                + rank
            )
        )


    for rank, index in enumerate(
        vector_top_indices,
        start=1
    ):

        rrf_scores[index] = (
            rrf_scores.get(
                index,
                0
            )

            + 1
            / (
                RRF_K
                + rank
            )
        )


    hybrid_results = sorted(
        rrf_scores.items(),

        key=lambda item:
            item[1],

        reverse=True
    )[
        :HYBRID_TOP_K
    ]


    # ========================================================
    # D. RERANKING
    # ========================================================

    pairs = []


    for index, _ in hybrid_results:

        pairs.append(
            [
                question,
                chunks[index]["text"]
            ]
        )


    rerank_scores = reranker.predict(
        pairs
    )


    reranked = []


    for i, (
        index,
        _
    ) in enumerate(
        hybrid_results
    ):

        reranked.append(
            (
                index,

                float(
                    rerank_scores[i]
                )
            )
        )


    reranked = sorted(
        reranked,

        key=lambda item:
            item[1],

        reverse=True
    )[
        :FINAL_TOP_K
    ]


    # ========================================================
    # E. RETRIEVED TIMESTAMPS
    # ========================================================

    retrieved_timestamps = [

        chunks[index][
            "start_time"
        ]

        for index, _
        in reranked
    ]


    print(
        "Retrieved timestamps:",
        retrieved_timestamps
    )


    # ========================================================
    # F. RETRIEVAL HIT@3
    # ========================================================

    if test[
        "should_answer"
    ]:

        total_positive += 1


        expected_timestamps = (
            test[
                "expected_timestamps"
            ]
        )


        retrieval_hit = any(

            timestamp
            in retrieved_timestamps

            for timestamp
            in expected_timestamps
        )


        print(
            "Expected timestamps:",
            expected_timestamps
        )


        print(
            "Hit@3:",
            retrieval_hit
        )


        if retrieval_hit:

            retrieval_hits += 1


    # ========================================================
    # G. BUILD CONTEXT
    # ========================================================

    context = build_context(
        reranked
    )


    # ========================================================
    # H. GENERATE ANSWER
    # ========================================================

    answer = generate_grounded_answer(
        question,
        context
    )


    if answer is None:

        generation_skipped += 1

        print(
            "Generation evaluation: skipped"
        )

        continue


    print(
        "\nGenerated answer:"
    )

    print(
        answer
    )


    # ========================================================
    # I. ANSWERABLE QUESTIONS
    # ========================================================

    if test[
        "should_answer"
    ]:


        # ----------------------------------------------------
        # CITATION EVALUATION
        # ----------------------------------------------------

        cited_timestamps = re.findall(
            r"\[(\d{1,2}:\d{2})\]",
            answer
        )


        total_citation_tests += 1


        citations_valid = (
            len(
                cited_timestamps
            ) > 0

            and all(

                timestamp
                in retrieved_timestamps

                for timestamp
                in cited_timestamps
            )
        )


        print(
            "Cited timestamps:",
            cited_timestamps
        )


        print(
            "Citation correctness:",
            citations_valid
        )


        if citations_valid:

            correct_citation_tests += 1


        else:

            print(
                "\n!!! CITATION FAILURE !!!"
            )

            print(
                "Question:",
                question
            )

            print(
                "Retrieved:",
                retrieved_timestamps
            )

            print(
                "Cited:",
                cited_timestamps
            )


        # ----------------------------------------------------
        # GROUNDEDNESS EVALUATION
        # ----------------------------------------------------

        grounded = evaluate_groundedness(
            question,
            context,
            answer
        )


        if grounded is None:

            groundedness_skipped += 1

            print(
                "Groundedness: skipped"
            )


        else:

            total_groundedness_tests += 1


            print(
                "Groundedness:",
                grounded
            )


            if grounded:

                correct_groundedness_tests += 1


    # ========================================================
    # J. UNANSWERABLE QUESTIONS
    # ========================================================

    else:

        total_negative_tests += 1


        expected_abstention = (
            "I don't have enough evidence "
            "in the FounderTechTok archive "
            "to answer that."
        )


        abstained = (
            expected_abstention
            in answer
        )


        print(
            "Expected behavior: abstain"
        )


        print(
            "Abstention correct:",
            abstained
        )


        if abstained:

            correct_abstentions += 1


# ============================================================
# 17. RETRIEVAL METRICS
# ============================================================

print(
    "\n================================"
)

print(
    "RETRIEVAL EVALUATION"
)

print(
    "================================"
)


print(
    "Hits:",
    retrieval_hits
)


print(
    "Positive questions:",
    total_positive
)


if total_positive > 0:

    hit_at_3 = (
        retrieval_hits
        / total_positive
    )


    print(
        "Hit@3:",
        round(
            hit_at_3,
            3
        )
    )


# ============================================================
# 18. ABSTENTION METRICS
# ============================================================

print(
    "\n================================"
)

print(
    "ABSTENTION EVALUATION"
)

print(
    "================================"
)


print(
    "Correct abstentions:",
    correct_abstentions
)


print(
    "Completed negative tests:",
    total_negative_tests
)


if total_negative_tests > 0:

    abstention_accuracy = (
        correct_abstentions
        / total_negative_tests
    )


    print(
        "Abstention accuracy:",
        round(
            abstention_accuracy,
            3
        )
    )

else:

    print(
        "Abstention accuracy: N/A"
    )


# ============================================================
# 19. CITATION METRICS
# ============================================================

print(
    "\n================================"
)

print(
    "CITATION EVALUATION"
)

print(
    "================================"
)


print(
    "Correct citation tests:",
    correct_citation_tests
)


print(
    "Completed citation tests:",
    total_citation_tests
)


if total_citation_tests > 0:

    citation_accuracy = (
        correct_citation_tests
        / total_citation_tests
    )


    print(
        "Citation accuracy:",
        round(
            citation_accuracy,
            3
        )
    )

else:

    print(
        "Citation accuracy: N/A"
    )


# ============================================================
# 20. GROUNDEDNESS METRICS
# ============================================================

print(
    "\n================================"
)

print(
    "GROUNDEDNESS EVALUATION"
)

print(
    "================================"
)


print(
    "Grounded answers:",
    correct_groundedness_tests
)


print(
    "Completed groundedness tests:",
    total_groundedness_tests
)


if total_groundedness_tests > 0:

    groundedness_accuracy = (
        correct_groundedness_tests
        / total_groundedness_tests
    )


    print(
        "Groundedness accuracy:",
        round(
            groundedness_accuracy,
            3
        )
    )

else:

    print(
        "Groundedness accuracy: N/A"
    )


# ============================================================
# 21. STATUS
# ============================================================

print(
    "\n================================"
)

print(
    "EVALUATION STATUS"
)

print(
    "================================"
)


print(
    "Generation tests skipped:",
    generation_skipped
)


print(
    "Groundedness tests skipped:",
    groundedness_skipped
)


print(
    "Cache:",
    CACHE_PATH
)


print(
    "\nEvaluation complete."
)