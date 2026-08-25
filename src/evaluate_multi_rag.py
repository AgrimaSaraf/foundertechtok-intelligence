# ============================================================
# FounderTechTok Intelligence
# Resumable Multi-Episode RAG Evaluation Harness
#
# Measures:
# - Retrieval Hit@1 / Hit@3 / Hit@5
# - Mean Reciprocal Rank (MRR)
# - Answerability accuracy
# - Abstention accuracy
# - Citation source correctness
# - Generation coverage
# - API / generation error counts
# - Average / median latency
#
# Reliability:
# - API_ERROR does NOT count as a wrong answer
# - GENERATION_ERROR does NOT count as a wrong answer
# - Successful generations are cached
# - Failed API calls are retried on future runs
# - Evaluation can resume without spending calls again
# - Daily quota exhaustion opens a circuit breaker
#
# Benchmark:
# data/evaluation/eval_questions.json
# ============================================================

from pathlib import Path
import json
import statistics
import time

from multi_episode_search import (
    chunks,
    search,
)

from multi_episode_rag import (
    ask,
)


# ============================================================
# CONFIG
# ============================================================

EVAL_PATH = Path(
    "data/evaluation/eval_questions.json"
)

OUTPUT_PATH = Path(
    "data/evaluation/multi_rag_results.json"
)

CACHE_PATH = Path(
    "data/evaluation/multi_rag_generation_cache.json"
)


# ------------------------------------------------------------
# Evaluation limit
#
# 1      -> first question only
# 5      -> first five
# None   -> full benchmark
# ------------------------------------------------------------

EVAL_LIMIT = None


# ------------------------------------------------------------
# Force Gemini regeneration
#
# False:
#   reuse successful cached generations
#
# True:
#   ignore cache and call Gemini again
#
# KEEP THIS FALSE during normal evaluation.
# ------------------------------------------------------------

FORCE_REGENERATE = False


# ============================================================
# LOAD BENCHMARK
# ============================================================

with open(
    EVAL_PATH,
    "r",
    encoding="utf-8"
) as file:

    test_cases = json.load(
        file
    )


# ============================================================
# SELECT EVALUATION SET
# ============================================================

if EVAL_LIMIT is None:

    evaluation_cases = test_cases

else:

    evaluation_cases = test_cases[
        :EVAL_LIMIT
    ]


print(
    "Benchmark questions:",
    len(test_cases)
)

print(
    "Questions being evaluated:",
    len(evaluation_cases)
)


# ============================================================
# CACHE HELPERS
# ============================================================

def load_cache():
    """
    Load successful generation results from disk.

    Infrastructure failures are not reusable successes.
    """

    if not CACHE_PATH.exists():

        return {}


    try:

        with open(
            CACHE_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        if isinstance(
            data,
            dict
        ):

            return data


    except Exception as error:

        print(
            "Warning: could not read cache:"
        )

        print(
            error
        )


    return {}


def save_cache(
    cache
):
    """
    Persist generation cache immediately.

    This makes the benchmark resumable.
    """

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cache,
            file,
            indent=2,
            ensure_ascii=False
        )


generation_cache = load_cache()


print(
    "Cached generations:",
    len(
        generation_cache
    )
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def timestamp_to_seconds(
    timestamp
):
    """
    Convert MM:SS or HH:MM:SS into total seconds.
    """

    if timestamp is None:

        return None


    parts = [
        int(part)
        for part in timestamp.split(":")
    ]


    if len(parts) == 2:

        minutes, seconds = parts

        return (
            minutes * 60
            + seconds
        )


    if len(parts) == 3:

        hours, minutes, seconds = parts

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )


    raise ValueError(
        f"Unsupported timestamp: {timestamp}"
    )


def chunk_matches_source(
    chunk,
    relevant_source
):
    """
    Determine whether a retrieved chunk overlaps
    a benchmark-labelled source window.
    """

    if (
        chunk.get(
            "episode_id"
        )
        !=
        relevant_source.get(
            "episode_id"
        )
    ):

        return False


    expected_start = (
        timestamp_to_seconds(
            relevant_source.get(
                "start_time"
            )
        )
    )


    expected_end = (
        timestamp_to_seconds(
            relevant_source.get(
                "end_time"
            )
        )
    )


    chunk_start = int(
        chunk.get(
            "start_seconds",
            0
        )
    )


    chunk_end = int(
        chunk.get(
            "end_seconds",
            chunk_start
        )
    )


    return (
        chunk_start <= expected_end
        and
        chunk_end >= expected_start
    )


def retrieved_rank_of_relevant_source(
    reranked_results,
    relevant_sources
):
    """
    Return rank of first relevant retrieved chunk.
    """

    for rank, (
        chunk_index,
        _
    ) in enumerate(
        reranked_results,
        start=1
    ):

        chunk = chunks[
            chunk_index
        ]


        for source in relevant_sources:

            if chunk_matches_source(
                chunk,
                source
            ):

                return rank


    return None


def citation_matches_source(
    citation,
    relevant_sources
):
    """
    Check whether a validated model citation overlaps
    one of the benchmark-labelled source windows.
    """

    citation_episode = (
        citation.get(
            "episode_id"
        )
    )


    citation_start = (
        timestamp_to_seconds(
            citation.get(
                "start_time"
            )
        )
    )


    citation_end = (
        timestamp_to_seconds(
            citation.get(
                "end_time"
            )
        )
    )


    if (
        citation_start is None
        or
        citation_end is None
    ):

        return False


    for source in relevant_sources:

        if (
            citation_episode
            !=
            source.get(
                "episode_id"
            )
        ):

            continue


        source_start = (
            timestamp_to_seconds(
                source.get(
                    "start_time"
                )
            )
        )


        source_end = (
            timestamp_to_seconds(
                source.get(
                    "end_time"
                )
            )
        )


        if (
            citation_start <= source_end
            and
            citation_end >= source_start
        ):

            return True


    return False


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def is_infrastructure_error(
    status
):
    """
    Infrastructure failures must not count as
    incorrect RAG answers.
    """

    return status in {
        "API_ERROR",
        "GENERATION_ERROR",
    }


def is_daily_quota_exhausted(
    error_text
):
    """
    Detect Gemini hard/free-tier quota exhaustion.

    A daily/request quota exhaustion cannot be fixed by
    repeatedly calling Gemini during the same benchmark run.

    We deliberately check several forms because Google API
    error messages can vary slightly.
    """

    if not error_text:

        return False


    text = str(
        error_text
    ).lower()


    resource_exhausted = (
        "resource_exhausted" in text
        or
        "resource exhausted" in text
        or
        "quota exceeded" in text
    )


    quota_signal = (
        "free_tier_requests" in text
        or
        "free tier" in text
        or
        "quota" in text
    )


    return (
        resource_exhausted
        and
        quota_signal
    )


# ============================================================
# CACHE CLASSIFICATION
# ============================================================

def is_cacheable_result(
    result
):
    """
    Cache legitimate RAG outcomes.

    Do not cache API_ERROR / GENERATION_ERROR because
    those should be retried later.
    """

    status = result.get(
        "status"
    )


    return not is_infrastructure_error(
        status
    )


def compact_result_for_cache(
    result
):
    """
    Store only data required for future evaluation.
    """

    return {
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

        "invalid_citations":
            result.get(
                "invalid_citations",
                []
            ),

        "error":
            result.get(
                "error"
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
    }


# ============================================================
# METRIC ACCUMULATORS
# ============================================================

positive_questions = 0

negative_questions = 0


hit_at_1 = 0

hit_at_3 = 0

hit_at_5 = 0


reciprocal_ranks = []


# ------------------------------------------------------------
# Generation metrics
# ------------------------------------------------------------

generation_evaluated_questions = 0

generation_positive_questions = 0

generation_negative_questions = 0


correct_answerability = 0

correct_abstentions = 0


citation_questions = 0

citation_correct_questions = 0


# ------------------------------------------------------------
# Reliability
# ------------------------------------------------------------

api_errors = 0

generation_errors = 0

cache_hits = 0

live_generation_calls = 0

quota_skipped_questions = 0


# ------------------------------------------------------------
# QUOTA CIRCUIT BREAKER
#
# Once a hard quota exhaustion is detected, no additional
# Gemini calls will be attempted during this run.
# ------------------------------------------------------------

quota_circuit_open = False


# ------------------------------------------------------------
# Latency
#
# Only successful live calls from THIS execution are included.
# ------------------------------------------------------------

successful_latencies = []


results_output = []


# ============================================================
# RUN EVALUATION
# ============================================================

for index, test in enumerate(
    evaluation_cases,
    start=1
):

    question_id = test[
        "id"
    ]


    question = test[
        "question"
    ]


    should_answer = test[
        "should_answer"
    ]


    relevant_sources = test.get(
        "relevant_sources",
        []
    )


    print(
        "\n"
        + "=" * 75
    )


    print(
        f"[{index}/"
        f"{len(evaluation_cases)}] "
        f"{question_id}"
    )


    print(
        question
    )


    print(
        "=" * 75
    )


    # ========================================================
    # RETRIEVAL EVALUATION
    #
    # This continues even if Gemini quota is exhausted.
    # ========================================================

    retrieval_results = search(
        question
    )


    reranked_results = retrieval_results[
        "reranked"
    ]


    retrieved_chunks = []


    for rank, (
        chunk_index,
        score
    ) in enumerate(
        reranked_results,
        start=1
    ):

        chunk = chunks[
            chunk_index
        ]


        retrieved_chunks.append(
            {
                "rank":
                    rank,

                "chunk_id":
                    chunk[
                        "chunk_id"
                    ],

                "episode_id":
                    chunk[
                        "episode_id"
                    ],

                "guest":
                    chunk[
                        "guest"
                    ],

                "start_time":
                    chunk[
                        "start_time"
                    ],

                "end_time":
                    chunk[
                        "end_time"
                    ],

                "reranker_score":
                    float(
                        score
                    ),
            }
        )


    relevant_rank = None


    # ========================================================
    # RETRIEVAL METRICS
    # ========================================================

    if should_answer:

        positive_questions += 1


        relevant_rank = (
            retrieved_rank_of_relevant_source(
                reranked_results,
                relevant_sources
            )
        )


        if (
            relevant_rank is not None
            and
            relevant_rank <= 1
        ):

            hit_at_1 += 1


        if (
            relevant_rank is not None
            and
            relevant_rank <= 3
        ):

            hit_at_3 += 1


        if (
            relevant_rank is not None
            and
            relevant_rank <= 5
        ):

            hit_at_5 += 1


        if relevant_rank is not None:

            reciprocal_ranks.append(
                1 / relevant_rank
            )

        else:

            reciprocal_ranks.append(
                0.0
            )


    else:

        negative_questions += 1


    # ========================================================
    # GENERATION
    # ========================================================

    used_cache = False

    skipped_due_to_quota = False

    rag_result = None

    elapsed = None


    # --------------------------------------------------------
    # FIRST: use successful cache if available.
    #
    # Cache still works even when circuit breaker is open.
    # --------------------------------------------------------

    if (
        not FORCE_REGENERATE
        and
        question_id in generation_cache
    ):

        cached_result = generation_cache[
            question_id
        ]


        cached_status = cached_result.get(
            "status"
        )


        if not is_infrastructure_error(
            cached_status
        ):

            rag_result = cached_result

            used_cache = True

            cache_hits += 1


            print(
                "Generation: CACHE HIT"
            )


    # --------------------------------------------------------
    # SECOND: if quota circuit is already open and no cache
    # exists, skip the Gemini call entirely.
    # --------------------------------------------------------

    if (
        rag_result is None
        and
        quota_circuit_open
    ):

        skipped_due_to_quota = True

        quota_skipped_questions += 1


        rag_result = {
            "status":
                "API_ERROR",

            "answer":
                "",

            "citations":
                [],

            "invalid_citations":
                [],

            "error":
                (
                    "Skipped because Gemini quota circuit "
                    "breaker is open."
                ),

            "retrieval_latency_sec":
                None,

            "generation_latency_sec":
                None,

            "total_latency_sec":
                None,
        }


        print(
            "Generation: SKIPPED — quota circuit open"
        )


    # --------------------------------------------------------
    # THIRD: make live Gemini call if:
    #
    # - no cache
    # - quota circuit is closed
    # --------------------------------------------------------

    if (
        rag_result is None
        and
        not quota_circuit_open
    ):

        live_generation_calls += 1


        generation_start = time.perf_counter()


        rag_result = ask(
            question
        )


        elapsed = (
            time.perf_counter()
            - generation_start
        )


        # ----------------------------------------------------
        # Detect hard quota exhaustion.
        #
        # Once detected, open circuit for remaining questions.
        # ----------------------------------------------------

        if (
            rag_result.get(
                "status"
            )
            ==
            "API_ERROR"
        ):

            error_text = rag_result.get(
                "error",
                ""
            )


            if is_daily_quota_exhausted(
                error_text
            ):

                quota_circuit_open = True


                print(
                    "\n"
                    + "!" * 75
                )


                print(
                    "GEMINI QUOTA EXHAUSTED"
                )


                print(
                    "Quota circuit breaker OPEN."
                )


                print(
                    "Remaining uncached generation "
                    "calls will be skipped this run."
                )


                print(
                    "Retrieval evaluation will continue."
                )


                print(
                    "!" * 75
                )


        # ----------------------------------------------------
        # Cache only legitimate RAG outcomes.
        # ----------------------------------------------------

        if is_cacheable_result(
            rag_result
        ):

            generation_cache[
                question_id
            ] = (
                compact_result_for_cache(
                    rag_result
                )
            )


            save_cache(
                generation_cache
            )


    # ========================================================
    # CLASSIFY RESULT
    # ========================================================

    status = rag_result.get(
        "status"
    )


    infrastructure_error = (
        is_infrastructure_error(
            status
        )
    )


    answerability_correct = None

    abstention_correct = None

    citation_correct = None


    # --------------------------------------------------------
    # Reliability counters
    # --------------------------------------------------------

    if status == "API_ERROR":

        api_errors += 1


    elif status == "GENERATION_ERROR":

        generation_errors += 1


    # ========================================================
    # LEGITIMATE RAG OUTCOME
    # ========================================================

    if not infrastructure_error:

        generation_evaluated_questions += 1


        if should_answer:

            generation_positive_questions += 1

        else:

            generation_negative_questions += 1


        system_answered = (
            status == "ANSWER"
        )


        # ----------------------------------------------------
        # Answerability
        # ----------------------------------------------------

        answerability_correct = (
            system_answered
            ==
            should_answer
        )


        if answerability_correct:

            correct_answerability += 1


        # ----------------------------------------------------
        # Abstention
        # ----------------------------------------------------

        if not should_answer:

            abstention_correct = (
                not system_answered
            )


            if abstention_correct:

                correct_abstentions += 1


        # ----------------------------------------------------
        # Citation correctness
        # ----------------------------------------------------

        if (
            should_answer
            and
            system_answered
        ):

            citation_questions += 1


            citations = rag_result.get(
                "citations",
                []
            )


            citation_correct = any(
                citation_matches_source(
                    citation,
                    relevant_sources
                )
                for citation in citations
            )


            if citation_correct:

                citation_correct_questions += 1


        # ----------------------------------------------------
        # Successful live latency
        # ----------------------------------------------------

        if (
            not used_cache
            and
            elapsed is not None
        ):

            successful_latencies.append(
                elapsed
            )


    # ========================================================
    # STORE QUESTION RESULT
    # ========================================================

    question_result = {

        "id":
            question_id,

        "question":
            question,

        "category":
            test.get(
                "category"
            ),

        "should_answer":
            should_answer,

        "used_cache":
            used_cache,

        "skipped_due_to_quota":
            skipped_due_to_quota,

        "system_status":
            status,

        "system_answer":
            rag_result.get(
                "answer"
            ),

        "infrastructure_error":
            infrastructure_error,

        "error":
            rag_result.get(
                "error"
            ),

        "answerability_correct":
            answerability_correct,

        "abstention_correct":
            abstention_correct,

        "relevant_rank":
            relevant_rank,

        "retrieved_chunks":
            retrieved_chunks,

        "citations":
            rag_result.get(
                "citations",
                []
            ),

        "citation_correct":
            citation_correct,

        "retrieval_latency_sec":
            rag_result.get(
                "retrieval_latency_sec"
            ),

        "generation_latency_sec":
            rag_result.get(
                "generation_latency_sec"
            ),

        "total_latency_sec":
            rag_result.get(
                "total_latency_sec"
            ),
    }


    results_output.append(
        question_result
    )


    # ========================================================
    # PRINT QUESTION SUMMARY
    # ========================================================

    print(
        "Should answer:",
        should_answer
    )


    if should_answer:

        print(
            "Relevant rank:",
            relevant_rank
        )


    print(
        "System status:",
        status
    )


    print(
        "Used cache:",
        used_cache
    )


    if skipped_due_to_quota:

        print(
            "Generation evaluation:"
            " SKIPPED — quota exhausted"
        )


    elif infrastructure_error:

        print(
            "Generation evaluation:"
            " EXCLUDED — infrastructure error"
        )


        print(
            "Error:",
            rag_result.get(
                "error"
            )
        )


    else:

        print(
            "Answerability correct:",
            answerability_correct
        )


        if abstention_correct is not None:

            print(
                "Abstention correct:",
                abstention_correct
            )


        if citation_correct is not None:

            print(
                "Citation correct:",
                citation_correct
            )


    if (
        not used_cache
        and
        elapsed is not None
    ):

        print(
            "Live call latency:",
            round(
                elapsed,
                3
            ),
            "sec"
        )


# ============================================================
# CALCULATE RETRIEVAL METRICS
# ============================================================

hit_1_score = (
    hit_at_1
    / positive_questions
    if positive_questions
    else 0
)


hit_3_score = (
    hit_at_3
    / positive_questions
    if positive_questions
    else 0
)


hit_5_score = (
    hit_at_5
    / positive_questions
    if positive_questions
    else 0
)


mrr = (
    sum(
        reciprocal_ranks
    )
    / len(
        reciprocal_ranks
    )
    if reciprocal_ranks
    else 0
)


# ============================================================
# GENERATION METRICS
#
# API / generation failures are excluded.
# ============================================================

answerability_accuracy = (
    correct_answerability
    / generation_evaluated_questions
    if generation_evaluated_questions
    else None
)


abstention_accuracy = (
    correct_abstentions
    / generation_negative_questions
    if generation_negative_questions
    else None
)


citation_accuracy = (
    citation_correct_questions
    / citation_questions
    if citation_questions
    else None
)


generation_coverage = (
    generation_evaluated_questions
    / len(
        evaluation_cases
    )
    if evaluation_cases
    else 0
)


# ------------------------------------------------------------
# API success rate is based only on calls actually attempted.
#
# Quota-skipped questions are NOT failed API calls because
# we deliberately did not send requests for them.
# ------------------------------------------------------------

attempted_generation_outcomes = (
    generation_evaluated_questions
    - cache_hits
    + api_errors
    - quota_skipped_questions
    + generation_errors
)


successful_live_generation_outcomes = (
    generation_evaluated_questions
    - cache_hits
)


api_success_rate = (
    successful_live_generation_outcomes
    / attempted_generation_outcomes
    if attempted_generation_outcomes > 0
    else None
)


# ============================================================
# LATENCY
# ============================================================

average_latency = (
    statistics.mean(
        successful_latencies
    )
    if successful_latencies
    else None
)


median_latency = (
    statistics.median(
        successful_latencies
    )
    if successful_latencies
    else None
)


# ============================================================
# ROUNDING HELPER
# ============================================================

def round_or_none(
    value,
    digits=4
):

    if value is None:

        return None


    return round(
        value,
        digits
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "benchmark_questions":
        len(
            test_cases
        ),

    "questions_selected":
        len(
            evaluation_cases
        ),

    "positive_questions":
        positive_questions,

    "negative_questions":
        negative_questions,


    "retrieval": {

        "hit_at_1":
            round(
                hit_1_score,
                4
            ),

        "hit_at_3":
            round(
                hit_3_score,
                4
            ),

        "hit_at_5":
            round(
                hit_5_score,
                4
            ),

        "mrr":
            round(
                mrr,
                4
            ),
    },


    "generation": {

        "successfully_evaluated":
            generation_evaluated_questions,

        "positive_evaluated":
            generation_positive_questions,

        "negative_evaluated":
            generation_negative_questions,

        "coverage":
            round(
                generation_coverage,
                4
            ),

        "answerability_accuracy":
            round_or_none(
                answerability_accuracy
            ),

        "abstention_accuracy":
            round_or_none(
                abstention_accuracy
            ),

        "citation_accuracy":
            round_or_none(
                citation_accuracy
            ),
    },


    "reliability": {

        "cache_hits":
            cache_hits,

        "live_generation_calls":
            live_generation_calls,

        "api_errors":
            api_errors,

        "generation_errors":
            generation_errors,

        "quota_skipped_questions":
            quota_skipped_questions,

        "quota_circuit_opened":
            quota_circuit_open,

        "api_success_rate":
            round_or_none(
                api_success_rate
            ),
    },


    "latency_successful_live_calls": {

        "count":
            len(
                successful_latencies
            ),

        "average_sec":
            round_or_none(
                average_latency,
                3
            ),

        "median_sec":
            round_or_none(
                median_latency,
                3
            ),
    },
}


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 75
)


print(
    "FOUNDERTECHTOK INTELLIGENCE"
)


print(
    "RESUMABLE MULTI-EPISODE RAG EVALUATION"
)


print(
    "=" * 75
)


print(
    "\nBenchmark questions:",
    summary[
        "benchmark_questions"
    ]
)


print(
    "Questions selected:",
    summary[
        "questions_selected"
    ]
)


print(
    "Positive:",
    positive_questions
)


print(
    "Negative:",
    negative_questions
)


# ============================================================
# RETRIEVAL
# ============================================================

print(
    "\n-------------------------"
)


print(
    "RETRIEVAL"
)


print(
    "-------------------------"
)


print(
    "Hit@1:",
    summary[
        "retrieval"
    ][
        "hit_at_1"
    ]
)


print(
    "Hit@3:",
    summary[
        "retrieval"
    ][
        "hit_at_3"
    ]
)


print(
    "Hit@5:",
    summary[
        "retrieval"
    ][
        "hit_at_5"
    ]
)


print(
    "MRR:",
    summary[
        "retrieval"
    ][
        "mrr"
    ]
)


# ============================================================
# GENERATION
# ============================================================

print(
    "\n-------------------------"
)


print(
    "GENERATION"
)


print(
    "-------------------------"
)


print(
    "Successfully evaluated:",
    summary[
        "generation"
    ][
        "successfully_evaluated"
    ]
)


print(
    "Generation coverage:",
    summary[
        "generation"
    ][
        "coverage"
    ]
)


print(
    "Answerability accuracy:",
    summary[
        "generation"
    ][
        "answerability_accuracy"
    ]
)


print(
    "Abstention accuracy:",
    summary[
        "generation"
    ][
        "abstention_accuracy"
    ]
)


print(
    "Citation accuracy:",
    summary[
        "generation"
    ][
        "citation_accuracy"
    ]
)


# ============================================================
# RELIABILITY
# ============================================================

print(
    "\n-------------------------"
)


print(
    "RELIABILITY"
)


print(
    "-------------------------"
)


print(
    "Cache hits:",
    summary[
        "reliability"
    ][
        "cache_hits"
    ]
)


print(
    "Live Gemini calls:",
    summary[
        "reliability"
    ][
        "live_generation_calls"
    ]
)


print(
    "API errors:",
    summary[
        "reliability"
    ][
        "api_errors"
    ]
)


print(
    "Generation errors:",
    summary[
        "reliability"
    ][
        "generation_errors"
    ]
)


print(
    "Quota-skipped questions:",
    summary[
        "reliability"
    ][
        "quota_skipped_questions"
    ]
)


print(
    "Quota circuit opened:",
    summary[
        "reliability"
    ][
        "quota_circuit_opened"
    ]
)


print(
    "API success rate:",
    summary[
        "reliability"
    ][
        "api_success_rate"
    ]
)


# ============================================================
# LATENCY
# ============================================================

print(
    "\n-------------------------"
)


print(
    "LATENCY — SUCCESSFUL LIVE CALLS"
)


print(
    "-------------------------"
)


print(
    "Calls measured:",
    summary[
        "latency_successful_live_calls"
    ][
        "count"
    ]
)


print(
    "Average:",
    summary[
        "latency_successful_live_calls"
    ][
        "average_sec"
    ],
    "sec"
)


print(
    "Median:",
    summary[
        "latency_successful_live_calls"
    ][
        "median_sec"
    ],
    "sec"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "summary":
                summary,

            "questions":
                results_output,
        },

        file,

        indent=2,

        ensure_ascii=False
    )


print(
    "\nResults saved:"
)


print(
    OUTPUT_PATH
)


print(
    "\nGeneration cache:"
)


print(
    CACHE_PATH
)


# ============================================================
# NEXT-RUN MESSAGE
# ============================================================

if quota_circuit_open:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "EVALUATION CHECKPOINT SAVED"
    )

    print(
        "=" * 75
    )

    print(
        "\nGemini quota was exhausted during this run."
    )

    print(
        "Successful generations remain cached."
    )

    print(
        "Run this evaluator again when quota is available."
    )

    print(
        "Cached questions will not call Gemini again."
    )