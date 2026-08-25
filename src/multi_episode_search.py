from pathlib import Path
import json

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# ============================================================
# CONFIG
# ============================================================

CHUNKS_PATH = Path(
    "data/processed/all_episode_chunks.json"
)

CHROMA_PATH = "data/chroma_db_multi"

COLLECTION_NAME = "foundertechtok_multi"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# RETRIEVAL SETTINGS
#
# Selected from candidate-depth experiment.
#
# Baseline:
# Hit@1 = 0.52
# Hit@3 = 0.72
# Hit@5 = 0.80
# MRR   = 0.6313
#
# Depth-20:
# Hit@1 = 0.52
# Hit@3 = 0.84
# Hit@5 = 0.88
# MRR   = 0.6700
# ============================================================

BM25_TOP_K = 20

VECTOR_TOP_K = 20

HYBRID_TOP_K = 20

FINAL_TOP_K = 5

RRF_K = 60


# ============================================================
# CONTEXT EXPANSION SETTINGS
#
# Stage 1:
# Add one neighboring transcript chunk on each side.
#
# Stage 2:
# If selected context chunks from the same episode are
# separated by a small time gap, fill the chunks between them.
#
# Retrieval ranking itself is NOT changed.
# ============================================================

NEIGHBOR_WINDOW = 1

MAX_CONTEXT_GAP_SECONDS = 120


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(
        file
    )


texts = [
    chunk["text"]
    for chunk in chunks
]


print(
    "Chunks loaded:",
    len(chunks)
)


# ============================================================
# BM25 SETUP
# ============================================================

tokenized_corpus = [
    text
    .lower()
    .split()
    for text in texts
]


bm25 = BM25Okapi(
    tokenized_corpus
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print(
    "Loading embedding model..."
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# VECTOR STORE
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# CROSS-ENCODER RERANKER
# ============================================================

print(
    "Loading cross-encoder reranker..."
)


reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


# ============================================================
# INDEX LOOKUPS
# ============================================================

chunk_id_to_index = {
    chunk["chunk_id"]:
        index
    for index, chunk in enumerate(
        chunks
    )
}


# ------------------------------------------------------------
# Build ordered chunk list for every episode.
# ------------------------------------------------------------

episode_to_indices = {}


for index, chunk in enumerate(
    chunks
):

    episode_id = chunk[
        "episode_id"
    ]

    episode_to_indices.setdefault(
        episode_id,
        []
    ).append(
        index
    )


for episode_id, indices in (
    episode_to_indices.items()
):

    indices.sort(
        key=lambda index: (
            int(
                chunks[
                    index
                ].get(
                    "start_seconds",
                    0
                )
            ),
            int(
                chunks[
                    index
                ].get(
                    "end_seconds",
                    0
                )
            ),
        )
    )


# ------------------------------------------------------------
# Global chunk index -> position in episode.
# ------------------------------------------------------------

index_to_episode_position = {}


for episode_id, indices in (
    episode_to_indices.items()
):

    for position, index in enumerate(
        indices
    ):

        index_to_episode_position[
            index
        ] = (
            episode_id,
            position
        )


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    question,
    top_k=BM25_TOP_K
):
    """
    Lexical retrieval using BM25.
    """

    tokenized_question = (
        question
        .lower()
        .split()
    )


    scores = bm25.get_scores(
        tokenized_question
    )


    top_indices = sorted(
        range(
            len(scores)
        ),
        key=lambda i: scores[i],
        reverse=True
    )[
        :top_k
    ]


    return [
        (
            index,
            float(
                scores[index]
            )
        )
        for index in top_indices
    ]


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(
    question,
    top_k=VECTOR_TOP_K
):
    """
    Dense semantic retrieval using ChromaDB.
    """

    question_embedding = (
        embedding_model.encode(
            question
        )
    )


    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=top_k
    )


    ids = results[
        "ids"
    ][0]


    distances = (
        results.get(
            "distances",
            [[]]
        )[0]
    )


    vector_results = []


    for position, chunk_id in enumerate(
        ids
    ):

        index = chunk_id_to_index.get(
            chunk_id
        )


        if index is None:

            continue


        distance = None


        if position < len(
            distances
        ):

            distance = distances[
                position
            ]


        vector_results.append(
            (
                index,
                distance
            )
        )


    return vector_results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    bm25_results,
    vector_results,
    k=RRF_K,
    top_k=HYBRID_TOP_K
):
    """
    Fuse lexical and semantic rankings using RRF.

    score += 1 / (k + rank)
    """

    rrf_scores = {}


    # --------------------------------------------------------
    # BM25 contribution
    # --------------------------------------------------------

    for rank, (
        index,
        _
    ) in enumerate(
        bm25_results,
        start=1
    ):

        rrf_scores[
            index
        ] = (
            rrf_scores.get(
                index,
                0
            )
            +
            1 / (
                k + rank
            )
        )


    # --------------------------------------------------------
    # Dense contribution
    # --------------------------------------------------------

    for rank, (
        index,
        _
    ) in enumerate(
        vector_results,
        start=1
    ):

        rrf_scores[
            index
        ] = (
            rrf_scores.get(
                index,
                0
            )
            +
            1 / (
                k + rank
            )
        )


    hybrid_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[
        :top_k
    ]


    return hybrid_results


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_results(
    question,
    hybrid_results,
    final_top_k=FINAL_TOP_K
):
    """
    Rerank RRF candidates with a cross-encoder.
    """

    if not hybrid_results:

        return []


    pairs = [
        [
            question,
            chunks[
                index
            ][
                "text"
            ]
        ]
        for index, _ in hybrid_results
    ]


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
                    rerank_scores[
                        i
                    ]
                )
            )
        )


    reranked = sorted(
        reranked,
        key=lambda item: item[1],
        reverse=True
    )[
        :final_top_k
    ]


    return reranked


# ============================================================
# NEIGHBOR EXPANSION
# ============================================================

def get_neighbor_indices(
    index,
    window=NEIGHBOR_WINDOW
):
    """
    Return current chunk plus nearby chunks from
    the SAME episode.
    """

    episode_info = (
        index_to_episode_position.get(
            index
        )
    )


    if episode_info is None:

        return [
            index
        ]


    episode_id, position = (
        episode_info
    )


    episode_indices = (
        episode_to_indices[
            episode_id
        ]
    )


    start_position = max(
        0,
        position - window
    )


    end_position = min(
        len(
            episode_indices
        ),
        position + window + 1
    )


    return episode_indices[
        start_position:
        end_position
    ]


def expand_reranked_context(
    reranked_results,
    window=NEIGHBOR_WINDOW
):
    """
    Add neighboring chunks around each Top-K result.

    Retrieval ranking remains unchanged.
    """

    expanded_indices = []

    seen = set()


    for index, _ in reranked_results:

        neighbor_indices = get_neighbor_indices(
            index,
            window=window
        )


        for neighbor_index in (
            neighbor_indices
        ):

            if neighbor_index in seen:

                continue


            seen.add(
                neighbor_index
            )


            expanded_indices.append(
                neighbor_index
            )


    return expanded_indices


# ============================================================
# SHORT-GAP SPAN FILLING
# ============================================================

def fill_short_context_gaps(
    context_indices,
    max_gap_seconds=MAX_CONTEXT_GAP_SECONDS
):
    """
    Fill short transcript gaps between already-selected
    context chunks belonging to the same episode.

    Example:

        selected chunk at 06:27
              ↓
        missing chunks at 06:37, 06:45, 06:54
              ↓
        selected chunk at 07:52

    If the time gap is small enough, the intermediate
    chunks are included.

    This is useful for conversational transcripts where
    one answer spans several small chunks.

    IMPORTANT:
    This modifies generation context only.
    It does NOT modify retrieval ranks.
    """

    if not context_indices:

        return []


    # --------------------------------------------------------
    # Group selected chunks by episode.
    # --------------------------------------------------------

    selected_by_episode = {}


    for index in context_indices:

        episode_id = chunks[
            index
        ][
            "episode_id"
        ]


        selected_by_episode.setdefault(
            episode_id,
            []
        ).append(
            index
        )


    final_indices = set(
        context_indices
    )


    # --------------------------------------------------------
    # Inspect neighboring selected chunks chronologically.
    # --------------------------------------------------------

    for episode_id, selected_indices in (
        selected_by_episode.items()
    ):

        selected_indices.sort(
            key=lambda index: int(
                chunks[
                    index
                ].get(
                    "start_seconds",
                    0
                )
            )
        )


        episode_indices = (
            episode_to_indices[
                episode_id
            ]
        )


        episode_position_lookup = {
            index: position
            for position, index
            in enumerate(
                episode_indices
            )
        }


        for i in range(
            len(
                selected_indices
            ) - 1
        ):

            left_index = (
                selected_indices[
                    i
                ]
            )

            right_index = (
                selected_indices[
                    i + 1
                ]
            )


            left_end = int(
                chunks[
                    left_index
                ].get(
                    "end_seconds",
                    chunks[
                        left_index
                    ].get(
                        "start_seconds",
                        0
                    )
                )
            )


            right_start = int(
                chunks[
                    right_index
                ].get(
                    "start_seconds",
                    0
                )
            )


            gap_seconds = (
                right_start
                - left_end
            )


            # ------------------------------------------------
            # Only fill short gaps.
            # ------------------------------------------------

            if gap_seconds < 0:

                gap_seconds = 0


            if (
                gap_seconds
                > max_gap_seconds
            ):

                continue


            left_position = (
                episode_position_lookup[
                    left_index
                ]
            )


            right_position = (
                episode_position_lookup[
                    right_index
                ]
            )


            if (
                right_position
                <= left_position
            ):

                continue


            # ------------------------------------------------
            # Add every intermediate transcript chunk.
            # ------------------------------------------------

            for position in range(
                left_position,
                right_position + 1
            ):

                final_indices.add(
                    episode_indices[
                        position
                    ]
                )


    # --------------------------------------------------------
    # Return deterministically sorted context.
    # --------------------------------------------------------

    final_indices = sorted(
        final_indices,
        key=lambda index: (
            chunks[
                index
            ][
                "episode_id"
            ],
            int(
                chunks[
                    index
                ].get(
                    "start_seconds",
                    0
                )
            ),
        )
    )


    return final_indices


# ============================================================
# FULL CONTEXT EXPANSION
# ============================================================

def build_generation_context(
    reranked_results,
    neighbor_window=NEIGHBOR_WINDOW,
    max_gap_seconds=MAX_CONTEXT_GAP_SECONDS
):
    """
    Generation context pipeline:

        Top-K reranked chunks
              ↓
        neighbor expansion
              ↓
        short-gap span filling
              ↓
        final grounded-generation context
    """

    neighbor_context = (
        expand_reranked_context(
            reranked_results,
            window=neighbor_window
        )
    )


    expanded_context = (
        fill_short_context_gaps(
            neighbor_context,
            max_gap_seconds=max_gap_seconds
        )
    )


    return expanded_context


# ============================================================
# CONFIGURABLE SEARCH
# ============================================================

def search_with_config(
    question,
    bm25_top_k=BM25_TOP_K,
    vector_top_k=VECTOR_TOP_K,
    hybrid_top_k=HYBRID_TOP_K,
    final_top_k=FINAL_TOP_K,
    rrf_k=RRF_K,
    neighbor_window=NEIGHBOR_WINDOW,
    max_context_gap_seconds=MAX_CONTEXT_GAP_SECONDS
):
    """
    Complete retrieval pipeline:

        Query
          ↓
        BM25 Top-K
          +
        Dense Top-K
          ↓
        RRF fusion
          ↓
        Cross-encoder reranking
          ↓
        Final retrieval Top-K
          ↓
        Neighbor expansion
          ↓
        Short-gap span filling
          ↓
        Generation context

    `reranked` remains the actual ranked retrieval result.

    `expanded_context` is only for grounded generation.
    """

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results = bm25_search(
        question,
        top_k=bm25_top_k
    )


    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    vector_results = vector_search(
        question,
        top_k=vector_top_k
    )


    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    hybrid_results = (
        reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            k=rrf_k,
            top_k=hybrid_top_k
        )
    )


    # --------------------------------------------------------
    # Cross-encoder
    # --------------------------------------------------------

    reranked = rerank_results(
        question,
        hybrid_results,
        final_top_k=final_top_k
    )


    # --------------------------------------------------------
    # Context expansion
    # --------------------------------------------------------

    expanded_context = (
        build_generation_context(
            reranked,
            neighbor_window=neighbor_window,
            max_gap_seconds=(
                max_context_gap_seconds
            )
        )
    )


    return {
        "bm25":
            bm25_results,

        "vector":
            vector_results,

        "hybrid":
            hybrid_results,

        "reranked":
            reranked,

        "expanded_context":
            expanded_context,
    }


# ============================================================
# DEFAULT SEARCH
# ============================================================

def search(
    question
):
    """
    Default FounderTechTok Intelligence search.

    Retrieval:
        BM25 Top-20
        Dense Top-20
        RRF Top-20
        Cross-encoder Top-5

    Generation context:
        one neighboring chunk each side
        +
        fill short transcript gaps <= 120 sec
    """

    return search_with_config(
        question=question,
        bm25_top_k=BM25_TOP_K,
        vector_top_k=VECTOR_TOP_K,
        hybrid_top_k=HYBRID_TOP_K,
        final_top_k=FINAL_TOP_K,
        rrf_k=RRF_K,
        neighbor_window=NEIGHBOR_WINDOW,
        max_context_gap_seconds=(
            MAX_CONTEXT_GAP_SECONDS
        )
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    question,
    results
):
    """
    Pretty-print retrieval + generation context.
    """

    print(
        "\n================================"
    )

    print(
        "QUESTION"
    )

    print(
        "================================"
    )

    print(
        question
    )


    # ========================================================
    # FINAL RETRIEVAL
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "FINAL RERANKED RESULTS"
    )

    print(
        "================================"
    )


    for rank, (
        index,
        score
    ) in enumerate(
        results[
            "reranked"
        ],
        start=1
    ):

        chunk = chunks[
            index
        ]


        print(
            f"\nRESULT {rank}"
        )


        print(
            "Reranker score:",
            round(
                score,
                4
            )
        )


        print(
            "Episode:",
            chunk[
                "episode_id"
            ]
        )


        print(
            "Guest:",
            chunk.get(
                "guest",
                ""
            )
        )


        print(
            "Timestamp:",
            chunk[
                "start_time"
            ],
            "-",
            chunk[
                "end_time"
            ]
        )


        print(
            "Chunk ID:",
            chunk[
                "chunk_id"
            ]
        )


        print(
            "Text:"
        )


        print(
            chunk[
                "text"
            ]
        )


    # ========================================================
    # GENERATION CONTEXT
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "EXPANDED GENERATION CONTEXT"
    )

    print(
        "================================"
    )


    print(
        "Chunks:",
        len(
            results[
                "expanded_context"
            ]
        )
    )


    for position, index in enumerate(
        results[
            "expanded_context"
        ],
        start=1
    ):

        chunk = chunks[
            index
        ]


        print(
            f"\nCONTEXT {position}"
        )


        print(
            "Episode:",
            chunk[
                "episode_id"
            ]
        )


        print(
            "Timestamp:",
            chunk[
                "start_time"
            ],
            "-",
            chunk[
                "end_time"
            ]
        )


        print(
            "Chunk ID:",
            chunk[
                "chunk_id"
            ]
        )


        print(
            chunk[
                "text"
            ]
        )


# ============================================================
# CLI
# ============================================================

def main():

    print(
        "\nFounderTechTok "
        "Multi-Episode Search"
    )


    print(
        "\nRetrieval configuration:"
    )


    print(
        "BM25 top-k:",
        BM25_TOP_K
    )


    print(
        "Vector top-k:",
        VECTOR_TOP_K
    )


    print(
        "Hybrid top-k:",
        HYBRID_TOP_K
    )


    print(
        "Final reranked top-k:",
        FINAL_TOP_K
    )


    print(
        "Neighbor window:",
        NEIGHBOR_WINDOW
    )


    print(
        "Max context gap:",
        MAX_CONTEXT_GAP_SECONDS,
        "seconds"
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


        results = search(
            question
        )


        print_results(
            question,
            results
        )


if __name__ == "__main__":

    main()