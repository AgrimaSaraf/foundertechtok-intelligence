from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

CHUNKS_PATH = Path(
    "data/processed/all_episode_chunks.json"
)

CHROMA_PATH = "data/chroma_db_multi"

COLLECTION_NAME = "foundertechtok_multi"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 64


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)


print(
    "Chunks loaded:",
    len(chunks)
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    "Embedding model loaded."
)


# ============================================================
# CHROMA SETUP
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# Delete old collection if it exists.
# This makes development runs deterministic.
try:
    client.delete_collection(
        name=COLLECTION_NAME
    )

    print(
        "Deleted existing collection."
    )

except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# PREPARE DATA
# ============================================================

ids = []

documents = []

metadatas = []


for chunk in chunks:

    ids.append(
        chunk["chunk_id"]
    )

    documents.append(
        chunk["text"]
    )

    metadata = {
        "episode_id":
            chunk.get(
                "episode_id",
                ""
            ),

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
            chunk.get(
                "start_time",
                ""
            ),

        "start_seconds":
            int(
                chunk.get(
                    "start_seconds",
                    0
                )
            ),

        "end_time":
            chunk.get(
                "end_time",
                ""
            ),

        "end_seconds":
            int(
                chunk.get(
                    "end_seconds",
                    0
                )
            ),
    }

    metadatas.append(
        metadata
    )


# ============================================================
# EMBED + UPSERT IN BATCHES
# ============================================================

total = len(documents)


for start in range(
    0,
    total,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total
    )

    batch_documents = documents[
        start:end
    ]

    batch_ids = ids[
        start:end
    ]

    batch_metadatas = metadatas[
        start:end
    ]


    print(
        f"Embedding chunks {start + 1}-{end} "
        f"of {total}"
    )


    embeddings = embedding_model.encode(
        batch_documents,
        show_progress_bar=False
    )


    collection.add(
        ids=batch_ids,
        documents=batch_documents,
        metadatas=batch_metadatas,
        embeddings=[
            embedding.tolist()
            for embedding in embeddings
        ]
    )


# ============================================================
# VERIFY
# ============================================================

stored_count = collection.count()


print(
    "\n================================"
)

print(
    "MULTI-EPISODE VECTOR STORE"
)

print(
    "================================"
)

print(
    "Input chunks:",
    len(chunks)
)

print(
    "Stored vectors:",
    stored_count
)

print(
    "Collection:",
    COLLECTION_NAME
)

print(
    "Database:",
    CHROMA_PATH
)


if stored_count != len(chunks):

    raise RuntimeError(
        "Vector-store validation failed: "
        f"expected {len(chunks)} vectors, "
        f"found {stored_count}."
    )


print(
    "\nVector-store validation: PASS"
)