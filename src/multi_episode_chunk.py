from pathlib import Path
import json


INGESTED_DIR = Path("data/ingested")
OUTPUT_PATH = Path(
    "data/processed/all_episode_chunks.json"
)

SEGMENTS_PER_CHUNK = 5


def build_chunks_from_episode(
    episode_data,
    segments_per_chunk=SEGMENTS_PER_CHUNK
):
    """
    Combine consecutive transcript segments into larger
    retrieval chunks while preserving episode metadata.
    """

    episode_id = episode_data["episode_id"]

    episode_title = episode_data.get(
        "episode_title"
    )

    guest = episode_data.get(
        "guest"
    )

    segments = episode_data.get(
        "segments",
        []
    )

    chunks = []

    chunk_number = 1


    for start_index in range(
        0,
        len(segments),
        segments_per_chunk
    ):

        segment_group = segments[
            start_index:
            start_index + segments_per_chunk
        ]

        if not segment_group:
            continue


        chunk_text = " ".join(
            segment["text"]
            for segment in segment_group
            if segment.get("text")
        ).strip()


        if not chunk_text:
            continue


        first_segment = segment_group[0]

        last_segment = segment_group[-1]


        chunk = {
            "chunk_id":
                (
                    f"{episode_id}"
                    f"_chunk_{chunk_number:03d}"
                ),

            "episode_id":
                episode_id,

            "episode_title":
                episode_title,

            "guest":
                guest,

            "start_time":
                first_segment.get(
                    "start_time"
                ),

            "start_seconds":
                first_segment.get(
                    "start_seconds"
                ),

            "end_time":
                last_segment.get(
                    "start_time"
                ),

            "end_seconds":
                last_segment.get(
                    "start_seconds"
                ),

            "text":
                chunk_text,
        }


        chunks.append(
            chunk
        )


        chunk_number += 1


    return chunks


def main():

    episode_files = sorted(
        INGESTED_DIR.glob(
            "episode_*.json"
        )
    )


    print(
        "Found ingested episodes:",
        len(episode_files)
    )


    all_chunks = []

    episode_summaries = []


    for episode_path in episode_files:

        print(
            "\nProcessing:",
            episode_path.name
        )


        with open(
            episode_path,
            "r",
            encoding="utf-8"
        ) as file:

            episode_data = json.load(
                file
            )


        episode_chunks = (
            build_chunks_from_episode(
                episode_data
            )
        )


        all_chunks.extend(
            episode_chunks
        )


        episode_summaries.append(
            {
                "episode_id":
                    episode_data[
                        "episode_id"
                    ],

                "guest":
                    episode_data.get(
                        "guest"
                    ),

                "segments":
                    episode_data.get(
                        "segment_count",
                        len(
                            episode_data.get(
                                "segments",
                                []
                            )
                        )
                    ),

                "chunks":
                    len(
                        episode_chunks
                    ),
            }
        )


        print(
            "Guest:",
            episode_data.get(
                "guest"
            )
        )

        print(
            "Chunks created:",
            len(
                episode_chunks
            )
        )


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
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        "\n================================"
    )

    print(
        "MULTI-EPISODE CHUNKING SUMMARY"
    )

    print(
        "================================"
    )


    print(
        "Episodes processed:",
        len(
            episode_files
        )
    )

    print(
        "Total chunks:",
        len(
            all_chunks
        )
    )


    print(
        "\nPer-episode breakdown:"
    )


    for summary in episode_summaries:

        print(
            summary[
                "episode_id"
            ],
            "| Guest:",
            summary[
                "guest"
            ],
            "| Segments:",
            summary[
                "segments"
            ],
            "| Chunks:",
            summary[
                "chunks"
            ],
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


    if all_chunks:

        print(
            "\nFirst chunk:"
        )

        print(
            all_chunks[0]
        )


if __name__ == "__main__":
    main()