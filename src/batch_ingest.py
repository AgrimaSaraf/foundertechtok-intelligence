from pathlib import Path
import json

from manual_ingest import ingest_file, save_episode


TRANSCRIPTS_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("data/ingested")
STATE_PATH = Path("data/ingestion_state.json")


def load_state():
    if STATE_PATH.exists():
        with open(
            STATE_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    return {
        "episodes": {}
    }


def save_state(state):
    with open(
        STATE_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False
        )


def episode_id_from_path(path: Path) -> str:
    return path.stem


def main():

    state = load_state()

    transcript_files = sorted(
        TRANSCRIPTS_DIR.glob(
            "episode_*.txt"
        )
    )

    print(
        "Found transcript files:",
        len(transcript_files)
    )

    successful = 0
    skipped = 0
    failed = 0

    for transcript_path in transcript_files:

        episode_id = (
            episode_id_from_path(
                transcript_path
            )
        )

        output_path = (
            OUTPUT_DIR
            / f"{episode_id}.json"
        )

        print(
            "\n================================"
        )
        print(
            "Processing:",
            episode_id
        )
        print(
            "================================"
        )

        existing_state = (
            state["episodes"].get(
                episode_id
            )
        )

        if (
            existing_state
            and existing_state.get(
                "status"
            ) == "ingested"
            and output_path.exists()
        ):
            print(
                "Already ingested. Skipping."
            )

            skipped += 1
            continue

        try:

            data = ingest_file(
                transcript_path=
                    transcript_path,
                episode_id=
                    episode_id
            )

            save_episode(
                data,
                output_path
            )

            state["episodes"][
                episode_id
            ] = {
                "status":
                    "ingested",

                "source_type":
                    data[
                        "source_type"
                    ],

                "episode_title":
                    data[
                        "episode_title"
                    ],

                "guest":
                    data[
                        "guest"
                    ],

                "segment_count":
                    data[
                        "segment_count"
                    ],

                "output_path":
                    str(
                        output_path
                    ),
            }

            save_state(
                state
            )

            print(
                "Success"
            )

            print(
                "Title:",
                data[
                    "episode_title"
                ]
            )

            print(
                "Guest:",
                data[
                    "guest"
                ]
            )

            print(
                "Segments:",
                data[
                    "segment_count"
                ]
            )

            successful += 1

        except Exception as error:

            print(
                "FAILED:",
                error
            )

            state["episodes"][
                episode_id
            ] = {
                "status":
                    "failed",

                "error":
                    str(error),
            }

            save_state(
                state
            )

            failed += 1


    print(
        "\n================================"
    )

    print(
        "BATCH INGESTION SUMMARY"
    )

    print(
        "================================"
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Skipped:",
        skipped
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Total discovered:",
        len(
            transcript_files
        )
    )

    print(
        "\nState file:",
        STATE_PATH
    )


if __name__ == "__main__":
    main()