from pathlib import Path
import json
import re
import sys


OUTPUT_DIR = Path("data/ingested")


def timestamp_to_seconds(timestamp: str) -> int:
    """
    Convert MM:SS or HH:MM:SS into total seconds.
    """

    parts = timestamp.split(":")

    if len(parts) == 2:
        minutes, seconds = parts

        return (
            int(minutes) * 60
            + int(seconds)
        )

    if len(parts) == 3:
        hours, minutes, seconds = parts

        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
        )

    raise ValueError(
        f"Unsupported timestamp format: {timestamp}"
    )


def parse_header(lines):
    """
    Extract episode title and guest
    from the beginning of the transcript.
    """

    episode_title = None
    guest = None

    for line in lines[:15]:

        stripped = line.strip()

        if (
            stripped.lower().startswith("episode")
            and episode_title is None
        ):
            episode_title = stripped

        if stripped.lower().startswith(
            "guest name:"
        ):
            guest = stripped.split(
                ":",
                1
            )[1].strip()

    return (
        episode_title,
        guest
    )


def normalize_timestamp_line(line: str):
    """
    Normalize timestamps from multiple transcript formats.

    Supported examples:

    Clean:
        0:17
        12:43

    Concatenated seconds:
        0:2424 seconds
        0:3232 seconds
        12:3434 seconds

    Verbose:
        1:041 minute, 4 seconds
        12:2212 minutes, 22 seconds
    """

    line = line.strip()


    # ========================================================
    # FORMAT 1
    #
    # Examples:
    #
    # 1:041 minute, 4 seconds
    # 12:2212 minutes, 22 seconds
    # ========================================================

    verbose_match = re.search(
        r"\b(\d{1,3}):(\d{2})"
        r"\d*\s+minutes?,\s*"
        r"(\d{1,2})\s+seconds?",
        line,
        re.IGNORECASE
    )

    if verbose_match:

        minutes = int(
            verbose_match.group(1)
        )

        seconds = int(
            verbose_match.group(3)
        )

        if 0 <= seconds < 60:

            return (
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )


    # ========================================================
    # FORMAT 2
    #
    # Examples:
    #
    # 0:2424 seconds
    # 0:3232 seconds
    # 12:3434 seconds
    # ========================================================

    duplicated_match = re.search(
        r"\b(\d{1,3}):(\d{2})"
        r"\d*\s+seconds?",
        line,
        re.IGNORECASE
    )

    if duplicated_match:

        minutes = int(
            duplicated_match.group(1)
        )

        seconds = int(
            duplicated_match.group(2)
        )

        if 0 <= seconds < 60:

            return (
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )


    # ========================================================
    # FORMAT 3
    #
    # Normal timestamps:
    #
    # 0:17
    # 12:43
    # ========================================================

    normal_match = re.search(
        r"\b(\d{1,3}):(\d{2})\b",
        line
    )

    if normal_match:

        minutes = int(
            normal_match.group(1)
        )

        seconds = int(
            normal_match.group(2)
        )

        if 0 <= seconds < 60:

            return (
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )


    return None


def remove_timestamp_prefix(
    line: str
) -> str:
    """
    Remove supported timestamp formats
    from the start of a transcript line.

    Examples:

    0:17 Hello
        -> Hello

    0:2424 secondsHello
        -> Hello

    1:041 minute, 4 secondsHello
        -> Hello
    """

    remaining_text = line.strip()


    # ========================================================
    # REMOVE VERBOSE FORMAT
    # ========================================================

    remaining_text = re.sub(
        r"^\s*"
        r"\d{1,3}:\d{2}\d*"
        r"\s+minutes?,\s*"
        r"\d{1,2}\s+seconds?",
        "",
        remaining_text,
        count=1,
        flags=re.IGNORECASE
    )


    # ========================================================
    # REMOVE DUPLICATED-SECONDS FORMAT
    # ========================================================

    remaining_text = re.sub(
        r"^\s*"
        r"\d{1,3}:\d{2}\d*"
        r"\s+seconds?",
        "",
        remaining_text,
        count=1,
        flags=re.IGNORECASE
    )


    # ========================================================
    # REMOVE ORDINARY TIMESTAMP
    # ========================================================

    remaining_text = re.sub(
        r"^\s*"
        r"\d{1,3}:\d{2}\b",
        "",
        remaining_text,
        count=1
    )


    return remaining_text.strip()


def parse_transcript(
    transcript_text: str
):
    """
    Convert raw transcript text into
    timestamped transcript segments.
    """

    lines = transcript_text.splitlines()


    episode_title, guest = (
        parse_header(
            lines
        )
    )


    segments = []


    current_timestamp = None

    current_text = []


    for line in lines:

        stripped = line.strip()


        if not stripped:
            continue


        # ----------------------------------------------------
        # Skip known metadata lines
        # ----------------------------------------------------

        if (
            episode_title
            and stripped == episode_title
        ):
            continue


        if stripped.lower().startswith(
            "guest name:"
        ):
            continue


        # ----------------------------------------------------
        # Detect timestamp
        # ----------------------------------------------------

        timestamp = (
            normalize_timestamp_line(
                stripped
            )
        )


        if timestamp is not None:

            # ------------------------------------------------
            # Save previous segment
            # ------------------------------------------------

            if (
                current_timestamp
                is not None
                and current_text
            ):

                text = " ".join(
                    current_text
                ).strip()


                if text:

                    segments.append(
                        {
                            "start_seconds":
                                timestamp_to_seconds(
                                    current_timestamp
                                ),

                            "start_time":
                                current_timestamp,

                            "text":
                                text,
                        }
                    )


            # ------------------------------------------------
            # Start new segment
            # ------------------------------------------------

            current_timestamp = timestamp


            remaining_text = (
                remove_timestamp_prefix(
                    stripped
                )
            )


            current_text = []


            if remaining_text:

                current_text.append(
                    remaining_text
                )


        else:

            # ------------------------------------------------
            # Continuation of current transcript segment
            # ------------------------------------------------

            if current_timestamp is not None:

                current_text.append(
                    stripped
                )


    # ========================================================
    # SAVE FINAL SEGMENT
    # ========================================================

    if (
        current_timestamp
        is not None
        and current_text
    ):

        text = " ".join(
            current_text
        ).strip()


        if text:

            segments.append(
                {
                    "start_seconds":
                        timestamp_to_seconds(
                            current_timestamp
                        ),

                    "start_time":
                        current_timestamp,

                    "text":
                        text,
                }
            )


    return (
        episode_title,
        guest,
        segments
    )


def validate_episode_data(
    data
):
    """
    Validate generated episode JSON.

    Prevents silent success when parsing
    produces an empty transcript.
    """

    errors = []


    if not data.get(
        "episode_id"
    ):

        errors.append(
            "Missing episode_id"
        )


    if not data.get(
        "episode_title"
    ):

        errors.append(
            "Missing episode_title"
        )


    if not data.get(
        "guest"
    ):

        errors.append(
            "Missing guest"
        )


    segments = data.get(
        "segments",
        []
    )


    if len(segments) == 0:

        errors.append(
            "No transcript segments were parsed"
        )


    for index, segment in enumerate(
        segments
    ):

        if (
            "start_seconds"
            not in segment
        ):

            errors.append(
                f"Segment {index} missing start_seconds"
            )


        if (
            "start_time"
            not in segment
        ):

            errors.append(
                f"Segment {index} missing start_time"
            )


        if not segment.get(
            "text"
        ):

            errors.append(
                f"Segment {index} has empty text"
            )


    if errors:

        raise ValueError(
            "Episode validation failed:\n"
            + "\n".join(
                f"- {error}"
                for error in errors
            )
        )


def ingest_file(
    transcript_path: Path,
    episode_id: str,
    youtube_url=None,
    video_id=None
):
    """
    Convert one raw transcript file
    into standardized episode JSON.
    """

    with open(
        transcript_path,
        "r",
        encoding="utf-8"
    ) as file:

        transcript_text = (
            file.read()
        )


    (
        episode_title,
        guest,
        segments
    ) = parse_transcript(
        transcript_text
    )


    data = {

        "episode_id":
            episode_id,

        "episode_title":
            episode_title,

        "guest":
            guest,

        "youtube_url":
            youtube_url,

        "video_id":
            video_id,

        "source_type":
            "manual_transcript",

        "segment_count":
            len(
                segments
            ),

        "segments":
            segments,
    }


    # --------------------------------------------------------
    # Do not silently allow broken ingestion
    # --------------------------------------------------------

    validate_episode_data(
        data
    )


    return data


def save_episode(
    data,
    output_path: Path
):
    """
    Save standardized episode JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def main():

    # ========================================================
    # CLI VALIDATION
    # ========================================================

    if len(sys.argv) < 3:

        print(
            "Usage:"
        )

        print(
            "python3 src/manual_ingest.py "
            "<transcript_path> "
            "<episode_id>"
        )


        print(
            "\nExample:"
        )

        print(
            "python3 src/manual_ingest.py "
            "data/transcripts/episode_01.txt "
            "episode_01"
        )


        sys.exit(1)


    transcript_path = Path(
        sys.argv[1]
    )


    episode_id = (
        sys.argv[2]
    )


    if not transcript_path.exists():

        print(
            "Transcript file not found:",
            transcript_path
        )

        sys.exit(1)


    # ========================================================
    # INGEST
    # ========================================================

    try:

        data = ingest_file(

            transcript_path=
                transcript_path,

            episode_id=
                episode_id
        )


    except Exception as error:

        print(
            "\nINGESTION FAILED"
        )

        print(
            "Episode:",
            episode_id
        )

        print(
            "File:",
            transcript_path
        )

        print(
            "Reason:",
            error
        )

        sys.exit(1)


    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / f"{episode_id}.json"
    )


    save_episode(
        data,
        output_path
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "INGESTION SUCCESSFUL"
    )

    print(
        "================================"
    )


    print(
        "Episode:",
        data[
            "episode_id"
        ]
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


    print(
        "Saved to:",
        output_path
    )


    if data[
        "segments"
    ]:

        print(
            "\nFirst segment:"
        )

        print(
            data[
                "segments"
            ][0]
        )


if __name__ == "__main__":

    main()