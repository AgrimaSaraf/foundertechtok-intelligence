from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import sys

from youtube_transcript_api import YouTubeTranscriptApi


OUTPUT_DIR = Path("data/transcripts_json")


def extract_video_id(url_or_id: str) -> str:
    """
    Accept either:
    - raw YouTube video ID
    - https://youtu.be/<id>
    - https://www.youtube.com/watch?v=<id>
    - https://youtube.com/shorts/<id>
    """

    value = url_or_id.strip()

    # Raw video ID
    if (
        "youtube.com" not in value
        and "youtu.be" not in value
        and "/" not in value
    ):
        return value

    parsed = urlparse(value)

    # youtu.be/<video_id>
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0]

    # youtube.com/watch?v=<video_id>
    if "youtube.com" in parsed.netloc:

        if parsed.path == "/watch":
            query = parse_qs(parsed.query)

            if "v" in query:
                return query["v"][0]

        # youtube.com/shorts/<id>
        # youtube.com/embed/<id>
        # youtube.com/live/<id>
        parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        if len(parts) >= 2:
            if parts[0] in {
                "shorts",
                "embed",
                "live"
            }:
                return parts[1]

    raise ValueError(
        f"Could not extract YouTube video ID from: {value}"
    )


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS or MM:SS.
    """

    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def fetch_transcript(video_id: str):
    """
    Fetch an English transcript from YouTube.
    """

    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id,
        languages=["en"]
    )

    return transcript


def normalize_transcript(
    video_id: str,
    source_url: str,
    transcript
):
    """
    Convert YouTube transcript snippets into
    our standardized episode JSON structure.
    """

    segments = []

    for snippet in transcript:

        segments.append(
            {
                "start_seconds": round(
                    float(snippet.start),
                    3
                ),
                "start_time": seconds_to_timestamp(
                    snippet.start
                ),
                "duration_seconds": round(
                    float(snippet.duration),
                    3
                ),
                "text": snippet.text.strip(),
            }
        )

    result = {
        "video_id": video_id,
        "youtube_url": source_url,
        "language": transcript.language,
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
        "segment_count": len(segments),
        "segments": segments,
    }

    return result


def save_transcript(
    data,
    output_path: Path
):
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

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python3 src/youtube_ingest.py "
            "<youtube_url_or_video_id> "
            "[episode_number]"
        )

        sys.exit(1)

    youtube_input = sys.argv[1]

    video_id = extract_video_id(
        youtube_input
    )

    print(
        "Video ID:",
        video_id
    )

    print(
        "Fetching transcript..."
    )

    transcript = fetch_transcript(
        video_id
    )

    print(
        "Transcript fetched successfully."
    )

    data = normalize_transcript(
        video_id=video_id,
        source_url=youtube_input,
        transcript=transcript
    )

    # Optional episode number
    if len(sys.argv) >= 3:

        episode_number = int(
            sys.argv[2]
        )

        output_filename = (
            f"episode_{episode_number:02d}.json"
        )

    else:

        output_filename = (
            f"{video_id}.json"
        )

    output_path = (
        OUTPUT_DIR
        / output_filename
    )

    save_transcript(
        data,
        output_path
    )

    print(
        "\nSaved transcript:"
    )

    print(
        output_path
    )

    print(
        "Segments:",
        data["segment_count"]
    )

    if data["segments"]:

        print(
            "\nFirst segment:"
        )

        print(
            data["segments"][0]
        )


if __name__ == "__main__":
    main()
    