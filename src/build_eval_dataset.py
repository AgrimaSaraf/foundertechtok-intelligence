import json
from pathlib import Path


INGESTED_DIR = Path("data/ingested")


def load_episodes():
    episodes = []

    for path in sorted(INGESTED_DIR.glob("episode_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        episodes.append(data)

    return episodes


def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def main():

    episodes = load_episodes()

    print("=" * 70)
    print("FOUNDERTECHTOK EVALUATION DATASET BUILDER")
    print("=" * 70)

    for episode in episodes:

        episode_id = episode.get("episode_id", "unknown")
        title = episode.get("title", "")
        guest = episode.get("guest", "")
        segments = episode.get("segments", [])

        print("\n")
        print("=" * 70)
        print(f"{episode_id}")
        print(f"Guest: {guest}")
        print(f"Title: {title}")
        print(f"Segments: {len(segments)}")
        print("=" * 70)

        if not segments:
            print("No segments found.")
            continue

        # Roughly 10 samples spread across the episode
        step = max(1, len(segments) // 10)

        for i in range(0, len(segments), step):

            segment = segments[i]

            start = segment.get("start_seconds", 0)
            text = segment.get("text", "").strip()

            if not text:
                continue

            print(
                f"\n[{format_timestamp(start)}] "
                f"{text[:500]}"
            )


if __name__ == "__main__":
    main()