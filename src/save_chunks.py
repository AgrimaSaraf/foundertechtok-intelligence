from pathlib import Path
import re
import json


# 1. Load transcript
transcript_path = Path("data/transcripts/episode_01.txt")

with open(transcript_path, "r", encoding="utf-8") as file:
    transcript = file.read()


# 2. Find timestamp + text segments
pattern = r'(?m)^(\d{1,2}:\d{2})\s*\n'

parts = re.split(pattern, transcript)


# 3. Create timestamped segments
segments = []

for i in range(1, len(parts), 2):
    timestamp = parts[i]
    text = parts[i + 1].strip()

    segments.append({
        "timestamp": timestamp,
        "text": text
    })


# 4. Chunk the segments
max_chunk_size = 1000
overlap_segments = 2

chunks = []
current_segments = []
current_length = 0

for segment in segments:

    segment_length = len(segment["text"])

    if current_length + segment_length <= max_chunk_size:
        current_segments.append(segment)
        current_length += segment_length

    else:

        if current_segments:

            chunks.append({
                "start_time": current_segments[0]["timestamp"],
                "text": " ".join(
                    segment["text"]
                    for segment in current_segments
                )
            })

        current_segments = current_segments[-overlap_segments:]
        current_segments.append(segment)

        current_length = sum(
            len(segment["text"])
            for segment in current_segments
        )


# 5. Add final chunk
if current_segments:

    chunks.append({
        "start_time": current_segments[0]["timestamp"],
        "text": " ".join(
            segment["text"]
            for segment in current_segments
        )
    })


# 6. Add metadata
episode_number = 1
guest_name = "Khushy Aggarwal"

chunks_with_metadata = []

for index, chunk in enumerate(chunks):

    chunks_with_metadata.append({
        "chunk_id": index + 1,
        "episode": episode_number,
        "guest": guest_name,
        "start_time": chunk["start_time"],
        "text": chunk["text"]
    })


# 7. Create processed folder
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)


# 8. Save chunks to JSON
output_path = output_folder / "episode_01_chunks.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        chunks_with_metadata,
        file,
        indent=2,
        ensure_ascii=False
    )


print("Saved successfully!")
print("Number of chunks:", len(chunks_with_metadata))
print("Output file:", output_path)
