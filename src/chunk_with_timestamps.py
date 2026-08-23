from pathlib import Path
import re


# 1. Load transcript
transcript_path = Path("data/transcripts/episode_01.txt")

with open(transcript_path, "r", encoding="utf-8") as file:
    transcript = file.read()


# 2. Find timestamp + text segments
pattern = r'(?m)^(\d{1,2}:\d{2})\s*\n'

parts = re.split(pattern, transcript)


# 3. Convert transcript into structured segments
segments = []

for i in range(1, len(parts), 2):
    timestamp = parts[i]
    text = parts[i + 1].strip()

    segments.append({
        "timestamp": timestamp,
        "text": text
    })


# 4. Build chunks from timestamped segments
max_chunk_size = 1000
overlap_segments = 2

chunks = []
current_segments = []
current_length = 0

for segment in segments:
    segment_text = segment["text"]
    segment_length = len(segment_text)

    if current_length + segment_length <= max_chunk_size:
        current_segments.append(segment)
        current_length += segment_length

    else:
        if current_segments:
            chunk_text = " ".join(
                segment["text"] for segment in current_segments
            )

            chunks.append({
                "start_time": current_segments[0]["timestamp"],
                "text": chunk_text
            })

        # Keep last 2 segments as overlap
        current_segments = current_segments[-overlap_segments:]
        current_segments.append(segment)

        current_length = sum(
            len(segment["text"]) for segment in current_segments
        )


# 5. Add final chunk
if current_segments:
    chunk_text = " ".join(
        segment["text"] for segment in current_segments
    )

    chunks.append({
        "start_time": current_segments[0]["timestamp"],
        "text": chunk_text
    })


# 6. Add episode metadata
episode_number = 1
guest_name = "Khushy Aggarwal"

chunks_with_metadata = []

for index, chunk in enumerate(chunks):

    chunk_data = {
        "chunk_id": index + 1,
        "episode": episode_number,
        "guest": guest_name,
        "start_time": chunk["start_time"],
        "text": chunk["text"]
    }

    chunks_with_metadata.append(chunk_data)


# 7. Inspect result
print("Number of chunks:", len(chunks_with_metadata))

print("\nFIRST CHUNK WITH TIMESTAMP:\n")
print(chunks_with_metadata[0])

print("\nSECOND CHUNK WITH TIMESTAMP:\n")
print(chunks_with_metadata[1])
