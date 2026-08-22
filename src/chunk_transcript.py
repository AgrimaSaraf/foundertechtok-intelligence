from pathlib import Path

transcript_path = Path("data/transcripts/episode_01.txt")

with open(transcript_path, "r", encoding="utf-8") as file:
    transcript = file.read()

chunk_size = 1000

chunks = []

for i in range(0, len(transcript), chunk_size):
    chunk = transcript[i:i + chunk_size]
    chunks.append(chunk)

print("Number of chunks:", len(chunks))

print("\nFIRST CHUNK:\n")
print(chunks[0])
