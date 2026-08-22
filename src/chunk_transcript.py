from pathlib import Path
import re


# Load the transcript
transcript_path = Path("data/transcripts/episode_01.txt")

with open(transcript_path, "r", encoding="utf-8") as file:
    transcript = file.read()


# Split the transcript into sentences
sentences = re.split(r'(?<=[.!?])\s+', transcript)


# Maximum size of each chunk
max_chunk_size = 1000

chunks = []
current_chunk = ""


# Build chunks sentence by sentence
for sentence in sentences:

    if len(current_chunk) + len(sentence) <= max_chunk_size:
        current_chunk += sentence + " "

    else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence + " "


# Add the final chunk
if current_chunk:
    chunks.append(current_chunk.strip())


# Show the results
print("Number of chunks:", len(chunks))

print("\nFIRST CHUNK:\n")
print(chunks[0])

print("\nSECOND CHUNK:\n")
print(chunks[1])

print("\nSECOND CHUNK:\n")
print(chunks[1])