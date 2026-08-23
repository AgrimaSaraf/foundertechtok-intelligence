from pathlib import Path
import re


# 1. Load transcript
transcript_path = Path("data/transcripts/episode_01.txt")

with open(transcript_path, "r", encoding="utf-8") as file:
    transcript = file.read()


# 2. Split transcript into sentences
sentences = re.split(r'(?<=[.!?])\s+', transcript)


# 3. Create chunks with overlap
max_chunk_size = 1000
overlap_sentences = 2

chunks = []
episode_number = 1
guest_name = "Khushy Aggarwal"

chunks_with_metadata = []

for index, chunk in enumerate(chunks):
    chunk_data = {
        "chunk_id": index + 1,
        "episode": episode_number,
        "guest": guest_name,
        "text": chunk
    }

    chunks_with_metadata.append(chunk_data)
current_sentences = []
current_length = 0

for sentence in sentences:
    sentence_length = len(sentence)

    if current_length + sentence_length <= max_chunk_size:
        current_sentences.append(sentence)
        current_length += sentence_length

    else:
        chunks.append(" ".join(current_sentences))

        current_sentences = current_sentences[-overlap_sentences:]
        current_sentences.append(sentence)

        current_length = sum(len(s) for s in current_sentences)

if current_sentences:
    chunks.append(" ".join(current_sentences))


# 4. Add metadata
episode_number = 1
guest_name = "Khushy Aggarwal"

chunks_with_metadata = []

for index, chunk in enumerate(chunks):

    chunk_data = {
        "chunk_id": index + 1,
        "episode": episode_number,
        "guest": guest_name,
        "text": chunk
    }

    chunks_with_metadata.append(chunk_data)



#5. Show the results
print("Number of chunks:", len(chunks))

print("\nFIRST CHUNK:\n")
print(chunks[0])

print("\nSECOND CHUNK:\n")
print(chunks[1])

print("\nTHIRD CHUNK:\n")
print(chunks[2])