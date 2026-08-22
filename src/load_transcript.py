from pathlib import Path

transcript_path = Path("data/transcripts/episode_01.txt")

with open(transcript_path, "r", encoding="utf-8") as file:
    transcript = file.read()
    
print("Transcript loaded successfully!")
print("Number of characters:", len(transcript))
