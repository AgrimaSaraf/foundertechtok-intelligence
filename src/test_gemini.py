from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain RAG in one simple sentence."
)

print(response.text)