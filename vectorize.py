import os
import pandas as pd
import requests
import pickle
from dotenv import load_dotenv

# Load .env and Cohere key
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found in .env")

# Read CSV
df = pd.read_csv("profiles.csv")

columns = [
    "1.  What do you love doing?",
    "2. What are you really good at?",
    "3. What does the world need that you care about?",
    "4. What kind of teammate are you looking for?",
    "5. How many hours per day can you commit?",
    "6. What is your preferred working style?",
    "7. What do you value most in life or work?",
    "8. When you're starting something new, which approach sounds most like you?"
]

# Join all answers into one text string
df["combined"] = df[columns].fillna("").agg(" ".join, axis=1)

print("⏳ Sending to Cohere for embedding...")
vectors = []

url = "https://api.cohere.ai/v1/embed"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

for i, text in enumerate(df["combined"]):
    data = {
        "model": "embed-english-v3.0",
        "texts": [text],
        "input_type": "search_document"
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    embedding = resp.json()["embeddings"][0]
    vectors.append(embedding)

# Save everything
with open("vectors.pkl", "wb") as f:
    pickle.dump({"df": df, "vectors": vectors}, f)

print(f"✅ Saved vectors.pkl with {len(vectors)} profiles.")
