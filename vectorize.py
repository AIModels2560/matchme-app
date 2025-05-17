import os
import pandas as pd
import requests
import pickle
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found in .env")

# Load CSV
df = pd.read_csv("profiles_dummy.csv")

# Define columns to use for embeddings
embed_cols = [
    "Energies", "Ambition", "Pride Point", "Work Style",
    "Co-Founder Trait", "Bad Day Response", "Top Skill",
    "Daily Commitment", "Fav Stage", "Co-Founder Values",
    "Fit Preference", "Red Flag", "Unfair Advantage"
]

# Clean text and combine for embedding
df["combined"] = df[embed_cols].fillna("").astype(str).agg(" ".join, axis=1)

# Generate embeddings
print("⏳ Embedding profiles...")
vectors = []
for i, text in enumerate(df["combined"]):
    try:
        response = requests.post(
            "https://api.cohere.ai/v1/embed",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "embed-english-v3.0",
                "texts": [text],
                "input_type": "search_document"
            }
        )
        response.raise_for_status()
        vectors.append(response.json()["embeddings"][0])
    except Exception as e:
        print(f"❌ Error on profile {i}: {e}")
        vectors.append([0.0]*768)

# Save full profile + embeddings
with open("vectors.pkl", "wb") as f:
    pickle.dump({"df": df, "vectors": vectors}, f)

print(f"✅ Saved vectors.pkl with {len(vectors)} profiles.")
