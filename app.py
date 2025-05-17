import pickle
import streamlit as st
import numpy as np
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -- SESSION STATE INIT --
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "name" not in st.session_state:
    st.session_state.name = ""
if "email" not in st.session_state:
    st.session_state.email = ""

# -- LOGIN PAGE --
if not st.session_state.logged_in:
    st.title("👋 Welcome to MatchMe")
    st.markdown("This tool helps you find the most compatible co-founders based on your Ikigai and profile data.")
    
    with st.form("login_form"):
        name = st.text_input("Your Full Name")
        email = st.text_input("Email Address")
        login = st.form_submit_button("Continue")

        if login:
            if name.strip() and email.strip():
                st.session_state.logged_in = True
                st.session_state.name = name.strip()
                st.session_state.email = email.strip()
                st.success("Login successful! Scroll down if not redirected.")
                st.rerun()
            else:
                st.error("Please enter both name and email to continue.")

# -- MAIN APP FLOW --
if st.session_state.logged_in:
    st.title("🧠 MatchMe: AI-Powered Founder Matchmaking")
    st.markdown(f"Hi **{st.session_state.name}**! Please answer a few quick questions.")

    with st.form("ikigai_form"):
        q1 = st.text_input("1. What do you love doing?")
        q2 = st.text_input("2. What are you really good at?")
        q3 = st.text_input("3. What does the world need that you care about?")
        q4 = st.text_input("4. What kind of teammate are you looking for?")
        q5 = st.selectbox("5. How many hours per day can you commit?", ["1-2", "2-4", "4-8", "8+"])

        # 🧠 Working style with definitions
        working_styles = {
            "Async – Work independently and communicate when needed (e.g., through messages, not live calls)": "Async",
            "Sync – Work in real-time with others, prefer meetings and live discussions": "Sync",
            "Hybrid – Mix of async focus and live sync-ups depending on task": "Hybrid",
            "Flexible – Open to any mode depending on team": "Flexible"
        }
        selected_style = st.selectbox("6. What is your preferred working style?", list(working_styles.keys()))
        q6 = working_styles[selected_style]  # Store clean version

        q7 = st.text_input("7. What do you value most in life or work?")
        q8 = st.text_input("8. When you're starting something new, which approach sounds most like you?")

        submitted = st.form_submit_button("🔍 Find My Matches")

    if submitted:
        combined_input = " ".join([q1, q2, q3, q4, q5, q6, q7, q8])

        if not combined_input.strip():
            st.warning("Please fill all fields before submitting.")
        else:
            with st.spinner("Finding your best-fit profiles..."):

                # Load vectors and profiles
                with open("vectors.pkl", "rb") as f:
                    data = pickle.load(f)
                df = data["df"]
                vectors = np.array(data["vectors"])

                # Call Cohere embed API
                embed_url = "https://api.cohere.ai/v1/embed"
                headers = {
                    "Authorization": f"Bearer {os.getenv('COHERE_API_KEY')}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "embed-english-v3.0",
                    "texts": [combined_input],
                    "input_type": "search_document"
                }
                response = requests.post(embed_url, headers=headers, json=payload)
                user_vector = np.array(response.json()["embeddings"][0]).reshape(1, -1)

                # Score similarity
                sims = vectors @ user_vector.T
                top_indices = sims.flatten().argsort()[::-1][:2]

                # Groq Explanation Function
                def explain_match(user_input, match_text):
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                        "Content-Type": "application/json"
                    }
                    prompt = f"""You are a startup team compatibility expert.

A user filled out the following self-discovery answers: 
\"\"\"{user_input}\"\"\"

You found this matching profile:
\"\"\"{match_text}\"\"\"

Write a short explanation (2 sentences) on why these two people might work well together as co-founders, based on personality, work style, and values.
"""
                    payload = {
                        "model": "llama3-70b-8192",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 200
                    }
                    try:
                        r = requests.post(url, headers=headers, json=payload)
                        return r.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        return f"❌ Explanation error: {e}"

                # Show Matches
                st.subheader("🎯 Your Best Matches")
                for i, idx in enumerate(top_indices):
                    name = df.iloc[idx]["Name"]
                    email = df.iloc[idx]["Email"]
                    match_score = float(sims[idx])
                    match_text = df.iloc[idx]["combined"]

                    explanation = explain_match(combined_input, match_text)

                    st.markdown(f"**Match #{i+1}: {name}**")
                    st.markdown(f"- ✉️ {email}")
                    st.markdown(f"- 🔢 Similarity score: `{match_score:.3f}`")
                    st.markdown(f"- 💬 Why this match: {explanation}")
                    st.markdown("---")
