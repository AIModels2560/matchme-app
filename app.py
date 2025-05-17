import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
from dotenv import load_dotenv
import time

# Load keys
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Question map with types and labels
questions = [
    {
        "key": "Energies",
        "title": "✨ What gives you energy? (Pick any 2)",
        "type": "multiselect",
        "options": [
            "Solving hard problems", "Creating beautiful things", "Helping people succeed",
            "Making money and building wealth", "Learning and exploring",
            "Leading teams to a vision", "Building systems that scale"
        ],
        "limit": 2
    },
    {
        "key": "Ambition",
        "title": "🚀 What best describes your ambition right now?",
        "type": "radio",
        "options": [
            "I want to bring something meaningful into the world.",
            "I want to build a rocketship and scale it fast.",
            "I want to experiment, learn, and grow as I build.",
            "I want to be my own boss and live on my terms."
        ]
    },
    {
        "key": "Pride Point",
        "title": "🏆 If your startup succeeds, what will you be most proud of?",
        "type": "radio",
        "options": [
            "The impact it created", "The innovation and product",
            "The revenue and market dominance", "The community and culture",
            "The journey and personal growth"
        ]
    },
    {
        "key": "Work Style",
        "title": "🕒 How do you prefer to work?",
        "type": "radio",
        "options": [
            "Async – Work independently, communicate flexibly (text/email)",
            "Sync – Prefer real-time collaboration (calls, meetings)",
            "Flexible – Comfortable with both async and sync"
        ]
    },
    {
        "key": "Co-Founder Trait",
        "title": "🧠 What trait best defines you as a co-founder?",
        "type": "radio",
        "options": [
            "Visionary", "Operator", "Product Builder",
            "Storyteller", "Technical Architect", "People Leader"
        ]
    },
    {
        "key": "Bad Day Response",
        "title": "💭 On a bad day, how do you usually respond?",
        "type": "radio",
        "options": [
            "Recharge solo", "Brainstorm with someone",
            "Push harder and fix it", "Step back and reflect"
        ]
    },
    {
        "key": "Top Skill",
        "title": "💡 What's your strongest current skill?",
        "type": "text"
    },
    {
        "key": "Daily Commitment",
        "title": "⏱️ How much time can you commit daily?",
        "type": "radio",
        "options": [
            "< 2 hours", "2–4 hours", "4–6 hours", "I can work for 25hrs everyday!!!"
        ]
    },
    {
        "key": "Fav Stage",
        "title": "📈 Which project stage excites you most?",
        "type": "radio",
        "options": [
            "0 → 1 Idea to MVP", "MVP → PMF",
            "Scaling and fundraising", "Growth & operations"
        ]
    },
    {
        "key": "Co-Founder Values",
        "title": "🤝 What do you value most in a co-founder? (Pick any 3)",
        "type": "multiselect",
        "options": [
            "Trust", "Speed of execution", "Creativity",
            "Technical strength", "Emotional intelligence", "Strategic thinking"
        ],
        "limit": 3
    },
    {
        "key": "Fit Preference",
        "title": "🧩 Which statement sounds most like you?",
        "type": "radio",
        "options": [
            "We don’t have to always agree, but we should respect each other’s craft.",
            "I want to build with someone who pushes me constantly.",
            "I value chemistry and shared ambition more than skills."
        ]
    },
    {
        "key": "Red Flag",
        "title": "🚫 What's a red flag for you in a co-founder?",
        "type": "text"
    }
]

# Initialize session state
if "stage" not in st.session_state:
    st.session_state.stage = "landing"
if "name" not in st.session_state:
    st.session_state.name = ""
if "email" not in st.session_state:
    st.session_state.email = ""
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "user_vector" not in st.session_state:
    st.session_state.user_vector = None
if "matches" not in st.session_state:
    st.session_state.matches = []

# Embedding
def embed_text(text):
    res = requests.post(
        "https://api.cohere.ai/v1/embed",
        headers={"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"},
        json={"model": "embed-english-v3.0", "texts": [text], "input_type": "search_document"}
    )
    res.raise_for_status()
    return res.json()["embeddings"][0]

# Match explanation using Groq
def explain_match(user_text, match_text):
    prompt = f"""
User: {user_text}

Match: {match_text}

Explain why these two would be great co-founders.
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 200
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# Stage: Landing
if st.session_state.stage == "landing":
    st.set_page_config(page_title="MatchMe AI", layout="centered")
    st.title("🚀 MatchMe AI")
    with st.form("user_details"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        submit = st.form_submit_button("Continue")
        if submit and name and email:
            st.session_state.name = name
            st.session_state.email = email
            st.session_state.stage = "questions"
            st.rerun()

# Stage: Questions
elif st.session_state.stage == "questions":
    q = questions[st.session_state.q_index]
    st.title(f"Step {st.session_state.q_index + 1} of {len(questions)}")
    st.header(q["title"])

    answer = None
    if q["type"] == "text":
        answer = st.text_input("Your answer", st.session_state.answers.get(q["key"], ""))
    elif q["type"] == "radio":
        answer = st.radio("Choose one", q["options"], index=q["options"].index(st.session_state.answers.get(q["key"], q["options"][0])) if st.session_state.answers.get(q["key"]) else 0)
    elif q["type"] == "multiselect":
        answer = st.multiselect("Choose options", q["options"], default=st.session_state.answers.get(q["key"], []))

        if "limit" in q and len(answer) > q["limit"]:
            st.warning(f"Please select only {q['limit']} option(s).")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back") and st.session_state.q_index > 0:
            st.session_state.q_index -= 1
            st.rerun()
    with col2:
        if st.button("Next ➡"):
            if not answer or (q["type"] == "multiselect" and "limit" in q and len(answer) > q["limit"]):
                st.warning("Please complete this step correctly.")
            else:
                st.session_state.answers[q["key"]] = answer
                st.session_state.q_index += 1
                if st.session_state.q_index >= len(questions):
                    st.session_state.stage = "loading"
                st.rerun()

# Stage: Loading
elif st.session_state.stage == "loading":
    st.title("🔍 Finding the partner for you...")
    with st.spinner("Analyzing your responses and scanning the network..."):
        time.sleep(3)
        user_input = " ".join(
            [", ".join(v) if isinstance(v, list) else v for v in st.session_state.answers.values()]
        )
        user_vector = np.array(embed_text(user_input))
        st.session_state.user_vector = user_vector

        with open("vectors.pkl", "rb") as f:
            data = pickle.load(f)
        df = data["df"]
        vectors = np.array(data["vectors"])
        similarities = vectors @ user_vector.T
        top_indices = similarities.flatten().argsort()[::-1][:5]

        results = []
        for idx in top_indices:
            match = df.iloc[idx]
            explanation = explain_match(user_input, match["combined"])
            results.append({
                "name": match["Name"],
                "email": match["Email"],
                "strengths": match["Top Skill"],
                "fit_score": round(float(similarities[idx]) * 100),
                "explanation": explanation
            })

        st.session_state.matches = results
        st.session_state.stage = "results"
        st.rerun()

# Stage: Results
elif st.session_state.stage == "results":
    st.title(f"Hi {st.session_state.name}, here are your top matches! 💡")

    st.subheader("🧠 Your strengths & interests")
    for k, v in st.session_state.answers.items():
        if k in ["Top Skill", "Energies", "Co-Founder Trait", "Work Style", "Fav Stage"]:
            st.markdown(f"**{k}:** {', '.join(v) if isinstance(v, list) else v}")

    st.subheader("🎯 Your Top 5 Matches")
    for i, match in enumerate(st.session_state.matches):
        with st.expander(f"#{i+1} {match['name']} — Score: {match['fit_score']}%"):
            st.markdown(f"**Strengths:** {match['strengths']}")
            st.markdown(f"**Why this match:** {match['explanation']}")
