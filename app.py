import streamlit as st

st.set_page_config(page_title="MatchMe - Founder Match", layout="centered")

st.title("👥 MatchMe - Your Ideal Founding Team")

# --- Your Profile ---
st.subheader("🧑 Your Profile")

st.markdown("""
**What you love doing:**  
Building something meaningful that solves real problems

**What you're good at:**  
Product strategy and communicating ideas clearly

**What the world needs:**  
More tools that simplify everyday life and help people save time

**Looking for:**  
A technical co-founder or creative partner who gets things done

**Hours per week:**  
20–30

**Working style:**  
I like working independently and checking in occasionally (Async)

**Core value:**  
Purpose & impact

**Startup approach:**  
I move when things are 70% ready — progress over perfection
""")

# --- Match Results ---
st.subheader("🔍 Your Best Matches")

with st.container():
    st.markdown("### 🧠 Match #1: Sneha AI")
    st.markdown("**Why this works:**")
    st.success("You and Sneha both thrive in fast-paced, purpose-driven environments — your bold execution mindset and her deep technical focus on AI could create a strong mission-aligned team.")

with st.container():
    st.markdown("### 🌐 Match #2: Rishi Connect")
    st.markdown("**Why this works:**")
    st.success("You and Rishi both care deeply about creating meaningful, people-first experiences — while you bring structured execution, Rishi adds community energy, making this a balanced and values-aligned team.")

st.markdown("---")
st.caption("Built for the 100X Hackathon 🚀")
