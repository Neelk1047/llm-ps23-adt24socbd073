import os
import re
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = list(dict.fromkeys([PRIMARY_MODEL, "gemini-3.5-flash-lite"]))
LEVELS = ["Kid", "Student", "Expert"]

SYSTEM_PROMPT = """You are a patient teacher who explains concepts at three levels.
Explain the concept given by the user at exactly three levels, using these exact headings:

### KID
Explain like the learner is 8 years old. Use a simple everyday analogy, very short
sentences and no jargon. Maximum 80 words.

### STUDENT
Explain for a college student who knows the basics. Use correct terms (define each new
term in brackets), give one real-world example. Maximum 150 words.

### EXPERT
Explain for a professional in the field. Use precise technical language, mention key
details, trade-offs or limitations. Maximum 200 words.

Rules: be factually correct, do not add any other headings, and if the input is not a
concept or topic (or is unsafe/meaningless), reply only with: INVALID INPUT"""


@st.cache_resource
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)


def explain(concept: str) -> str:
    client = get_client()
    last_error = None

    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=f"Concept: {concept}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT, temperature=0.4
                    ),
                )
                return resp.text or ""
            except APIError as e:
                last_error = e
                if e.code == 503:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise

    raise last_error


def split_levels(text: str) -> dict:
    """Split the model reply into {'Kid':..., 'Student':..., 'Expert':...}."""
    parts = re.split(r"###\s*(KID|STUDENT|EXPERT)\s*", text, flags=re.I)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].capitalize()] = parts[i + 1].strip()
    return out


st.set_page_config(page_title="Concept Explainer (ELI5)", page_icon="💡")
st.title("💡 Concept Explainer (ELI5)")
st.caption("Type any concept and get explanations for a Kid, a Student and an Expert.")

if get_client() is None:
    st.error("GEMINI_API_KEY not found. Copy .env.example to .env and add your key.")
    st.stop()

examples = ["Blockchain", "Photosynthesis", "Black holes", "Recursion", "Inflation"]
choice = st.selectbox("Or pick an example", [""] + examples)
concept = st.text_input("Enter a concept", value=choice, placeholder="e.g. Machine learning")

if st.button("Explain", type="primary"):
    if not concept.strip():
        st.warning("Please enter a concept first.")
    elif len(concept) > 200:
        st.warning("Please keep the concept under 200 characters.")
    else:
        try:
            with st.spinner("Thinking..."):
                raw = explain(concept.strip())
            if "INVALID INPUT" in raw.upper()[:30]:
                st.warning("That does not look like a concept. Try something like 'Photosynthesis'.")
            else:
                levels = split_levels(raw)
                if len(levels) < 3:
                    st.info("Showing the raw answer:")
                    st.markdown(raw)
                else:
                    tabs = st.tabs(["🧒 Kid", "🎓 Student", "🧑‍🔬 Expert"])
                    for tab, name in zip(tabs, LEVELS):
                        with tab:
                            st.markdown(levels[name])
        except Exception as e:  # network error, quota, bad key
            st.error(f"Could not get an answer: {e}")