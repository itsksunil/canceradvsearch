# app.py

import streamlit as st
import json
from difflib import get_close_matches

# Load the JSON dataset
@st.cache_data
def load_data():
    with open("cancer_clinical_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()

# App title and description
st.title("🧬 Atezolizumab Clinical Q&A Assistant")
st.markdown("Ask questions about Atezolizumab trials, immune mechanisms, PD-L1, or clinical results.")

# User input
user_question = st.text_input("🔍 Ask your clinical question:")

# Search logic using fuzzy matching
def find_best_match(question, dataset):
    prompts = [entry["prompt"] for entry in dataset]
    matches = get_close_matches(question, prompts, n=1, cutoff=0.4)
    if matches:
        for entry in dataset:
            if entry["prompt"] == matches[0]:
                return entry
    return None

# Show answer
if user_question:
    result = find_best_match(user_question, data)
    if result:
        st.success("✅ Match found!")
        st.markdown(f"**Prompt:** {result['prompt']}")
        st.markdown(f"**Answer:** {result['completion']}")
        st.markdown("---")
        st.json(result)
    else:
        st.error("❌ No match found. Try rephrasing your question.")
