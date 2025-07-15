import streamlit as st
import json
from difflib import get_close_matches
import pandas as pd
from datetime import datetime

# Constants
DATA_FILE = "cancer_clinical_dataset.json"
DEFAULT_CUTOFF = 0.4

# Initialize session state
if 'search_input' not in st.session_state:
    st.session_state.search_input = ""
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# Load and validate data with error handling
@st.cache_data
def load_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # Validate and clean data
        clean_data = []
        for entry in raw_data:
            if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                # Clean whitespace and ensure string type
                entry["prompt"] = str(entry["prompt"]).strip()
                entry["completion"] = str(entry["completion"]).strip()
                clean_data.append(entry)
        
        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset.")
            return None
        
        return clean_data
    
    except FileNotFoundError:
        st.error(f"Data file '{DATA_FILE}' not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON format in '{DATA_FILE}'.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Enhanced fuzzy matching with multiple results
def find_matches(question, dataset, cutoff=DEFAULT_CUTOFF, max_results=3):
    prompts = [entry["prompt"] for entry in dataset]
    matches = get_close_matches(question, prompts, n=max_results, cutoff=cutoff)
    
    results = []
    for match in matches:
        for entry in dataset:
            if entry.get("prompt") == match:
                results.append(entry)
                break
    
    return results

# Generate alternative suggestions
def generate_suggestions(question, dataset):
    # Extract keywords from question
    keywords = set(question.lower().split())
    stop_words = {"what", "how", "when", "why", "which", "where", "who"}
    keywords = keywords - stop_words
    
    # Find similar questions in dataset
    all_prompts = [entry["prompt"] for entry in dataset]
    similar_questions = []
    
    for prompt in all_prompts:
        prompt_keywords = set(prompt.lower().split())
        common = keywords & prompt_keywords
        if len(common) >= 1:  # At least one keyword match
            similar_questions.append(prompt)
    
    return similar_questions[:5]  # Return top 5 similar questions

# Streamlit UI Configuration
st.set_page_config(
    page_title="🧬 Cancer Clinical Trial Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for advanced controls
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Cancer+Search", width=150)
    st.title("Search Settings")
    
    cutoff = st.slider(
        "Match Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CUTOFF,
        step=0.05,
        help="Higher values require closer matches"
    )
    
    max_results = st.slider(
        "Maximum Results",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of results to display"
    )
    
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown("""
    This platform helps researchers find:
    - Clinical trial details
    - Drug mechanisms
    - Treatment outcomes
    - Biomarker data
    """)

# Main content area
st.title("🧬 Precision Cancer Clinical Search")
st.markdown("""
**Find precise answers about cancer treatments and clinical trials**  
This tool uses advanced pattern matching to surface relevant clinical information from structured datasets.
""")

# Load data with error handling
data = load_data()
if data is None:
    st.stop()

# Search interface - using session state to maintain query
user_question = st.text_input(
    "Search clinical questions:",
    value=st.session_state.search_input,
    placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high patients?",
    help="Enter your clinical question or keywords",
    key="search_input"
)

# Handle suggestion clicks
def update_search(question):
    st.session_state.search_input = question
    st.session_state.last_query = question  # Store the last query used

# Display results
if st.session_state.search_input:
    current_query = st.session_state.search_input
    with st.spinner("Searching clinical knowledge base..."):
        results = find_matches(current_query, data, cutoff=cutoff, max_results=max_results)
        
        if results:
            st.success(f"🔍 Found {len(results)} matching results")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"Result #{i}: {result['prompt'][:50]}...", expanded=(i == 1)):
                    st.markdown(f"**Question:**\n\n{result['prompt']}")
                    st.markdown(f"**Answer:**\n\n{result['completion']}")
                    
                    # Add metadata if available
                    if "source" in result:
                        st.caption(f"Source: {result['source']}")
                    if "trial_id" in result:
                        st.caption(f"Clinical Trial: {result['trial_id']}")
        else:
            st.error("No direct matches found. Here are some suggestions:")
            
            # Show alternative suggestions
            similar_questions = generate_suggestions(current_query, data)
            
            if similar_questions:
                st.markdown("**Try these similar questions:**")
                cols = st.columns(2)
                col_idx = 0
                for i, question in enumerate(similar_questions, 1):
                    with cols[col_idx]:
                        if st.button(
                            f"{question[:60]}...",
                            key=f"suggestion_{i}",
                            help="Click to try this search",
                            on_click=update_search,
                            args=(question,)
                        ):
                            pass
                    col_idx = 1 - col_idx  # Alternate between columns
            
            # General search tips
            st.markdown("""
            **Search tips:**
            1. Use specific drug names (e.g., 'atezolizumab' instead of 'PD-L1 inhibitor')
            2. Include cancer types (e.g., 'NSCLC' or 'triple-negative breast cancer')
            3. Focus on one aspect at a time (mechanism, efficacy, or safety)
            4. Try both abbreviations and full names (e.g., 'OS' and 'overall survival')
            """)
            
            # Show sample questions if no similar questions found
            if not similar_questions:
                st.markdown("**Sample questions you could try:**")
                samples = [
                    "What is the recommended dose for atezolizumab in bladder cancer?",
                    "How does PD-L1 expression affect atezolizumab response?",
                    "What are common adverse events with atezolizumab combination therapy?"
                ]
                for sample in samples:
                    if st.button(
                        sample,
                        key=f"sample_{sample[:10]}",
                        on_click=update_search,
                        args=(sample,)
                    ):
                        pass

# Footer
st.markdown("---")
st.markdown("""
<small>© 2023 Cancer Clinical Search | For research use only</small>
""", unsafe_allow_html=True)
