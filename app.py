import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_home' not in st.session_state:
    st.session_state.show_home = True

# Constants
DATA_FILE = "cancer_clinical_dataset.json"

# Load and preprocess data with caching
@st.cache_data
def load_and_index_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        clean_data = []
        word_index = defaultdict(list)

        for idx, entry in enumerate(raw_data):
            if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                entry["prompt"] = str(entry["prompt"]).strip()
                entry["completion"] = str(entry["completion"]).strip()
                clean_data.append(entry)

                for word in set(entry["prompt"].lower().split()):
                    if len(word) > 2:
                        word_index[word].append(idx)
                for word in set(entry["completion"].lower().split()):
                    if len(word) > 2:
                        word_index[word].append(idx)

        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset.")
            return None, None

        return clean_data, word_index

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Safe keyword search function
def keyword_search(query, dataset, word_index):
    query_words = set(word.lower() for word in query.split() if len(word) > 2)
    if not query_words:
        return []

    doc_matches = defaultdict(int)
    for word in query_words:
        if word in word_index:
            for doc_id in word_index[word]:
                if doc_id < len(dataset):
                    doc_matches[doc_id] += 1

    ranked_results = []
    for doc_id, count in doc_matches.items():
        entry = dataset[doc_id]
        prompt_words = set(entry["prompt"].lower().split())
        completion_words = set(entry["completion"].lower().split())

        prompt_matches = len(query_words & prompt_words)
        completion_matches = len(query_words & completion_words)
        total_score = (prompt_matches * 2) + completion_matches

        ranked_results.append({
            "entry": entry,
            "score": total_score,
            "prompt_matches": prompt_matches,
            "completion_matches": completion_matches
        })

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results

# Home page layout (fast search mode)
def show_home():
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("""
    **Find precise answers about cancer treatments and clinical trials**  
    This tool helps researchers access structured clinical trial information.
    """)

    query = st.text_input(
        "Search clinical questions:",
        value=st.session_state.current_query,
        placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high patients?",
        help="Enter your clinical question or keywords"
    )

    if st.button("Search", type="primary") and query.strip():
        st.session_state.current_query = query
        st.session_state.show_home = False
        st.session_state.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        st.experimental_rerun()  # ✅ Instant refresh for fast search

# Results page layout
def show_results():
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.experimental_rerun()
        return

    st.title("🔍 Search Results")

    if st.session_state.search_history:
        with st.expander("📚 Search History", expanded=False):
            for i, search in enumerate(reversed(st.session_state.search_history), 1):
                if st.button(f"{i}. {search['query']}", key=f"history_{i}"):
                    st.session_state.current_query = search["query"]
                    st.session_state.show_home = False
                    st.experimental_rerun()
                    return

    data, word_index = load_and_index_data()
    if data is None:
        return

    st.markdown(f"**Current Search:** {st.session_state.current_query}")

    with st.spinner("Searching clinical knowledge base..."):
        ranked_results = keyword_search(st.session_state.current_query, data, word_index)

        if ranked_results:
            display_results(ranked_results)
        else:
            show_no_results(data, word_index)

def display_results(ranked_results):
    st.success(f"Found {len(ranked_results)} relevant results")
    all_results = [result["entry"] for result in ranked_results]

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download All as JSON",
            json.dumps(all_results, indent=2),
            file_name="cancer_search_results.json",
            mime="application/json"
        )
    with col2:
        st.download_button(
            "Download All as CSV",
            pd.DataFrame(all_results).to_csv(index=False),
            file_name="cancer_search_results.csv",
            mime="text/csv"
        )

    for i, result in enumerate(ranked_results, 1):
        entry = result["entry"]
        with st.expander(f"#{i} | Score: {result['score']} (Prompt: {result['prompt_matches']}, Answer: {result['completion_matches']}) - {entry['prompt'][:50]}...", expanded=(i == 1)):
            st.markdown(f"**Question:** {entry['prompt']}")
            st.markdown(f"**Answer:** {entry['completion']}")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download as JSON",
                    json.dumps(entry, indent=2),
                    file_name=f"cancer_result_{i}.json",
                    mime="application/json",
                    key=f"json_{i}"
                )
            with col2:
                st.download_button(
                    "Download as CSV",
                    pd.DataFrame([entry]).to_csv(index=False),
                    file_name=f"cancer_result_{i}.csv",
                    mime="text/csv",
                    key=f"csv_{i}"
                )

def show_no_results(data, word_index):
    st.error("No matches found. Try these suggestions:")

    query_words = set(word.lower() for word in st.session_state.current_query.split() if len(word) > 3)
    suggestions = set()

    if query_words:
        doc_ids = set()
        for word in query_words:
            if word in word_index:
                doc_ids.update(word_index[word])

        for doc_id in list(doc_ids)[:50]:
            if doc_id < len(data):
                suggestions.add(data[doc_id]["prompt"])
                if len(suggestions) >= 5:
                    break

    if suggestions:
        st.write("**Similar questions in our database:**")
        for suggestion in list(suggestions)[:5]:
            if st.button(suggestion[:100], key=f"suggestion_{hash(suggestion)}"):
                st.session_state.current_query = suggestion
                st.session_state.show_home = False
                st.experimental_rerun()
                return

    st.markdown("""
    **Search Tips:**
    - Use specific drug names (e.g., "atezolizumab")
    - Include cancer types (e.g., "NSCLC")
    - Try both abbreviations and full terms (e.g., "OS" and "overall survival")
    - Use at least 3-4 keywords for better results
    """)

# Main app flow
def main():
    st.set_page_config(
        page_title="🧬 Cancer Clinical Trial Search",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Cancer+Search", width=150)
        st.title("Navigation")

        if not st.session_state.show_home:
            if st.button("🏠 Home"):
                st.session_state.show_home = True
                st.experimental_rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This platform helps researchers:
        - Find clinical trial details
        - Access treatment outcomes
        - Download structured data
        """)

    if st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
