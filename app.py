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
DEFAULT_CUTOFF = 0.4

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

# Keyword-based search with ranking
def keyword_search(query, dataset):
    query_words = set(word.lower() for word in query.split())
    ranked_results = []
    
    for entry in dataset:
        prompt_words = set(word.lower() for word in entry["prompt"].split())
        completion_words = set(word.lower() for word in entry["completion"].split())
        
        # Count matches in both prompt and completion
        prompt_matches = len(query_words & prompt_words)
        completion_matches = len(query_words & completion_words)
        
        # Higher weight for prompt matches
        total_score = (prompt_matches * 2) + completion_matches
        
        if total_score > 0:
            ranked_results.append({
                "entry": entry,
                "score": total_score,
                "prompt_matches": prompt_matches,
                "completion_matches": completion_matches
            })
    
    # Sort by score (descending)
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results

# Generate alternative suggestions
def generate_suggestions(query, dataset):
    query_words = set(word.lower() for word in query.split())
    stop_words = {"what", "how", "when", "why", "which", "where", "who", "is", "are", "the"}
    keywords = query_words - stop_words
    
    suggestions = []
    for entry in dataset:
        prompt_words = set(word.lower() for word in entry["prompt"].split())
        if keywords & prompt_words:
            suggestions.append(entry["prompt"])
    
    return list(set(suggestions))[:5]  # Return unique suggestions

# Home page layout
def show_home():
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("""
    **Find precise answers about cancer treatments and clinical trials**  
    This tool helps researchers access structured clinical trial information.
    """)
    
    # Search input
    st.session_state.current_query = st.text_input(
        "Search clinical questions:",
        placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high patients?",
        help="Enter your clinical question or keywords",
        key="search_input"
    )
    
    # Search button
    if st.button("Search", type="primary"):
        if st.session_state.current_query.strip():
            st.session_state.show_home = False
            st.session_state.search_history.append({
                "query": st.session_state.current_query,
                "timestamp": datetime.now().isoformat()
            })

# Results page layout
def show_results():
    # Back button
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        return
    
    st.title("🔍 Search Results")
    
    # Show search history
    if st.session_state.search_history:
        with st.expander("📚 Search History", expanded=False):
            for i, search in enumerate(reversed(st.session_state.search_history), 1):
                if st.button(f"{i}. {search['query']}", key=f"history_{i}"):
                    st.session_state.current_query = search["query"]
                    st.experimental_rerun()
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Current query display
    st.markdown(f"**Current Search:** {st.session_state.current_query}")
    
    # Perform search
    ranked_results = keyword_search(st.session_state.current_query, data)
    
    if ranked_results:
        st.success(f"Found {len(ranked_results)} relevant results")
        
        # Download all button
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
        
        # Display results
        for i, result in enumerate(ranked_results, 1):
            entry = result["entry"]
            with st.expander(f"#{i} | Score: {result['score']} (Prompt: {result['prompt_matches']}, Answer: {result['completion_matches']}) - {entry['prompt'][:50]}...", expanded=(i==1)):
                st.markdown(f"**Question:** {entry['prompt']}")
                st.markdown(f"**Answer:** {entry['completion']}")
                
                # Individual download buttons
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
    else:
        st.error("No matches found. Try these suggestions:")
        
        # Generate and show suggestions
        suggestions = generate_suggestions(st.session_state.current_query, data)
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.current_query = suggestion
                    st.experimental_rerun()
        
        st.markdown("""
        **Search Tips:**
        - Use specific drug names (e.g., "atezolizumab")
        - Include cancer types (e.g., "NSCLC")
        - Try both abbreviations and full terms (e.g., "OS" and "overall survival")
        """)

# Main app flow
def main():
    st.set_page_config(
        page_title="🧬 Cancer Clinical Trial Search",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
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
    
    # Show appropriate page
    if st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
