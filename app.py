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
if 'min_score' not in st.session_state:
    st.session_state.min_score = 1
if 'keyword_filters' not in st.session_state:
    st.session_state.keyword_filters = []

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

# Enhanced keyword search with score calculation
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
            "completion_matches": completion_matches,
            "all_keywords": query_words,
            "matched_keywords": query_words & (prompt_words | completion_words)
        })

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results

# Filter results based on score and keywords
def filter_results(results, min_score, keyword_filters):
    filtered = []
    for result in results:
        # Apply score filter
        if result["score"] < min_score:
            continue
        
        # Apply keyword filters if any
        if keyword_filters:
            matched = False
            for kw in keyword_filters:
                if kw.lower() in result["matched_keywords"]:
                    matched = True
                    break
            if not matched:
                continue
        
        filtered.append(result)
    return filtered

# Home page layout
def show_home():
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("""
    **Find precise answers about cancer treatments and clinical trials**  
    This tool helps researchers access structured clinical trial information.
    """)

    with st.form("search_form"):
        query = st.text_input(
            "Search clinical questions:",
            value=st.session_state.current_query,
            placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high patients?",
            help="Enter your clinical question or keywords"
        )

        if st.form_submit_button("Search", type="primary") and query.strip():
            st.session_state.current_query = query
            st.session_state.show_home = False
            st.session_state.search_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat()
            })

# Results page layout
def show_results():
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        return

    st.title("🔍 Search Results")
    
    # Filters sidebar
    with st.sidebar:
        st.subheader("🔎 Refine Results")
        
        # Score filter
        st.session_state.min_score = st.slider(
            "Minimum Match Score",
            min_value=0,
            max_value=20,
            value=1,
            help="Higher scores mean more keywords matched"
        )
        
        # Keyword filters
        st.markdown("**Filter by specific keywords:**")
        new_keyword = st.text_input("Add keyword filter", key="new_keyword")
        if st.button("Add Filter") and new_keyword.strip():
            if new_keyword.lower() not in [k.lower() for k in st.session_state.keyword_filters]:
                st.session_state.keyword_filters.append(new_keyword.strip())
        
        # Display active filters
        if st.session_state.keyword_filters:
            st.markdown("**Active Filters:**")
            cols = st.columns(3)
            for i, keyword in enumerate(st.session_state.keyword_filters):
                with cols[i % 3]:
                    if st.button(f"❌ {keyword}", key=f"remove_{keyword}"):
                        st.session_state.keyword_filters.remove(keyword)
                        st.rerun()
        
        if st.button("Clear All Filters"):
            st.session_state.keyword_filters = []
            st.session_state.min_score = 1
            st.rerun()

    if st.session_state.search_history:
        with st.expander("📚 Search History", expanded=False):
            for i, search in enumerate(reversed(st.session_state.search_history), 1):
                if st.button(f"{i}. {search['query']}", key=f"history_{i}"):
                    st.session_state.current_query = search["query"]
                    st.session_state.show_home = False

    data, word_index = load_and_index_data()
    if data is None:
        return

    st.markdown(f"**Current Search:** {st.session_state.current_query}")

    with st.spinner("Searching clinical knowledge base..."):
        ranked_results = keyword_search(st.session_state.current_query, data, word_index)
        filtered_results = filter_results(
            ranked_results,
            st.session_state.min_score,
            st.session_state.keyword_filters
        )

        if filtered_results:
            display_results(filtered_results, ranked_results)
        else:
            show_no_results(data, word_index)

def display_results(results, all_results):
    st.success(f"Found {len(results)} relevant results (filtered from {len(all_results)})")
    
    # Score distribution chart
    if len(results) > 1:
        scores = [r["score"] for r in results]
        st.bar_chart(pd.DataFrame({"Score": scores}), use_container_width=True)
    
    all_results_data = [result["entry"] for result in results]
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download All as JSON",
            json.dumps(all_results_data, indent=2),
            file_name="cancer_search_results.json",
            mime="application/json"
        )
    with col2:
        st.download_button(
            "Download All as CSV",
            pd.DataFrame(all_results_data).to_csv(index=False),
            file_name="cancer_search_results.csv",
            mime="text/csv"
        )

    for i, result in enumerate(results, 1):
        entry = result["entry"]
        with st.expander(f"#{i} | Score: {result['score']} (Keywords: {', '.join(result['matched_keywords'])}) - {entry['prompt'][:50]}...", expanded=(i==1)):
            st.markdown(f"**Question:** {entry['prompt']}")
            st.markdown(f"**Answer:** {entry['completion']}")
            
            # Score details
            with st.expander("🔍 Match Details"):
                st.markdown(f"**Total Score:** {result['score']}")
                st.markdown(f"**Prompt Matches:** {result['prompt_matches']}")
                st.markdown(f"**Answer Matches:** {result['completion_matches']}")
                st.markdown(f"**Matched Keywords:** {', '.join(result['matched_keywords']) or 'None'}")

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
    st.error("No matches found with current filters. Try these suggestions:")
    
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

    st.markdown("""
    **Search Tips:**
    - Try lowering the minimum score filter
    - Remove some keyword filters
    - Use more general search terms
    - Check for typos in your search
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
