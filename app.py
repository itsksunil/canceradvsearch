import streamlit as st
import json
from difflib import get_close_matches
import pandas as pd
from datetime import datetime

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

# Enhanced fuzzy matching with multiple results
def find_matches(question, dataset, cutoff=DEFAULT_CUTOFF, max_results=3):
    prompts = [entry["prompt"] for entry in dataset]
    matches = get_close_matches(question, prompts, n=max_results, cutoff=cutoff)
    
    results = []
    for match in matches:
        for entry in dataset:
            if entry["prompt"] == match if "prompt" in entry else entry["prompt"] == match:
                results.append(entry)
                break
    
    return results

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
    
    st.markdown("---")
    st.markdown("**Example searches:**")
    st.markdown("""
    - "Atezolizumab PD-L1 efficacy"
    - "Immune-related adverse events"
    - "NSCLC combination therapies"
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

# Search interface
col1, col2 = st.columns([3, 1])
with col1:
    user_question = st.text_input(
        "Search clinical questions:",
        placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high patients?",
        help="Enter your clinical question or keywords"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("Search", type="primary")

# Display results
if user_question or search_btn:
    with st.spinner("Searching clinical knowledge base..."):
        results = find_matches(user_question, data, cutoff=cutoff, max_results=max_results)
        
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
                    
                    # Export options for each result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download as JSON",
                            data=json.dumps(result, indent=2),
                            file_name=f"clinical_result_{i}.json",
                            mime="application/json",
                            key=f"json_{i}"
                        )
                    with col2:
                        st.download_button(
                            label="Download as CSV",
                            data=pd.DataFrame([result]).to_csv(index=False),
                            file_name=f"clinical_result_{i}.csv",
                            mime="text/csv",
                            key=f"csv_{i}"
                        )
            
            # Export all results
            st.markdown("---")
            st.subheader("Export All Results")
            
            export_data = {
                "query": user_question,
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "cutoff": cutoff,
                    "max_results": max_results
                },
                "results": results
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download All as JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name="clinical_search_results.json",
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    label="Download All as CSV",
                    data=pd.DataFrame(results).to_csv(index=False),
                    file_name="clinical_search_results.csv",
                    mime="text/csv"
                )
        else:
            st.error("No matches found. Try:")
            st.markdown("""
            - Using different keywords
            - Lowering the match sensitivity
            - Making your query more specific
            """)

# Value proposition section
st.markdown("---")
st.subheader("Key Features for Researchers")
cols = st.columns(3)
with cols[0]:
    st.markdown("**🔬 Precise Matching**")
    st.markdown("Find clinically relevant answers with pattern recognition")
with cols[1]:
    st.markdown("**💾 Structured Data**")
    st.markdown("Export results for analysis or reporting")
with cols[2]:
    st.markdown("**⚡ Fast Results**")
    st.markdown("Get answers without complex search syntax")

# Footer
st.markdown("---")
st.markdown("""
<small>© 2023 Cancer Clinical Search | For research use only</small>
""", unsafe_allow_html=True)
