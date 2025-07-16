import streamlit as st
import json
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import defaultdict
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain  # python-louvain package
import matplotlib.pyplot as plt

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
if 'cancer_type_filter' not in st.session_state:
    st.session_state.cancer_type_filter = []
if 'gene_filter' not in st.session_state:
    st.session_state.gene_filter = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'cancer_types' not in st.session_state:
    st.session_state.cancer_types = []
if 'genes' not in st.session_state:
    st.session_state.genes = []
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'concept_network' not in st.session_state:
    st.session_state.concept_network = None

# Constants
DATA_FILE = "cancer_clinical_dataset.json"
HISTORY_FILE = "search_history.json"
GRAPH_FILE = "knowledge_graph.gexf"

# Load and save search history
def load_search_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_search_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.search_history, f)
    except:
        pass

# Build knowledge graph from the dataset
def build_knowledge_graph(data):
    G = nx.Graph()
    
    # Extract entities and relationships
    for entry in data:
        # Add nodes for each entry
        entry_id = f"entry_{hash(entry['prompt'])}"
        G.add_node(entry_id, type="entry", prompt=entry["prompt"], completion=entry["completion"])
        
        # Process cancer types
        if entry["cancer_type"]:
            for ct in entry["cancer_type"].split(","):
                ct = ct.strip()
                if ct:
                    G.add_node(ct, type="cancer_type")
                    G.add_edge(entry_id, ct, relationship="about")
        
        # Process genes
        if entry["genes"]:
            for gene in entry["genes"].split(","):
                gene = gene.strip()
                if gene:
                    G.add_node(gene, type="gene")
                    G.add_edge(entry_id, gene, relationship="involves")
        
        # Extract key terms from prompt and completion
        text = f"{entry['prompt']} {entry['completion']}"
        terms = set(word.lower() for word in text.split() if len(word) > 3 and word.isalpha())
        
        for term in terms:
            G.add_node(term, type="term")
            G.add_edge(entry_id, term, relationship="mentions")
    
    return G

# Load or create knowledge graph
@st.cache_data
def load_or_create_graph(data):
    if os.path.exists(GRAPH_FILE):
        try:
            return nx.read_gexf(GRAPH_FILE)
        except:
            pass
    
    G = build_knowledge_graph(data)
    nx.write_gexf(G, GRAPH_FILE)
    return G

# Find related concepts in the knowledge graph
def find_related_concepts(graph, query, top_n=5):
    if not graph or not query:
        return []
    
    query_terms = set(word.lower() for word in query.split() if len(word) > 3)
    related = defaultdict(float)
    
    for term in query_terms:
        if term in graph:
            for neighbor in graph.neighbors(term):
                if graph.nodes[neighbor]["type"] != "entry":
                    weight = graph[term][neighbor].get("weight", 1.0)
                    related[neighbor] += weight
    
    return sorted(related.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Load and preprocess data with caching
@st.cache_data
def load_and_index_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        clean_data = []
        word_index = defaultdict(list)
        cancer_types = set()
        genes = set()
        all_prompts = []

        for idx, entry in enumerate(raw_data):
            if isinstance(entry, dict) and "prompt" in entry and "completion" in entry:
                # Clean and standardize data
                entry["prompt"] = str(entry["prompt"]).strip()
                entry["completion"] = str(entry["completion"]).strip()
                
                # Extract metadata
                entry_cancer_types = []
                if "cancer_type" in entry:
                    entry_cancer_types = [ct.strip() for ct in str(entry["cancer_type"]).split(",")]
                    cancer_types.update(entry_cancer_types)
                
                entry_genes = []
                if "genes" in entry:
                    entry_genes = [g.strip() for g in str(entry["genes"]).split(",")]
                    genes.update(entry_genes)
                
                # Store cleaned entry
                clean_data.append({
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "cancer_type": ", ".join(entry_cancer_types) if entry_cancer_types else "",
                    "genes": ", ".join(entry_genes) if entry_genes else ""
                })
                all_prompts.append(entry["prompt"])

                # Index words
                for word in set(entry["prompt"].lower().split()):
                    if len(word) > 2:
                        word_index[word].append(idx)
                for word in set(entry["completion"].lower().split()):
                    if len(word) > 2:
                        word_index[word].append(idx)

        if not clean_data:
            st.error("No valid Q&A pairs found in the dataset.")
            return None, None, [], [], [], None
        
        # Generate random suggestions
        random_suggestions = random.sample(all_prompts, min(10, len(all_prompts))) if all_prompts else []
        
        # Build knowledge graph
        knowledge_graph = load_or_create_graph(clean_data)
        
        return clean_data, word_index, sorted(cancer_types), sorted(genes), random_suggestions, knowledge_graph
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, [], [], [], None

# Enhanced keyword search with filters and knowledge graph
def keyword_search(query, dataset, word_index, knowledge_graph):
    if not query or not dataset:
        return [], []
    
    query_words = set(word.lower() for word in query.split() if len(word) > 2)
    
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
            "matched_keywords": query_words & (prompt_words | completion_words)
        })

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Find related concepts from knowledge graph
    related_concepts = find_related_concepts(knowledge_graph, query)
    
    return ranked_results, related_concepts

# Filter results based on multiple criteria
def filter_results(results, min_score, keyword_filters, cancer_types, genes):
    filtered = []
    for result in results:
        entry = result["entry"]
        
        # Apply score filter
        if result["score"] < min_score:
            continue
        
        # Apply keyword filters
        if keyword_filters:
            matched = any(kw.lower() in result["matched_keywords"] for kw in keyword_filters)
            if not matched:
                continue
        
        # Apply cancer type filter
        if cancer_types and entry["cancer_type"]:
            entry_types = set(ct.strip() for ct in entry["cancer_type"].split(","))
            if not any(ct in entry_types for ct in cancer_types):
                continue
        
        # Apply gene filter
        if genes and entry["genes"]:
            entry_genes = set(g.strip() for g in entry["genes"].split(","))
            if not any(g in entry_genes for g in genes):
                continue
        
        filtered.append(result)
    return filtered

# Home page layout
def show_home():
    st.title("🧬 Precision Cancer Clinical Search")
    st.markdown("""
    **Find precise answers about cancer treatments and clinical trials**  
    This tool helps researchers discover connections between cancer types, genes, and treatments.
    """)

    # Display random suggestions
    if st.session_state.suggestions:
        st.markdown("**💡 Try these sample questions:**")
        cols = st.columns(2)
        for i, suggestion in enumerate(st.session_state.suggestions[:6]):
            with cols[i % 2]:
                if st.button(suggestion[:50] + "..." if len(suggestion) > 50 else suggestion, 
                            key=f"suggestion_{i}"):
                    st.session_state.current_query = suggestion
                    st.session_state.show_home = False
                    st.rerun()

    with st.form("search_form"):
        query = st.text_input(
            "Search clinical questions:",
            value=st.session_state.current_query,
            placeholder="e.g., What is the response rate for atezolizumab in PD-L1 high NSCLC patients?",
            help="Enter your clinical question or keywords"
        )

        if st.form_submit_button("Search", type="primary") and query.strip():
            st.session_state.current_query = query
            st.session_state.show_home = False
            st.session_state.search_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat()
            })
            save_search_history()
            st.rerun()

# Results page layout
def show_results():
    if st.button("← Back to Home"):
        st.session_state.show_home = True
        st.rerun()

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
        st.markdown("**Filter by keywords:**")
        new_keyword = st.text_input("Add keyword filter", key="new_keyword")
        if st.button("Add Keyword Filter") and new_keyword.strip():
            if new_keyword.lower() not in [k.lower() for k in st.session_state.keyword_filters]:
                st.session_state.keyword_filters.append(new_keyword.strip())
                st.rerun()
        
        # Cancer type filter
        if st.session_state.cancer_types:
            st.markdown("**Filter by cancer type:**")
            selected_cancers = st.multiselect(
                "Select cancer types",
                st.session_state.cancer_types,
                default=st.session_state.cancer_type_filter,
                key="cancer_type_select"
            )
            if selected_cancers != st.session_state.cancer_type_filter:
                st.session_state.cancer_type_filter = selected_cancers
                st.rerun()
        
        # Gene filter
        if st.session_state.genes:
            st.markdown("**Filter by genes:**")
            selected_genes = st.multiselect(
                "Select genes",
                st.session_state.genes,
                default=st.session_state.gene_filter,
                key="gene_select"
            )
            if selected_genes != st.session_state.gene_filter:
                st.session_state.gene_filter = selected_genes
                st.rerun()
        
        # Display active filters
        active_filters = []
        if st.session_state.keyword_filters:
            active_filters.extend([f"Keyword: {kw}" for kw in st.session_state.keyword_filters])
        if st.session_state.cancer_type_filter:
            active_filters.extend([f"Cancer: {ct}" for ct in st.session_state.cancer_type_filter])
        if st.session_state.gene_filter:
            active_filters.extend([f"Gene: {g}" for g in st.session_state.gene_filter])
        
        if active_filters:
            st.markdown("**Active Filters:**")
            for i, f in enumerate(active_filters):
                cols = st.columns([1, 4])
                with cols[0]:
                    if st.button("❌", key=f"remove_{i}"):
                        if f.startswith("Keyword:"):
                            st.session_state.keyword_filters.remove(f[8:])
                        elif f.startswith("Cancer:"):
                            st.session_state.cancer_type_filter.remove(f[7:])
                        elif f.startswith("Gene:"):
                            st.session_state.gene_filter.remove(f[5:])
                        st.rerun()
                with cols[1]:
                    st.markdown(f)
        
        if st.button("Clear All Filters"):
            st.session_state.keyword_filters = []
            st.session_state.cancer_type_filter = []
            st.session_state.gene_filter = []
            st.session_state.min_score = 1
            st.rerun()

    # Load data
    data, word_index, cancer_types, genes, _, knowledge_graph = load_and_index_data()
    if data is None:
        return

    # Store cancer types and genes in session state
    if cancer_types:
        st.session_state.cancer_types = cancer_types
    if genes:
        st.session_state.genes = genes
    if knowledge_graph:
        st.session_state.knowledge_graph = knowledge_graph

    # Display search history
    if st.session_state.search_history:
        with st.expander("📚 Search History", expanded=False):
            history_cols = st.columns(2)
            for i, search in enumerate(reversed(st.session_state.search_history)):
                with history_cols[i % 2]:
                    if st.button(f"{search['query'][:50]}{'...' if len(search['query']) > 50 else ''}",
                               key=f"history_{i}"):
                        st.session_state.current_query = search["query"]
                        st.rerun()

    st.markdown(f"**Current Search:** {st.session_state.current_query}")

    with st.spinner("Searching clinical knowledge base..."):
        ranked_results, related_concepts = keyword_search(
            st.session_state.current_query, 
            data, 
            word_index,
            st.session_state.knowledge_graph
        )
        filtered_results = filter_results(
            ranked_results,
            st.session_state.min_score,
            st.session_state.keyword_filters,
            st.session_state.cancer_type_filter,
            st.session_state.gene_filter
        ) if ranked_results else []

        if filtered_results:
            display_results(filtered_results, ranked_results, related_concepts)
        else:
            show_no_results(data, word_index)

def display_results(results, all_results, related_concepts):
    st.success(f"Found {len(results)} relevant results (from {len(all_results)} total matches)")
    
    # Show related concepts from knowledge graph
    if related_concepts:
        with st.expander("🔗 Related Concepts", expanded=True):
            st.markdown("**These concepts are frequently associated with your search:**")
            cols = st.columns(4)
            for i, (concept, score) in enumerate(related_concepts):
                with cols[i % 4]:
                    if st.button(f"{concept}", key=f"concept_{i}"):
                        st.session_state.current_query = concept
                        st.rerun()
    
    # Score distribution chart
    if len(results) > 1:
        scores = [r["score"] for r in results]
        st.bar_chart(pd.DataFrame({"Score": scores}), use_container_width=True)
    
    # Download buttons
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

    # Display results
    for i, result in enumerate(results, 1):
        entry = result["entry"]
        with st.expander(f"#{i} | Score: {result['score']} - {entry['prompt'][:50]}...", expanded=(i==1)):
            st.markdown(f"**Question:** {entry['prompt']}")
            st.markdown(f"**Answer:** {entry['completion']}")
            
            # Metadata
            metadata = []
            if entry["cancer_type"]:
                metadata.append(f"**Cancer Type:** {entry['cancer_type']}")
            if entry["genes"]:
                metadata.append(f"**Genes:** {entry['genes']}")
            if metadata:
                st.markdown(" | ".join(metadata))
            
            # Score details
            with st.expander("🔍 Match Details"):
                st.markdown(f"**Total Score:** {result['score']}")
                st.markdown(f"**Prompt Matches:** {result['prompt_matches']}")
                st.markdown(f"**Answer Matches:** {result['completion_matches']}")
                if result["matched_keywords"]:
                    st.markdown(f"**Matched Keywords:** {', '.join(result['matched_keywords'])}")
            
            # Download buttons
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
    
    # Generate suggestions from query
    query_words = set(word.lower() for word in st.session_state.current_query.split() if len(word) > 3)
    suggestions = set()

    if query_words and word_index and data:
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
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion[:50] + "..." if len(suggestion) > 50 else suggestion, 
                            key=f"nores_sugg_{i}"):
                    st.session_state.current_query = suggestion
                    st.rerun()
    
    st.markdown("""
    **Search Tips:**
    - Try lowering the minimum score filter
    - Remove some filters to broaden your search
    - Check for typos in your search terms
    - Use more general terms if your search is too specific
    """)

# Main app flow
def main():
    st.set_page_config(
        page_title="🧬 Cancer Clinical Trial Search",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load search history
    if not st.session_state.search_history:
        st.session_state.search_history = load_search_history()

    # Load data and suggestions
    if not st.session_state.suggestions or not st.session_state.cancer_types or not st.session_state.genes:
        data, word_index, cancer_types, genes, suggestions, knowledge_graph = load_and_index_data()
        if suggestions:
            st.session_state.suggestions = suggestions
        if cancer_types:
            st.session_state.cancer_types = cancer_types
        if genes:
            st.session_state.genes = genes
        if knowledge_graph:
            st.session_state.knowledge_graph = knowledge_graph

    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Cancer+Search", width=150)
        st.title("Navigation")

        if not st.session_state.show_home:
            if st.button("🏠 Home"):
                st.session_state.show_home = True
                st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This platform helps researchers:
        - Find clinical trial details
        - Discover connections between concepts
        - Access treatment outcomes
        - Download structured data
        """)

    if st.session_state.show_home:
        show_home()
    else:
        show_results()

if __name__ == "__main__":
    main()
