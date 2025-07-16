import streamlit as st
import json
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# Load JSON data
@st.cache_data
def load_data(file_path="cancer_clinical_dataset.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Build a co-occurrence graph
def build_graph(data, min_word_length=3):
    G = nx.Graph()

    for entry in data:
        keywords = set()

        # Extract and clean keywords from prompt
        if "prompt" in entry:
            keywords |= set(w.lower().strip(".,:;") for w in entry["prompt"].split() if len(w) >= min_word_length)

        # From cancer_type
        if "cancer_type" in entry and entry["cancer_type"]:
            keywords |= set(w.lower().strip() for w in entry["cancer_type"].split(",") if w.strip())

        # From genes
        if "genes" in entry and entry["genes"]:
            keywords |= set(w.lower().strip() for w in entry["genes"].split(",") if w.strip())

        # Create co-occurrence edges
        for kw1 in keywords:
            for kw2 in keywords:
                if kw1 != kw2:
                    if G.has_edge(kw1, kw2):
                        G[kw1][kw2]['weight'] += 1
                    else:
                        G.add_edge(kw1, kw2, weight=1)

    return G

# Show interactive graph using pyvis
def show_graph(G, selected_keyword=None):
    net = Network(height="600px", width="100%", notebook=False)
    net.barnes_hut()

    for node in G.nodes():
        net.add_node(node, label=node, title=node)

    for source, target, data in G.edges(data=True):
        weight = data.get('weight', 1)
        net.add_edge(source, target, value=weight)

    if selected_keyword:
        net.focus(selected_keyword)

    # Save and display HTML
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)
    st.components.v1.html(open(tmp_file.name, "r", encoding="utf-8").read(), height=600, scrolling=True)

# Main Streamlit UI
def main():
    st.set_page_config(page_title="🧠 Cancer Keyword Graph", layout="wide")
    st.title("🧠 Connect the Dots: Cancer Clinical Knowledge Graph")

    st.markdown("""
    This tool builds a keyword co-occurrence graph from your clinical Q&A dataset.  
    Explore how concepts like **genes**, **cancer types**, and **therapies** are connected.
    """)

    data = load_data()

    if not data:
        st.error("Dataset is empty or missing.")
        return

    G = build_graph(data)

    st.sidebar.header("🔎 Graph Filter")
    selected_keyword = st.sidebar.text_input("Focus on a keyword (optional)", value="")

    if selected_keyword and selected_keyword.lower() not in G.nodes:
        st.sidebar.warning("Keyword not found in graph.")

    st.markdown("### 🔗 Keyword Relationship Network")
    show_graph(G, selected_keyword.lower() if selected_keyword else None)

if __name__ == "__main__":
    main()
