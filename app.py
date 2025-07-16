import streamlit as st
import json
import networkx as nx
from pyvis.network import Network
import tempfile

# -------------------------------
# Load JSON data
# -------------------------------
@st.cache_data
def load_data(file_path="cancer_clinical_dataset.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load JSON file: {e}")
        return []

# -------------------------------
# Build co-occurrence graph
# -------------------------------
def build_graph(data, min_word_length=3):
    G = nx.Graph()

    for entry in data:
        keywords = set()

        # From prompt
        if "prompt" in entry and isinstance(entry["prompt"], str):
            words = entry["prompt"].split()
            keywords |= set(w.lower().strip(".,:;()[]") for w in words if len(w) >= min_word_length)

        # From cancer_type
        if "cancer_type" in entry and entry["cancer_type"]:
            types = [t.strip().lower() for t in entry["cancer_type"].split(",")]
            keywords |= set(types)

        # From genes
        if "genes" in entry and entry["genes"]:
            genes = [g.strip().lower() for g in entry["genes"].split(",")]
            keywords |= set(genes)

        # Build edges
        for kw1 in keywords:
            for kw2 in keywords:
                if kw1 != kw2:
                    if G.has_edge(kw1, kw2):
                        G[kw1][kw2]["weight"] += 1
                    else:
                        G.add_edge(kw1, kw2, weight=1)
    return G

# -------------------------------
# Display interactive pyvis graph
# -------------------------------
def display_graph(G, focus_keyword=None):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()

    for node in G.nodes():
        net.add_node(node, label=node)

    for source, target, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(source, target, value=weight)

    if focus_keyword and focus_keyword in G.nodes():
        net.focus(focus_keyword)

    # Save graph to HTML
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)

    # Display in Streamlit
    with open(tmp_file.name, "r", encoding="utf-8") as f:
        html = f.read()
        st.components.v1.html(html, height=600, scrolling=True)

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="🧠 Cancer Knowledge Graph", layout="wide")
    st.title("🧠 Cancer Clinical Keyword Graph")
    st.markdown("""
    This tool visualizes co-occurring keywords, genes, and cancer types  
    extracted from your clinical Q&A JSON file.
    """)

    data = load_data("cancer_clinical_dataset.json")

    if not data:
        st.warning("Upload or provide a valid JSON file.")
        return

    st.sidebar.header("🔍 Graph Filters")
    focus_kw = st.sidebar.text_input("Focus on a keyword (optional)", "")

    G = build_graph(data)

    st.subheader("🔗 Interactive Concept Graph")
    display_graph(G, focus_kw.strip().lower() if focus_kw else None)

if __name__ == "__main__":
    main()
