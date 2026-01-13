import streamlit as st
import sys
import os

# 1. Setup Path to find 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_engine import get_query_engine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nvidia Financial Analyst",
    page_icon="📊",
    layout="wide"
)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg", width=200)
    st.title("🤖 AI Analyst Settings")
    st.markdown("---")
    st.markdown("**Model:** Llama 3.2 (3B)")
    st.markdown("**Data Source:** Nvidia 2025 10-K")
    st.markdown("**Status:** ✅ Online")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT LOGIC ---
st.title("📊 Nvidia FY2025 Financial Analyst")
st.markdown("""
Ask questions about the **Fiscal Year 2025 10-K Report**. 
*Try asking: "What was the Data Center revenue?" or "List the primary risk factors."*
""")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Engine (Cached so it doesn't reload every click)
@st.cache_resource(show_spinner=False)
def load_engine():
    return get_query_engine()

with st.spinner("⏳ Waking up the AI Analyst... (This might take a moment)"):
    engine = load_engine()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask your financial question..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing 10-K Documents..."):
            try:
                # Query the RAG Engine
                response = engine.query(prompt)
                
                # Display Answer
                st.markdown(response.response)
                
                # Display Sources (The Trust Feature)
                with st.expander("📚 View Source Documents (Click to Verify)"):
                    for i, node in enumerate(response.source_nodes):
                        # Extract Metadata
                        page_label = node.node.metadata.get('page_label', 'N/A')
                        file_name = node.node.metadata.get('file_name', 'N/A')
                        snippet = node.node.get_content()[:300] + "..." # First 300 chars
                        
                        st.markdown(f"**Source {i+1}:** `{file_name}` (Page {page_label})")
                        st.caption(f"\"{snippet}\"")
                        st.markdown("---")

                # Save to History
                st.session_state.messages.append({"role": "assistant", "content": response.response})

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")