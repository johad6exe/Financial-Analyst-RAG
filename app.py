import streamlit as st
import sys
import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# Load Env (for local testing)
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_engine import get_query_engine
from src.db_manager import init_db, save_message, load_history

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = Groq(
        model="llama-3.3-70b-versatile", 
        api_key=GROQ_API_KEY,
        temperature=0.0
    )

def rewrite_query(llm, query: str):
    prompt = f"Rewrite this query for better semantic retrieval: {query}"
    return str(llm.complete(prompt))

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nvidia Analyst (Cloud Edition)", page_icon="☁️")

# Initialize DB on startup
init_db()

st.title("☁️ Nvidia Financial RAG (Live)")

# --- LOAD HISTORY ---
if "messages" not in st.session_state:
    # Try to load from Cloud DB, fallback to empty list
    try:
        st.session_state.messages = load_history(10) or []
    except Exception:
        st.session_state.messages = []

# --- LOAD ENGINE ---
@st.cache_resource
def load_engine():
    return get_query_engine()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Engine Failed to Load: {e}")
    st.stop()

# --- CHAT LOOP ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about Nvidia's finances...")

prompt_master = rewrite_query(llm,prompt)

if prompt:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt_master})
    save_message("user", prompt_master) # Save to Cloud
    with st.chat_message("user"):
        st.markdown(prompt_master)

    # 2. Assistant Response
    with st.chat_message("assistant"):
        response = engine.query(prompt_master)
        st.markdown(response.response)
        
        # Save to Cloud
        st.session_state.messages.append({"role": "assistant", "content": response.response})
        save_message("assistant", response.response)