import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq  # <--- NEW: Cloud LLM
import chromadb
from src.prompts import STRICT_QA_TEMPLATE

# Load environment variables
load_dotenv()   

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Local fallback for DB path, but for Cloud we need to handle paths carefully
DB_PATH = "./database" 
COLLECTION_NAME = "nvidia_financials"

def get_query_engine():
    # 1. Database Setup
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 2. Embedding Model (This still runs on CPU, might be slow on free tier)
    # Optimization: On Render free tier, HuggingFace Local Embeddings might OOM (Out of Memory).
    # If it crashes, we might need a smaller model or an API for embeddings too.
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )

    # ... inside get_query_engine() ...

    # 3. LLM Setup (Switched to Groq)
    if not GROQ_API_KEY:
        raise ValueError("❌ MISSING GROQ_API_KEY!")

    llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer(
            llm=llm,
            text_qa_template=STRICT_QA_TEMPLATE
        )
    )

    return query_engine