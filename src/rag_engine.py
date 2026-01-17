import os
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq
import chromadb
from src.prompts import STRICT_QA_TEMPLATE
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

DB_PATH = "./database" 
COLLECTION_NAME = "nvidia_financials"

def get_query_engine():
    # 1. Database Setup
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 2. Embedding Model (CLOUD BASED - SAVES RAM)
    # We use the API so we don't load PyTorch on Render
    if not HF_TOKEN:
        raise ValueError("❌ MISSING HF_TOKEN! Add it to Render Environment Variables.")

    embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        token=HF_TOKEN
    )
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )

    # 3. LLM Setup (Groq)
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