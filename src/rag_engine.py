import os
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

# --- CONFIGURATION ---
DB_PATH = "./database"
COLLECTION_NAME = "nvidia_financials"

def get_query_engine():
    # 1. Connect to the existing Database
    # We use the same path where we stored the data
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 2. Load the Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 3. Load the Index from the Vector Store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    # 4. Setup the LLM (Ollama)
    # request_timeout=360.0 ensures it doesn't crash if your laptop is slow
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=360.0,
        context_window=3072) 

    # 5. Configure Retrieval
    # "similarity_top_k=5" means: "Find the 5 most relevant paragraphs"
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1,
    )

    # 6. Build the Engine
    # This combines: Finding Data (Retriever) + Answering (LLM)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer(llm=llm)
    )

    return query_engine

# --- TEST SECTION ---
if __name__ == "__main__":
    print("⏳ Loading RAG Engine...")
    engine = get_query_engine()
    
    # --- NEW QUESTION FOR 2025 REPORT ---
    # This targets the specific "Data Center" segment which drives Nvidia's 2025 growth.
    question = "What was the specific dollar amount for Data Center revenue in Fiscal Year 2025?"
    
    print(f"❓ Question: {question}")
    print("🤖 Thinking...")
    
    response = engine.query(question)
    print(f"🤖 Answer: {response}")