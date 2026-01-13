import os
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownNodeParser  # <--- CHANGED THIS IMPORT
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Apply nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Use raw string (r"...") for Windows paths
PDF_PATH = r"D:\FRAG\Financial-Analyst-RAG\data\nvidia_10k.pdf"
DB_PATH = "./database"
COLLECTION_NAME = "nvidia_financials"

def ingest_data():
    print("🚀 Starting Ingestion Pipeline...")

    # Verification: Check if API Key is loaded
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("❌ Error: LLAMA_CLOUD_API_KEY not found in .env file!")
        return
    else:
        print("🔑 API Key found and loaded.")

    # 1. Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: File not found at {PDF_PATH}")
        return

    # 2. Setup Embedding Model
    print("📥 Loading Embedding Model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 3. Parse PDF
    print("📄 Parsing PDF with LlamaParse (This will take 1-2 mins)...")
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True,
        language="en",
    )
    
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        input_files=[PDF_PATH], 
        file_extractor=file_extractor
    ).load_data()
    print(f"✅ Parsed {len(documents)} pages/sections.")

    # 4. Setup ChromaDB
    print("💾 Initializing ChromaDB...")
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Process & Index
    print("🧠 Splitting text into chunks...")
    # CHANGED: Using MarkdownNodeParser instead of ElementNodeParser
    # This respects the markdown structure but doesn't need OpenAI
    node_parser = MarkdownNodeParser()
    
    print("Cx Building Vector Index...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        embed_model=embed_model,
        transformations=[node_parser]
    )
    
    print("🎉 Ingestion Complete! Your financial data is now stored in './database'.")

if __name__ == "__main__":
    ingest_data()