import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PATH = "./chroma_db"
PDF_PATH = "./data/knowledge_base.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"      
LLM_MODEL = "gemma-3-27b-it"            
RETRIEVAL_K = 6
CONFIDENCE_THRESHOLD = 0.6