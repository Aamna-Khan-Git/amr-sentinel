import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-5")
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/processed/amr_data.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/embeddings/chroma_db")
