"""
config.py
---------
Central configuration for AMR Sentinel.
All settings are read from environment variables (loaded from .env).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

# ── Models ────────────────────────────────────────────────────────────────────
INTENT_MODEL    = os.getenv("INTENT_MODEL",    "llama3.2:3b")
NARRATIVE_MODEL = os.getenv("NARRATIVE_MODEL", "mistral:7b")

# ── NCBI ──────────────────────────────────────────────────────────────────────
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL   = os.getenv("NCBI_EMAIL",   "")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATABASE_PATH = os.getenv("DATABASE_PATH", "amr_sentinel.db")
CHROMA_PATH   = os.getenv("CHROMA_PATH",   "chroma_store")

# ── Server ────────────────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "8000"))
