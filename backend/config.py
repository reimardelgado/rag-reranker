import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    CHROMA_HOST     = os.getenv("CHROMA_HOST", "chroma")
    CHROMA_PORT     = int(os.getenv("CHROMA_PORT", 8000))
    COLLECTION      = os.getenv("COLLECTION", "kb_docs")
    EMBED_MODEL     = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
    GEM_BASE        = os.getenv("GEM_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
    DOCS_PATH       = os.getenv("DOCS_PATH", "/app/docs")
    DEFAULT_TOP_K   = int(os.getenv("DEFAULT_TOP_K", 6))
    DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.8-flash")

settings = Settings()
