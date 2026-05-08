import os
import tempfile
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- Model settings ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"

    # --- Groq API ---
    GROQ_API_KEY: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.2
    GROQ_MAX_TOKENS: int = 1024

    # --- PDF processing ---
    MAX_PDFS: int = 10
    MAX_PDF_SIZE_MB: int = 50
    OCR_LANGUAGES: List[str] = field(default_factory=lambda: ["en"])
    USE_GPU: bool = False

    # --- Chunking ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_LENGTH: int = 50

    # --- Retrieval ---
    MAX_CHUNKS_RETRIEVE: int = 50
    TOP_K_AFTER_RERANK: int = 8
    LEXICAL_WEIGHT: float = 0.3
    SEMANTIC_WEIGHT: float = 0.7

    # --- Memory ---
    MEMORY_MAX_TURNS: int = 6

    # --- Storage ---
    # Use the OS system temp directory so Gradio 6 can always serve generated files.
    # Windows: C:\Users\<user>\AppData\Local\Temp\docvision_*
    # Linux / macOS: /tmp/docvision_*
    CACHE_DIR: str = field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "docvision_cache")
    )
    REPORTS_DIR: str = field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "docvision_reports")
    )
    USE_CACHE: bool = True

    # --- FastAPI ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Gradio ---
    GRADIO_PORT: int = 7860


config = Config()