"""
Core document processing and retrieval engine.
Handles PDF extraction, OCR, chunking, indexing, and hybrid retrieval.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import fitz  # PyMuPDF

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import easyocr
    easyocr_available = True
except ImportError:
    easyocr_available = False
    logger.warning("easyocr not available. Scanned PDFs will not be OCR-processed.")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    st_available = True
except ImportError:
    st_available = False
    logger.warning("sentence-transformers / FAISS not available. Semantic retrieval disabled.")

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.tokenize import sent_tokenize
    nltk_available = True
except Exception:
    nltk_available = False


def _sentence_split(text: str) -> List[str]:
    if nltk_available:
        return sent_tokenize(text)
    return re.split(r"(?<=[.!?])\s+", text)


# =============================================================================
# OCR PROCESSOR
# =============================================================================

class OCRProcessor:
    def __init__(self):
        self._reader = None
        self._initialized = False

    def _init(self) -> bool:
        if self._initialized:
            return True
        if not easyocr_available:
            return False
        try:
            self._reader = easyocr.Reader(
                config.OCR_LANGUAGES,
                gpu=config.USE_GPU,
                model_storage_directory="/tmp/easyocr_models",
                download_enabled=True,
            )
            self._initialized = True
            logger.info("OCR processor initialized.")
            return True
        except Exception as e:
            logger.error(f"OCR initialization failed: {e}")
            return False

    def extract(self, image_path: str) -> str:
        if not self._init():
            return ""
        try:
            results = self._reader.readtext(image_path, detail=0)
            return " ".join(results) if results else ""
        except Exception as e:
            logger.error(f"OCR failed on {image_path}: {e}")
            return ""


# =============================================================================
# PDF TEXT EXTRACTOR
# =============================================================================

class PDFExtractor:
    def __init__(self):
        self._ocr = OCRProcessor()

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a PDF file.
        Returns a dict with keys: text, total_pages, has_ocr_content, pages.
        """
        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            full_text = ""
            has_ocr = False

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                ocr_text = ""
                if len(text.strip()) < 50:
                    img_path = f"/tmp/page_{page_num}.png"
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        pix.save(img_path)
                        ocr_text = self._ocr.extract(img_path)
                        if ocr_text:
                            text = ocr_text
                            has_ocr = True
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num}: {e}")
                    finally:
                        if os.path.exists(img_path):
                            os.remove(img_path)

                page_record = {
                    "page_number": page_num + 1,
                    "text": text,
                    "has_ocr": bool(ocr_text),
                    "char_count": len(text),
                }
                pages_data.append(page_record)
                full_text += f"\n[Page {page_num + 1}]\n{text}\n"

            doc.close()

            return {
                "text": full_text,
                "pages": pages_data,
                "total_pages": len(pages_data),
                "has_ocr_content": has_ocr,
                "total_chars": len(full_text),
            }

        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise


# =============================================================================
# TEXT CHUNKER
# =============================================================================

class TextChunker:
    def create_chunks(self, text: str, doc_name: str) -> List[Dict]:
        """Split text into overlapping sentence-based chunks."""
        try:
            sentences = _sentence_split(text)
            chunks: List[Dict] = []
            current: List[str] = []
            current_len = 0

            for sentence in sentences:
                slen = len(sentence)
                if current_len + slen > config.CHUNK_SIZE and current:
                    chunk_text = " ".join(current)
                    if len(chunk_text) >= config.MIN_CHUNK_LENGTH:
                        chunks.append({
                            "text": chunk_text,
                            "document": doc_name,
                            "chunk_id": len(chunks),
                            "length": len(chunk_text),
                        })
                    current = current[-1:] if config.CHUNK_OVERLAP > 0 else []
                    current_len = sum(len(s) for s in current)

                current.append(sentence)
                current_len += slen

            if current:
                chunk_text = " ".join(current)
                if len(chunk_text) >= config.MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": chunk_text,
                        "document": doc_name,
                        "chunk_id": len(chunks),
                        "length": len(chunk_text),
                    })

            return chunks
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            return []


# =============================================================================
# RETRIEVER
# =============================================================================

class HybridRetriever:
    """
    Combines TF-IDF lexical search with FAISS-backed semantic search.
    Score = LEXICAL_WEIGHT * tfidf + SEMANTIC_WEIGHT * embedding_similarity
    """

    def __init__(self):
        self._embedding_model: Optional[Any] = None
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self.chunks: List[Dict] = []

    def initialize(self):
        if st_available:
            try:
                logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
                self._embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
                logger.info("Embedding model loaded.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")

    def index(self, chunks: List[Dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        # TF-IDF index
        self._tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(texts)

        # Semantic index
        if self._embedding_model:
            self._embeddings = self._embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=8,
            )
            dim = self._embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self._embeddings.astype("float32"))

        logger.info(f"Indexed {len(chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or config.MAX_CHUNKS_RETRIEVE
        if not self.chunks:
            return []

        n = len(self.chunks)
        lex_scores = np.zeros(n)
        sem_scores = np.zeros(n)

        if self._tfidf and self._tfidf_matrix is not None:
            q_vec = self._tfidf.transform([query])
            lex_scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]

        if self._embedding_model and self._embeddings is not None:
            q_emb = self._embedding_model.encode([query], convert_to_numpy=True)
            if self._faiss_index:
                sims, idxs = self._faiss_index.search(
                    q_emb.astype("float32"), min(n, top_k)
                )
                for idx, sim in zip(idxs[0], sims[0]):
                    if idx < n:
                        sem_scores[idx] = sim
            else:
                sem_scores = cosine_similarity(q_emb, self._embeddings)[0]

        combined = config.LEXICAL_WEIGHT * lex_scores + config.SEMANTIC_WEIGHT * sem_scores
        top_idx = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if combined[idx] > 0:
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(combined[idx])
                results.append(chunk)

        return results


# =============================================================================
# RERANKER
# =============================================================================

class CrossEncoderReranker:
    def __init__(self):
        self._model: Optional[Any] = None

    def initialize(self):
        if st_available:
            try:
                logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
                self._model = CrossEncoder(config.RERANKER_MODEL)
                logger.info("Reranker loaded.")
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        top_k = config.TOP_K_AFTER_RERANK
        if not self._model or not chunks:
            return chunks[:top_k]

        pairs = [[query, c["text"]] for c in chunks]
        try:
            scores = self._model.predict(pairs)
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)
            reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]


# =============================================================================
# DOCUMENT STORE (in-memory)
# =============================================================================

class DocumentStore:
    """Holds all processed document state for the current session."""

    def __init__(self):
        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []
        self.processed_file_paths: List[str] = []

    def clear(self):
        self.documents.clear()
        self.chunks.clear()
        self.processed_file_paths.clear()

    def add_document(self, doc_meta: Dict, chunks: List[Dict], file_path: str):
        self.documents.append(doc_meta)
        self.chunks.extend(chunks)
        self.processed_file_paths.append(file_path)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0