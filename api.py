import logging
import os
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agents import (
    AgentContext,
    ConversationMemory,
    InsightsAgent,
    OrchestratorAgent,
    ReportAgent,
    RetrievalAgent,
)
from config import config
from core import DocumentStore, PDFExtractor, TextChunker
from llm import LLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# APPLICATION SETUP

app = FastAPI(
    title="DocVision OCR API",
    description="Multi-agent AI document intelligence and question-answering API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# SHARED STATE
# =============================================================================

store = DocumentStore()
retrieval_agent = RetrievalAgent()
orchestrator = OrchestratorAgent()
orchestrator._retrieval = retrieval_agent

insights_agent = InsightsAgent()
report_agent = ReportAgent()
extractor = PDFExtractor()
chunker = TextChunker()
memory = ConversationMemory(max_turns=config.MEMORY_MAX_TURNS)

qa_history: List[Dict] = []
last_rag_debug: Dict = {}

# Rolling window for metrics
response_times: deque = deque(maxlen=100)
query_count: int = 0

os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)

# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class QueryRequest(BaseModel):
    question: str
    use_memory: bool = True
    tone: str = "professional"
    custom_system_prompt: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    query_type: str
    confidence: float
    processing_time: float
    sources: List[Dict[str, Any]]
    suggestions: List[str]
    hallucination_flag: bool
    hallucination_reason: str
    report_available: bool
    report_filename: Optional[str]
    metadata: Dict[str, Any]
    rag_debug: Dict[str, Any]
    error: Optional[str]


class InsightsRequest(BaseModel):
    insight_type: str


class InsightsResponse(BaseModel):
    insight_type: str
    content: str
    processing_time: float


class DocumentSummary(BaseModel):
    name: str
    pages: int
    chunks: int
    has_ocr: bool


class UploadResponse(BaseModel):
    success: bool
    documents: List[DocumentSummary]
    total_chunks: int
    message: str


class HistoryItem(BaseModel):
    question: str
    answer: str
    query_type: str
    confidence: float
    hallucination_flag: bool
    timestamp: str


class MetricsResponse(BaseModel):
    total_queries: int
    avg_response_time: float
    documents_loaded: int
    total_chunks: int
    memory_turns: int
    llm_backend: str


# =============================================================================
# HELPERS
# =============================================================================

def _validate_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail=f"Only PDF files accepted. Got: {file.filename}")


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health():
    llm = LLMFactory.get_llm()
    return {
        "status": "ok",
        "version": "3.0.0",
        "llm_backend": llm.name(),
        "llm_available": llm.is_available(),
        "documents_loaded": len(store.documents),
        "total_chunks": len(store.chunks),
        "memory_turns": len(memory.turns),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    global query_count
    if len(files) > config.MAX_PDFS:
        raise HTTPException(400, detail=f"Maximum {config.MAX_PDFS} files per upload.")

    for f in files:
        _validate_pdf(f)

    store.clear()
    memory.clear()
    qa_history.clear()
    processed: List[DocumentSummary] = []
    tmp_dir = tempfile.mkdtemp()

    for upload in files:
        tmp_path = os.path.join(tmp_dir, upload.filename)
        with open(tmp_path, "wb") as fh:
            fh.write(await upload.read())

        if _file_size_mb(tmp_path) > config.MAX_PDF_SIZE_MB:
            raise HTTPException(400, detail=f"{upload.filename} exceeds {config.MAX_PDF_SIZE_MB}MB.")

        try:
            pdf_data = extractor.extract(tmp_path)
            chunks = chunker.create_chunks(pdf_data["text"], upload.filename)
            store.add_document(
                {
                    "name": upload.filename,
                    "path": tmp_path,
                    "pages": pdf_data["total_pages"],
                    "has_ocr": pdf_data["has_ocr_content"],
                    "chunks": len(chunks),
                },
                chunks,
                tmp_path,
            )
            processed.append(DocumentSummary(
                name=upload.filename,
                pages=pdf_data["total_pages"],
                chunks=len(chunks),
                has_ocr=pdf_data["has_ocr_content"],
            ))
            logger.info(f"Processed {upload.filename}: {pdf_data['total_pages']} pages, {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process {upload.filename}: {e}")
            raise HTTPException(500, detail=f"Failed to process {upload.filename}: {e}")

    retrieval_agent.index_store(store)
    query_count = 0

    return UploadResponse(
        success=True,
        documents=processed,
        total_chunks=len(store.chunks),
        message=f"Successfully processed {len(processed)} document(s).",
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    global query_count, last_rag_debug

    if not req.question.strip():
        raise HTTPException(400, detail="Question must not be empty.")
    if store.is_empty():
        raise HTTPException(400, detail="No documents uploaded. Use POST /upload first.")

    tone = req.tone if req.tone in ("professional", "simple", "technical", "academic") else "professional"

    ctx = AgentContext(
        question=req.question,
        store=store,
        memory=memory if req.use_memory else None,
        custom_system_prompt=req.custom_system_prompt,
        tone=tone,
    )

    t0 = time.time()
    result = orchestrator.run(ctx)
    elapsed = time.time() - t0

    response_times.append(elapsed)
    query_count += 1
    last_rag_debug = result.rag_debug

    qa_history.append({
        "question": req.question,
        "answer": result.answer,
        "query_type": result.query_type,
        "confidence": result.confidence,
        "hallucination_flag": result.hallucination_flag,
        "timestamp": datetime.now().isoformat(),
    })

    report_filename = None
    if result.report_path and os.path.exists(result.report_path):
        report_filename = Path(result.report_path).name

    return QueryResponse(
        question=req.question,
        answer=result.answer,
        query_type=result.query_type,
        confidence=result.confidence,
        processing_time=result.processing_time,
        sources=result.sources,
        suggestions=result.suggestions,
        hallucination_flag=result.hallucination_flag,
        hallucination_reason=result.hallucination_reason,
        report_available=bool(result.report_path),
        report_filename=report_filename,
        metadata=result.metadata,
        rag_debug=result.rag_debug,
        error=result.error,
    )


@app.get("/stream")
def stream_query(question: str, tone: str = "professional"):
    """
    Server-Sent Events endpoint for streaming LLM responses.
    The Gradio UI polls this for token-by-token display.
    """
    if store.is_empty():
        raise HTTPException(400, detail="No documents uploaded.")

    from agents import (
        ReasoningAgent, SummarizerAgent, detect_query_type, HallucinationGuard,
        TYPE_PROMPT_INSTRUCTIONS, TONE_INSTRUCTIONS, BASE_SUMMARIZER_SYSTEM,
    )
    from llm import GroqLLM
    import json

    def generate():
        ctx = AgentContext(question=question, store=store, memory=memory, tone=tone)
        ctx.query_type = detect_query_type(question)

        # Run retrieval synchronously
        ctx = retrieval_agent.run(ctx)
        if not ctx.reranked_chunks:
            yield f"data: {json.dumps({'token': 'No relevant content found in documents.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Build prompt
        context_block = "\n\n".join(
            f"[Source {i}: {c.get('document', '')}]\n{c['text'][:500]}"
            for i, c in enumerate(ctx.reranked_chunks, 1)
        )
        type_instruction = TYPE_PROMPT_INSTRUCTIONS.get(ctx.query_type, "")
        tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["professional"])
        system_prompt = (
            f"{BASE_SUMMARIZER_SYSTEM}\n"
            f"Tone: {tone_instruction}\n"
            f"Format: {type_instruction}"
        )
        prompt = (
            f"QUESTION: {question}\n"
            f"DOCUMENT CONTEXT:\n{'--'*20}\n{context_block}\n{'--'*20}\n\n"
            f"Answer the question based solely on the above context."
        )

        llm = LLMFactory.get_llm()

        # If Groq is available, stream using the requests stream param
        if isinstance(llm, GroqLLM) and llm.is_available():
            import requests as req_lib
            headers = {
                "Authorization": f"Bearer {llm.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": llm.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
                "stream": True,
            }
            try:
                with req_lib.post(
                    GroqLLM.API_URL, headers=headers, json=payload, stream=True, timeout=90
                ) as resp:
                    for line in resp.iter_lines():
                        if line:
                            line_str = line.decode("utf-8")
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str.strip() == "[DONE]":
                                    yield "data: [DONE]\n\n"
                                    return
                                try:
                                    chunk = json.loads(data_str)
                                    token = chunk["choices"][0]["delta"].get("content", "")
                                    if token:
                                        yield f"data: {json.dumps({'token': token})}\n\n"
                                except Exception:
                                    pass
            except Exception as e:
                yield f"data: {json.dumps({'token': f'Stream error: {e}'})}\n\n"
                yield "data: [DONE]\n\n"
        else:
            # Fallback: generate full response and send as single chunk
            try:
                answer = llm.generate(prompt, system_prompt=system_prompt)
                for word in answer.split(" "):
                    yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'token': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/insights", response_model=InsightsResponse)
def generate_insights(req: InsightsRequest):
    valid_types = list(
        ["summary", "key_topics", "smart_notes", "short_questions", "long_questions", "mcq", "difficulty"]
    )
    if req.insight_type not in valid_types:
        raise HTTPException(400, detail=f"insight_type must be one of: {valid_types}")

    if store.is_empty():
        raise HTTPException(400, detail="No documents uploaded.")

    t0 = time.time()
    content = insights_agent.generate_insight(store, req.insight_type)
    return InsightsResponse(
        insight_type=req.insight_type,
        content=content,
        processing_time=round(time.time() - t0, 2),
    )


@app.post("/insights/report")
def insights_report(insights: Dict[str, str]):
    path = report_agent.generate_insights_report(store, insights)
    if not path or not os.path.exists(path):
        raise HTTPException(500, detail="Report generation failed.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=Path(path).name,
    )


@app.get("/history", response_model=List[HistoryItem])
def get_history():
    return [HistoryItem(**item) for item in qa_history]


@app.get("/documents", response_model=List[DocumentSummary])
def list_documents():
    return [
        DocumentSummary(name=d["name"], pages=d["pages"], chunks=d["chunks"], has_ocr=d["has_ocr"])
        for d in store.documents
    ]


@app.delete("/documents")
def clear_documents():
    store.clear()
    memory.clear()
    qa_history.clear()
    last_rag_debug.clear()
    return {"message": "All documents, memory, and history cleared."}


@app.delete("/memory")
def clear_memory():
    memory.clear()
    return {"message": "Conversation memory cleared."}


@app.get("/rag-debug")
def rag_debug():
    if not last_rag_debug:
        return {"message": "No query has been run yet."}
    return last_rag_debug


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    llm = LLMFactory.get_llm()
    avg_rt = round(sum(response_times) / len(response_times), 3) if response_times else 0.0
    return MetricsResponse(
        total_queries=query_count,
        avg_response_time=avg_rt,
        documents_loaded=len(store.documents),
        total_chunks=len(store.chunks),
        memory_turns=len(memory.turns),
        llm_backend=llm.name(),
    )


@app.get("/reports/{filename}")
def download_report(filename: str):
    path = os.path.join(config.REPORTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, detail="Report not found.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=False, log_level="info")