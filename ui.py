"""
Gradio frontend for DocVision OCR - Version 3.0
Compatible with Gradio 6.0+

Fixes applied for Gradio 6.0 breaking changes:
    1. theme and css moved from gr.Blocks() to demo.launch()
    2. gr.update() removed from .click() outputs lists entirely
    3. process_documents() returns 2 outputs instead of 3
    4. All lambda wrappers replaced with named functions
       (Gradio 6 does not support progress= inside lambdas)
    5. load_metrics() now returns 2 values to match 2 output components

Tabs:
    1. Chat & Q&A             - Streaming Q&A with memory
    2. Document Insights      - Auto summary, topics, difficulty
    3. Smart Notes            - Notes and exam questions generator
    4. Report & Export Center - Word document downloads
    5. RAG Debug Viewer       - Retrieval pipeline transparency
    6. Evaluation Dashboard   - System metrics
    7. Settings               - Memory/document controls
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import requests

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API = f"http://127.0.0.1:{config.API_PORT}"

# =============================================================================
# API CLIENT HELPERS
# =============================================================================

def _upload(files):
    file_handles = []
    try:
        file_tuples = []
        for f in files:
            fh = open(f.name, "rb")
            file_handles.append(fh)
            file_tuples.append(("files", (Path(f.name).name, fh, "application/pdf")))
        r = requests.post(f"{API}/upload", files=file_tuples, timeout=300)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"message": str(e)}
    finally:
        for fh in file_handles:
            try:
                fh.close()
            except Exception:
                pass


def _query(question, use_memory=True, tone="professional", custom_prompt=None):
    try:
        payload = {
            "question": question,
            "use_memory": use_memory,
            "tone": tone,
            "custom_system_prompt": custom_prompt or None,
        }
        r = requests.post(f"{API}/query", json=payload, timeout=120)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {
            "answer": str(e),
            "sources": [],
            "suggestions": [],
            "metadata": {},
            "rag_debug": {},
            "hallucination_flag": False,
            "hallucination_reason": "",
            "confidence": 0,
            "processing_time": 0,
            "query_type": "",
            "report_filename": "",
        }


def _stream_query(question, tone="professional"):
    """Yields tokens from the SSE /stream endpoint."""
    try:
        url = f"{API}/stream?question={requests.utils.quote(question)}&tone={tone}"
        with requests.get(url, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        if data.strip() == "[DONE]":
                            return
                        try:
                            token = json.loads(data).get("token", "")
                            yield token
                        except Exception:
                            pass
    except Exception as e:
        yield f"\n[Stream error: {e}]"


def _insights(insight_type):
    try:
        r = requests.post(
            f"{API}/insights",
            json={"insight_type": insight_type},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"content": str(e), "processing_time": 0}


def _get_history():
    try:
        r = requests.get(f"{API}/history", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _get_rag_debug():
    try:
        r = requests.get(f"{API}/rag-debug", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _get_metrics():
    try:
        r = requests.get(f"{API}/metrics", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _clear_memory():
    try:
        r = requests.delete(f"{API}/memory", timeout=10)
        r.raise_for_status()
        return "Conversation memory cleared."
    except Exception as e:
        return f"Failed: {e}"


def _clear_documents():
    try:
        r = requests.delete(f"{API}/documents", timeout=10)
        r.raise_for_status()
        return "All documents and memory cleared."
    except Exception as e:
        return f"Failed: {e}"


def _download_report(filename):
    if not filename:
        return None
    try:
        r = requests.get(f"{API}/reports/{filename}", timeout=30)
        r.raise_for_status()
        # Save into system temp so Gradio 6 can always serve the file
        tmp_dir = tempfile.gettempdir()
        out_path = os.path.join(tmp_dir, filename)
        with open(out_path, "wb") as fh:
            fh.write(r.content)
        return out_path
    except Exception:
        return None


# =============================================================================
# TAB 1: CHAT & Q&A
# =============================================================================

def process_documents(files, progress=gr.Progress()):
    """
    Returns (upload_status, doc_summary_state).
    Only 2 outputs — gr.update() has been removed entirely.
    """
    if not files:
        return "Please select at least one PDF file.", ""

    progress(0.2, desc="Uploading files...")
    ok, data = _upload(files)
    progress(0.9, desc="Building index...")

    if not ok:
        return f"Upload failed: {data.get('message', 'Unknown error')}", ""

    docs = data.get("documents", [])
    lines = [
        f"Processed {len(docs)} document(s) | "
        f"{data.get('total_chunks', 0)} chunks indexed",
        "",
    ]
    for i, d in enumerate(docs, 1):
        ocr = " [OCR]" if d.get("has_ocr") else ""
        lines.append(
            f"  {i}. {d['name']} | {d['pages']} pages | "
            f"{d['chunks']} chunks{ocr}"
        )

    progress(1.0)
    summary = "\n".join(lines)
    return summary, summary


def stream_answer(question, use_memory, tone, custom_prompt, progress=gr.Progress()):
    """
    Single-call answer function.
    Uses only POST /query — no SSE double-call to avoid doubling token usage.
    The generator wrapper is kept so Gradio event wiring stays unchanged.
    """
    if not question.strip():
        yield "", "", "", ""
        return

    progress(0.2, desc="Processing question...")

    ok, data = _query(
        question,
        use_memory=use_memory,
        tone=tone,
        custom_prompt=custom_prompt if custom_prompt and custom_prompt.strip() else None,
    )

    progress(0.9, desc="Formatting response...")

    if not ok:
        error_msg = data.get("answer", "An error occurred. Please try again.")
        yield error_msg, "", "", ""
        return

    sources_text = _format_sources(data.get("sources", []))
    sugg_text    = _format_suggestions(data.get("suggestions", []))

    meta        = data.get("metadata", {})
    conf        = data.get("confidence", 0)
    pt          = data.get("processing_time", 0)
    qt          = data.get("query_type", "")
    chunks_used = meta.get("chunks_used", 0)
    llm_name    = meta.get("llm_backend", "")
    halluc      = data.get("hallucination_flag", False)
    halluc_rsn  = data.get("hallucination_reason", "")

    info_lines = [
        f"Query type:      {qt}",
        f"Confidence:      {conf:.3f}",
        f"Processing time: {pt:.2f}s",
        f"Chunks used:     {chunks_used}",
        f"LLM backend:     {llm_name}",
    ]
    if halluc:
        info_lines.append(f"\nHallucination warning: {halluc_rsn}")

    answer       = data.get("answer", "No answer returned.")
    full_answer  = f"{answer}\n\n{'-' * 40}\n" + "\n".join(info_lines)
    report_filename = data.get("report_filename") or ""

    progress(1.0)
    yield full_answer, sources_text, sugg_text, report_filename


def _format_sources(sources):
    if not sources:
        return "No sources returned."
    lines = ["Retrieved Sources:\n"]
    for src in sources:
        lines.append(
            f"  {src['id']}. {src['document_name']}\n"
            f"     Score: {src.get('score', 0):.3f}\n"
            f"     Preview: {src.get('text_preview', '')[:200]}\n"
        )
    return "\n".join(lines)


def _format_suggestions(suggestions):
    if not suggestions:
        return ""
    return "Suggested follow-up questions:\n\n" + "\n".join(
        f"  - {s}" for s in suggestions
    )


def load_history_tab():
    items = _get_history()
    if not items:
        return "No conversation history yet."
    lines = []
    for item in reversed(items[-15:]):
        lines.append(f"[{item['timestamp'][:19]}]")
        lines.append(f"Q: {item['question']}")
        lines.append(f"A: {item['answer'][:300]}...")
        lines.append(
            f"   Type: {item['query_type']} | "
            f"Confidence: {item['confidence']:.3f} | "
            f"Hallucination: {item['hallucination_flag']}"
        )
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# TAB 2: DOCUMENT INSIGHTS
# Named functions — Gradio 6 does not support progress= in lambdas
# =============================================================================

def run_insight_base(insight_type, progress):
    labels = {
        "summary":    "Generating document summary...",
        "key_topics": "Extracting key topics...",
        "difficulty": "Analyzing difficulty levels...",
    }
    progress(0.2, desc=labels.get(insight_type, "Processing..."))
    data = _insights(insight_type)
    progress(1.0)
    pt      = data.get("processing_time", 0)
    content = data.get("content", "Failed to generate insight.")
    return f"{content}\n\n[Generated in {pt:.2f}s]"


def run_summary(progress=gr.Progress()):
    return run_insight_base("summary", progress)


def run_key_topics(progress=gr.Progress()):
    return run_insight_base("key_topics", progress)


def run_difficulty(progress=gr.Progress()):
    return run_insight_base("difficulty", progress)


def run_all_insights(progress=gr.Progress()):
    results = {}
    types = ["summary", "key_topics", "difficulty"]
    for i, t in enumerate(types):
        progress((i + 1) / len(types), desc=f"Generating {t}...")
        data    = _insights(t)
        results[t] = data.get("content", "")

    combined = ""
    if results.get("summary"):
        combined += f"DOCUMENT SUMMARY\n{'=' * 50}\n{results['summary']}\n\n"
    if results.get("key_topics"):
        combined += f"KEY TOPICS\n{'=' * 50}\n{results['key_topics']}\n\n"
    if results.get("difficulty"):
        combined += f"DIFFICULTY ANALYSIS\n{'=' * 50}\n{results['difficulty']}"
    return combined


# =============================================================================
# TAB 3: SMART NOTES & EXAM QUESTIONS
# =============================================================================

def run_notes(progress=gr.Progress()):
    progress(0.3, desc="Generating smart notes...")
    data = _insights("smart_notes")
    progress(1.0)
    return data.get("content", "Failed to generate notes.")


def run_short_questions(progress=gr.Progress()):
    progress(0.3, desc="Generating short-answer questions...")
    data = _insights("short_questions")
    progress(1.0)
    return data.get("content", "")


def run_long_questions(progress=gr.Progress()):
    progress(0.3, desc="Generating long-answer questions...")
    data = _insights("long_questions")
    progress(1.0)
    return data.get("content", "")


def run_mcq(progress=gr.Progress()):
    progress(0.3, desc="Generating MCQs...")
    data = _insights("mcq")
    progress(1.0)
    return data.get("content", "")


def run_all_questions(progress=gr.Progress()):
    combined = ""
    pairs = [
        ("Short Answer", "short_questions"),
        ("Long Answer",  "long_questions"),
        ("MCQ",          "mcq"),
    ]
    for i, (label, key) in enumerate(pairs):
        progress((i + 1) / len(pairs), desc=f"Generating {label}...")
        data      = _insights(key)
        combined += f"{label.upper()} QUESTIONS\n{'=' * 50}\n{data.get('content', '')}\n\n"
    return combined


# =============================================================================
# TAB 4: REPORT & EXPORT CENTER
# =============================================================================

def build_and_download_insights_report(progress=gr.Progress()):
    progress(0.1, desc="Generating all insights...")
    all_insights = {}
    types = [
        "summary", "key_topics", "smart_notes",
        "short_questions", "long_questions", "mcq", "difficulty",
    ]
    for i, t in enumerate(types):
        progress((i + 1) / len(types) * 0.8, desc=f"Generating {t}...")
        data            = _insights(t)
        all_insights[t] = data.get("content", "")
        # Small delay between calls to avoid hitting Groq TPM rate limits
        if i < len(types) - 1:
            time.sleep(8)

    progress(0.9, desc="Writing Word document...")
    try:
        r = requests.post(f"{API}/insights/report", json=all_insights, timeout=60)
        r.raise_for_status()
        out_path = os.path.join(tempfile.gettempdir(), "DocVision_Insights_Report.docx")
        with open(out_path, "wb") as fh:
            fh.write(r.content)
        progress(1.0)
        return out_path, "Full insights report generated."
    except Exception as e:
        return None, f"Failed: {e}"


def get_last_qa_report(report_filename):
    if not report_filename or not report_filename.strip():
        return None, "No Q&A report available yet. Ask a question first."
    path = _download_report(report_filename)
    if path:
        return path, f"Report ready: {report_filename}"
    return None, "Report file not found on server."


# =============================================================================
# TAB 5: RAG DEBUG VIEWER
# =============================================================================

def load_rag_debug():
    debug = _get_rag_debug()
    if not debug or "message" in debug:
        return "No query has been run yet. Ask a question first.", ""

    lines = [
        f"Retrieved chunks:    {debug.get('retrieved_count', 0)}",
        f"Reranked to:         {debug.get('reranked_count', 0)}",
        f"Reasoning generated: {debug.get('reasoning_generated', False)}",
        f"Grounding overlap:   {debug.get('grounding_overlap', 'N/A')}",
        "",
    ]

    chunk_lines = []
    for c in debug.get("top_scores", []):
        chunk_lines.append(
            f"  Chunk {c.get('chunk_id', '?')} | {c.get('document', '')} | "
            f"TF-IDF: {c.get('tfidf_score', 0):.4f} | "
            f"Rerank: {c.get('rerank_score', 0):.4f}\n"
            f"    Preview: {c.get('preview', '')[:120]}"
        )

    return "\n".join(lines), "\n\n".join(chunk_lines)


# =============================================================================
# TAB 6: EVALUATION DASHBOARD
# Returns 2 values to match 2 output components
# =============================================================================

def load_metrics():
    m = _get_metrics()
    if not m:
        return "Metrics unavailable.", ""

    metrics_lines = [
        "System Metrics",
        "=" * 40,
        f"Total queries run:    {m.get('total_queries', 0)}",
        f"Avg response time:    {m.get('avg_response_time', 0):.3f}s",
        f"Documents loaded:     {m.get('documents_loaded', 0)}",
        f"Total chunks:         {m.get('total_chunks', 0)}",
        f"Memory turns:         {m.get('memory_turns', 0)}",
        f"LLM backend:          {m.get('llm_backend', 'unknown')}",
        "",
        "Retrieval Settings",
        "=" * 40,
        f"Lexical weight:       {config.LEXICAL_WEIGHT}",
        f"Semantic weight:      {config.SEMANTIC_WEIGHT}",
        f"Top K after rerank:   {config.TOP_K_AFTER_RERANK}",
        f"Embedding model:      {config.EMBEDDING_MODEL}",
        f"Reranker model:       {config.RERANKER_MODEL}",
    ]

    items = _get_history()
    history_lines = [
        f"{'#':<4} {'Type':<14} {'Conf':<8} {'Halluc':<8} Question",
        "-" * 70,
    ]
    for i, item in enumerate(items, 1):
        h    = "YES" if item.get("hallucination_flag") else "no"
        conf = f"{item.get('confidence', 0):.3f}"
        q    = item.get("question", "")[:40]
        qt   = item.get("query_type", "")[:12]
        history_lines.append(f"{i:<4} {qt:<14} {conf:<8} {h:<8} {q}")

    return "\n".join(metrics_lines), "\n".join(history_lines)


# =============================================================================
# UI CONSTANTS
# =============================================================================

HEADER_HTML = """
<div style="background:linear-gradient(135deg,#1a202c,#2d3748);
            padding:24px;border-radius:12px;margin-bottom:16px;text-align:center;">
  <h1 style="color:#fff;font-size:28px;margin:0;">DocVision OCR</h1>
  <p style="color:#a0aec0;margin:6px 0 0;">
    Multi-Agent AI Document Intelligence Platform v3.0
  </p>
  <p style="color:#718096;font-size:13px;margin:4px 0 0;">
    Groq LLaMA 3 &nbsp;|&nbsp; FAISS Retrieval &nbsp;|&nbsp; OCR
    &nbsp;|&nbsp; Streaming &nbsp;|&nbsp; Memory &nbsp;|&nbsp;
    Hallucination Detection
  </p>
</div>
"""

PIPELINE_HTML = """
<div style="background:#f7fafc;padding:14px;border-radius:8px;
            font-size:13px;line-height:1.8;color:#1a202c;">
  <strong style="color:#1a202c;">Pipeline</strong><br>
  PDF + OCR Extraction<br>
  Sentence Chunking<br>
  TF-IDF + Semantic Retrieval<br>
  Cross-Encoder Reranking<br>
  Reasoning Agent<br>
  Groq LLM (LLaMA 3) Generation<br>
  Hallucination Guard<br>
  Word Report Export<br><br>
  <strong style="color:#1a202c;">Key Aspects</strong><br>
  Streaming responses<br>
  Conversation memory<br>
  Document insights<br>
  RAG debug viewer<br>
  Evaluation dashboard<br>
  Custom prompts
</div>
"""

CSS = """
.gradio-container { font-family: 'Segoe UI', Arial, sans-serif !important; }
.tab-nav button   { font-size: 13px !important; }
"""

# =============================================================================
# INTERFACE BUILDER
# =============================================================================

def create_interface() -> gr.Blocks:
    # Gradio 6.0: title only in gr.Blocks(); theme and css go to launch()
    with gr.Blocks(
    title="DocVision OCR v3",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:

        gr.HTML(HEADER_HTML)

        # ------------------------------------------------------------------ #
        # UPLOAD PANEL
        # ------------------------------------------------------------------ #
        with gr.Row():
            with gr.Column(scale=3):
                file_upload = gr.File(
                    label=f"Upload PDF files (max {config.MAX_PDFS})",
                    file_count="multiple",
                    file_types=[".pdf"],
                )
                process_btn  = gr.Button(
                    "Process Documents", variant="primary", size="lg",
                )
                upload_status = gr.Textbox(
                    label="Upload Status", interactive=False, lines=5,
                )
            with gr.Column(scale=1):
                gr.HTML(PIPELINE_HTML)

        # Hidden state
        doc_summary_state   = gr.State("")
        last_report_filename = gr.State("")

        # ------------------------------------------------------------------ #
        # TABS
        # ------------------------------------------------------------------ #
        with gr.Tabs():

            # ---- TAB 1: Chat & Q&A ----------------------------------------
            with gr.Tab("Chat & Q&A"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="Ask a question about your documents",
                            placeholder="What is the main topic discussed?",
                            lines=3,
                        )
                        with gr.Row():
                            ask_btn    = gr.Button("Get Answer",    variant="primary")
                            stream_btn = gr.Button("Generate Answer", variant="secondary")
                        answer_output = gr.Textbox(
                            label="Answer", lines=14, interactive=False,
                        )
                    with gr.Column(scale=2):
                        sources_output = gr.Textbox(
                            label="Sources", lines=8, interactive=False,
                        )
                        suggestions_output = gr.Textbox(
                            label="Follow-up Suggestions", lines=5, interactive=False,
                        )

                with gr.Row():
                    history_btn = gr.Button("Load Conversation History")
                history_output = gr.Textbox(
                    label="History", lines=10, interactive=False,
                )

                with gr.Accordion("Answer Settings", open=False):
                    use_memory_check = gr.Checkbox(
                        label="Use conversation memory", value=True,
                    )
                    tone_dropdown = gr.Dropdown(
                        choices=["professional", "simple", "technical", "academic"],
                        value="professional",
                        label="Response tone",
                    )
                    custom_prompt_input = gr.Textbox(
                        label="Custom system prompt (optional)",
                        placeholder="You are a helpful assistant...",
                        lines=3,
                    )

            # ---- TAB 2: Document Insights ---------------------------------
            with gr.Tab("Document Insights"):
                gr.Markdown("### One-click document analysis")
                with gr.Row():
                    with gr.Column():
                        summary_btn      = gr.Button("Auto Summary",     variant="primary")
                        topics_btn       = gr.Button("Key Topics")
                        difficulty_btn   = gr.Button("Difficulty Analysis")
                        all_insights_btn = gr.Button("Run All Insights", variant="secondary")
                    with gr.Column(scale=2):
                        insights_output = gr.Textbox(
                            label="Insights Output", lines=20, interactive=False,
                        )

            # ---- TAB 3: Smart Notes & Questions ---------------------------
            with gr.Tab("Smart Notes & Questions"):
                gr.Markdown("### Generate study materials from your documents")
                with gr.Row():
                    notes_btn   = gr.Button("Generate Smart Notes",   variant="primary")
                    short_q_btn = gr.Button("Short Answer Questions")
                    long_q_btn  = gr.Button("Long Answer Questions")
                    mcq_btn     = gr.Button("MCQ Questions")
                    all_q_btn   = gr.Button("All Question Types",     variant="secondary")
                notes_output = gr.Textbox(label="Output", lines=22, interactive=False)

            # ---- TAB 4: Report & Export Center ----------------------------
            with gr.Tab("Report & Export Center"):
                gr.Markdown("### Download your analysis as Word documents")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Last Q&A Report**")
                        download_qa_btn  = gr.Button("Download Q&A Report")
                        qa_report_file   = gr.File(label="Q&A Report")
                        qa_report_status = gr.Textbox(
                            label="Status", lines=1, interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("**Full Insights Report**")
                        gr.Markdown(
                            "Generates summary, topics, notes, all questions, "
                            "and difficulty into one Word document. "
                            "May take 1-2 minutes."
                        )
                        insights_report_btn    = gr.Button(
                            "Build & Download Insights Report", variant="primary",
                        )
                        insights_report_file   = gr.File(label="Insights Report")
                        insights_report_status = gr.Textbox(
                            label="Status", lines=1, interactive=False,
                        )

            # ---- TAB 5: RAG Debug Viewer ----------------------------------
            with gr.Tab("RAG Debug Viewer"):
                gr.Markdown(
                    "### Live retrieval pipeline transparency\n"
                    "Shows what the retrieval system found and scored "
                    "for the last query."
                )
                rag_refresh_btn = gr.Button("Refresh Debug Data")
                with gr.Row():
                    rag_summary_output = gr.Textbox(
                        label="Pipeline Summary", lines=8, interactive=False,
                    )
                    rag_chunks_output  = gr.Textbox(
                        label="Retrieved Chunks Detail", lines=20, interactive=False,
                    )

            # ---- TAB 6: Evaluation Dashboard ------------------------------
            with gr.Tab("Evaluation Dashboard"):
                gr.Markdown("### System performance and retrieval metrics")
                metrics_refresh_btn = gr.Button("Refresh Metrics")
                with gr.Row():
                    metrics_output       = gr.Textbox(
                        label="System Metrics", lines=18, interactive=False,
                    )
                    query_history_output = gr.Textbox(
                        label="Query Log", lines=18, interactive=False,
                    )

            # ---- TAB 7: Settings ------------------------------------------
            with gr.Tab("Settings"):
                gr.Markdown("### System controls")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Memory**")
                        clear_memory_btn = gr.Button("Clear Conversation Memory")
                        memory_status    = gr.Textbox(
                            label="Memory Status", lines=1, interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("**Documents**")
                        clear_docs_btn = gr.Button(
                            "Clear All Documents", variant="stop",
                        )
                        docs_status = gr.Textbox(
                            label="Document Status", lines=1, interactive=False,
                        )
                gr.Markdown("---")
                gr.Markdown(
                    "**API base:** `http://localhost:8000`  \n"
                    "**API docs:** `http://localhost:8000/docs`  \n"
                    "**Gradio:**   `http://localhost:7860`"
                )

        # ------------------------------------------------------------------ #
        # EVENT WIRING
        # ------------------------------------------------------------------ #

        # Upload — exactly 2 real component outputs, no gr.update()
        process_btn.click(
            fn=process_documents,
            inputs=[file_upload],
            outputs=[upload_status, doc_summary_state],
            show_progress=True,
        )

        # Q&A streaming
        ask_btn.click(
            fn=stream_answer,
            inputs=[question_input, use_memory_check, tone_dropdown, custom_prompt_input],
            outputs=[answer_output, sources_output, suggestions_output, last_report_filename],
            show_progress=True,
        )
        stream_btn.click(
            fn=stream_answer,
            inputs=[question_input, use_memory_check, tone_dropdown, custom_prompt_input],
            outputs=[answer_output, sources_output, suggestions_output, last_report_filename],
        )
        question_input.submit(
            fn=stream_answer,
            inputs=[question_input, use_memory_check, tone_dropdown, custom_prompt_input],
            outputs=[answer_output, sources_output, suggestions_output, last_report_filename],
        )

        # History
        history_btn.click(fn=load_history_tab, outputs=[history_output])

        # Insights — all named functions, no lambdas
        summary_btn.click(
            fn=run_summary, outputs=[insights_output], show_progress=True,
        )
        topics_btn.click(
            fn=run_key_topics, outputs=[insights_output], show_progress=True,
        )
        difficulty_btn.click(
            fn=run_difficulty, outputs=[insights_output], show_progress=True,
        )
        all_insights_btn.click(
            fn=run_all_insights, outputs=[insights_output], show_progress=True,
        )

        # Notes & Questions
        notes_btn.click(
            fn=run_notes, outputs=[notes_output], show_progress=True,
        )
        short_q_btn.click(
            fn=run_short_questions, outputs=[notes_output], show_progress=True,
        )
        long_q_btn.click(
            fn=run_long_questions, outputs=[notes_output], show_progress=True,
        )
        mcq_btn.click(
            fn=run_mcq, outputs=[notes_output], show_progress=True,
        )
        all_q_btn.click(
            fn=run_all_questions, outputs=[notes_output], show_progress=True,
        )

        # Reports
        download_qa_btn.click(
            fn=get_last_qa_report,
            inputs=[last_report_filename],
            outputs=[qa_report_file, qa_report_status],
            show_progress=True,
        )
        insights_report_btn.click(
            fn=build_and_download_insights_report,
            outputs=[insights_report_file, insights_report_status],
            show_progress=True,
        )

        # RAG debug
        rag_refresh_btn.click(
            fn=load_rag_debug,
            outputs=[rag_summary_output, rag_chunks_output],
        )

        # Dashboard — load_metrics() returns 2 values
        metrics_refresh_btn.click(
            fn=load_metrics,
            outputs=[metrics_output, query_history_output],
        )

        # Settings
        clear_memory_btn.click(fn=_clear_memory, outputs=[memory_status])
        clear_docs_btn.click(fn=_clear_documents, outputs=[docs_status])

    return demo


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    demo = create_interface()
    # Gradio 6.0: theme and css passed to launch(), not gr.Blocks()
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.GRADIO_PORT,
        share=False,
        show_error=True,
        inbrowser=False,
    )