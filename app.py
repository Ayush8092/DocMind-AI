"""
app.py - Single entry point for DocVision OCR v3.
Run with: python app.py
"""

import logging
import os
import sys
import threading
import time

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from config import config


def validate_environment():
    if not config.GROQ_API_KEY or config.GROQ_API_KEY in ("", "your_groq_api_key_here"):
        logger.warning("=" * 60)
        logger.warning("GROQ_API_KEY not set in .env file.")
        logger.warning("The system will use the extractive fallback LLM.")
        logger.warning("Set GROQ_API_KEY in .env for full LLM quality.")
        logger.warning("=" * 60)


def run_api():
    logger.info(f"Starting FastAPI backend on port {config.API_PORT} ...")
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="warning",
    )


def run_ui():
    import gradio as gr
    import requests

    health_url = f"http://127.0.0.1:{config.API_PORT}/health"
    for _ in range(30):
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                logger.info("FastAPI backend is ready.")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        logger.warning("API did not start in 30s. UI may not function correctly.")

    import tempfile
    from ui import CSS, create_interface
    demo = create_interface()
    logger.info(f"Starting Gradio UI on port {config.GRADIO_PORT} ...")
    # allowed_paths lets Gradio 6 serve files from the system temp directory.
    # This is required for report downloads on Windows where /tmp is not used.
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.GRADIO_PORT,
        share=False,
        show_error=True,
        inbrowser=True,
        allowed_paths=[tempfile.gettempdir()],
    )


if __name__ == "__main__":
    validate_environment()

    logger.info("DocVision OCR v3 - Multi-Agent AI Document Intelligence")
    logger.info(f"  FastAPI  -> http://localhost:{config.API_PORT}")
    logger.info(f"  API docs -> http://localhost:{config.API_PORT}/docs")
    logger.info(f"  Gradio   -> http://localhost:{config.GRADIO_PORT}")

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    api_thread = threading.Thread(target=run_api, daemon=True, name="fastapi")
    api_thread.start()

    try:
        run_ui()
    except KeyboardInterrupt:
        logger.info("Shutting down DocVision OCR.")
        sys.exit(0)