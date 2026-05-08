"""
start_render.py - Production start script for Render deployment.

Render assigns a single PORT environment variable.
This script:
  1. Reads PORT from environment (set by Render)
  2. Runs FastAPI on that port (Render routes external traffic here)
  3. Runs Gradio on PORT+1 internally (accessible via /gradio path or directly)

On Render free/starter tier only one port is exposed externally.
We serve FastAPI on the main port and mount Gradio as a sub-application.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# Render sets PORT automatically — we must bind to it
RENDER_PORT = int(os.environ.get("PORT", 8000))

# Patch config to use Render's port
os.environ["API_PORT"] = str(RENDER_PORT)

from config import config

# Validate key
if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
    logger.warning("GROQ_API_KEY not set. Using extractive fallback LLM.")

logger.info(f"Starting DocVision OCR on port {RENDER_PORT}")

# Mount Gradio inside FastAPI so both are served on the same port
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the FastAPI app
from api import app as fastapi_app

# Build Gradio interface
from ui import create_interface
gradio_demo = create_interface()

# Mount Gradio at /gradio path inside FastAPI
gradio_app = gr.mount_gradio_app(fastapi_app, gradio_demo, path="/gradio")

logger.info(f"FastAPI docs:   http://0.0.0.0:{RENDER_PORT}/docs")
logger.info(f"Gradio UI:      http://0.0.0.0:{RENDER_PORT}/gradio")
logger.info(f"Health check:   http://0.0.0.0:{RENDER_PORT}/health")

if __name__ == "__main__":
    uvicorn.run(
        gradio_app,
        host="0.0.0.0",
        port=RENDER_PORT,
        log_level="info",
    )