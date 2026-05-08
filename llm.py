"""
LLM abstraction layer.
Provides a common interface for all LLM backends.
Primary: Groq API (LLaMA 3)
Fallback: Extractive summarization (no external model required)

Fix log (400 Bad Request):
    1. Model name was "llama3-70b-8192" which Groq has deprecated.
       Correct current names: "llama-3.3-70b-versatile" (primary)
       and "llama-3.1-8b-instant" (fallback).

    2. Prompt was sent without a token-length guard. Groq returns 400 when
       prompt_tokens + max_tokens exceeds the model context window.
       We now estimate token count and hard-truncate the prompt before sending.

    3. The Groq error response body was never logged, making the root cause
       invisible. We now log response.text on every HTTP error.

    4. Fallback model retry only triggered on 429/503, not 400.
       A 400 from a deprecated/wrong model name also needs a retry.
       Updated condition to include 400.
"""

import logging
import time
import re
from abc import ABC, abstractmethod
from typing import Optional

import requests

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq model context window sizes (in tokens)
# ---------------------------------------------------------------------------
GROQ_MODEL_CONTEXT_LIMITS = {
    "llama-3.3-70b-versatile":  128_000,
    "llama-3.1-70b-versatile":  128_000,
    "llama-3.1-8b-instant":     128_000,
    "llama3-8b-8192":             8_192,
    "llama3-70b-8192":            8_192,
    "mixtral-8x7b-32768":        32_768,
    "gemma2-9b-it":               8_192,
}
DEFAULT_CONTEXT_LIMIT = 8_192

# Reserve this many tokens for the completion response
COMPLETION_RESERVE_TOKENS = 1_200

# Conservative characters-per-token estimate for English text
CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _truncate_prompt(prompt: str, model: str, max_completion_tokens: int) -> str:
    """
    Truncate the prompt so that:
        estimated_prompt_tokens + max_completion_tokens + COMPLETION_RESERVE_TOKENS
        <= model context window
    This prevents Groq 400 errors caused by exceeding the context limit.
    """
    context_limit = GROQ_MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)
    available_for_prompt = context_limit - max_completion_tokens - COMPLETION_RESERVE_TOKENS
    max_prompt_chars = int(available_for_prompt * CHARS_PER_TOKEN)

    if len(prompt) <= max_prompt_chars:
        return prompt

    truncated = prompt[:max_prompt_chars]
    logger.warning(
        f"Prompt truncated from {len(prompt)} to {len(truncated)} chars "
        f"to fit within {model} context window ({context_limit} tokens)."
    )
    return truncated + "\n\n[Context was truncated to fit within model limits.]"


# =============================================================================
# BASE INTERFACE
# =============================================================================

class BaseLLM(ABC):
    """Abstract base class for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def name(self) -> str:
        return self.__class__.__name__


# =============================================================================
# GROQ LLM
# =============================================================================

class GroqLLM(BaseLLM):
    """
    Groq API integration using LLaMA 3.
    Uses the OpenAI-compatible chat completions endpoint.
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    DEFAULT_SYSTEM = (
        "You are DocVision, an expert document analyst. "
        "Answer questions strictly based on the provided document context. "
        "If the context does not contain sufficient information, clearly say so. "
        "Do not hallucinate or fabricate information. "
        "Be concise and accurate."
    )

    def __init__(self):
        self.api_key = config.GROQ_API_KEY
        self.model = config.GROQ_MODEL
        self.fallback_model = config.GROQ_FALLBACK_MODEL
        self.temperature = config.GROQ_TEMPERATURE
        self.max_tokens = config.GROQ_MAX_TOKENS

    def is_available(self) -> bool:
        return bool(
            self.api_key
            and self.api_key.strip()
            and self.api_key != "your_groq_api_key_here"
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Groq API key not configured. Set GROQ_API_KEY in your .env file."
            )

        system = system_prompt or self.DEFAULT_SYSTEM
        safe_prompt = _truncate_prompt(prompt, self.model, self.max_tokens)
        return self._call_api(safe_prompt, system, self.model)

    def _call_api(self, prompt: str, system: str, model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            # Log full Groq error body — this is what was hiding the real cause
            if not response.ok:
                logger.error(
                    f"Groq API returned {response.status_code} for model {model!r}. "
                    f"Full response: {response.text}"
                )

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0

            if status == 429:
                # Parse the exact wait time from Groq's error message.
                # Example: "Please try again in 11.675s."
                wait_seconds = self._parse_retry_after(e.response)
                logger.warning(
                    f"Groq rate limit on {model!r}. "
                    f"Waiting {wait_seconds:.1f}s then retrying "
                    f"({'same model' if model == self.fallback_model else 'fallback model'})."
                )
                time.sleep(wait_seconds)
                if model != self.fallback_model:
                    safe_prompt = _truncate_prompt(prompt, self.fallback_model, self.max_tokens)
                    return self._call_api(safe_prompt, system, self.fallback_model)
                # Already on fallback — re-raise so caller gets a clean error
                logger.error(f"Groq API HTTP error (no retry): {e}")
                raise

            # 400 bad model/context, 502/503/520 server/Cloudflare errors — switch to fallback
            if status in (400, 502, 503, 520) and model != self.fallback_model:
                logger.warning(
                    f"Groq model {model!r} returned {status}. "
                    f"Retrying with fallback model {self.fallback_model!r}."
                )
                safe_prompt = _truncate_prompt(prompt, self.fallback_model, self.max_tokens)
                return self._call_api(safe_prompt, system, self.fallback_model)

            logger.error(f"Groq API HTTP error (no retry): {e}")
            raise

        except requests.exceptions.Timeout:
            logger.error("Groq API request timed out after 60 seconds.")
            raise

        except Exception as e:
            logger.error(f"Groq API call failed unexpectedly: {e}")
            raise


    @staticmethod
    def _parse_retry_after(response) -> float:
        """
        Extract the wait duration from a Groq 429 response.
        Groq embeds it in the error message as: "Please try again in 11.675s."
        Falls back to 20 seconds if unparseable.
        """
        try:
            # Check standard HTTP Retry-After header first
            retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
            if retry_after:
                return float(retry_after) + 1.0

            # Parse from Groq's JSON error message body
            body = response.json()
            msg = body.get("error", {}).get("message", "")
            match = re.search(r'try again in ([0-9.]+)s', msg)
            if match:
                return float(match.group(1)) + 1.5  # add 1.5s buffer
        except Exception:
            pass
        return 20.0  # safe default


# =============================================================================
# EXTRACTIVE FALLBACK LLM
# =============================================================================

class ExtractiveLLM(BaseLLM):
    """
    Pure-Python extractive fallback — no API key or model download required.
    Scores sentences by keyword overlap with the question and returns the top ones.
    Used when Groq is unavailable or when all API retries are exhausted.
    """

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        context = self._extract_context(prompt)
        if not context:
            return "Unable to extract an answer from the provided context."

        question = self._extract_question(prompt)
        sentences = self._split_sentences(context)

        if not sentences:
            return context[:500]

        q_words = set(re.findall(r"\w+", question.lower()))
        scored = []
        for sentence in sentences:
            s_words = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(q_words & s_words) if q_words else 0
            scored.append((overlap, sentence))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:5] if s.strip()]
        answer = " ".join(top).strip()
        return answer if answer else sentences[0]

    @staticmethod
    def _extract_context(prompt: str) -> str:
        match = re.search(r"DOCUMENT CONTEXT:\s*-{5,}(.*?)-{5,}", prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        parts = re.split(r"-{5,}", prompt)
        return parts[-1].strip() if len(parts) > 1 else prompt.strip()

    @staticmethod
    def _extract_question(prompt: str) -> str:
        match = re.search(r"QUESTION:\s*(.+)", prompt)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _split_sentences(text: str):
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            return re.split(r"(?<=[.!?])\s+", text)


# =============================================================================
# LLM FACTORY
# =============================================================================

class LLMFactory:
    """Returns the best available LLM as a singleton."""

    _instance: Optional[BaseLLM] = None

    @classmethod
    def get_llm(cls) -> BaseLLM:
        if cls._instance is not None:
            return cls._instance

        groq = GroqLLM()
        if groq.is_available():
            logger.info(
                f"LLM backend: Groq API | "
                f"primary={groq.model} | fallback={groq.fallback_model}"
            )
            cls._instance = groq
        else:
            logger.warning(
                "GROQ_API_KEY not set or is still the placeholder value. "
                "Using extractive fallback LLM. "
                "Set GROQ_API_KEY in .env for full answer quality."
            )
            cls._instance = ExtractiveLLM()

        return cls._instance

    @classmethod
    def reset(cls):
        """Force re-evaluation (useful after updating env vars at runtime)."""
        cls._instance = None