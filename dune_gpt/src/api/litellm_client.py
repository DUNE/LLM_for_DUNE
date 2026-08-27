import json
from typing import Optional

import requests

from config import (
    LITELLM_API_KEY,
    LITELLM_API_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LiteLLMClient:
    """Client for LiteLLM's OpenAI-compatible chat completions API."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = self._normalize_chat_url(base_url or LITELLM_API_URL)
        self.api_key = api_key if api_key is not None else LITELLM_API_KEY

        if not self.base_url:
            raise ValueError("LITELLM_API_URL is not configured")

    def _normalize_chat_url(self, url: str) -> str:
        url = (url or "").strip().rstrip("/")
        if not url:
            return ""
        if url.endswith("/v1/chat/completions") or url.endswith("/chat/completions"):
            return url
        return f"{url}/v1/chat/completions"

    def chat_completion(
        self,
        question: str,
        context: str,
        links: list[str] = None,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        model: str = LLM_MODEL,
        timeout: int = 60,
    ):
        """Send a streaming OpenAI-compatible chat completion request."""
        yield "STATUS:PROCESSING"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant specialized in scientific "
                        "documentation for the DUNE experiment. Provide succinct answers. "
                        "If the answer is not supported by the context, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ],
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        try:
            logger.info(f"Sending request to LiteLLM API with model {model}")
            with requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    if not line.startswith("data: "):
                        continue

                    data = line[len("data: ") :].strip()
                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content

            if links:
                sources = [{"url": url} for url in links]
                yield "SOURCES: " + json.dumps(sources)

        except requests.Timeout:
            logger.error("LiteLLM API request timed out")
            yield f"[ERROR] Request to LiteLLM API timed out after {timeout} seconds."

        except requests.RequestException as e:
            logger.error(f"LiteLLM API request failed: {e}")
            yield f"[ERROR] Failed to get response from LiteLLM API: {e}"

        except Exception as e:
            logger.error(f"Unexpected error with LiteLLM API: {e}")
            yield f"[ERROR] Unexpected error: {e}"

    def health_check(self) -> bool:
        """Check if the LiteLLM API is accessible."""
        try:
            chunks = self.chat_completion(
                question="Hello",
                context="This is a test",
                timeout=10,
            )
            for chunk in chunks:
                if str(chunk).startswith("STATUS:"):
                    continue
                return not str(chunk).startswith("[ERROR]")
            return False
        except Exception as e:
            logger.error(f"LiteLLM API health check failed: {e}")
            return False
