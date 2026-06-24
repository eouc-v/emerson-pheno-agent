"""
llm.py — Unified Ollama LLM interface for the agentic system.

Provides a single ``OllamaHandler`` class that wraps ``ollama.chat`` with:
- Model-per-call selection
- ``<think>…</think>`` tag stripping (for reasoning models like deepseek-r1)
- Robust JSON extraction from free-form responses
- Retry logic with exponential back-off
- Per-call timing / token logging
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

import ollama

from pheno_agent.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response post-processing helpers
# ---------------------------------------------------------------------------

def strip_think_tags(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_from_response(raw: str) -> str:
    """
    Extract the first JSON object ``{…}`` from a model response.

    Handles surrounding markdown fences, ``<think>`` blocks, and preamble text.
    """
    cleaned = strip_think_tags(raw)

    # Strip markdown code fences  (```json … ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)

    # Find outermost JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return match.group(0) if match else cleaned


def parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parse from a model response.

    Returns ``None`` if parsing fails completely.
    """
    json_str = extract_json_from_response(raw)
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        logger.warning("JSON parse failed. Attempting line-by-line repair…")

    # Last-resort: try to fix common issues (trailing commas, single quotes)
    try:
        repaired = re.sub(r",\s*}", "}", json_str)
        repaired = re.sub(r",\s*]", "]", repaired)
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        logger.error("Could not parse JSON from model response.")
        return None


# ---------------------------------------------------------------------------
# Ollama handler
# ---------------------------------------------------------------------------

class OllamaHandler:
    """
    Thin wrapper around ``ollama.chat`` for the agentic system.

    Parameters
    ----------
    default_model : str
        Fallback model name when ``model`` is not passed to ``get_completion``.
    temperature : float
        Sampling temperature (0.0 for deterministic).
    seed : int
        Random seed for reproducibility.
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Number of retries on transient failures.
    """

    def __init__(
        self,
        default_model: str = cfg.models.reasoning_model,
        temperature: float = cfg.agent.temperature,
        seed: int = cfg.agent.seed,
        timeout: int = cfg.agent.ollama_timeout,
        max_retries: int = 3,
    ):
        self.default_model = default_model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize client pointing to correct host
        self.client = ollama.Client(host=cfg.agent.ollama_host)

        # Cumulative stats
        self.total_calls = 0
        self.total_time = 0.0

    # ---- core completion ---------------------------------------------------

    def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        expect_json: bool = False,
    ) -> str:
        """
        Send a chat completion request to Ollama.

        Parameters
        ----------
        system_prompt : str
            The system-level instruction.
        user_prompt : str
            The user message.
        model : str, optional
            Override the default model for this call.
        temperature : float, optional
            Override the default temperature.
        expect_json : bool
            If True, set ``format="json"`` in the Ollama request.

        Returns
        -------
        str
            Raw model response text (with ``<think>`` tags already stripped).
        """
        model = model or self.default_model
        temp = temperature if temperature is not None else self.temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        options = {
            "temperature": temp,
            "seed": self.seed,
        }

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options,
        }
        if expect_json:
            kwargs["format"] = "json"

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.time()
                response = self.client.chat(**kwargs)
                elapsed = time.time() - t0

                raw = response["message"]["content"]
                cleaned = strip_think_tags(raw)

                self.total_calls += 1
                self.total_time += elapsed
                logger.debug(
                    "LLM call #%d  model=%s  time=%.1fs  chars=%d",
                    self.total_calls, model, elapsed, len(cleaned),
                )
                return cleaned

            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        # All retries exhausted
        logger.error("Ollama call failed after %d retries: %s", self.max_retries, last_error)
        return ""

    # ---- convenience wrappers ----------------------------------------------

    def get_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM and parse the response as JSON.

        Returns None if parsing fails.
        """
        raw = self.get_completion(
            system_prompt, user_prompt, model=model, expect_json=True,
        )
        return parse_json_response(raw)

    def stats(self) -> Dict[str, Any]:
        """Return cumulative call statistics."""
        return {
            "total_calls": self.total_calls,
            "total_time_s": round(self.total_time, 1),
            "avg_time_s": round(self.total_time / max(self.total_calls, 1), 1),
        }
