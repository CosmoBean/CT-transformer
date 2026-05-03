"""
CMU AI Gateway client for OpenAI-compatible chat completions.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class GatewayError(RuntimeError):
    """Raised when the CMU gateway request fails."""


@dataclass(frozen=True)
class GatewayResponse:
    raw_json: dict[str, Any]
    text: str


class CmuGatewayClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("CMU_LLM_GATEWAY_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise GatewayError(
                "No CMU gateway API key found. Set CMU_LLM_GATEWAY_API_KEY or ANTHROPIC_API_KEY."
            )

    def chat_completion(self, messages: list[dict[str, Any]]) -> GatewayResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise GatewayError(f"Gateway HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise GatewayError(f"Gateway request failed: {exc}") from exc

        response_json = json.loads(body)
        try:
            text = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayError(f"Unexpected gateway response format: {body}") from exc

        return GatewayResponse(raw_json=response_json, text=text)

    def chat_completion_with_retries(
        self,
        messages: list[dict[str, Any]],
        max_retries: int,
        retry_delay_seconds: float = 1.5,
    ) -> GatewayResponse:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.chat_completion(messages)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(retry_delay_seconds)

        raise GatewayError(f"Gateway request failed after retries: {last_error}") from last_error
