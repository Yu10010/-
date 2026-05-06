"""Adapters for sending CNN outputs and images to Llama-compatible APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.schemas import RescuePlanRequest


SYSTEM_PROMPT = """You are an emergency-response planning assistant for wildfire rescue.
Use the CNN detections, image context, and provided metadata to produce a concise,
actionable rescue support plan. Treat the output as decision support only: do not
claim certainty, replace incident command, or recommend unsafe actions. Prioritize
life safety, evacuation routes, triage, communications, staging areas, resource
allocation, uncertainty, and data gaps. If information is missing, say what is
needed before responders act.
"""


USER_PROMPT_TEMPLATE = """Build a wildfire rescue support plan from this CNN output and external context.

CNN JSON:
{cnn_output}

External context from LANDFIRE, OpenStreetMap, USGS National Map, and OpenWeatherMap:
{external_context}

Return these sections:
1. Situation summary
2. People-at-risk assessment
3. Immediate priorities
4. Rescue and evacuation plan
5. Resources to dispatch
6. Communication plan
7. Hazards, uncertainty, and missing data
8. Safety note for human incident commanders
"""


class LLMClient(Protocol):
    """Protocol shared by LLM provider adapters."""

    provider: str
    model: str

    async def build_plan(self, request: RescuePlanRequest) -> tuple[str, dict[str, Any]]:
        """Return generated plan text and raw upstream response."""


@dataclass(frozen=True)
class LLMConfig:
    """Environment-driven configuration for the LLM adapter."""

    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower()
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434")
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2-vision"))
    api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    )


class OllamaClient:
    """Client for Ollama's Llama-compatible chat API."""

    provider = "ollama"

    def __init__(self, config: LLMConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.timeout_seconds = config.timeout_seconds

    async def build_plan(self, request: RescuePlanRequest) -> tuple[str, dict[str, Any]]:
        user_message: dict[str, Any] = {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                cnn_output=request.cnn_output,
                external_context=request.external_context or {},
            ),
        }
        if request.image_base64:
            user_message["images"] = [request.image_base64]

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                user_message,
            ],
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            raw = response.json()

        return raw.get("message", {}).get("content", ""), raw


class OpenAICompatibleClient:
    """Client for OpenAI-compatible Llama gateways such as vLLM or Together-style APIs."""

    provider = "openai_compatible"

    def __init__(self, config: LLMConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.api_key = config.api_key
        self.timeout_seconds = config.timeout_seconds

    async def build_plan(self, request: RescuePlanRequest) -> tuple[str, dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": USER_PROMPT_TEMPLATE.format(
                    cnn_output=request.cnn_output,
                    external_context=request.external_context or {},
                ),
            }
        ]
        if request.image_base64 and request.image_mime_type:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{request.image_mime_type};base64,{request.image_base64}"
                    },
                }
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            raw = response.json()

        return raw.get("choices", [{}])[0].get("message", {}).get("content", ""), raw


def create_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """Create the configured LLM adapter."""

    config = config or LLMConfig()
    if config.provider == "ollama":
        return OllamaClient(config)
    if config.provider in {"openai", "openai_compatible", "vllm"}:
        return OpenAICompatibleClient(config)
    raise ValueError(
        "Unsupported LLM_PROVIDER. Use 'ollama' or 'openai_compatible'."
    )
