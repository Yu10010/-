"""LLM client helpers for Ollama-backed wildfire rescue planning."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.schemas import RescuePlanRequest


SYSTEM_PROMPT = """You are an emergency-response planning assistant.

You generate wildfire rescue-support plans from structured CNN classifier output and external context.

Critical rules:
- This is decision support only. A human incident commander must review and approve all actions.
- For the final rescue-plan step, no image is provided. Do not say "the image shows" or claim visual details.
- Use only the CNN JSON and external_context.
- Do not invent object counts, bounding boxes, roads, buildings, residential areas, landmarks, people, or nearby assets unless they appear explicitly in the CNN JSON or external_context.
- If the CNN JSON does not include people, buildings, roads, bounding boxes, or object detections, say those details are missing.
- If an external context source has status "error", "skipped", or "configured_request_only", treat that source as unavailable.
- The CNN output should be treated as classification output unless the JSON explicitly contains detections or bounding boxes.
- Prioritize life safety, evacuation, responder safety, and uncertainty management.
- Do not claim certainty when the data is uncertain.
- Do not summarize the JSON as a map or dashboard. Produce an emergency-support plan.

Return exactly these eight numbered sections:
1. Situation summary
2. People-at-risk assessment
3. Immediate priorities
4. Rescue and evacuation plan
5. Resources to dispatch
6. Communication plan
7. Hazards, uncertainty, and missing data
8. Safety note for human incident commanders
"""


USER_PROMPT_TEMPLATE = """Generate a wildfire rescue-support plan using only the following structured data.

CNN output:
{cnn_output}

External context:
{external_context}

Instructions:
- Use the CNN prediction and confidence as the main fire/no-fire evidence.
- Use external_context only when its status is "ok" or when specific fields are available.
- If location, weather, roads, hospitals, fire stations, water sources, people count, or threatened assets are missing, say they are missing.
- Do not say "the image shows".
- Do not describe screenshots, dashboards, maps, buttons, or visual labels.
- Do not invent people, buildings, or evacuation routes.
- The response must use exactly the eight required numbered sections.
"""

VISION_ASSESSMENT_PROMPT = """You are checking a wildfire classifier result.

Use the uploaded normal image only as supporting visual evidence.
Return ONLY valid JSON. Do not include markdown.

CNN output:
{cnn_output}

Return this JSON shape:
{{
  "prediction": "fire" or "nofire",
  "fire_probability": number between 0 and 1,
  "nofire_probability": number between 0 and 1,
  "confidence": number between 0 and 1,
  "rationale": "one short sentence"
}}

Rules:
- Do not invent object counts, bounding boxes, roads, buildings, map labels, or landmarks.
- If uncertain, use probabilities close to 0.5.
- fire_probability + nofire_probability should be approximately 1.
"""


def _json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return JSON: {text}")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM JSON was not an object: {text}")

    return parsed


def _clamp_probability(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    return max(0.0, min(1.0, number))


def _normalize_visual_assessment(raw_text: str) -> dict[str, Any]:
    data = _extract_json_object(raw_text)

    fire_probability = _clamp_probability(data.get("fire_probability"), 0.5)
    nofire_probability = _clamp_probability(
        data.get("nofire_probability"),
        1.0 - fire_probability,
    )

    total = fire_probability + nofire_probability
    if total > 0:
        fire_probability = fire_probability / total
        nofire_probability = nofire_probability / total
    else:
        fire_probability = 0.5
        nofire_probability = 0.5

    prediction = "fire" if fire_probability >= nofire_probability else "nofire"
    confidence = max(fire_probability, nofire_probability)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "class_probabilities": {
            "fire": fire_probability,
            "nofire": nofire_probability,
        },
        "rationale": str(data.get("rationale", "")),
    }


class LLMClient(Protocol):
    provider: str
    model: str

    async def build_plan(
        self,
        request: RescuePlanRequest,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a wildfire rescue-support plan."""

    async def assess_image_fire_probability(
        self,
        image_base64: str,
        image_mime_type: str | None,
        cnn_output: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Use the LLM vision model to estimate fire/nofire probability."""


@dataclass
class OllamaClient:
    base_url: str
    model: str
    timeout_seconds: float
    provider: str = "ollama"

    async def build_plan(
        self,
        request: RescuePlanRequest,
    ) -> tuple[str, dict[str, Any]]:
        cnn_output = request.cnn_output
        external_context = request.external_context

        user_message: dict[str, Any] = {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                cnn_output=_json_dumps(cnn_output),
                external_context=_json_dumps(external_context),
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

        plan = raw.get("message", {}).get("content", "")
        return plan, raw

    async def assess_image_fire_probability(
        self,
        image_base64: str,
        image_mime_type: str | None,
        cnn_output: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        user_message: dict[str, Any] = {
            "role": "user",
            "content": VISION_ASSESSMENT_PROMPT.format(
                cnn_output=_json_dumps(cnn_output)
            ),
            "images": [image_base64],
        }

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON for wildfire image verification.",
                },
                user_message,
            ],
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            raw = response.json()

        content = raw.get("message", {}).get("content", "")
        assessment = _normalize_visual_assessment(content)
        return assessment, raw


def create_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    model = os.getenv("LLM_MODEL", "llava:7b")
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))

    if provider != "ollama":
        raise ValueError(
            f"Unsupported LLM_PROVIDER={provider!r}. This version supports 'ollama'."
        )

    return OllamaClient(
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )