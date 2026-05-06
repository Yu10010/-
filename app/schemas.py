"""Request and response models for the wildfire rescue planning API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RescuePlanRequest(BaseModel):
    """Normalized payload sent to the language model."""

    cnn_output: dict[str, Any] = Field(
        ..., description="Structured JSON produced by the trained CNN pipeline."
    )
    image_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded image emitted with the CNN JSON output.",
    )
    image_mime_type: str | None = Field(
        default=None, description="MIME type for the optional image."
    )
    external_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional geospatial/weather context from LANDFIRE, OSM, USGS, and OpenWeather.",
    )


class RescuePlanResponse(BaseModel):
    """Decision-support plan produced by the selected Llama model."""

    model: str = Field(..., description="LLM model used to produce the plan.")
    provider: str = Field(..., description="Configured LLM provider adapter.")
    plan: str = Field(..., description="LLM-generated wildfire rescue plan.")
    external_context: dict[str, Any] | None = Field(
        default=None,
        description="External context included in the LLM prompt; omitted unless requested.",
    )
    raw_response: dict[str, Any] | None = Field(
        default=None,
        description="Raw upstream response for debugging; omitted unless requested.",
    )
