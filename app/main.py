"""FastAPI app that bridges CNN wildfire outputs to a Llama planning model."""

from __future__ import annotations

import base64
import json
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.context_clients import ExternalContextClient, IncidentLocation, extract_location
from app.llm_client import create_llm_client
from app.schemas import RescuePlanRequest, RescuePlanResponse

app = FastAPI(
    title="Wildfire Rescue Planning Bridge",
    description=(
        "Accepts JSON and image outputs from a trained CNN and forwards them to a "
        "configured Llama-compatible LLM to generate a human-reviewed rescue plan."
    ),
    version="0.1.0",
)


async def _load_json_payload(
    cnn_payload: str | None, cnn_json: UploadFile | None
) -> dict[str, Any]:
    if cnn_payload and cnn_json:
        raise HTTPException(
            status_code=400,
            detail="Send either cnn_payload form text or cnn_json file, not both.",
        )
    if not cnn_payload and not cnn_json:
        raise HTTPException(
            status_code=400,
            detail="A CNN JSON payload is required as cnn_payload or cnn_json.",
        )

    try:
        if cnn_payload is not None:
            raw_payload = cnn_payload
        elif cnn_json is not None:
            raw_payload = (await cnn_json.read()).decode()
        else:
            raw_payload = ""
        parsed = json.loads(raw_payload)
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="CNN JSON file must be UTF-8."
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid CNN JSON: {exc.msg}"
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="CNN JSON must be an object.")
    return parsed


async def _encode_image(image: UploadFile | None) -> tuple[str | None, str | None]:
    if image is None:
        return None, None
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    return base64.b64encode(content).decode("ascii"), image.content_type


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health for container and load-balancer checks."""

    return {"status": "ok"}


@app.post("/wildfire/rescue-plan", response_model=RescuePlanResponse)
async def wildfire_rescue_plan(
    cnn_payload: Annotated[
        str | None,
        Form(description="CNN output JSON as text. Use this or cnn_json."),
    ] = None,
    cnn_json: Annotated[
        UploadFile | None,
        File(description="CNN output JSON file. Use this or cnn_payload."),
    ] = None,
    image: Annotated[
        UploadFile | None,
        File(description="Image emitted by the CNN pipeline."),
    ] = None,
    latitude: Annotated[
        float | None,
        Form(description="Incident latitude. If omitted, common CNN JSON keys are used."),
    ] = None,
    longitude: Annotated[
        float | None,
        Form(description="Incident longitude. If omitted, common CNN JSON keys are used."),
    ] = None,
    context_radius_meters: Annotated[
        int,
        Form(
            description="Search radius for OSM, LANDFIRE AOI, and terrain context.",
            gt=0,
            le=50000,
        ),
    ] = 5000,
    include_external_context: Annotated[
        bool,
        Form(description="Include external context in the API response."),
    ] = False,
    gather_external_context: Annotated[
        bool,
        Form(description="Fetch LANDFIRE, OSM, USGS, and OpenWeatherMap context."),
    ] = True,
    include_raw_response: Annotated[
        bool,
        Form(description="Include the raw upstream LLM response for debugging."),
    ] = False,
) -> RescuePlanResponse:
    """Generate a Llama-backed wildfire rescue plan from CNN JSON plus optional image."""

    cnn_output = await _load_json_payload(cnn_payload, cnn_json)
    image_base64, image_mime_type = await _encode_image(image)

    external_context = None
    location = extract_location(cnn_output, latitude, longitude)
    if location is not None:
        location = IncidentLocation(
            latitude=location.latitude,
            longitude=location.longitude,
            radius_meters=context_radius_meters,
        )
    if gather_external_context and location is not None:
        external_context = await ExternalContextClient().gather(location)
    elif gather_external_context:
        external_context = {
            "status": "skipped",
            "detail": "No incident latitude/longitude was provided or found in CNN JSON.",
        }

    try:
        request = RescuePlanRequest(
            cnn_output=cnn_output,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            external_context=external_context,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    client = create_llm_client()
    try:
        plan, raw_response = await client.build_plan(request)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider returned {exc.response.status_code}: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"LLM provider error: {exc}"
        ) from exc

    return RescuePlanResponse(
        model=client.model,
        provider=client.provider,
        plan=plan,
        external_context=external_context if include_external_context else None,
        raw_response=raw_response if include_raw_response else None,
    )
