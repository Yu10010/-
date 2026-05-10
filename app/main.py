"""FastAPI app that bridges CNN wildfire outputs to a Llama planning model."""

from __future__ import annotations

import base64
import json
import os
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.cnn_inference import predict_wildfire
from app.context_clients import ExternalContextClient, IncidentLocation, extract_location
from app.llm_client import create_llm_client
from app.schemas import RescuePlanRequest, RescuePlanResponse


app = FastAPI(
    title="Wildfire Rescue Planning Bridge",
    description=(
        "Accepts image inputs for a weighted CNN ensemble and forwards the "
        "structured result plus external context to an Ollama-compatible LLM."
    ),
    version="0.3.0",
)


CNN_HIGH_CONFIDENCE_THRESHOLD = float(
    os.getenv("CNN_HIGH_CONFIDENCE_THRESHOLD", "0.85")
)

LLM_VISUAL_WEIGHT = float(
    os.getenv("LLM_VISUAL_WEIGHT", "0.25")
)

FINAL_MIN_CONFIDENCE_THRESHOLD = float(
    os.getenv("FINAL_MIN_CONFIDENCE_THRESHOLD", "0.65")
)


async def _load_json_payload(
    cnn_payload: str | None,
    cnn_json: UploadFile | None,
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
            status_code=400,
            detail="CNN JSON file must be UTF-8.",
        ) from exc

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CNN JSON: {exc.msg}",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="CNN JSON must be an object.")

    return parsed


async def _read_image_upload(
    image: UploadFile | None,
) -> tuple[bytes | None, str | None]:
    if image is None:
        return None, None

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    return content, image.content_type


def _encode_image_bytes(image_bytes: bytes | None) -> str | None:
    if image_bytes is None:
        return None

    return base64.b64encode(image_bytes).decode("ascii")


def _normalize_class_name(class_name: str) -> str:
    value = class_name.lower().strip()
    if value in {"no_fire", "no fire", "nonfire", "not_fire"}:
        return "nofire"
    return value


def _should_use_llm_vision(cnn_output: dict[str, Any]) -> bool:
    try:
        confidence = float(cnn_output.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return confidence < CNN_HIGH_CONFIDENCE_THRESHOLD


def _get_final_confidence(cnn_output: dict[str, Any]) -> float:
    try:
        return float(cnn_output.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _build_human_assistance_plan(cnn_output: dict[str, Any]) -> str:
    confidence = _get_final_confidence(cnn_output)
    prediction = cnn_output.get("prediction", "unknown")

    return (
        "Human assistance required.\n\n"
        "The automated wildfire classification confidence is below the minimum "
        "threshold required for reliable rescue-plan generation.\n\n"
        f"Final prediction: {prediction}\n"
        f"Final confidence: {confidence:.4f}\n"
        f"Required minimum confidence: {FINAL_MIN_CONFIDENCE_THRESHOLD:.4f}\n\n"
        "Recommended action:\n"
        "1. Send the normal image, heat map, CNN output, and available external "
        "context to a human incident commander or trained reviewer.\n"
        "2. Do not rely on the automated rescue plan as the primary decision source.\n"
        "3. Collect additional evidence if possible, such as a clearer image, "
        "a second camera angle, updated weather data, or field confirmation.\n"
    )


def _combine_cnn_and_llm_confidence(
    cnn_output: dict[str, Any],
    llm_visual_assessment: dict[str, Any],
) -> dict[str, Any]:
    cnn_probs_raw = cnn_output.get("class_probabilities", {})
    llm_probs_raw = llm_visual_assessment.get("class_probabilities", {})

    cnn_probs = {
        _normalize_class_name(str(class_name)): float(probability)
        for class_name, probability in cnn_probs_raw.items()
    }

    llm_probs = {
        _normalize_class_name(str(class_name)): float(probability)
        for class_name, probability in llm_probs_raw.items()
    }

    all_classes = sorted(set(cnn_probs) | set(llm_probs) | {"fire", "nofire"})

    llm_weight = max(0.0, min(1.0, LLM_VISUAL_WEIGHT))
    cnn_weight = 1.0 - llm_weight

    combined_probs: dict[str, float] = {}

    for class_name in all_classes:
        combined_probs[class_name] = (
            cnn_weight * cnn_probs.get(class_name, 0.0)
            + llm_weight * llm_probs.get(class_name, 0.0)
        )

    total = sum(combined_probs.values())
    if total > 0:
        combined_probs = {
            class_name: probability / total
            for class_name, probability in combined_probs.items()
        }
    else:
        combined_probs = {"fire": 0.5, "nofire": 0.5}

    final_prediction = max(combined_probs, key=combined_probs.get)
    final_confidence = combined_probs[final_prediction]

    updated_output = dict(cnn_output)

    updated_output["cnn_only_result"] = {
        "prediction": cnn_output.get("prediction"),
        "confidence": cnn_output.get("confidence"),
        "class_probabilities": cnn_output.get("class_probabilities"),
    }

    updated_output["llm_visual_used"] = True
    updated_output["llm_visual_assessment"] = llm_visual_assessment

    updated_output["confidence_fusion"] = {
        "method": "weighted_average",
        "cnn_weight": cnn_weight,
        "llm_visual_weight": llm_weight,
        "reason": (
            f"CNN confidence was below threshold "
            f"{CNN_HIGH_CONFIDENCE_THRESHOLD}."
        ),
    }

    updated_output["prediction"] = final_prediction
    updated_output["confidence"] = final_confidence
    updated_output["class_probabilities"] = combined_probs

    return updated_output

def _compact_external_context(
    external_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Create a shorter context object for the LLM prompt.

    The full external_context can be returned to the API caller, but the LLM
    should receive a compact version so it does not waste context window on
    dozens of raw OSM features.
    """

    if external_context is None:
        return None

    compact: dict[str, Any] = {}

    incident_location = external_context.get("incident_location")
    if incident_location is not None:
        compact["incident_location"] = incident_location

    openweather = external_context.get("openweather")
    if isinstance(openweather, dict):
        compact["openweather"] = openweather

    usgs = external_context.get("usgs_national_map")
    if isinstance(usgs, dict):
        compact["usgs_national_map"] = usgs

    osm = external_context.get("openstreetmap")
    if isinstance(osm, dict):
        if osm.get("status") != "ok":
            compact["openstreetmap"] = osm
        else:
            categories = osm.get("categories", {})

            compact["openstreetmap"] = {
                "status": "ok",
                "radius_meters": osm.get("radius_meters"),
                "feature_count": osm.get("feature_count"),
                "summary": {
                    "fire_station_count": len(categories.get("fire_stations", [])),
                    "hospital_count": len(categories.get("hospitals", [])),
                    "road_count": len(categories.get("roads", [])),
                    "water_source_count": len(categories.get("water_sources", [])),
                    "threatened_asset_count": len(categories.get("threatened_assets", [])),
                },
                "nearest_fire_stations": categories.get("fire_stations", [])[:3],
                "main_roads": categories.get("roads", [])[:8],
                "water_sources": categories.get("water_sources", [])[:5],
                "threatened_assets": categories.get("threatened_assets", [])[:5],
            }

    return compact

@app.get("/health")
def health() -> dict[str, str]:
    """Return service health for container and load-balancer checks."""

    return {"status": "ok"}


@app.post("/wildfire/classify")
async def wildfire_classify(
    image: Annotated[
        UploadFile,
        File(description="Normal image for the resnet18_i CNN model."),
    ],
    heatmap: Annotated[
        UploadFile,
        File(description="Heat map image for the resnet18 CNN model."),
    ],
) -> dict[str, Any]:
    """Classify wildfire risk using the local two-input CNN ensemble."""

    image_bytes, _ = await _read_image_upload(image)
    heatmap_bytes, _ = await _read_image_upload(heatmap)

    if image_bytes is None:
        raise HTTPException(status_code=400, detail="Normal image is required.")

    if heatmap_bytes is None:
        raise HTTPException(status_code=400, detail="Heat map image is required.")

    return predict_wildfire(image_bytes, heatmap_bytes)


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
        File(description="Normal image for resnet18_i and optional LLM visual check."),
    ] = None,
    heatmap: Annotated[
        UploadFile | None,
        File(description="Heat map image for the resnet18 CNN model."),
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
            description="Search radius for OSM, weather, and terrain context.",
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
        Form(description="Fetch OSM, USGS, and OpenWeatherMap context."),
    ] = True,
    include_raw_response: Annotated[
        bool,
        Form(description="Include raw upstream LLM responses for debugging."),
    ] = False,
) -> RescuePlanResponse:
    """Generate a wildfire rescue plan from CNN ensemble output and optional context."""

    image_bytes, image_mime_type = await _read_image_upload(image)
    heatmap_bytes, _ = await _read_image_upload(heatmap)

    if cnn_payload is None and cnn_json is None:
        if image_bytes is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Provide cnn_payload/cnn_json or upload a normal image "
                    "for CNN inference."
                ),
            )

        if heatmap_bytes is None:
            raise HTTPException(
                status_code=400,
                detail="Heat map image is required for CNN ensemble inference.",
            )

        cnn_output = predict_wildfire(image_bytes, heatmap_bytes)
    else:
        cnn_output = await _load_json_payload(cnn_payload, cnn_json)

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

    client = create_llm_client()

    image_base64_for_vision = _encode_image_bytes(image_bytes)
    llm_visual_raw_response = None

    if (
        image_base64_for_vision is not None
        and _should_use_llm_vision(cnn_output)
    ):
        try:
            llm_visual_assessment, llm_visual_raw_response = (
                await client.assess_image_fire_probability(
                    image_base64=image_base64_for_vision,
                    image_mime_type=image_mime_type,
                    cnn_output=cnn_output,
                )
            )

            cnn_output = _combine_cnn_and_llm_confidence(
                cnn_output,
                llm_visual_assessment,
            )

        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "LLM visual verifier returned "
                    f"{exc.response.status_code}: {exc.response.text}"
                ),
            ) from exc

        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"LLM visual verifier error: {type(exc).__name__}: {repr(exc)}",
            ) from exc

        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"LLM visual verifier returned invalid JSON: {exc}",
            ) from exc

    else:
        cnn_output["llm_visual_used"] = False

        if image_base64_for_vision is None:
            cnn_output["llm_visual_skip_reason"] = (
                "No normal image was provided for LLM visual verification."
            )
        else:
            cnn_output["llm_visual_skip_reason"] = (
                f"CNN confidence was high enough: "
                f"{cnn_output.get('confidence')} >= {CNN_HIGH_CONFIDENCE_THRESHOLD}"
            )

    final_confidence = _get_final_confidence(cnn_output)

    if final_confidence < FINAL_MIN_CONFIDENCE_THRESHOLD:
        cnn_output["human_assistance_required"] = True
        cnn_output["human_assistance_reason"] = (
            f"Final confidence {final_confidence:.4f} is below "
            f"minimum threshold {FINAL_MIN_CONFIDENCE_THRESHOLD:.4f}."
        )

        raw_response = {
            "llm_visual_response": llm_visual_raw_response,
            "plan_response": None,
            "manual_review": {
                "required": True,
                "final_confidence": final_confidence,
                "minimum_required_confidence": FINAL_MIN_CONFIDENCE_THRESHOLD,
                "reason": cnn_output["human_assistance_reason"],
            },
        }

        return RescuePlanResponse(
            model=client.model,
            provider=client.provider,
            plan=_build_human_assistance_plan(cnn_output),
            external_context=external_context if include_external_context else None,
            raw_response=raw_response if include_raw_response else None,
        )

    # Important:
    # The final planning prompt does not include the image.
    # The image is used only for low-confidence visual verification above.
    image_base64_for_plan = None

    compact_external_context_for_llm = _compact_external_context(external_context)

    try:
        request = RescuePlanRequest(
            cnn_output=cnn_output,
            image_base64=image_base64_for_plan,
            image_mime_type=None,
            external_context=compact_external_context_for_llm,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    try:
        plan, plan_raw_response = await client.build_plan(request)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider returned {exc.response.status_code}: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error: {type(exc).__name__}: {repr(exc)}",
        ) from exc

    raw_response = {
        "llm_visual_response": llm_visual_raw_response,
        "plan_response": plan_raw_response,
    }

    return RescuePlanResponse(
        model=client.model,
        provider=client.provider,
        plan=plan,
        external_context=external_context if include_external_context else None,
        raw_response=raw_response if include_raw_response else None,
    )