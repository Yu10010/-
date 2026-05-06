# Wildfire Rescue Planning Bridge

This repository contains a FastAPI service that connects your trained CNN output to a Llama-compatible LLM. The CNN sends a JSON object and an optional image; the service enriches that inference with external wildfire context, forwards everything to the configured LLM, and returns a human-reviewed wildfire rescue support plan.

> Safety note: this service is decision support only. Generated plans must be reviewed by trained emergency responders and incident command before action.

## API flow

1. Your CNN pipeline produces:
   - a JSON object with detections, coordinates, confidence scores, people count, fire/smoke regions, timestamps, weather, or any other metadata you have; and
   - an optional image from the same inference pass.
2. The bridge accepts those artifacts at `POST /wildfire/rescue-plan`.
3. If an incident latitude/longitude is available, the bridge gathers external context from:
   - **LANDFIRE Product Service (LFPS)** for vegetation cover, canopy/fuel layers, and fuel model context;
   - **OpenStreetMap / Overpass** for nearby hospitals, fire stations, roads, water sources, and threatened assets;
   - **USGS National Map EPQS** for elevation and sampled local slope/terrain; and
   - **OpenWeatherMap** for current wind direction, wind speed, humidity, temperature, and weather.
4. The bridge sends the CNN JSON, optional image, and external context to Llama.
5. The response contains a rescue planning draft with priorities, evacuation guidance, resources, communications, hazards, and missing data.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Configure Llama

The default adapter targets Ollama:

```bash
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=llama3.2-vision
```

For an OpenAI-compatible Llama gateway, such as vLLM or a hosted provider:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://your-llama-gateway.example.com
export LLM_MODEL=your-llama-vision-model
export LLM_API_KEY=your-api-key
```

Optional timeout:

```bash
export LLM_TIMEOUT_SECONDS=120
```

## Configure external context APIs

OpenStreetMap Overpass and USGS EPQS work without API keys by default. OpenWeatherMap requires an API key. LANDFIRE LFPS job submission requires an email address; without it, the service prepares the LANDFIRE request payload and includes it in the LLM context as a data gap instead of submitting a job.

```bash
# OpenWeatherMap: wind direction, wind speed, humidity, temperature
export OPENWEATHER_API_KEY=your-openweather-key
export OPENWEATHER_BASE_URL=https://api.openweathermap.org

# OpenStreetMap / Overpass: hospitals, fire stations, roads, water, threatened assets
export OVERPASS_URL=https://overpass-api.de/api/interpreter

# USGS National Map: elevation and sampled terrain/slope
export USGS_EPQS_URL=https://epqs.nationalmap.gov/v1/json
export TERRAIN_SAMPLE_SPACING_METERS=90

# LANDFIRE Product Service: vegetation cover/canopy/fuel layers
export LANDFIRE_BASE_URL=https://lfps.usgs.gov
export LANDFIRE_EMAIL=your-email@example.com
export LANDFIRE_LAYERS='EVT;EVC;FBFM40;CC;CH'
export LANDFIRE_RESAMPLE_RESOLUTION=90
export LANDFIRE_AUTO_SUBMIT=false

# Shared timeout for external context calls
export CONTEXT_TIMEOUT_SECONDS=20
```

Set `LANDFIRE_AUTO_SUBMIT=true` only when you want the service to submit asynchronous LFPS jobs automatically for each request.

## Request examples

Send JSON as a form field with explicit coordinates:

```bash
curl -X POST http://localhost:8000/wildfire/rescue-plan \
  -F 'cnn_payload={"people_detected":3,"smoke":"heavy","fire_front":"northwest","confidence":0.91}' \
  -F 'latitude=34.3917' \
  -F 'longitude=-118.5426' \
  -F 'context_radius_meters=5000' \
  -F 'image=@frame.jpg'
```

Send JSON as a file and ask the bridge to return the external context it included in the LLM prompt:

```bash
curl -X POST http://localhost:8000/wildfire/rescue-plan \
  -F 'cnn_json=@cnn_output.json;type=application/json' \
  -F 'image=@frame.jpg' \
  -F 'include_external_context=true'
```

If the CNN JSON already contains common coordinate keys such as `latitude`/`longitude`, `lat`/`lon`, `center_lat`/`center_lon`, or a `location` object, you can omit the coordinate form fields.

Add `-F 'gather_external_context=false'` to skip LANDFIRE, OSM, USGS, and OpenWeatherMap lookups. Add `-F 'include_raw_response=true'` when debugging the upstream LLM payload.

## Response shape

```json
{
  "model": "llama3.2-vision",
  "provider": "ollama",
  "plan": "...generated rescue support plan...",
  "external_context": null,
  "raw_response": null
}
```

## Endpoint summary

- `GET /health` returns service status.
- `POST /wildfire/rescue-plan` accepts one CNN JSON payload (`cnn_payload` or `cnn_json`), an optional `image` upload, optional coordinates, and external context controls.
