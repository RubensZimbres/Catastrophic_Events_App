# Run Instructions — Catastrophic Events App

A Gemini-powered multi-agent app for monitoring extreme weather and geophysical
hazards — hurricanes, earthquakes, tsunamis, volcanic activity and floods — for
a given address.

---

## 1. Prerequisites

- **Python 3.11+** (the Dockerfile uses `python:3.11-slim`)
- A **Google Cloud project** with these APIs enabled:
  - Vertex AI
  - Secret Manager
  - Geocoding API
  - Places API
- Application Default Credentials:

  ```bash
  gcloud auth application-default login
  gcloud config set project YOUR-PROJECT
  ```

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt
```

---

## 2. Secrets and configuration

API keys come from **Google Cloud Secret Manager** at runtime, with an
environment-variable fallback if Secret Manager is unreachable.

Secrets to create:

| Secret ID | Purpose |
|---|---|
| `OPENWEATHER_API_KEY` | Current conditions and 5-day forecast |
| `GOOGLE_MAPS_API_KEY` | Geocoding and Places (emergency resources) |
| `GOOGLE_AI_API_KEY` | Only needed for the LangChain agent path, see §6 |

Then create `app/.env` from the template:

```bash
cp app/.env.example app/.env
```

```
GCP_PROJECT_ID=<numeric project id, used for Secret Manager>
GCP_PROJECT_NAME=<project id, used for Vertex AI>
```

Secrets are read from the `latest` version, so rotating one takes effect on the
next restart.

---

## 3. Run it

```bash
python3 -m streamlit run app/app.py
```

Opens on <http://localhost:8501>. Enter any address; the pipeline geocodes it,
then fetches weather, seismic, tsunami, hurricane and flood data before asking
Gemini for a briefing.

---

## 4. Tests

```bash
pytest -q
```

**112 tests**, no network and no GCP credentials required — `tests/conftest.py`
mocks Streamlit, `google.genai` and Secret Manager before importing the app.

```bash
pytest tests/test_app.py -q                 # tools, fetchers, prompt assembly
pytest tests/test_weather_dashboard.py -q   # dashboard rendering
pytest tests/test_regressions.py -q         # audited defects
```

### Linting

```bash
flake8 app tests --select=E9,F63,F7,F82   # must be clean
flake8 app tests --max-line-length=120 --exit-zero
```

---

## 5. Docker

The Dockerfile's build context is the **`app/` directory**, where it lives:

```bash
cd app
docker build -t xtreme-weather .
docker run --rm -p 8080:8080 \
  -e GCP_PROJECT_ID=your-numeric-id \
  -e GCP_PROJECT_NAME=your-project \
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds/sa.json \
  -v /path/to/service-account.json:/creds/sa.json:ro \
  xtreme-weather
```

The image runs as a non-root `appuser` and healthchecks `/_stcore/health`.
`.dockerignore` excludes `pics/`, `frontend/` and `.env` — the screenshots are
documentation and the React component is not served by this image (§6).

---

## 6. Architecture

```
app/app.py
  ├── GeocodingTool            Google Maps Geocoding
  ├── WeatherTool              OpenWeatherMap current + forecast
  │     ├── _get_earthquake_data   USGS, 300 km radius
  │     ├── _get_hurricane_data    NWS active alerts, filtered
  │     ├── _get_flood_data        NWS active alerts, filtered
  │     ├── _get_tsunami_data      NOAA PAAQ CAP feed
  │     └── _get_volcano_data      USGS volcanic events, 500 km radius
  ├── EmergencyResourcesTool   Google Places: hospitals, police, fire, shelters
  └── ExplanationAgent         Vertex AI (gemini-3.7-flash), streams the briefing
```

`DisasterAdvisorAgent` also builds a LangChain `ZERO_SHOT_REACT_DESCRIPTION`
agent, but **the UI never reaches it**: the prompt it sends always contains
"climate threats", which takes a deterministic branch calling the three tools
directly. The ReAct agent is reachable only via `get_response()` with a
different query, and is the only thing that needs `GOOGLE_AI_API_KEY`.

`app/frontend/` holds a React dashboard component that is **not built or
served**. `app/components/weather_dashboard.py` renders natively in Streamlit
instead.

---

## 7. Coverage — read this before trusting an "all clear"

| Hazard | Source | Coverage |
|---|---|---|
| Weather, forecast | OpenWeatherMap | Global |
| Earthquakes | USGS | Global |
| Volcanic activity | USGS | Global, filtered to 500 km |
| **Tsunami** | NOAA **PAAQ** (National Tsunami Warning Center) | **US + Canada Pacific/Atlantic coasts only** |
| **Hurricane / tropical storm** | NWS active alerts | **US only** |
| **Flood** | NWS active alerts | **US only** |

**For a non-US address, the tsunami, hurricane and flood sections will always be
empty — that reflects feed coverage, not absence of risk.** The example
addresses in the README are in Indonesia, which none of those three feeds cover.

Tsunami alerts are matched against the CAP area geometry (`<circle>` via
haversine, `<polygon>` via ray casting). An alert whose area carries no usable
geometry is **not** reported, rather than being assumed local.

---

## 8. Deploy to Cloud Run

```bash
export GCP_PROJECT='your-project'
export GCP_REGION='us-central1'
export AR_REPO='repo-weather'
export SERVICE_NAME='weather-app'

cd app
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"

gcloud run deploy "$SERVICE_NAME" \
  --port=8080 \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --platform=managed \
  --region=$GCP_REGION \
  --set-env-vars=GCP_PROJECT_ID=$GCP_PROJECT,GCP_PROJECT_NAME=$GCP_PROJECT \
  --min-instances 0 --max-instances 3 --cpu 4 --memory 8192Mi --concurrency 40
```

`cd app` matters — the Dockerfile's context is that directory.

---

## 9. Troubleshooting

| Symptom | Cause |
|---|---|
| "We could not complete the analysis for that address" | Check the server log; detail is deliberately not shown in the browser |
| Tsunami / hurricane / flood always empty | Expected for a non-US address, see §7 |
| Every hazard section empty for a US address | Geocoding likely failed — verify the address resolves |
| Sunrise/sunset look wrong | They are rendered in the *location's* timezone from OpenWeather's `timezone` offset, not the server's |
| `Secret [...] not found` at startup | `GCP_PROJECT_ID` unset or the secret is absent; the code then falls back to environment variables |
| Blank dashboard panels | A fetcher returned nothing; each has a timeout and logs its own failure |

---

## 10. Before you change anything

- **Every outbound call has a 10 s timeout** (`HTTP_TIMEOUT`). Keep it that way:
  one unresponsive upstream previously pinned a Streamlit worker indefinitely.
- **Never pass a location payload through `eval()`.** Use `_coerce_location()`,
  which accepts JSON or a Python literal via `ast.literal_eval`. The string
  reaching it is produced by an LLM from user-supplied address text.
- **Hazard dictionaries are rendered by key name.** `_block()` in
  `explain_weather_data` reads the keys the fetchers actually produce; inventing
  a key silently renders nothing, which previously dropped the evacuation
  `instruction` text from the model's briefing entirely.
- **Nothing is cached.** Every click re-runs geocoding, two OpenWeather calls,
  USGS, NWS and four Places lookups. A `st.cache_data` layer keyed by address
  would cut both latency and API spend.
