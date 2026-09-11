# Catastrophic Events App

A Gemini-powered multi-agent application for real-time monitoring and analysis of extreme weather events — hurricanes, earthquakes, tsunamis, volcanic activity, and floods — for any location worldwide.

---

## Features

### Multi-Agent Architecture
The app uses two cooperating AI agents:

- **DisasterAdvisorAgent** — Owns the three data tools (geocoding, weather/hazards, emergency resources). It also builds a LangChain `ZERO_SHOT_REACT_DESCRIPTION` agent with conversation memory for free-form queries, but note that **the UI never reaches that path**: it always sends a prompt containing "climate threats", which takes a deterministic branch that calls the three tools directly. The ReAct agent is reachable only via `get_response()` with a different query.
- **ExplanationAgent** — Connects to Gemini (`gemini-3.7-flash`) via Vertex AI and produces a structured markdown analysis of all collected data, including actionable safety recommendations.

### Data Sources & Tools

| Tool | API | Data |
|---|---|---|
| Geocoding | Google Maps Geocoding API | Converts any address to lat/lng coordinates |
| Weather | OpenWeatherMap API | Current conditions + 5-day forecast |
| Seismic Activity | USGS Earthquake API | Earthquakes M2.5+ within 300 km, last 7 days |
| Tsunami Alerts | NOAA Tsunami Warning Center (CAP/XML feed) | Active tsunami warnings whose CAP `<circle>`/`<polygon>` geometry actually covers the location |
| Hurricane/Tropical Storm Alerts | National Weather Service Alerts API | Active tropical storm and hurricane warnings |
| Flood Alerts | National Weather Service Alerts API | Active flood, flash flood and coastal flood warnings |
| Volcanic Activity | USGS Seismic API (volcanic eruption filter) | Volcanic eruptions in the last 10 days within 500 km |
| Emergency Resources | Google Maps Places API | Nearby hospitals, police stations, fire stations, and emergency shelters within 5 km |

### Dashboard & UI

Built with **Streamlit**, the interface displays:

- **Current conditions**: temperature, feels-like, humidity, wind speed/direction, pressure, visibility
- **5-day forecast**: expandable entries per timestamp with temperature, humidity, wind, and conditions
- **Seismic activity**: magnitude, location, timestamp, and link to USGS detail page
- **Active alerts**: hurricane and tsunami warning panels with severity, headline, and instructions
- **Emergency resources**: tabbed view (Hospital / Police / Fire Station / Shelter) with address and interactive map per facility
- **AI analysis**: streaming Gemini response covering all hazards with a structured markdown report including an emergency status summary and safety recommendations

### Secrets Management

API keys are stored in **Google Cloud Secret Manager** and fetched at runtime — no credentials are stored in code or environment files.

---

## Architecture

```
app/
├── app.py                        # Main Streamlit app, agents, and tools
├── components/
│   └── weather_dashboard.py      # Weather dashboard Streamlit component
├── frontend/
│   └── src/components/
│       └── weather_dashboard.jsx # React weather dashboard component
├── Dockerfile
├── requirements.txt
└── .env                          # Local environment variables (not committed)
```

**AI model**: `gemini-3.7-flash` via Vertex AI (`us-central1`)

---

## Coverage caveat

The tsunami feed is NOAA's **PAAQ** (National Tsunami Warning Center), which covers
the US and Canadian Pacific and Atlantic coasts. The NWS alerts feed used for
hurricane and flood warnings is **US-only**. Geocoding, weather and seismic data are
global, so a non-US address returns real weather and earthquake data but will never
show a hurricane, flood or tsunami alert — the feeds simply do not cover it.

Previously the tsunami area check returned `True` for every alert regardless of
location, so users anywhere in the world were shown Alaskan warnings as local.
Alerts are now matched against the CAP area geometry, which means **absence of an
alert outside the US reflects feed coverage, not absence of risk.**

---

## Prerequisites

- Python 3.9+
- A Google Cloud project with the following APIs enabled:
  - Vertex AI
  - Secret Manager
  - Maps Geocoding API
  - Maps Places API
- Secrets stored in Secret Manager: `OPENWEATHER_API_KEY`, `GOOGLE_MAPS_API_KEY`, `GOOGLE_AI_API_KEY`
  (fetched from the `latest` version, so rotating a key takes effect on the next restart)
- Application Default Credentials configured (`gcloud auth application-default login`)

---

## Environment Variables

Create an `app/.env` file for local development (already in `.gitignore`):

```
GCP_PROJECT_ID=<your-numeric-project-id>
GCP_PROJECT_NAME=<your-project-name>
```

---

## Local Development

```bash
# 1. Create and activate a virtual environment
python3 -m venv astros-env
. astros-env/bin/activate

# 2. Install dependencies
pip install -r app/requirements.txt

# 3. Authenticate with Google Cloud
gcloud auth application-default login
gcloud config set project <your-project-name>

# 4. Run the app
python3 -m streamlit run app/app.py
```

---

## Cloud Deployment (Google Cloud Run)

### 1. Set environment variables in Cloud Shell

```bash
export GCP_PROJECT='<your-project-name>'
export GCP_REGION='us-central1'
export AR_REPO='repo-weather'
export SERVICE_NAME='weather-app'
```

### 2. Build and push the Docker image to Artifact Registry

The Dockerfile's build context is the `app/` directory (where it lives):

```bash
cd app
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy "$SERVICE_NAME" \
  --port=8080 \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --platform=managed \
  --region=$GCP_REGION \
  --project=$GCP_PROJECT \
  --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION \
  --min-instances 0 --max-instances 3 --cpu 4 --memory 8192Mi --concurrency 40
```

---

## Dependencies

Key packages (see `requirements.txt` for full list):

- `streamlit` — UI framework
- `langchain`, `langchain-google-genai` — agent orchestration
- `google-genai` — Vertex AI / Gemini client
- `google-cloud-secret-manager` — secrets at runtime
- `python-dotenv` — local `.env` support
- `requests`, `pandas` — data fetching and processing

---

## Running the tests

```bash
pip install pytest pytest-mock
pytest -q
```

112 tests, no network access and no GCP credentials required — `tests/conftest.py`
injects mocks for Streamlit, `google.genai` and Secret Manager before importing
the app.

> `tests/test_regressions.py` deliberately asserts against the real SDK and
> against the source itself. The mocks in `conftest.py` cover exactly the
> boundaries where several shipped bugs lived, so a mock-only suite could not
> have caught them.

---

## Known limitations

- **Emergency resources come from Google Places** with a fixed 5 km radius and no
  distance sorting, so the nearest hospital is not necessarily listed first.
- **The React dashboard in `frontend/` is not built or served.** `components/
  weather_dashboard.py` renders natively in Streamlit instead; the JSX and
  `package.json` are unwired.
- **No caching.** Every click re-runs geocoding, two OpenWeather calls, USGS,
  NWS and four Places lookups. A `st.cache_data` layer keyed by address would
  cut both latency and API spend substantially.
- **Secrets are fetched at import time**, so the module cannot be imported
  without either GCP credentials or the matching environment variables.

---

## Example Locations to Try

- `Datah Village in Bali`
- `6Q9M+CCC, Pelangai, Ranah Pesisir, South Pesisir Regency, West Sumatra 25666, Indonesia`
- Any street address worldwide
