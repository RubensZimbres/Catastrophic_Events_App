"""Weather dashboard component.

Two defects lived here:

1. `import streamlit as components` aliased the top-level streamlit module, so
   `components.html(..., height=600)` resolved to `st.html()`, which takes no
   `height` argument — the call raised TypeError. The intended module is
   `streamlit.components.v1`.
2. The rendered HTML set `window.weatherData` and created an empty div, but
   nothing ever loaded the React bundle in `frontend/`, so the component was a
   blank 600px iframe.

The React path is not wired into this deployment (the Dockerfile does not build
`frontend/`), so this renders natively in Streamlit instead.
"""
import json
import logging

import streamlit as st
import streamlit.components.v1 as components  # noqa: F401  (kept for the React path)

logger = logging.getLogger(__name__)


def render_weather_dashboard(weather_data):
    """Render the weather dashboard natively in Streamlit."""
    try:
        if isinstance(weather_data, str):
            try:
                weather_data = json.loads(weather_data)
            except json.JSONDecodeError:
                st.error("Weather data could not be read.")
                return

        if not isinstance(weather_data, dict):
            st.error("Weather data is unavailable for this location.")
            return

        current = weather_data.get("current_conditions") or {}
        forecast = weather_data.get("forecast") or []
        seismic = weather_data.get("seismic_activity") or []

        cols = st.columns(4)
        cols[0].metric("Temperature", f"{current.get('temperature', '--')}°C",
                       f"Feels like {current.get('feels_like', '--')}°C")
        cols[1].metric("Humidity", f"{current.get('humidity', '--')}%")
        cols[2].metric("Wind", f"{current.get('wind_speed', '--')} m/s")
        cols[3].metric("Pressure", f"{current.get('pressure', '--')} hPa")

        if current.get("weather"):
            st.caption(f"Conditions: {current['weather']}")

        if forecast:
            st.markdown("**Forecast**")
            for entry in forecast:
                st.write(
                    f"- {entry.get('datetime', 'n/a')}: "
                    f"{entry.get('temperature', '--')}°C, {entry.get('weather', 'n/a')}"
                )

        if seismic:
            st.markdown("**Recent seismic activity**")
            for quake in seismic:
                st.write(f"- M{quake.get('magnitude', '?')} — {quake.get('location', 'unknown')}")

    except Exception:
        # Debug dumps of the full payload used to be written straight to the UI
        logger.exception("Failed to render weather dashboard")
        st.error("The weather dashboard could not be displayed.")
