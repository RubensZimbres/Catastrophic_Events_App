"""Tests for the weather dashboard component.

The previous suite asserted `st.html` was called. That assertion only held
because `weather_dashboard.py` did `import streamlit as components`, so
`components.html(...)` resolved to `st.html()` — the very aliasing bug that
made the component raise TypeError at runtime (st.html takes no `height`).
The tests now assert on what the user actually sees.
"""
import json

import streamlit as st

from components.weather_dashboard import render_weather_dashboard


def _rendered_text():
    """Everything the component wrote, including via st.columns() children.

    st.columns() hands back child mocks, so cols[0].metric(...) never reaches
    st.metric; mock_calls records the full nested call tree instead.
    """
    return " ".join(str(c) for c in st.mock_calls)


class TestRenderWeatherDashboard:
    def setup_method(self):
        st.reset_mock()

    def test_valid_dict_does_not_call_error(self):
        render_weather_dashboard({
            "current_conditions": {
                "temperature": 25.0, "weather": "sunny", "humidity": 60,
                "wind_speed": 5.0, "wind_direction": 180,
                "pressure": 1013, "feels_like": 24.0,
            },
            "forecast": [], "seismic_activity": [],
        })
        st.error.assert_not_called()

    def test_json_string_is_parsed_and_rendered(self):
        render_weather_dashboard(json.dumps({
            "current_conditions": {"temperature": 22.0},
            "forecast": [], "seismic_activity": [],
        }))
        st.error.assert_not_called()
        assert "22.0" in _rendered_text()

    def test_invalid_json_string_shows_an_error(self):
        render_weather_dashboard("not valid json {{{{")
        st.error.assert_called()

    def test_non_dict_integer_shows_an_error(self):
        render_weather_dashboard(42)
        st.error.assert_called()

    def test_non_dict_list_shows_an_error(self):
        render_weather_dashboard([1, 2, 3])
        st.error.assert_called()

    def test_missing_current_conditions_uses_defaults_no_error(self):
        render_weather_dashboard({"forecast": [], "seismic_activity": []})
        st.error.assert_not_called()

    def test_empty_dict_uses_defaults_no_error(self):
        render_weather_dashboard({})
        st.error.assert_not_called()

    def test_does_not_use_st_html(self):
        """st.html() has no `height` parameter, so the old call raised
        TypeError; the React bundle it pointed at is not built or served."""
        render_weather_dashboard({"current_conditions": {"temperature": 20}})
        st.html.assert_not_called()

    def test_seismic_activity_is_shown_to_the_user(self):
        render_weather_dashboard({
            "current_conditions": {}, "forecast": [],
            "seismic_activity": [{"magnitude": 3.5, "location": "5km N of Shinjuku"}],
        })
        st.error.assert_not_called()
        text = _rendered_text()
        assert "3.5" in text and "Shinjuku" in text

    def test_forecast_is_shown_to_the_user(self):
        render_weather_dashboard({
            "current_conditions": {},
            "forecast": [{"datetime": "2024-01-01 12:00:00", "temperature": 23.0,
                          "weather": "light rain"}],
            "seismic_activity": [],
        })
        st.error.assert_not_called()
        text = _rendered_text()
        assert "2024-01-01 12:00:00" in text and "23.0" in text

    def test_no_debug_dump_of_the_raw_payload(self):
        """The component used to st.write() the entire weather payload and its
        Python type to the page as 'Debug -' output."""
        render_weather_dashboard({"current_conditions": {}, "forecast": [],
                                  "seismic_activity": []})
        assert "Debug" not in _rendered_text()
