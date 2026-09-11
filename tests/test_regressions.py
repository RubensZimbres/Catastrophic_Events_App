"""Regression tests for defects found during the code audit.

Most of these survived a 78-test green suite because conftest mocks
`google.genai`, `google.genai.types` and `streamlit` — precisely the boundaries
where the bugs lived. Where that applies, these tests assert against the real
dependency or against the source itself.
"""
import inspect
import re
from xml.etree import ElementTree

import pytest

import app as A
from app import WeatherTool, ExplanationAgent, _coerce_location, _haversine_km

CAP_NS = 'xmlns="urn:oasis:names:tc:emergency:cap:1.2"'
SRC = inspect.getsource(A)


# ---------------------------------------------------------------------------
# eval() on LLM-derived strings was a remote code execution path.
# ---------------------------------------------------------------------------

class TestNoEval:

    def test_eval_is_gone_from_the_tools(self):
        assert "eval(location_dict)" not in SRC.replace("`eval(location_dict)`", "")

    def test_dict_passes_through(self):
        assert _coerce_location({"lat": 1.0, "lng": 2.0}) == {"lat": 1.0, "lng": 2.0}

    def test_json_string_is_parsed(self):
        assert _coerce_location('{"lat": 1.0, "lng": 2.0}') == {"lat": 1.0, "lng": 2.0}

    def test_python_literal_dict_is_parsed(self):
        assert _coerce_location("{'lat': 1.0, 'lng': 2.0}") == {"lat": 1.0, "lng": 2.0}

    @pytest.mark.parametrize("payload", [
        "__import__('os').system('touch /tmp/pwned')",
        "[].__class__.__mro__[1].__subclasses__()",
        "open('/etc/passwd').read()",
    ])
    def test_code_payloads_are_rejected_not_executed(self, payload):
        with pytest.raises(ValueError):
            _coerce_location(payload)


# ---------------------------------------------------------------------------
# The life-safety fields were being dropped before reaching the model.
# ---------------------------------------------------------------------------

class TestHazardFieldsReachTheModel:

    def setup_method(self):
        self.agent = ExplanationAgent.__new__(ExplanationAgent)
        self.agent.generate_content_config = None
        self.agent.client = type("C", (), {"models": type("M", (), {
            "generate_content_stream": staticmethod(lambda **kw: iter(()))})()})()
        A.types.Part.from_text.reset_mock()

    def _prompt_for(self, data):
        self.agent.explain_weather_data(data, "Somewhere")
        calls = A.types.Part.from_text.call_args_list
        args, kwargs = calls[-1][0], calls[-1][1]
        return kwargs.get("text", args[0] if args else "")

    def test_hurricane_instruction_reaches_the_prompt(self):
        """hurricane_info was built twice; the second version read keys the
        fetcher never produces, so headline/description/instruction — the
        evacuation order — were replaced with None and Unknown."""
        prompt = self._prompt_for({"hurricane_alerts": [{
            "event": "Hurricane Warning", "severity": "Extreme",
            "headline": "Hurricane Warning in effect",
            "description": "Catastrophic winds expected",
            "instruction": "EVACUATE IMMEDIATELY to higher ground.",
            "onset": "2026-09-11T06:00", "expires": "2026-09-12T06:00"}]})
        assert "EVACUATE IMMEDIATELY to higher ground." in prompt
        assert "Hurricane Warning in effect" in prompt
        assert "Catastrophic winds expected" in prompt

    def test_tsunami_instruction_reaches_the_prompt(self):
        prompt = self._prompt_for({"tsunami_alerts": [{
            "event": "Tsunami Warning", "severity": "Extreme", "urgency": "Immediate",
            "description": "Tsunami waves expected", "instruction": "MOVE INLAND NOW.",
            "effective": "2026-09-11T05:00", "expires": "2026-09-11T11:00"}]})
        assert "MOVE INLAND NOW." in prompt
        assert "Immediate" in prompt

    def test_no_placeholder_unknowns_leak_into_the_prompt(self):
        prompt = self._prompt_for({"hurricane_alerts": [{
            "event": "Hurricane Warning", "severity": "Extreme",
            "headline": "H", "description": "D", "instruction": "I",
            "onset": "o", "expires": "e"}]})
        hurricane_block = prompt[prompt.index("HURRICANE/CYCLONE ALERTS:"):
                                 prompt.index("FLOOD ALERTS:")]
        assert "Unknown" not in hurricane_block
        assert "None" not in hurricane_block

    def test_flood_alerts_are_rendered_when_present(self):
        prompt = self._prompt_for({"flood_alerts": [{
            "event": "Flash Flood Warning", "severity": "Severe",
            "instruction": "Move to higher ground immediately."}]})
        assert "Flash Flood Warning" in prompt
        assert "Move to higher ground immediately." in prompt


# ---------------------------------------------------------------------------
# Alerts must respect geography.
# ---------------------------------------------------------------------------

class TestAlertsAreLocal:

    def setup_method(self):
        self.tool = WeatherTool()

    def _area(self, inner):
        return ElementTree.fromstring(f"<area {CAP_NS}>{inner}</area>")

    def test_area_without_geometry_does_not_match_everywhere(self):
        area = self._area("<areaDesc>Pacific Coast</areaDesc>")
        for lat, lng in [(61.2, -149.9), (-8.4, 115.2), (47.4, 8.5)]:
            assert self.tool._location_in_alert_area(lat, lng, area) is False

    def test_circle_respects_its_radius(self):
        area = self._area("<areaDesc>A</areaDesc><circle>61.2,-149.9 300</circle>")
        assert self.tool._location_in_alert_area(61.2, -149.9, area) is True
        assert self.tool._location_in_alert_area(64.8, -147.7, area) is False   # ~420 km
        assert self.tool._location_in_alert_area(-8.4, 115.2, area) is False    # Bali

    def test_polygon_respects_its_ring(self):
        area = self._area("<areaDesc>A</areaDesc>"
                          "<polygon>-9.5,113.0 -9.5,117.0 -7.0,117.0 -7.0,113.0 -9.5,113.0</polygon>")
        assert self.tool._location_in_alert_area(-8.4, 115.2, area) is True
        assert self.tool._location_in_alert_area(61.2, -149.9, area) is False

    def test_volcano_query_is_location_scoped(self):
        """The query had no location filter, so every eruption on Earth was
        returned and rendered as local activity."""
        src = inspect.getsource(WeatherTool._get_volcano_data)
        assert "maxradiuskm" in src
        assert '"latitude"' in src and '"longitude"' in src

    def test_haversine_is_sane(self):
        # Anchorage -> Mt Merapi, Java: roughly 11,000 km
        assert 10_500 < _haversine_km(61.2, -149.9, -7.54, 110.44) < 12_000
        assert _haversine_km(0, 0, 0, 0) == 0


# ---------------------------------------------------------------------------
# Flood monitoring was advertised but never implemented.
# ---------------------------------------------------------------------------

class TestFloodAlerts:

    def test_flood_fetcher_exists(self):
        assert hasattr(WeatherTool, "_get_flood_data")

    def test_weather_tool_populates_flood_alerts(self):
        src = inspect.getsource(WeatherTool._run)
        assert 'weather_info["flood_alerts"]' in src

    def test_flood_terms_cover_the_common_nws_events(self):
        terms = WeatherTool.FLOOD_TERMS
        for event in ["Flood Warning", "Flash Flood Warning", "Coastal Flood Advisory"]:
            assert any(t in event.lower() for t in terms), event


# ---------------------------------------------------------------------------
# Outbound calls, secrets and error handling.
# ---------------------------------------------------------------------------

class TestOutboundCalls:

    def test_every_requests_get_sets_a_timeout(self):
        body = SRC[SRC.index("class GeocodingTool"):]
        calls = re.findall(r"requests\.get\((?:[^()]|\([^()]*\))*\)", body, re.S)
        assert calls
        assert all("timeout" in c for c in calls)

    def test_no_api_key_is_concatenated_into_a_url(self):
        assert "key={GOOGLE_MAPS_API_KEY}" not in SRC
        assert "appid={OPENWEATHER_API_KEY}" not in SRC

    def test_scrub_removes_keys_from_exception_text(self):
        text = A._scrub(Exception(
            "401 for url: https://api.openweathermap.org/x?lat=1&appid=SUPERSECRET123"))
        assert "SUPERSECRET123" not in text
        assert "appid=***" in text

    def test_secret_version_defaults_to_latest(self):
        default = inspect.signature(
            A.get_secret.__wrapped__).parameters["version_id"].default
        assert default == "latest"


class TestNoSilentFailure:

    def test_top_level_handler_tells_the_user(self):
        """The outer handler was `except Exception as e: print(e)`, so a failed
        analysis ended the spinner and showed the user nothing at all."""
        main_src = inspect.getsource(A.main)
        assert "print(e)" not in main_src
        assert "st.error" in main_src

    def test_explanation_is_requested_once(self):
        main_src = inspect.getsource(A.main)
        assert main_src.count("explain_weather_data") == 1

    def test_streaming_tolerates_chunks_without_text(self):
        """Safety/metadata chunks carry text=None; `full_response += chunk.text`
        raised TypeError and lost the whole briefing."""
        main_src = inspect.getsource(A.main)
        assert 'getattr(chunk, "text", None)' in main_src

    def test_model_output_is_not_rendered_as_raw_html(self):
        assert "unsafe_allow_html=True" not in inspect.getsource(A.main)


class TestDeploymentConfig:

    def test_dockerfile_copies_resolve_from_one_context(self):
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dockerfile = open(os.path.join(here, "app", "Dockerfile")).read()
        app_dir = os.path.join(here, "app")
        # Check only real directives, not the explanatory comments
        copies = [ln.split()[1] for ln in dockerfile.splitlines()
                  if ln.strip().startswith("COPY ")]
        assert copies
        for src in copies:
            assert os.path.exists(os.path.join(app_dir, src)), (
                f"COPY {src} does not resolve from the Dockerfile's own directory")

    def test_genai_pin_supports_the_configured_model(self):
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reqs = open(os.path.join(here, "app", "requirements.txt")).read()
        assert "google-genai>=1.0" in reqs
        assert "google-genai==0.6.0" not in reqs

    def test_part_from_text_is_called_by_keyword(self):
        assert "Part.from_text(text=" in SRC
        assert "Part.from_text(prompt)" not in SRC
