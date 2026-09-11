import pytest
from unittest.mock import MagicMock, patch

import app
from app import (
    get_secret,
    get_secret_or_env,
    LocationInfo,
    WeatherInfo,
    GeocodingTool,
    WeatherTool,
    EmergencyResourcesTool,
    ExplanationAgent,
)

LOCATION_DICT = {"lat": 35.6762, "lng": 139.6503}


def _mock_response(status_code=200, json_data=None, content=None):
    r = MagicMock()
    r.status_code = status_code
    if json_data is not None:
        r.json.return_value = json_data
    if content is not None:
        r.content = content
    return r


class TestSecretManagement:
    def setup_method(self):
        # get_secret / _secret_client are lru_cached so results survive between
        # calls; clear them so each test exercises a real lookup.
        import app as _app
        _app.get_secret.cache_clear()
        _app._secret_client.cache_clear()

    def test_get_secret_raises_when_client_call_fails(self):
        with patch('app._secret_client') as mock_client:
            mock_client.return_value.access_secret_version.side_effect = Exception("Auth error")
            with pytest.raises(Exception, match="Auth error"):
                get_secret("proj", "FAILING_KEY")

    def test_get_secret_or_env_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_KEY", "env_value_abc")
        with patch('app.get_secret', side_effect=Exception("No creds")):
            result = get_secret_or_env("proj", "FALLBACK_KEY")
        assert result == "env_value_abc"

    def test_get_secret_or_env_returns_empty_string_when_unset(self, monkeypatch):
        monkeypatch.delenv("TOTALLY_ABSENT_KEY", raising=False)
        with patch('app.get_secret', side_effect=Exception("No creds")):
            result = get_secret_or_env("proj", "TOTALLY_ABSENT_KEY")
        assert result == ""

    def test_get_secret_returns_decoded_payload(self):
        with patch('app._secret_client') as mock_client:
            mock_client.return_value.access_secret_version.return_value.payload.data = b"secret123"
            result = get_secret("proj", "DECODE_KEY")
        assert result == "secret123"

    def test_get_secret_defaults_to_latest_version(self):
        """Pinning to version "1" meant rotating a leaked key had no effect."""
        import inspect
        import app as _app
        default = inspect.signature(_app.get_secret.__wrapped__).parameters['version_id'].default
        assert default == "latest"


class TestDataModels:
    def test_location_info_stores_fields(self):
        loc = LocationInfo(address="Tokyo, Japan", lat=35.6762, lng=139.6503)
        assert loc.address == "Tokyo, Japan"
        assert loc.lat == 35.6762
        assert loc.lng == 139.6503

    def test_location_info_requires_address(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LocationInfo(lat=35.6762, lng=139.6503)

    def test_location_info_requires_lat(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LocationInfo(address="Tokyo", lng=139.6503)

    def test_weather_info_stores_all_fields(self):
        info = WeatherInfo(
            current_conditions={"temp": 22.5},
            forecast=[{"day": "Mon", "temp": 23.0}],
            alerts=[{"type": "flood"}],
        )
        assert info.current_conditions == {"temp": 22.5}
        assert len(info.forecast) == 1
        assert info.alerts[0]["type"] == "flood"

    def test_weather_info_accepts_empty_collections(self):
        info = WeatherInfo(current_conditions={}, forecast=[], alerts=[])
        assert info.forecast == []
        assert info.alerts == []


class TestGeocodingTool:
    def setup_method(self):
        self.tool = GeocodingTool()

    def test_tool_name(self):
        assert self.tool.name == "geocoding_tool"

    def test_tool_description_mentions_coordinates(self):
        desc = self.tool.description.lower()
        assert "latitude" in desc or "coordinates" in desc

    @patch('app.requests.get')
    def test_run_success_returns_address_lat_lng(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "status": "OK",
            "results": [{
                "formatted_address": "Tokyo, Japan",
                "geometry": {"location": {"lat": 35.6762, "lng": 139.6503}},
            }],
        })
        result = self.tool._run("Tokyo, Japan")
        assert result["address"] == "Tokyo, Japan"
        assert result["lat"] == 35.6762
        assert result["lng"] == 139.6503

    @patch('app.requests.get')
    def test_run_zero_results_raises_exception(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"status": "ZERO_RESULTS", "results": []})
        with pytest.raises(Exception, match="Geocoding failed: ZERO_RESULTS"):
            self.tool._run("NonExistentXYZ")

    @patch('app.requests.get')
    def test_run_request_denied_raises_exception(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"status": "REQUEST_DENIED", "results": []})
        with pytest.raises(Exception, match="Geocoding failed: REQUEST_DENIED"):
            self.tool._run("Tokyo")

    @patch('app.requests.get')
    def test_run_passes_api_key_as_a_parameter_not_in_the_url(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "status": "OK",
            "results": [{"formatted_address": "Tokyo", "geometry": {"location": {"lat": 35.6, "lng": 139.6}}}],
        })
        self.tool._run("Tokyo")
        called_url = mock_get.call_args[0][0]
        params = mock_get.call_args[1]["params"]
        # The key must not be embedded in the URL string: requests copies the
        # full URL into HTTPError messages, which are logged and shown.
        assert "key=" not in called_url
        assert params["key"]
        assert params["address"] == "Tokyo"

    @patch('app.requests.get')
    def test_run_sets_a_timeout(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "status": "OK",
            "results": [{"formatted_address": "Tokyo", "geometry": {"location": {"lat": 35.6, "lng": 139.6}}}],
        })
        self.tool._run("Tokyo")
        assert mock_get.call_args[1].get("timeout")

    @patch('app.requests.get')
    def test_address_with_ampersand_is_encoded_not_injected(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "status": "OK",
            "results": [{"formatted_address": "X", "geometry": {"location": {"lat": 1, "lng": 2}}}],
        })
        self.tool._run("Tom & Jerry St&key=attacker")
        params = mock_get.call_args[1]["params"]
        # The whole string stays one parameter value instead of splicing in
        # extra query parameters
        assert params["address"] == "Tom & Jerry St&key=attacker"

    def test_arun_raises_not_implemented(self):
        import asyncio
        with pytest.raises(NotImplementedError):
            asyncio.run(self.tool._arun("Tokyo"))


class TestWeatherToolGetWeatherData:
    def setup_method(self):
        self.tool = WeatherTool()
        self.current_data = {
            "main": {"temp": 22.5, "humidity": 65, "pressure": 1013, "feels_like": 21.0},
            "weather": [{"description": "clear sky"}],
            "wind": {"speed": 5.2, "deg": 180},
            "visibility": 10000,
        }
        self.forecast_data = {
            "list": [{
                "dt_txt": "2024-01-01 12:00:00",
                "main": {"temp": 23.0, "humidity": 60, "pressure": 1012, "feels_like": 22.0},
                "weather": [{"description": "partly cloudy"}],
                "wind": {"speed": 4.0, "deg": 200},
            }]
        }

    @patch('app.requests.get')
    def test_returns_structured_weather_dict(self, mock_get):
        mock_get.side_effect = [
            _mock_response(json_data=self.current_data),
            _mock_response(json_data=self.forecast_data),
        ]
        result = self.tool._get_weather_data(LOCATION_DICT)
        assert result["current_conditions"]["temperature"] == 22.5
        assert result["current_conditions"]["humidity"] == 65
        assert result["current_conditions"]["weather"] == "clear sky"
        assert result["current_conditions"]["wind_speed"] == 5.2
        assert len(result["forecast"]) == 1
        assert result["forecast"][0]["datetime"] == "2024-01-01 12:00:00"
        assert result["alerts"] == []

    @patch('app.requests.get')
    def test_raises_on_current_weather_api_failure(self, mock_get):
        mock_get.return_value = _mock_response(status_code=401)
        with pytest.raises(Exception, match="Current weather API failed with status 401"):
            self.tool._get_weather_data(LOCATION_DICT)

    @patch('app.requests.get')
    def test_raises_on_forecast_api_failure(self, mock_get):
        mock_get.side_effect = [
            _mock_response(json_data=self.current_data),
            _mock_response(status_code=429),
        ]
        with pytest.raises(Exception, match="Forecast API failed with status 429"):
            self.tool._get_weather_data(LOCATION_DICT)

    @patch('app.requests.get')
    def test_forecast_capped_at_five_items(self, mock_get):
        eight_items = [
            {
                "dt_txt": f"2024-01-0{i} 12:00:00",
                "main": {"temp": 22.0, "humidity": 60, "pressure": 1013, "feels_like": 21.0},
                "weather": [{"description": "sunny"}],
                "wind": {"speed": 3.0, "deg": 90},
            }
            for i in range(1, 9)
        ]
        mock_get.side_effect = [
            _mock_response(json_data=self.current_data),
            _mock_response(json_data={"list": eight_items}),
        ]
        result = self.tool._get_weather_data(LOCATION_DICT)
        assert len(result["forecast"]) == 5

    @patch('app.requests.get')
    def test_pressure_and_feels_like_in_current_conditions(self, mock_get):
        mock_get.side_effect = [
            _mock_response(json_data=self.current_data),
            _mock_response(json_data=self.forecast_data),
        ]
        result = self.tool._get_weather_data(LOCATION_DICT)
        assert result["current_conditions"]["pressure"] == 1013
        assert result["current_conditions"]["feels_like"] == 21.0
        assert result["current_conditions"]["visibility"] == 10000


class TestWeatherToolGetEarthquakeData:
    def setup_method(self):
        self.tool = WeatherTool()

    @patch('app.requests.get')
    def test_returns_list_of_earthquakes(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [{
                "properties": {
                    "mag": 4.5,
                    "place": "10km NE of Tokyo",
                    "time": 1704067200000,
                    "url": "https://earthquake.usgs.gov/events/12345",
                }
            }]
        })
        result = self.tool._get_earthquake_data(LOCATION_DICT)
        assert len(result) == 1
        assert result[0]["magnitude"] == 4.5
        assert result[0]["location"] == "10km NE of Tokyo"
        assert "time" in result[0]
        assert "url" in result[0]

    @patch('app.requests.get')
    def test_empty_features_returns_empty_list(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"features": []})
        result = self.tool._get_earthquake_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_returns_multiple_earthquakes(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [
                {"properties": {"mag": 3.1, "place": "5km W of Chiba", "time": 1704067200000, "url": "https://usgs.gov/1"}},
                {"properties": {"mag": 5.2, "place": "20km S of Yokohama", "time": 1704153600000, "url": "https://usgs.gov/2"}},
            ]
        })
        result = self.tool._get_earthquake_data(LOCATION_DICT)
        assert len(result) == 2
        mags = {r["magnitude"] for r in result}
        assert 3.1 in mags
        assert 5.2 in mags

    @patch('app.requests.get')
    def test_request_carries_location_params(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"features": []})
        self.tool._get_earthquake_data(LOCATION_DICT)
        params = mock_get.call_args[1]["params"]
        assert params["latitude"] == LOCATION_DICT["lat"]
        assert params["longitude"] == LOCATION_DICT["lng"]
        assert mock_get.call_args[1].get("timeout")


class TestWeatherToolGetHurricaneData:
    def setup_method(self):
        self.tool = WeatherTool()

    @patch('app.requests.get')
    def test_returns_hurricane_alerts(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [{
                "properties": {
                    "event": "Hurricane Warning",
                    "severity": "Extreme",
                    "headline": "Hurricane Warning in Effect",
                    "description": "Category 4 hurricane approaching",
                    "instruction": "Evacuate immediately",
                    "onset": "2024-01-01T00:00:00Z",
                    "expires": "2024-01-02T00:00:00Z",
                }
            }]
        })
        result = self.tool._get_hurricane_data(LOCATION_DICT)
        assert len(result) == 1
        assert result[0]["event"] == "Hurricane Warning"
        assert result[0]["severity"] == "Extreme"

    @patch('app.requests.get')
    def test_filters_out_non_hurricane_alerts(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [{
                "properties": {
                    "event": "Frost Advisory",
                    "severity": "Minor",
                    "headline": None,
                    "description": None,
                    "instruction": None,
                    "onset": None,
                    "expires": None,
                }
            }]
        })
        result = self.tool._get_hurricane_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_tropical_storm_term_matches(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [{
                "properties": {
                    "event": "Tropical Storm Watch",
                    "severity": "Severe",
                    "headline": None,
                    "description": None,
                    "instruction": None,
                    "onset": None,
                    "expires": None,
                }
            }]
        })
        result = self.tool._get_hurricane_data(LOCATION_DICT)
        assert len(result) == 1

    @patch('app.requests.get')
    def test_tropical_cyclone_term_matches(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [{
                "properties": {
                    "event": "Tropical Cyclone Warning",
                    "severity": "Extreme",
                    "headline": None,
                    "description": None,
                    "instruction": None,
                    "onset": None,
                    "expires": None,
                }
            }]
        })
        result = self.tool._get_hurricane_data(LOCATION_DICT)
        assert len(result) == 1

    @patch('app.requests.get')
    def test_exception_returns_empty_list(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = self.tool._get_hurricane_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_empty_features_returns_empty_list(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"features": []})
        result = self.tool._get_hurricane_data(LOCATION_DICT)
        assert result == []


class TestWeatherToolGetTsunamiData:
    def setup_method(self):
        self.tool = WeatherTool()

    TSUNAMI_XML_WITH_ALERT = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
      <info>
        <event>Tsunami Warning</event>
        <severity>Extreme</severity>
        <urgency>Immediate</urgency>
        <description>Tsunami waves expected</description>
        <instruction>Move to higher ground</instruction>
        <effective>2024-01-01T00:00:00Z</effective>
        <expires>2024-01-01T12:00:00Z</expires>
        <area>
          <areaDesc>Pacific Coast</areaDesc>
          <circle>35.6762,139.6503 500</circle>
        </area>
      </info>
    </alert>
  </entry>
</feed>"""

    # Same alert, but with no machine-readable geometry: it must NOT be
    # reported, because we cannot establish that it covers the caller.
    TSUNAMI_XML_NO_GEOMETRY = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
      <info>
        <event>Tsunami Warning</event>
        <severity>Extreme</severity>
        <area>
          <areaDesc>Pacific Coast</areaDesc>
        </area>
      </info>
    </alert>
  </entry>
</feed>"""

    TSUNAMI_XML_NO_AREA_DESC = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
      <info>
        <event>Tsunami Watch</event>
        <severity>Moderate</severity>
        <urgency>Expected</urgency>
        <area>
        </area>
      </info>
    </alert>
  </entry>
</feed>"""

    TSUNAMI_XML_NO_ALERTS = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""

    @patch('app.requests.get')
    def test_parses_tsunami_alert_from_xml(self, mock_get):
        r = MagicMock()
        r.content = self.TSUNAMI_XML_WITH_ALERT
        mock_get.return_value = r

        result = self.tool._get_tsunami_data(LOCATION_DICT)

        assert len(result) == 1
        assert result[0]["event"] == "Tsunami Warning"
        assert result[0]["severity"] == "Extreme"
        assert result[0]["urgency"] == "Immediate"
        assert result[0]["description"] == "Tsunami waves expected"
        assert result[0]["instruction"] == "Move to higher ground"

    @patch('app.requests.get')
    def test_alert_without_area_desc_is_excluded(self, mock_get):
        r = MagicMock()
        r.content = self.TSUNAMI_XML_NO_AREA_DESC
        mock_get.return_value = r

        result = self.tool._get_tsunami_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_empty_feed_returns_empty_list(self, mock_get):
        r = MagicMock()
        r.content = self.TSUNAMI_XML_NO_ALERTS
        mock_get.return_value = r

        result = self.tool._get_tsunami_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_request_exception_returns_empty_list(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        result = self.tool._get_tsunami_data(LOCATION_DICT)
        assert result == []


class TestWeatherToolLocationInAlertArea:
    def setup_method(self):
        self.tool = WeatherTool()

    def test_area_desc_alone_no_longer_matches_every_location(self):
        """An <areaDesc> with no geometry used to return True for every point
        on Earth, so one Alaskan warning alerted every user worldwide."""
        from xml.etree import ElementTree
        xml = b"""<area xmlns="urn:oasis:names:tc:emergency:cap:1.2">
            <areaDesc>Pacific Coast</areaDesc>
        </area>"""
        element = ElementTree.fromstring(xml)
        assert self.tool._location_in_alert_area(35.0, 139.0, element) is False

    def test_circle_matches_only_inside_its_radius(self):
        from xml.etree import ElementTree
        xml = b"""<area xmlns="urn:oasis:names:tc:emergency:cap:1.2">
            <areaDesc>Coastal Alaska</areaDesc>
            <circle>61.2,-149.9 300</circle>
        </area>"""
        element = ElementTree.fromstring(xml)
        assert self.tool._location_in_alert_area(61.2, -149.9, element) is True
        assert self.tool._location_in_alert_area(-8.4, 115.2, element) is False

    def test_polygon_matches_only_inside_the_ring(self):
        from xml.etree import ElementTree
        xml = b"""<area xmlns="urn:oasis:names:tc:emergency:cap:1.2">
            <areaDesc>Indonesian coast</areaDesc>
            <polygon>-9.5,113.0 -9.5,117.0 -7.0,117.0 -7.0,113.0 -9.5,113.0</polygon>
        </area>"""
        element = ElementTree.fromstring(xml)
        assert self.tool._location_in_alert_area(-8.4, 115.2, element) is True
        assert self.tool._location_in_alert_area(61.2, -149.9, element) is False

    def test_returns_false_when_area_desc_absent(self):
        from xml.etree import ElementTree
        xml = b"""<area xmlns="urn:oasis:names:tc:emergency:cap:1.2"></area>"""
        element = ElementTree.fromstring(xml)
        assert self.tool._location_in_alert_area(35.0, 139.0, element) is False


class TestWeatherToolGetVolcanoData:
    def setup_method(self):
        self.tool = WeatherTool()

    @patch('app.requests.get')
    def test_returns_volcano_activity_list(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [{
                "properties": {
                    "place": "Mount Fuji area",
                    "mag": 2.1,
                    "time": 1704067200000,
                    "detail": "Volcanic activity detected",
                },
                "geometry": {"type": "Point", "coordinates": [138.73, 35.36]},
            }]
        })
        result = self.tool._get_volcano_data(LOCATION_DICT)
        assert len(result) == 1
        assert result[0]["name"] == "Mount Fuji area"
        assert result[0]["type"] == "Volcanic Activity"
        assert result[0]["status"] == "Active"
        assert result[0]["alert_level"] == "Warning"
        assert result[0]["magnitude"] == 2.1

    @patch('app.requests.get')
    def test_non_200_status_returns_empty_list(self, mock_get):
        mock_get.return_value = _mock_response(status_code=500)
        result = self.tool._get_volcano_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_non_point_geometry_is_skipped(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [{
                "properties": {"place": "Region X", "mag": 3.0, "time": 1704067200000},
                "geometry": {"type": "Polygon", "coordinates": []},
            }]
        })
        result = self.tool._get_volcano_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_request_exception_returns_empty_list(self, mock_get):
        mock_get.side_effect = Exception("Timeout")
        result = self.tool._get_volcano_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_empty_features_returns_empty_list(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"features": []})
        result = self.tool._get_volcano_data(LOCATION_DICT)
        assert result == []

    @patch('app.requests.get')
    def test_time_is_converted_to_iso_format(self, mock_get):
        mock_get.return_value = _mock_response(json_data={
            "features": [{
                "properties": {"place": "Somewhere", "mag": 1.5, "time": 1704067200000, "detail": ""},
                "geometry": {"type": "Point", "coordinates": [138.0, 35.0]},
            }]
        })
        result = self.tool._get_volcano_data(LOCATION_DICT)
        assert result[0]["time"] is not None
        assert "T" in result[0]["time"]


class TestWeatherToolRun:
    def setup_method(self):
        self.tool = WeatherTool()

    @patch.object(WeatherTool, '_get_volcano_data', return_value=[])
    @patch.object(WeatherTool, '_get_tsunami_data', return_value=[])
    @patch.object(WeatherTool, '_get_hurricane_data', return_value=[])
    @patch.object(WeatherTool, '_get_earthquake_data', return_value=[{"magnitude": 3.1}])
    @patch.object(WeatherTool, '_get_weather_data', return_value={"current_conditions": {"temp": 22}, "forecast": [], "alerts": []})
    def test_run_aggregates_all_data_sources(self, m_w, m_eq, m_hu, m_ts, m_vo):
        result = self.tool._run(LOCATION_DICT)
        assert "current_conditions" in result
        assert "seismic_activity" in result
        assert "hurricane_alerts" in result
        assert "tsunami_alerts" in result
        assert "volcano_activity" in result
        assert result["seismic_activity"] == [{"magnitude": 3.1}]

    @patch.object(WeatherTool, '_get_volcano_data', return_value=[])
    @patch.object(WeatherTool, '_get_tsunami_data', return_value=[])
    @patch.object(WeatherTool, '_get_hurricane_data', return_value=[])
    @patch.object(WeatherTool, '_get_earthquake_data', return_value=[])
    @patch.object(WeatherTool, '_get_weather_data', return_value={"current_conditions": {}, "forecast": [], "alerts": []})
    def test_run_accepts_string_input(self, m_w, m_eq, m_hu, m_ts, m_vo):
        result = self.tool._run(str(LOCATION_DICT))
        assert "current_conditions" in result

    @patch.object(WeatherTool, '_get_weather_data', side_effect=Exception("API down"))
    def test_run_wraps_exception_with_context(self, _):
        with pytest.raises(Exception, match="Weather data fetch failed"):
            self.tool._run(LOCATION_DICT)

    @patch.object(WeatherTool, '_get_volcano_data', return_value=[])
    @patch.object(WeatherTool, '_get_tsunami_data', return_value=[])
    @patch.object(WeatherTool, '_get_hurricane_data', return_value=[])
    @patch.object(WeatherTool, '_get_earthquake_data', return_value=[])
    @patch.object(WeatherTool, '_get_weather_data', return_value={"current_conditions": {}, "forecast": [], "alerts": []})
    def test_run_calls_all_sub_methods(self, m_w, m_eq, m_hu, m_ts, m_vo):
        self.tool._run(LOCATION_DICT)
        m_w.assert_called_once()
        m_eq.assert_called_once()
        m_hu.assert_called_once()
        m_ts.assert_called_once()
        m_vo.assert_called_once()


class TestEmergencyResourcesTool:
    def setup_method(self):
        self.tool = EmergencyResourcesTool()

    @patch('app.requests.get')
    def test_run_returns_resources_from_all_place_types(self, mock_get):
        hospital_resp = _mock_response(json_data={"status": "OK", "results": [{
            "name": "Tokyo Hospital", "vicinity": "1-1-1 Shinjuku",
            "geometry": {"location": {"lat": 35.69, "lng": 139.70}},
        }]})
        police_resp = _mock_response(json_data={"status": "ZERO_RESULTS", "results": []})
        fire_resp = _mock_response(json_data={"status": "ZERO_RESULTS", "results": []})
        shelter_resp = _mock_response(json_data={"status": "OK", "results": [{
            "name": "City Shelter", "vicinity": "2-2-2 Shibuya",
            "geometry": {"location": {"lat": 35.66, "lng": 139.71}},
        }]})
        mock_get.side_effect = [hospital_resp, police_resp, fire_resp, shelter_resp]

        result = self.tool._run(LOCATION_DICT)

        assert len(result) == 2
        types_in_result = {r["type"] for r in result}
        assert "hospital" in types_in_result
        assert "shelter" in types_in_result

    @patch('app.requests.get')
    def test_run_assigns_correct_resource_type_labels(self, mock_get):
        hospital_resp = _mock_response(json_data={"status": "OK", "results": [{
            "name": "Tokyo General", "vicinity": "1-1",
            "geometry": {"location": {"lat": 35.69, "lng": 139.70}},
        }]})
        police_resp = _mock_response(json_data={"status": "OK", "results": [{
            "name": "Shinjuku Police", "vicinity": "3-3",
            "geometry": {"location": {"lat": 35.70, "lng": 139.69}},
        }]})
        fire_resp = _mock_response(json_data={"status": "ZERO_RESULTS", "results": []})
        shelter_resp = _mock_response(json_data={"status": "ZERO_RESULTS", "results": []})
        mock_get.side_effect = [hospital_resp, police_resp, fire_resp, shelter_resp]

        result = self.tool._run(LOCATION_DICT)

        by_name = {r["name"]: r["type"] for r in result}
        assert by_name["Tokyo General"] == "hospital"
        assert by_name["Shinjuku Police"] == "police"

    @patch('app.requests.get')
    def test_run_accepts_string_input(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"status": "ZERO_RESULTS", "results": []})
        result = self.tool._run(str(LOCATION_DICT))
        assert isinstance(result, list)

    @patch('app.requests.get')
    def test_run_raises_on_unexpected_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        with pytest.raises(Exception, match="Emergency resources fetch failed"):
            self.tool._run(LOCATION_DICT)

    @patch('app.requests.get')
    def test_run_makes_four_api_calls(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"status": "ZERO_RESULTS", "results": []})
        self.tool._run(LOCATION_DICT)
        assert mock_get.call_count == 4

    @patch('app.requests.get')
    def test_result_includes_name_address_location_type(self, mock_get):
        mock_get.side_effect = [
            _mock_response(json_data={"status": "OK", "results": [{
                "name": "My Hospital", "vicinity": "Main St",
                "geometry": {"location": {"lat": 35.7, "lng": 139.7}},
            }]}),
            _mock_response(json_data={"status": "ZERO_RESULTS", "results": []}),
            _mock_response(json_data={"status": "ZERO_RESULTS", "results": []}),
            _mock_response(json_data={"status": "ZERO_RESULTS", "results": []}),
        ]
        result = self.tool._run(LOCATION_DICT)
        assert result[0]["name"] == "My Hospital"
        assert result[0]["address"] == "Main St"
        assert "lat" in result[0]["location"]
        assert result[0]["type"] == "hospital"


class TestExplanationAgent:
    def setup_method(self):
        app.types.Part.from_text.reset_mock()
        self.agent = ExplanationAgent()
        self.agent.client.models.generate_content_stream.reset_mock()

    def _get_last_prompt(self):
        calls = app.types.Part.from_text.call_args_list
        if calls:
            args, kwargs = calls[-1][0], calls[-1][1]
        return kwargs.get('text', args[0] if args else '')
        return ""

    def _base_weather_data(self, **overrides):
        data = {
            "current_conditions": {
                "temperature": 25, "weather": "sunny", "humidity": 60,
                "wind_speed": 5, "wind_direction": 90, "pressure": 1013,
                "visibility": 10000, "feels_like": 24,
            },
            "forecast": [],
            "seismic_activity": [],
            "tsunami_alerts": [],
            "hurricane_alerts": [],
            "volcano_activity": [],
            "flood_alerts": [],
            "emergency_resources": [],
        }
        data.update(overrides)
        return data

    def test_calls_llm_generate_stream_once(self):
        self.agent.explain_weather_data(self._base_weather_data(), "Tokyo, Japan")
        self.agent.client.models.generate_content_stream.assert_called_once()

    def test_seismic_data_appears_in_prompt(self):
        data = self._base_weather_data(seismic_activity=[
            {"magnitude": 4.5, "location": "Near Osaka", "time": "2024-01-01T00:00:00", "url": "http://usgs.gov/1"}
        ])
        self.agent.explain_weather_data(data, "Osaka, Japan")
        prompt = self._get_last_prompt()
        assert "4.5" in prompt
        assert "Near Osaka" in prompt

    def test_no_seismic_message_when_empty(self):
        self.agent.explain_weather_data(self._base_weather_data(), "Nowhere")
        prompt = self._get_last_prompt()
        assert "No recent seismic activity reported" in prompt

    def test_no_tsunami_message_when_empty(self):
        self.agent.explain_weather_data(self._base_weather_data(), "Somewhere")
        prompt = self._get_last_prompt()
        assert "No active tsunami alerts reported" in prompt

    def test_no_volcano_message_when_empty(self):
        self.agent.explain_weather_data(self._base_weather_data(), "Somewhere")
        prompt = self._get_last_prompt()
        assert "No volcanic eruptions reported within range of this location" in prompt

    def test_no_hurricane_message_when_empty(self):
        self.agent.explain_weather_data(self._base_weather_data(), "Somewhere")
        prompt = self._get_last_prompt()
        assert "No active hurricane/cyclone alerts reported" in prompt

    def test_no_flood_message_when_empty(self):
        self.agent.explain_weather_data(self._base_weather_data(), "Somewhere")
        prompt = self._get_last_prompt()
        assert "No active flood alerts reported" in prompt

    def test_emergency_resources_present_does_not_crash(self):
        data = self._base_weather_data(emergency_resources=[
            {"type": "hospital", "name": "City Medical Center", "address": "1-1 Main St"}
        ])
        self.agent.explain_weather_data(data, "Tokyo")
        self.agent.client.models.generate_content_stream.assert_called_once()

    def test_empty_emergency_resources_does_not_crash(self):
        self.agent.explain_weather_data(self._base_weather_data(), "Somewhere")
        self.agent.client.models.generate_content_stream.assert_called_once()

    def test_location_name_appears_in_prompt(self):
        self.agent.explain_weather_data(self._base_weather_data(), "Unique Location Name XYZ")
        prompt = self._get_last_prompt()
        assert "Unique Location Name XYZ" in prompt

    def test_handles_completely_empty_weather_data(self):
        self.agent.explain_weather_data({}, "Unknown")
        self.agent.client.models.generate_content_stream.assert_called_once()

    def test_current_temperature_in_prompt(self):
        data = self._base_weather_data()
        self.agent.explain_weather_data(data, "Tokyo")
        prompt = self._get_last_prompt()
        assert "25" in prompt

    def test_tsunami_alert_details_in_prompt(self):
        data = self._base_weather_data(tsunami_alerts=[
            {"event": "Tsunami Warning", "severity": "Extreme", "status": "Active",
             "expected_time": None, "affected_areas": None, "instructions": None}
        ])
        self.agent.explain_weather_data(data, "Pacific Coast")
        prompt = self._get_last_prompt()
        assert "Tsunami Warning" in prompt
