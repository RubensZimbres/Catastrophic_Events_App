import ast
import logging
import os
import re
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Any, ClassVar
from functools import lru_cache
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import json
from datetime import datetime, timedelta, timezone
import math
from google.cloud import secretmanager
import pandas as pd

logger = logging.getLogger(__name__)

# Every outbound call needs a ceiling. Eight of the nine requests.get() calls
# had none, so one unresponsive upstream pinned a Streamlit worker forever.
HTTP_TIMEOUT = 10

project_id = os.environ.get("GCP_PROJECT_ID", "")

GEMINI_MODEL = "gemini-3.7-flash"
VERTEX_LOCATION = os.environ.get("GCP_REGION", "us-central1")


def _utcnow() -> datetime:
    """Timezone-aware UTC now (datetime.utcnow() is deprecated in 3.12+)."""
    return datetime.now(timezone.utc)


def _scrub(text) -> str:
    """Strip API keys out of text before it reaches a log or the UI.

    Every upstream URL here is built by f-string with the key inline, and
    requests puts the full URL into its exception messages.
    """
    text = str(text)
    text = re.sub(r'(appid|key|api_key|API_KEY)=[^&\s\'"]+', r'\1=***', text)
    for secret in (OPENWEATHER_API_KEY, GOOGLE_MAPS_API_KEY, GOOGLE_AI_API_KEY):
        if secret and len(secret) > 6:
            text = text.replace(secret, '***')
    return text


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _coerce_location(value: Any) -> Dict:
    """Turn a tool argument into a location dict without executing it.

    This was `eval(location_dict)`. The string reaching it is produced by the
    LLM from user-supplied address text, so eval() made prompt injection a
    remote code execution path.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Unsupported location payload: {type(value).__name__}")
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(value)          # literals only, never code
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Could not parse location payload") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Location payload is not an object")
    return parsed


@lru_cache(maxsize=1)
def _secret_client():
    return secretmanager.SecretManagerServiceClient()


# "latest" rather than "1": pinning to version 1 meant rotating a leaked key in
# Secret Manager had no effect on the running app.
@lru_cache(maxsize=16)
def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = _secret_client().access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')


def get_secret_or_env(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    if project_id:
        try:
            return get_secret(project_id, secret_id, version_id)
        except Exception as exc:
            logger.info("Secret Manager lookup for %s failed (%s); falling back to env",
                        secret_id, type(exc).__name__)
    return os.environ.get(secret_id, "")


OPENWEATHER_API_KEY = get_secret_or_env(project_id, 'OPENWEATHER_API_KEY')
GOOGLE_MAPS_API_KEY = get_secret_or_env(project_id, 'GOOGLE_MAPS_API_KEY')
GOOGLE_AI_API_KEY = get_secret_or_env(project_id, 'GOOGLE_AI_API_KEY')

from google import genai
from google.genai import types

class LocationInfo(BaseModel):
    address: str = Field(description="User's address")
    lat: float = Field(description="Latitude")
    lng: float = Field(description="Longitude")

class WeatherInfo(BaseModel):
    current_conditions: dict
    forecast: List[dict]
    alerts: List[dict]

class GeocodingTool(BaseTool):
    name: str = Field(default="geocoding_tool")
    description: str = Field(default="Get latitude and longitude coordinates for a given address")
    return_direct: bool = Field(default=False)

    def _run(self, address: str) -> Dict:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        response = requests.get(
            url,
            params={"address": address, "key": GOOGLE_MAPS_API_KEY},
            timeout=HTTP_TIMEOUT,
        )
        data = response.json()

        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return {
                "address": data['results'][0]['formatted_address'],
                "lat": location['lat'],
                "lng": location['lng']
            }
        else:
            raise Exception(f"Geocoding failed: {data['status']}")

    async def _arun(self, address: str) -> Dict:
        raise NotImplementedError("Async not implemented")

class WeatherTool(BaseTool):
    name: str = Field(default="weather_tool")
    description: str = Field(default="Get weather information and alerts for a specific location")
    return_direct: bool = Field(default=False)

    def _run(self, location_dict: Any) -> Dict:
        try:
            location_dict = _coerce_location(location_dict)

            # Get basic weather data
            weather_info = self._get_weather_data(location_dict)

            # Add earthquake data
            earthquake_data = self._get_earthquake_data(location_dict)
            weather_info["seismic_activity"] = earthquake_data

            # Add hurricane data
            hurricane_data = self._get_hurricane_data(location_dict)
            weather_info["hurricane_alerts"] = hurricane_data

            # Add tsunami data
            tsunami_data = self._get_tsunami_data(location_dict)
            weather_info["tsunami_alerts"] = tsunami_data

            volcano_data = self._get_volcano_data(location_dict)
            weather_info["volcano_activity"] = volcano_data

            flood_data = self._get_flood_data(location_dict)
            weather_info["flood_alerts"] = flood_data

            return weather_info

        except Exception as e:
            print(f"Debug - Error in WeatherTool: {str(e)}")
            print(f"Debug - Error type: {type(e)}")
            import traceback
            print(f"Debug - Traceback: {traceback.format_exc()}")
            raise Exception(f"Weather data fetch failed: {str(e)}")


    VOLCANO_RADIUS_KM: ClassVar[int] = 500

    def _get_volcano_data(self, location_dict: Dict) -> List[Dict]:
        """Volcanic eruptions from USGS, restricted to the caller's vicinity.

        The query previously had no location filter, so it returned every
        volcanic eruption worldwide and each was presented to the user (and to
        the model) as local activity.
        """
        try:
            lat = float(location_dict["lat"])
            lng = float(location_dict["lng"])

            end_date = _utcnow()
            start_date = end_date - timedelta(days=10)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            headers = {
                'User-Agent': 'DisasterAdvisor/1.0',
                'Accept': 'application/json'
            }

            response = requests.get(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                params={
                    "format": "geojson",
                    "starttime": start_str,
                    "endtime": end_str,
                    "eventtype": "volcanic eruption",
                    "latitude": lat,
                    "longitude": lng,
                    "maxradiuskm": self.VOLCANO_RADIUS_KM,
                },
                headers=headers, timeout=HTTP_TIMEOUT)

            if response.status_code != 200:
                logger.warning("Volcano API returned status %s", response.status_code)
                return []

            data = response.json()
            nearby_volcanoes = []

            # Process features from the GeoJSON
            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})

                if geometry and geometry.get('type') == 'Point':
                    coordinates = geometry.get('coordinates', [0, 0, 0])
                    ev_lng, ev_lat = coordinates[0], coordinates[1]
                    distance_km = _haversine_km(lat, lng, ev_lat, ev_lng)

                    nearby_volcanoes.append({
                        "name": properties.get('place', 'Unknown Location'),
                        "type": "Volcanic Activity",
                        "status": "Active",
                        "alert_level": "Warning",
                        "magnitude": properties.get('mag'),
                        # Real distance; the UI used to print a hardcoded 0.0 km
                        "distance_km": round(distance_km, 1),
                        "time": (datetime.fromtimestamp(
                            properties.get('time', 0) / 1000, timezone.utc).isoformat()
                            if properties.get('time') else None),
                        "details": properties.get('detail', 'Volcanic activity detected')
                    })

            return nearby_volcanoes

        except Exception as e:
            logger.warning("Volcano data fetch failed: %s", _scrub(e))
            return []

    def _get_weather_data(self, location_dict: Dict) -> Dict:
        """Get weather information from OpenWeatherMap"""
        lat = float(location_dict["lat"])
        lng = float(location_dict["lng"])

        ow_params = {"lat": lat, "lon": lng, "appid": OPENWEATHER_API_KEY, "units": "metric"}

        # Current weather
        current_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=ow_params, timeout=HTTP_TIMEOUT)
        if current_response.status_code != 200:
            raise Exception(f"Current weather API failed with status {current_response.status_code}")
        current_data = current_response.json()

        # 5-day forecast
        forecast_response = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params=ow_params, timeout=HTTP_TIMEOUT)
        if forecast_response.status_code != 200:
            raise Exception(f"Forecast API failed with status {forecast_response.status_code}")
        forecast_data = forecast_response.json()

        # Process forecast data
        forecast_list = []
        if isinstance(forecast_data, dict) and 'list' in forecast_data:
            forecast_list = forecast_data['list']

        weather_data = {
            "current_conditions": {
                "temperature": current_data.get('main', {}).get('temp'),
                "weather": current_data.get('weather', [{}])[0].get('description'),
                "humidity": current_data.get('main', {}).get('humidity'),
                "wind_speed": current_data.get('wind', {}).get('speed'),
                "wind_direction": current_data.get('wind', {}).get('deg'),
                "pressure": current_data.get('main', {}).get('pressure'),
                "visibility": current_data.get('visibility'),
                "feels_like": current_data.get('main', {}).get('feels_like')
            },
            "forecast": [
                {
                    "datetime": item.get('dt_txt'),
                    "temperature": item.get('main', {}).get('temp'),
                    "weather": item.get('weather', [{}])[0].get('description'),
                    "humidity": item.get('main', {}).get('humidity'),
                    "wind_speed": item.get('wind', {}).get('speed'),
                    "wind_direction": item.get('wind', {}).get('deg'),
                    "pressure": item.get('main', {}).get('pressure'),
                    "feels_like": item.get('main', {}).get('feels_like')
                }
                for item in forecast_list[:5]
            ],
            "alerts": []
        }

        return weather_data
    def _get_earthquake_data(self, location_dict: Dict) -> List[Dict]:
        """Fetch recent earthquake data from USGS"""
        lat = location_dict["lat"]
        lng = location_dict["lng"]

        response = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={"format": "geojson", "latitude": lat, "longitude": lng,
                    "maxradiuskm": 300, "minmagnitude": 2.5, "orderby": "time"},
            timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        earthquakes = []
        for feature in data["features"]:
            earthquakes.append({
                "magnitude": feature["properties"]["mag"],
                "location": feature["properties"]["place"],
                "time": datetime.fromtimestamp(
                    feature["properties"]["time"] / 1000.0, timezone.utc).isoformat(),
                "url": feature["properties"]["url"]
            })

        return earthquakes

    HURRICANE_TERMS: ClassVar[tuple] = ('hurricane', 'tropical storm', 'tropical cyclone')
    FLOOD_TERMS: ClassVar[tuple] = ('flood', 'flash flood', 'coastal flood', 'river flood')

    def _fetch_nws_alerts(self, lat, lng) -> List[Dict]:
        """Active NWS alerts for a point. Cached per instance per location so
        the hurricane and flood filters share a single request."""
        cache_key = (lat, lng)
        if getattr(self, "_nws_cache_key", None) == cache_key:
            return self._nws_cache
        headers = {
            "Accept": "application/geo+json",
            "User-Agent": "(disaster-advisor-app.com, contact@disaster-advisor-app.com)"
        }
        try:
            response = requests.get(
                "https://api.weather.gov/alerts/active",
                params={"point": f"{lat},{lng}"}, headers=headers, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            features = response.json().get('features', [])
        except Exception as e:
            logger.warning("NWS alert fetch failed: %s", _scrub(e))
            features = []
        object.__setattr__(self, "_nws_cache_key", cache_key)
        object.__setattr__(self, "_nws_cache", features)
        return features

    @staticmethod
    def _alert_fields(properties: Dict) -> Dict:
        return {
            "event": properties.get('event'),
            "severity": properties.get('severity'),
            "headline": properties.get('headline'),
            "description": properties.get('description'),
            "instruction": properties.get('instruction'),
            "onset": properties.get('onset'),
            "expires": properties.get('expires'),
        }

    def _filter_alerts(self, lat, lng, terms) -> List[Dict]:
        matches = []
        for feature in self._fetch_nws_alerts(lat, lng):
            properties = feature.get('properties', {})
            event = (properties.get('event') or '').lower()
            if any(term in event for term in terms):
                matches.append(self._alert_fields(properties))
        return matches

    def _get_hurricane_data(self, location_dict: Dict) -> List[Dict]:
        """Hurricane and tropical storm warnings from the NWS alerts feed."""
        return self._filter_alerts(location_dict["lat"], location_dict["lng"],
                                   self.HURRICANE_TERMS)

    def _get_flood_data(self, location_dict: Dict) -> List[Dict]:
        """Flood warnings from the NWS alerts feed.

        The app advertised flood monitoring and the prompt had a FLOOD ALERTS
        section, but nothing ever populated weather_data['flood_alerts'], so
        every report emitted a standing "No active flood alerts reported".
        """
        return self._filter_alerts(location_dict["lat"], location_dict["lng"],
                                   self.FLOOD_TERMS)

    def _get_tsunami_data(self, location_dict: Dict) -> List[Dict]:
        """Fetch tsunami warnings from NOAA's Tsunami Warning System"""
        lat = location_dict["lat"]
        lng = location_dict["lng"]

        # NOAA Tsunami Warning Center API
        url = "https://www.tsunami.gov/events/xml/PAAQAtom.xml"
        headers = {
            "User-Agent": "(disaster-advisor-app.com, contact@disaster-advisor-app.com)"
        }

        try:
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)

            # The feed is in XML format
            from xml.etree import ElementTree
            root = ElementTree.fromstring(response.content)

            # Parse tsunami alerts
            tsunami_alerts = []

            # XML namespaces used in the feed
            namespaces = {
                'cap': 'urn:oasis:names:tc:emergency:cap:1.2',
                'atom': 'http://www.w3.org/2005/Atom'
            }

            for entry in root.findall('.//atom:entry', namespaces):
                # Get the CAP alert
                cap_alert = entry.find('.//cap:alert', namespaces)
                if cap_alert is not None:
                    info = cap_alert.find('.//cap:info', namespaces)
                    if info is not None:
                        # Check if this alert affects our location
                        area = info.find('.//cap:area', namespaces)
                        if area is not None:
                            # Convert the area description to a rough bounding box
                            if self._location_in_alert_area(lat, lng, area):
                                tsunami_alerts.append({
                                    "event": info.find('.//cap:event', namespaces).text if info.find('.//cap:event', namespaces) is not None else None,
                                    "severity": info.find('.//cap:severity', namespaces).text if info.find('.//cap:severity', namespaces) is not None else None,
                                    "urgency": info.find('.//cap:urgency', namespaces).text if info.find('.//cap:urgency', namespaces) is not None else None,
                                    "description": info.find('.//cap:description', namespaces).text if info.find('.//cap:description', namespaces) is not None else None,
                                    "instruction": info.find('.//cap:instruction', namespaces).text if info.find('.//cap:instruction', namespaces) is not None else None,
                                    "effective": info.find('.//cap:effective', namespaces).text if info.find('.//cap:effective', namespaces) is not None else None,
                                    "expires": info.find('.//cap:expires', namespaces).text if info.find('.//cap:expires', namespaces) is not None else None
                                })

            return tsunami_alerts

        except Exception as e:
            logger.warning("Tsunami data fetch failed: %s", _scrub(e))
            return []

    TSUNAMI_RADIUS_KM: ClassVar[int] = 1000
    _CAP_NS: ClassVar[dict] = {'cap': 'urn:oasis:names:tc:emergency:cap:1.2'}

    def _location_in_alert_area(self, lat: float, lng: float, area_element) -> bool:
        """Does this CAP alert area actually cover (lat, lng)?

        The previous implementation returned True whenever an <areaDesc>
        element existed, which is every alert — so users anywhere in the world
        were shown every warning in the feed as if it were local.

        CAP areas carry machine-readable geometry: <polygon> (a lat,lon ring)
        and/or <circle> ("lat,lon radius_km"). Both are honoured here. When an
        area carries neither, we cannot place it and return False rather than
        raising a false alarm.
        """
        matched_geometry = False

        for circle in area_element.findall('./cap:circle', self._CAP_NS):
            text = (circle.text or "").strip()
            try:
                centre, radius = text.split()
                clat, clng = (float(v) for v in centre.split(","))
                if _haversine_km(lat, lng, clat, clng) <= float(radius):
                    return True
                matched_geometry = True
            except (ValueError, AttributeError):
                continue

        for polygon in area_element.findall('./cap:polygon', self._CAP_NS):
            points = []
            for pair in (polygon.text or "").split():
                try:
                    plat, plng = (float(v) for v in pair.split(","))
                    points.append((plat, plng))
                except ValueError:
                    continue
            if len(points) >= 3:
                matched_geometry = True
                if self._point_in_polygon(lat, lng, points):
                    return True

        if matched_geometry:
            return False

        # No usable geometry. Fall back to proximity against any geocode the
        # area advertises, and otherwise decline to claim the alert is local.
        for geocode in area_element.findall('./cap:geocode', self._CAP_NS):
            value = geocode.find('./cap:value', self._CAP_NS)
            if value is not None and value.text:
                try:
                    glat, glng = (float(v) for v in value.text.split(","))
                except ValueError:
                    continue
                if _haversine_km(lat, lng, glat, glng) <= self.TSUNAMI_RADIUS_KM:
                    return True
        return False

    @staticmethod
    def _point_in_polygon(lat: float, lng: float, points: List) -> bool:
        """Ray-casting test. `points` are (lat, lng) pairs as CAP orders them."""
        inside = False
        n = len(points)
        for i in range(n):
            y1, x1 = points[i]
            y2, x2 = points[(i + 1) % n]
            if (y1 > lat) != (y2 > lat):
                x_at = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
                if lng < x_at:
                    inside = not inside
        return inside

    async def _arun(self, location: Dict) -> Dict:
        raise NotImplementedError("Async not implemented")


class EmergencyResourcesTool(BaseTool):
    name: str = Field(default="emergency_resources_tool")
    description: str = Field(default="Find nearby emergency resources and shelters")
    return_direct: bool = Field(default=False)

    def _run(self, location_dict: Dict) -> List[Dict]:
        try:
            location_dict = _coerce_location(location_dict)

            lat = location_dict["lat"]
            lng = location_dict["lng"]
            place_types = ["hospital", "police", "fire_station"]
            all_resources = []

            places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            for place_type in place_types:
                response = requests.get(
                    places_url,
                    params={"location": f"{lat},{lng}", "radius": 5000,
                            "type": place_type, "key": GOOGLE_MAPS_API_KEY},
                    timeout=HTTP_TIMEOUT,
                )
                data = response.json()

                if data['status'] == 'OK':
                    for place in data['results']:
                        all_resources.append({
                            'name': place['name'],
                            'address': place.get('vicinity', ''),
                            'location': place['geometry']['location'],
                            'type': place_type
                        })

            # Additionally search for emergency shelters using keyword
            shelter_response = requests.get(
                places_url,
                params={"location": f"{lat},{lng}", "radius": 5000,
                        "keyword": "emergency shelter", "key": GOOGLE_MAPS_API_KEY},
                timeout=HTTP_TIMEOUT,
            )
            shelter_data = shelter_response.json()

            if shelter_data['status'] == 'OK':
                for place in shelter_data['results']:
                    all_resources.append({
                        'name': place['name'],
                        'address': place.get('vicinity', ''),
                        'location': place['geometry']['location'],
                        'type': 'shelter'
                    })

            return all_resources

        except Exception as e:
            print(f"Error in emergency resources fetch: {str(e)}")
            raise Exception(f"Emergency resources fetch failed: {str(e)}")

    async def _arun(self, location: Dict) -> List[Dict]:
        raise NotImplementedError("Async not implemented")


class DisasterAdvisorAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_AI_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        # Initialize tools
        self.geocoding_tool = GeocodingTool()
        self.weather_tool = WeatherTool()
        self.emergency_resources_tool = EmergencyResourcesTool()

        self.tools = [
            Tool(
                name="Geocoding",
                func=self.geocoding_tool._run,
                description="Convert address to coordinates. Input should be a string address."
            ),
             Tool(
                name="Weather",
                func=self.weather_tool._run,
                description="Get weather information and alerts for a location. Input should be the direct output from the Geocoding tool."
            ),
            Tool(
                name="Emergency Resources",
                func=self.emergency_resources_tool._run,
                description="Find nearby emergency resources. Input should be a dictionary containing 'lat' and 'lng' keys."
            )
        ]

        # Initialize the agent
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def get_response(self, user_input: str) -> str:
        """
        Process user input and return a response
        """
        try:
            if "climate threats" in user_input.lower() or "weather" in user_input.lower():
                # First get location data
                location_response = self.geocoding_tool._run(
                    user_input.split("address is ")[-1].strip())

                # Then get weather data using the location
                weather_data = self.weather_tool._run(location_response)

                # Add emergency resources data
                try:
                    emergency_resources = self.emergency_resources_tool._run(location_response)
                    weather_data['emergency_resources'] = emergency_resources
                except Exception as e:
                    print(f"Error fetching emergency resources: {str(e)}")
                    weather_data['emergency_resources'] = []

                # Return the combined data
                return weather_data

            else:
                # For non-weather queries, use the normal agent response
                response = self.agent_executor.run(user_input)
                return response

        except Exception as e:
            return f"An error occurred: {str(e)}"


class ExplanationAgent:
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("GCP_PROJECT_NAME", ""),
            location=VERTEX_LOCATION,
        )

        self.generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ]
        )

    def explain_weather_data(self, weather_data: dict, location: str):
        # Extract data with safe fallbacks
        current = weather_data.get('current_conditions', {})
        forecast = weather_data.get('forecast', [])
        seismic = weather_data.get('seismic_activity', [])
        tsunamis = weather_data.get('tsunami_alerts', [])
        volcanoes = weather_data.get('volcano_activity', [])  # Add this line
        hurricanes = weather_data.get('hurricane_alerts', [])  # Add this line
        floods = weather_data.get('flood_alerts', [])  # Add flood data



        emergency_resources = weather_data.get('emergency_resources', [])

        # Format emergency resources information
        emergency_info = ""
        if emergency_resources:
            emergency_info = "Nearby Emergency Resources:\n"
            resources_by_type = {}

            # Group resources by type
            for resource in emergency_resources:
                resource_type = resource.get('type', 'other')
                if resource_type not in resources_by_type:
                    resources_by_type[resource_type] = []
                resources_by_type[resource_type].append(resource)

            # Format each type of resource
            for resource_type, resources in resources_by_type.items():
                emergency_info += f"\n{resource_type.upper()}:\n"
                for resource in resources:
                    emergency_info += f"""
        - Name: {resource.get('name')}
        Address: {resource.get('address')}
        Distance: {resource.get('distance', 'N/A')}
        """
        else:
            emergency_info = "No emergency resource information available."




        # Format the seismic activity data for better readability
        seismic_info = ""
        if seismic:
            seismic_info = "Recent earthquakes:\n"
            for quake in seismic:
                seismic_info += f"""
    - Magnitude: {quake.get('magnitude')}
    Location: {quake.get('location')}
    Time: {quake.get('time')}
    More info: {quake.get('url')}
    """
        else:
            seismic_info = "No recent seismic activity reported."

        # --- Hazard blocks -------------------------------------------------
        # These read the keys the fetchers ACTUALLY produce. Previously
        # hurricane_info was built twice and the second version (reading
        # 'source', 'distance_km', 'wind_speed', 'pressure', 'movement',
        # 'details' — none of which _get_hurricane_data returns) overwrote the
        # correct one, so the headline, description and above all the
        # *instruction* text never reached the model. tsunami_info had the same
        # defect ('status'/'expected_time'/'affected_areas'/'instructions'
        # vs the real 'urgency'/'description'/'instruction'/'effective').
        # In a disaster app those dropped fields are the evacuation orders.

        def _block(title: str, items: list, fields: list, empty: str) -> str:
            if not items:
                return empty
            out = f"{title}\n"
            for item in items:
                out += "\n"
                for label, key in fields:
                    value = item.get(key)
                    if value not in (None, ""):
                        out += f"    {label}: {value}\n"
            return out

        tsunami_info = _block(
            "Active tsunami alerts:", tsunamis,
            [("Event", "event"), ("Severity", "severity"), ("Urgency", "urgency"),
             ("Description", "description"), ("Instruction", "instruction"),
             ("Effective", "effective"), ("Expires", "expires")],
            "No active tsunami alerts reported.")

        hurricane_info = _block(
            "Active hurricane/cyclone alerts:", hurricanes,
            [("Event", "event"), ("Severity", "severity"), ("Headline", "headline"),
             ("Description", "description"), ("Instruction", "instruction"),
             ("Onset", "onset"), ("Expires", "expires")],
            "No active hurricane/cyclone alerts reported.")

        volcano_info = _block(
            "Recent volcanic eruptions near this location:", volcanoes,
            [("Name", "name"), ("Type", "type"), ("Status", "status"),
             ("Alert Level", "alert_level"), ("Magnitude", "magnitude"),
             ("Distance (km)", "distance_km"), ("Time", "time"),
             ("Activity Details", "details")],
            "No volcanic eruptions reported within range of this location.")

        flood_info = _block(
            "Active flood alerts:", floods,
            [("Event", "event"), ("Severity", "severity"), ("Headline", "headline"),
             ("Description", "description"), ("Instruction", "instruction"),
             ("Onset", "onset"), ("Expires", "expires")],
            "No active flood alerts reported.")

        prompt = f"""
        Analyze the following weather and emergency data for {location}:

        CURRENT CONDITIONS:
        Temperature: {current.get('temperature')}°C
        Weather: {current.get('weather')}
        Humidity: {current.get('humidity')}%
        Wind Speed: {current.get('wind_speed')} m/s
        Wind Direction: {current.get('wind_direction')}°
        Pressure: {current.get('pressure')} hPa
        Visibility: {current.get('visibility')} m
        Feels Like: {current.get('feels_like')}°C

        FORECAST:
        {json.dumps(forecast, indent=2)}

        SEISMIC ACTIVITY:
        {seismic_info}

        TSUNAMI ALERTS:
        {tsunami_info}

        VOLCANIC ALERTS:
        {volcano_info}

        HURRICANE/CYCLONE ALERTS:
        {hurricane_info}

        FLOOD ALERTS:
        {flood_info}

        Please provide a concise analysis formatted in markdown:

        # ⚠️ Emergency Status Summary
        [ Overview of immediate risks]

        # Current Conditions
        - Highlight anomalies in temperature, humidity, wind conditions
        - Highlight any severe weather conditions
        - Highlight any immediate concerns

        # 📈 Seismic Activity
        - List all recent earthquakes with magnitude and location
        - Evaluate potential aftershock risks
        - Note proximity to populated areas

        # 🌊 Tsunami Alerts
        - List tsunamis, if applicable
        - Note areas at risk
        - Include evacuation instructions if provided

        # 🌋 Volcanic Alerts
        - List any active volcanoes in the area
        - Note current alert levels and activity status
        - Include distance from location and potential risks
        - Highlight any significant recent changes in activity

        # 🌀 Hurricane/Cyclone Alerts
        - List any active storms
        - Note severity, wind speeds, and movement patterns
        - Include specific threat levels for the location
        - Highlight expected timeline and progression

        # 🌊 Flood Alerts
        - List active flood warnings
        - Note severity and affected areas
        - Include water levels and forecasts if available
        - Highlight areas at immediate risk

        # ⛑️ Safety Recommendations
        1. [Immediate actions needed]
        2. [Preparation steps]
        3. [Emergency supplies if needed]
        4. [Evacuation considerations if relevant]

        Include all numerical data where available and be specific about potential risks.
        Prioritize immediate threats and provide clear, actionable guidance.
        """

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        return self.client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=self.generate_content_config
        )


import streamlit as st


def _render_alerts(weather_data):
    """Show active alerts with the fields that matter in an emergency."""
    groups = [
        ("🌀 Hurricane / Tropical Storm", weather_data.get('hurricane_alerts') or []),
        ("🌊 Tsunami", weather_data.get('tsunami_alerts') or []),
        ("🌊 Flood", weather_data.get('flood_alerts') or []),
    ]
    if not any(alerts for _, alerts in groups):
        return

    st.markdown("### ⚠️ Active Alerts ⚠️")
    for title, alerts in groups:
        if not alerts:
            continue
        st.error(f"{title} — {len(alerts)} active")
        for alert in alerts:
            headline = alert.get('headline') or alert.get('event') or 'Alert'
            with st.expander(f"{headline}", expanded=True):
                for label, key in (("Event", "event"), ("Severity", "severity"),
                                   ("Urgency", "urgency"), ("Onset", "onset"),
                                   ("Expires", "expires")):
                    if alert.get(key):
                        st.write(f"**{label}:** {alert[key]}")
                if alert.get('description'):
                    st.write(alert['description'])
                if alert.get('instruction'):
                    # The single most important field on the page
                    st.warning(f"**What to do:** {alert['instruction']}")


def main():
    # Add a container for better spacing
    main_container = st.container()

    with main_container:
        st.title("🌪️ Catastrophic Events App")
        st.write("")
        st.markdown("**Your Gemini multi-agent app for extreme events: hurricanes, earthquakes and tsunamis**")
        st.write("")
        st.write("")
#        st.write("Try any address or enter:")
#        st.write("Datah Village in Bali")

        # Initialize agents
        if 'disaster_advisor' not in st.session_state:
            st.session_state.disaster_advisor = DisasterAdvisorAgent()
        if 'explanation_agent' not in st.session_state:
            st.session_state.explanation_agent = ExplanationAgent()

        # Create columns for better layout
        col1, col2 = st.columns([2, 1])

        with col1:
            address = st.text_input("📍 Enter your address:",
                                  placeholder="e.g., 123 Main St, City, Country",
                                  key="address_input")

        with col2:
            st.write("")
            st.write("")
            analyze_button = st.button("🔍 Get Analysis", type="primary")


        if analyze_button and address:
            with st.spinner("📊 Our agents are analyzing weather and emergency data..."):
                try:


                    # Get weather data
                    response = st.session_state.disaster_advisor.get_response(
                        f"What are the extreme climate threats in my area for the next week? My address is {address}"
                    )

                    # Convert response to structured data
                    try:
                        # If response is a string that looks like a dict
                        if isinstance(response, str) and '{' in response:
                            # Clean up the string if needed (remove any escape characters)
                            cleaned_response = response.replace('\n', '').replace('\\', '')
                            weather_data = json.loads(cleaned_response)
                        # If response is already a dict
                        elif isinstance(response, dict):
                            weather_data = response
                        else:
                            # Create a basic structure for text responses
                            weather_data = {
                                "current_conditions": {
                                    "weather": response
                                }
                            }

                        # Display the weather dashboard
                        weather_dashboard_container = st.container()
                        with weather_dashboard_container:
                            st.markdown("### Current Weather Dashboard")

                            # Create three columns for current conditions
                            col1, col2, col3 = st.columns(3)

                            cc = weather_data.get('current_conditions', {})

                            with col1:
                                st.metric("Temperature", f"{cc.get('temperature', '--')}°C",
                                          f"Feels like {cc.get('feels_like', '--')}°C")

                            with col2:
                                st.metric("Humidity", f"{cc.get('humidity', '--')}%")

                            with col3:
                                st.metric("Wind", f"{cc.get('wind_speed', '--')} m/s",
                                          f"Direction {cc.get('wind_direction', '--')}°")

                            # Current weather condition
                            st.info(f"Current weather: {cc.get('weather', 'unavailable')}")

                            # Forecast section
                            st.markdown("### 5-Day Forecast")
                            for forecast in weather_data.get('forecast', []):
                                with st.expander(f"Forecast for {forecast['datetime']}"):
                                    cols = st.columns(4)
                                    with cols[0]:
                                        st.metric("Temperature", f"{forecast['temperature']}°C")
                                    with cols[1]:
                                        st.metric("Humidity", f"{forecast['humidity']}%")
                                    with cols[2]:
                                        st.metric("Wind", f"{forecast['wind_speed']} m/s")
                                    with cols[3]:
                                        st.write("Conditions:", forecast['weather'])

                            # Seismic activity section
                            if weather_data.get('seismic_activity'):
                                st.markdown("### Recent Seismic Activity")
                                for quake in weather_data.get('seismic_activity', []):
                                    with st.expander(f"Magnitude {quake['magnitude']} - {quake['location']}"):
                                        st.write(f"Time: {quake['time']}")
                                        st.write(f"Location: {quake['location']}")
                                        st.markdown(f"[More details]({quake['url']})")



                            _render_alerts(weather_data)

                            if weather_data.get('volcano_activity'):
                                st.markdown("### 🌋 Recent Volcanic Activity")
                                for v in weather_data['volcano_activity']:
                                    label = v.get('name', 'Volcanic activity')
                                    dist = v.get('distance_km')
                                    suffix = f" — {dist} km away" if dist is not None else ""
                                    with st.expander(f"{label}{suffix}"):
                                        st.write(f"Magnitude: {v.get('magnitude', 'n/a')}")
                                        st.write(f"Time: {v.get('time', 'n/a')}")

                            # Add Emergency Resources section here
                            st.markdown("### 🚑 Emergency Resources")

                            # Create tabs for different types of emergency resources
                            resource_types = ['hospital', 'police', 'fire_station', 'shelter']
                            tabs = st.tabs([resource.replace('_', ' ').title() for resource in resource_types])

                            # Group resources by type
                            resources_by_type = {}
                            for resource in weather_data.get('emergency_resources', []):
                                resource_type = resource.get('type', 'other')
                                if resource_type not in resources_by_type:
                                    resources_by_type[resource_type] = []
                                resources_by_type[resource_type].append(resource)

                            # Display resources in respective tabs
                            for tab, resource_type in zip(tabs, resource_types):
                                with tab:
                                    resources = resources_by_type.get(resource_type, [])
                                    if resources:
                                        for resource in resources:
                                            with st.expander(f"📍 {resource['name']}"):
                                                st.write(f"**Address:** {resource['address']}")
                                                if 'location' in resource:
                                                    st.write(f"**Coordinates:** Lat {resource['location']['lat']}, Lng {resource['location']['lng']}")

                                                # Create a map for this resource
                                                map_data = pd.DataFrame({
                                                    'lat': [resource['location']['lat']],
                                                    'lon': [resource['location']['lng']]
                                                })
                                                st.map(map_data)
                                    else:
                                        st.info(f"No {resource_type.replace('_', ' ')} facilities found nearby.")

                            # Add disclaimer
                            st.caption("⚠️ Emergency resource information is provided for reference only. In case of emergency, always call your local emergency number (e.g., 911 in the US).")

                            # One call only: this used to be invoked twice,
                            # with the first stream created, billed and discarded.
                            explanation_stream = st.session_state.explanation_agent.explain_weather_data(
                                weather_data,
                                address
                            )

                            st.markdown("### Detailed Analysis")
                            placeholder = st.empty()
                            full_response = ""
                            for chunk in explanation_stream:
                                # Safety/metadata chunks carry text=None, which
                                # used to raise TypeError mid-stream and lose
                                # the whole briefing.
                                if getattr(chunk, "text", None):
                                    full_response += chunk.text
                                    placeholder.markdown(full_response)
                            if not full_response.strip():
                                placeholder.warning(
                                    "The AI briefing came back empty. The measured data above "
                                    "is unaffected — follow official guidance for any alert shown."
                                )





                    except Exception:
                        logger.exception("Could not render weather data")
                        st.warning(
                            "Some data could not be displayed for this location. "
                            "Any alerts shown above are still valid."
                        )

                except Exception:
                    logger.exception("Analysis failed for address %r", address)
                    st.error(
                        "We could not complete the analysis for that address. "
                        "Please check the address and try again. If an emergency "
                        "is in progress, contact your local emergency services."
                    )

if __name__ == "__main__":
    main()


# Try any address or enter:

# Datah Village in Bali
# 6Q9M+CCC, Pelangai, Ranah Pesisir, South Pesisir Regency, West Sumatra 25666, Indonesia
