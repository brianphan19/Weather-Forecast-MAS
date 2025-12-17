import requests
from typing import Dict, Any, Optional
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_client import BaseWeatherClient, WeatherData, DataSource


class VisualCrossingClient(BaseWeatherClient):
    """
    Visual Crossing Weather API client.

    Responsibilities:
    - Fetch current weather conditions from Visual Crossing
    - Normalize fields into the shared WeatherData schema
    - Handle missing API keys and request failures gracefully
    """

    BASE_URL = (
        "https://weather.visualcrossing.com/"
        "VisualCrossingWebServices/rest/services/timeline"
    )

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._is_available = bool(api_key)

    @property
    def source_name(self) -> DataSource:
        """Identifier used by the aggregation/consensus layer."""
        return DataSource.VISUAL_CROSSING

    @property
    def is_available(self) -> bool:
        """Indicates whether this client is usable based on configuration."""
        return self._is_available

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5),
    )
    def get_weather(self, location: str) -> WeatherData:
        """
        Fetch current weather for a given location.

        Visual Crossing uses a timeline endpoint; the `currentConditions`
        field is extracted and normalized.
        """
        if not self.is_available:
            return self._error_result(
                location,
                "Visual Crossing API key not configured",
            )

        try:
            params = {
                "key": self.api_key,
                "unitGroup": "us",      # imperial units
                "include": "current",
            }

            url = f"{self.BASE_URL}/{location}"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            return self._parse_response(response.json(), location)

        except Exception as e:
            return self._error_result(
                location,
                f"Visual Crossing request failed: {e}",
            )

    def _parse_response(
        self,
        raw_data: Dict[str, Any],
        location: str,
    ) -> WeatherData:
        """
        Convert Visual Crossing response JSON into WeatherData.

        Notes:
        - Min/max temperatures are not available for current conditions
        - Wind, temperature, and visibility are already in imperial units
        """
        current = raw_data.get("currentConditions", {})
        address = raw_data.get("resolvedAddress", location)

        city = self._extract_city(address)

        return WeatherData(
            source=self.source_name,
            location=city,
            country=self._extract_country(address),
            temperature=current.get("temp", 0),
            feels_like=current.get(
                "feelslike",
                current.get("temp", 0),
            ),
            temp_min=current.get("temp", 0),
            temp_max=current.get("temp", 0),
            humidity=current.get("humidity", 0),
            pressure=current.get("pressure", 0),
            wind_speed=current.get("windspeed", 0),
            wind_direction=current.get("winddir", 0),
            wind_gust=current.get("windgust"),
            description=current.get("conditions", ""),
            conditions=self._map_conditions(
                current.get("conditions", "")
            ),
            visibility=current.get("visibility"),
            cloud_cover=current.get("cloudcover", 0),
            precipitation=current.get("precip", 0),
            uv_index=current.get("uvindex", 0),
            sunrise=self._parse_time(current.get("sunrise")),
            sunset=self._parse_time(current.get("sunset")),
            raw_data=raw_data,
        )

    def _extract_city(self, address: str) -> str:
        """
        Extract city name from Visual Crossing resolved address.

        The API usually returns a comma-separated string.
        """
        parts = address.split(",")
        return parts[0].strip() if parts else address

    def _extract_country(self, address: str) -> str:
        """
        Extract country from resolved address.

        Conventionally the country appears as the last segment.
        """
        parts = address.split(",")
        return parts[-1].strip() if len(parts) > 1 else ""

    def _map_conditions(self, conditions: str) -> str:
        """
        Map Visual Crossing condition strings into normalized categories.

        This enables consistent comparison across providers.
        """
        text = conditions.lower()

        if "clear" in text:
            return "Clear"
        if "partly cloudy" in text:
            return "Clouds"
        if "cloudy" in text or "overcast" in text:
            return "Clouds"
        if "rain" in text or "drizzle" in text:
            return "Rain"
        if "snow" in text or "ice" in text:
            return "Snow"
        if "storm" in text or "thunder" in text:
            return "Thunderstorm"
        if "fog" in text or "mist" in text:
            return "Mist"

        return "Unknown"

    def _parse_time(self, time_str: Optional[str]) -> Optional[int]:
        """
        Parse a Visual Crossing time string (HH:MM:SS) into a Unix timestamp.

        The API provides time-only values, so today's date is assumed.
        """
        if not time_str:
            return None

        try:
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            today = datetime.now().date()
            return int(datetime.combine(today, time_obj).timestamp())
        except ValueError:
            return None

    def _error_result(self, location: str, message: str) -> WeatherData:
        """
        Construct a standardized error WeatherData object.
        """
        return WeatherData(
            source=self.source_name,
            location=location,
            country="",
            temperature=0,
            feels_like=0,
            temp_min=0,
            temp_max=0,
            humidity=0,
            pressure=0,
            wind_speed=0,
            wind_direction=0,
            error=message,
        )
