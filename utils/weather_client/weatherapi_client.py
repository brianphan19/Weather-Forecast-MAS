import requests
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_client import BaseWeatherClient, WeatherData, DataSource


class WeatherAPIClient(BaseWeatherClient):
    """
    WeatherAPI.com client.

    Responsibilities:
    - Fetch current weather data from WeatherAPI
    - Normalize units and fields into the shared WeatherData schema
    - Gracefully handle missing API keys and request failures
    """

    BASE_URL = "http://api.weatherapi.com/v1/current.json"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._is_available = bool(api_key)

    @property
    def source_name(self) -> DataSource:
        """Identifier used by the aggregation/consensus layer."""
        return DataSource.WEATHERAPI

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
        Fetch current weather for a given location string.

        Retries are handled by tenacity for transient failures.
        All errors are returned as WeatherData with the error field populated,
        allowing the caller to continue aggregating other sources.
        """
        if not self.is_available:
            return self._error_result(
                location,
                "WeatherAPI key not configured",
            )

        try:
            params = {
                "key": self.api_key,
                "q": location,
                "aqi": "no",
            }

            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=10,
            )
            response.raise_for_status()

            return self._parse_response(response.json(), location)

        except Exception as e:
            return self._error_result(
                location,
                f"WeatherAPI request failed: {e}",
            )

    def _parse_response(
        self,
        raw_data: Dict[str, Any],
        location: str,
    ) -> WeatherData:
        """
        Convert WeatherAPI response JSON into WeatherData.

        Notes:
        - WeatherAPI returns imperial units directly (temp_f)
        - Wind is converted from km/h to mph for consistency
        - Some fields (min/max temp) are not available for current conditions
        """
        current = raw_data.get("current", {})
        location_info = raw_data.get("location", {})

        # Unit conversions
        wind_mph = current.get("wind_kph", 0) * 0.621371
        precip_inches = current.get("precip_mm", 0) * 0.0393701
        visibility_m = (
            current.get("vis_km", 0) * 1000
            if current.get("vis_km") is not None
            else None
        )

        return WeatherData(
            source=self.source_name,
            location=location_info.get("name", location),
            country=location_info.get("country", ""),
            temperature=current.get("temp_f", 0),
            feels_like=current.get(
                "feelslike_f",
                current.get("temp_f", 0),
            ),
            temp_min=current.get("temp_f", 0),
            temp_max=current.get("temp_f", 0),
            humidity=current.get("humidity", 0),
            pressure=current.get("pressure_mb", 0),
            wind_speed=wind_mph,
            wind_direction=current.get("wind_degree", 0),
            wind_gust=(
                current.get("gust_kph", 0) * 0.621371
                if current.get("gust_kph") is not None
                else None
            ),
            description=current.get("condition", {}).get("text", ""),
            conditions=self._map_condition(
                current.get("condition", {}).get("code", 1000)
            ),
            visibility=visibility_m,
            cloud_cover=current.get("cloud", 0),
            precipitation=precip_inches,
            uv_index=current.get("uv", 0),
            raw_data=raw_data,
        )

    def _map_condition(self, code: int) -> str:
        """
        Map WeatherAPI condition codes into coarse-grained condition buckets.

        This normalization allows multiple providers to be compared consistently
        during consensus analysis.
        """
        if code in {1000, 1003}:
            return "Clear"

        if code in {1006, 1009, 1030, 1135, 1147}:
            return "Clouds"

        if code in {
            1063, 1066, 1069, 1072, 1087, 1114, 1117,
            1150, 1153, 1168, 1171,
            1180, 1183, 1186, 1189,
            1192, 1195, 1198, 1201,
            1204, 1207, 1210, 1213,
            1216, 1219, 1222, 1225,
            1237, 1240, 1243, 1246,
            1249, 1252, 1255, 1258,
            1261, 1264,
        }:
            return "Rain" if code < 1200 else "Snow"

        if code in {1087, 1273, 1276, 1279, 1282}:
            return "Thunderstorm"

        return "Unknown"

    def _error_result(self, location: str, message: str) -> WeatherData:
        """
        Construct a standardized error WeatherData object.

        Centralizing this avoids duplication and keeps error handling consistent
        across all code paths.
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
