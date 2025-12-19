import requests
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_client import BaseWeatherClient, WeatherData, DataSource


class OpenWeatherClient(BaseWeatherClient):
    """
    OpenWeather API client.

    Responsibilities:
    - Fetch current weather data from OpenWeather
    - Normalize response fields into WeatherData
    - Handle API errors without raising exceptions upstream
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._is_available = bool(
            api_key and api_key != "your_actual_openweather_api_key_here"
        )

    @property
    def source_name(self) -> DataSource:
        """Identifier used by the aggregation/consensus layer."""
        return DataSource.OPENWEATHER

    @property
    def is_available(self) -> bool:
        """Indicates whether this client is usable based on configuration."""
        return self._is_available

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_weather(self, location: str) -> WeatherData:
        """
        Fetch current weather for a given location.

        Uses imperial units to align with other providers.
        Retries are applied for transient request failures.
        """
        if not self.is_available:
            return self._error_result(
                location,
                "OpenWeather API key not configured",
            )

        try:
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "imperial",
            }

            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=10,
            )
            response.raise_for_status()

            return self._parse_response(response.json(), location)

        except requests.exceptions.RequestException as e:
            return self._error_result(
                location,
                f"API request failed: {e}",
            )
        except Exception as e:
            return self._error_result(
                location,
                f"Unexpected error: {e}",
            )

    def _parse_response(
        self,
        raw_data: Dict[str, Any],
        location: str,
    ) -> WeatherData:
        """
        Convert OpenWeather response JSON into WeatherData.

        Notes:
        - Precipitation is optional and only included if present
        - Visibility is reported in meters
        """
        main = raw_data.get("main", {})
        weather = raw_data.get("weather", [{}])[0]
        wind = raw_data.get("wind", {})
        sys = raw_data.get("sys", {})

        precipitation = self._extract_precipitation(raw_data)

        return WeatherData(
            source=self.source_name,
            location=raw_data.get("name", location),
            country=sys.get("country", ""),
            temperature=main.get("temp", 0),
            feels_like=main.get(
                "feels_like",
                main.get("temp", 0),
            ),
            temp_min=main.get(
                "temp_min",
                main.get("temp", 0),
            ),
            temp_max=main.get(
                "temp_max",
                main.get("temp", 0),
            ),
            humidity=main.get("humidity", 0),
            pressure=main.get("pressure", 0),
            wind_speed=wind.get("speed", 0),
            wind_direction=wind.get("deg", 0),
            wind_gust=wind.get("gust"),
            description=weather.get("description", "").title(),
            conditions=weather.get("main", ""),
            visibility=raw_data.get("visibility"),
            cloud_cover=raw_data.get("clouds", {}).get("all"),
            precipitation=precipitation,
            sunrise=sys.get("sunrise"),
            sunset=sys.get("sunset"),
            timestamp=raw_data.get("dt"),
            raw_data=raw_data,
        )

    def _extract_precipitation(self, raw_data: Dict[str, Any]) -> float:
        """
        Extract hourly precipitation if present.

        OpenWeather only includes rain/snow fields when applicable.
        Values are in millimeters for the past hour.
        """
        if "rain" in raw_data:
            return raw_data["rain"].get("1h", 0.0)
        if "snow" in raw_data:
            return raw_data["snow"].get("1h", 0.0)
        return 0.0

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
