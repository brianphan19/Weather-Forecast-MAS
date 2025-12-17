from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class DataSource(Enum):
    """
    Supported weather data providers.

    Values are stable identifiers used throughout the system
    (logging, aggregation, reporting).
    """
    OPENWEATHER = "openweather"
    WEATHERAPI = "weatherapi"
    VISUAL_CROSSING = "visualcrossing"
    ACCUWEATHER = "accuweather"
    NOAA = "noaa"


@dataclass
class WeatherData:
    """
    Standardized weather data model shared across all providers.

    Design goals:
    - Single schema for aggregation and consensus logic
    - Provider-specific quirks normalized at the client level
    - Missing or failed data represented via the `error` field
    """

    source: DataSource
    location: str
    country: str

    temperature: float          # degrees Fahrenheit
    feels_like: float           # degrees Fahrenheit
    temp_min: float             # degrees Fahrenheit
    temp_max: float             # degrees Fahrenheit

    humidity: int               # percent
    pressure: int               # hPa
    wind_speed: float           # mph
    wind_direction: int         # degrees (0–360)

    wind_gust: Optional[float] = None      # mph
    description: str = ""
    conditions: str = ""                   # normalized condition bucket
    visibility: Optional[int] = None       # meters
    cloud_cover: Optional[int] = None      # percent
    precipitation: Optional[float] = None  # inches
    uv_index: Optional[float] = None

    sunrise: Optional[int] = None           # unix timestamp
    sunset: Optional[int] = None            # unix timestamp
    timestamp: Optional[int] = None         # observation time (unix)

    raw_data: Optional[Dict] = None         # original provider payload
    error: Optional[str] = None             # populated on failure


class BaseWeatherClient(ABC):
    """
    Abstract base class for all weather API clients.

    Each concrete client is responsible for:
    - Fetching data from its provider
    - Converting provider-specific fields into WeatherData
    - Never raising provider errors upward (return error-filled WeatherData)
    """

    @abstractmethod
    def get_weather(self, location: str) -> WeatherData:
        """
        Fetch weather data for a given location.

        Implementations must return a WeatherData instance even on failure.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_name(self) -> DataSource:
        """Unique identifier for the data source."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        Indicates whether the client can be used.

        Typically depends on API key availability or configuration.
        """
        raise NotImplementedError

    def _degrees_to_cardinal(self, degrees: Optional[float]) -> str:
        """
        Convert wind direction in degrees to a cardinal direction.

        Used for presentation-layer convenience.
        """
        if degrees is None:
            return "Unknown"

        directions = [
            "N", "NNE", "NE", "ENE",
            "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW",
            "W", "WNW", "NW", "NNW",
        ]
        index = round(degrees / (360.0 / len(directions))) % len(directions)
        return directions[index]
