from typing import Any, Dict

import requests


class OpenWeatherClient:
    """Client for OpenWeather API"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str, temperature_unit='imperial'):
        if not api_key or api_key == 'your_actual_openweather_api_key_here':
            raise ValueError("Please set a valid OPENWEATHER_API_KEY in your .env file")
        self.api_key = api_key
        self.temperature_unit = temperature_unit
    
    def get_weather_by_location(self, location: str) -> Dict[str, Any]:
        """
        Get weather data for a location (city name, state, country)
        
        Args:
            location: City name (e.g., "New York, NY, USA")
            
        Returns:
            Dictionary containing weather data
        """
        params = {
            'q': location,
            'appid': self.api_key,
            'units': self.temperature_unit
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {}

