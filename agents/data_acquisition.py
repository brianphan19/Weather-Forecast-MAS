from typing import Any, Dict, Optional
from config.settings import Config
from tools.open_weather_client import OpenWeatherClient

class DataAcquisitionAgent:
    """Agent responsible for fetching weather data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.weather_client = OpenWeatherClient(api_key=config.openweather_api_key,temperature_unit=config.temp_unit)
    
    def fetch_weather_data(self, location: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch weather data for a location
        
        Args:
            location: Optional location string. Uses default if not provided
            
        Returns:
            Weather data dictionary
        """
        target_location = location or self.config.default_location
        print(f"🌤️  Fetching weather data for: {target_location}")
        
        weather_data = self.weather_client.get_weather_by_location(target_location)
        
        if not weather_data:
            print("❌ Failed to fetch weather data")
            return {}
        
        return self._extract_relevant_data(weather_data)
    
    def _extract_relevant_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format relevant weather data"""
        try:
            main = raw_data.get('main', {})
            weather = raw_data.get('weather', [{}])[0]
            wind = raw_data.get('wind', {})
            
            return {
                'location': raw_data.get('name', 'Unknown'),
                'country': raw_data.get('sys', {}).get('country', 'Unknown'),
                'temperature': main.get('temp'),
                'feels_like': main.get('feels_like'),
                'temp_min': main.get('temp_min'),
                'temp_max': main.get('temp_max'),
                'humidity': main.get('humidity'),
                'pressure': main.get('pressure'),
                'description': weather.get('description', '').title(),
                'wind_speed': wind.get('speed'),
                'wind_direction': wind.get('deg'),
                'clouds': raw_data.get('clouds', {}).get('all'),
                'visibility': raw_data.get('visibility'),
                'timestamp': raw_data.get('dt'),
                'sunrise': raw_data.get('sys', {}).get('sunrise'),
                'sunset': raw_data.get('sys', {}).get('sunset')
            }
        except (KeyError, IndexError) as e:
            print(f"Error extracting weather data: {e}")
            return {}


