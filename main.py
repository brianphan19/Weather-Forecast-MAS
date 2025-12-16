from typing import Any, Dict
from agents.data_acquisition import DataAcquisitionAgent
from config.settings import Config 


def main():
    """Main function to test the weather agent"""

    # Load configuration
    print("🔧 Loading configuration...")
    config = Config.from_env()

    # Create the data acquisition agent
    print("🤖 Initializing Weather Data Acquisition Agent...")
    fetcher_agent = DataAcquisitionAgent(config)
    
    # Test with default location
    print("\n📡 Testing with default location...")
    weather_data = fetcher_agent.fetch_weather_data()
    print_weather_summary(weather_data)
    
def print_weather_summary(weather_data: Dict[str, Any]):
    """Print a formatted weather summary"""
    if not weather_data:
        print("No weather data available")
        return
    
    print("\n" + "="*50)
    print("WEATHER SUMMARY")
    print("="*50)
    print(f"    Location: {weather_data.get('location', 'Unknown')}, {weather_data.get('country', 'Unknown')}")
    print(f"    Temperature: {weather_data.get('temperature', 'N/A')}°F")
    print(f"    Feels like: {weather_data.get('feels_like', 'N/A')}°F")
    print(f"    Min/Max: {weather_data.get('temp_min', 'N/A')}°F / {weather_data.get('temp_max', 'N/A')}°F")
    print(f"    Humidity: {weather_data.get('humidity', 'N/A')}%")
    print(f"    Wind: {weather_data.get('wind_speed', 'N/A')} mph")
    print(f"    Conditions: {weather_data.get('description', 'N/A')}")
    print(f"    Visibility: {weather_data.get('visibility', 'N/A')} meters")
    print("="*50)


if __name__ == "__main__":
    main()