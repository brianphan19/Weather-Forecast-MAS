import math
from typing import Dict, List, Optional
import statistics
from dataclasses import dataclass
import time

from config.settings import Config

from weather_client import (
    BaseWeatherClient, OpenWeatherClient, WeatherAPIClient, 
    VisualCrossingClient, DataSource, WeatherData
)


@dataclass
class SourceResult:
    """Result from a single source"""
    source: DataSource
    data: Optional['WeatherData']
    response_time: float  # seconds
    success: bool
    error: Optional[str] = None


@dataclass
class ConsensusData:
    """Consensus data from multiple sources"""
    temperature: float                 # degrees Fahrenheit
    feels_like: float                  # degrees Fahrenheit
    temp_min: float                    # degrees Fahrenheit
    temp_max: float                    # degrees Fahrenheit
    humidity: float                    # percent
    pressure: float                    # hpa
    wind_speed: float                  # mph
    wind_direction: float              # Degree
    wind_gust: Optional[float]         # mph
    description: str
    conditions: str
    cloud_cover: Optional[float]       # percent
    precipitation: Optional[float]     # inches
    uv_index: Optional[float]           
    location: str
    country: str
    sources_used: int
    total_sources: int
    confidence: float  
    disagreements: List[str]
    source_details: Dict[DataSource, 'WeatherData']


class DataAcquisitonAgent:
    """
    Multi-source weather data acquisition agent.

    Responsibilities:
    - Initialize available weather API clients
    - Fetch data from all sources
    - Aggregate results into a consensus forecast
    - Compute confidence scores and detect disagreements
    """
    def __init__(self, config: Config):
        self.config = config
        self.clients = self._initialize_clients(config)
        self.source_weights = self._get_source_weights()
    
    def _initialize_clients(self, config: Config) -> Dict[DataSource, BaseWeatherClient]:
        """
        Initialize the agent with API keys and default location.

        Args:
            config (Config): Configuration containing API keys and default location.
        """
        clients = {}
        
        clients[DataSource.OPENWEATHER] = OpenWeatherClient(config.openweather_api_key)
        if config.weatherapi_key:
            clients[DataSource.WEATHERAPI] = WeatherAPIClient(config.weatherapi_key)
        
        if config.visual_crossing_api_key:
            clients[DataSource.VISUAL_CROSSING] = VisualCrossingClient(config.visual_crossing_api_key)
                
        print(f"MultiSourceDataAgent initialized with {len(clients)} sources:")
        for source, client in clients.items():
            status = "Available" if client.is_available else "Not available"
            print(f"   - {source.value}: {status}")
        
        return clients
    
    def _get_source_weights(self) -> Dict[DataSource, float]:
        """
        Assign and normalize reliability weights for available sources.

        Returns:
            Dict[DataSource, float]: Normalized weights summing to 1.0.
        """       
        weights = {
            DataSource.OPENWEATHER: 0.35,
            DataSource.WEATHERAPI: 0.30,
            DataSource.VISUAL_CROSSING: 0.25,
            DataSource.ACCUWEATHER: 0.10,
        }
        
        available_sources = [s for s, c in self.clients.items() if c.is_available]
        if not available_sources:
            return {}
        
        total_weight = sum(weights.get(s, 0.1) for s in available_sources)
        normalized_weights = {
            s: weights.get(s, 0.1) / total_weight 
            for s in available_sources
        }
        
        return normalized_weights
    
    def fetch_all_sources(self, location: Optional[str] = None) -> Dict[DataSource, SourceResult]:
        """
        Fetch weather data from all available sources concurrently
        
        Args:
            location: Optional location string. Uses default if not provided
            
        Returns:
            Dictionary mapping source to result
        """
        target_location = location or self.config.default_location
        results = {}
        
        print(f"   Fetching weather data for: {target_location}")
        print(f"   Using {len([c for c in self.clients.values() if c.is_available])} available sources")
        
        for source, client in self.clients.items():
            if not client.is_available:
                continue
            
            start_time = time.time()
            try:
                weather_data = client.get_weather(target_location)
                response_time = time.time() - start_time
                
                if weather_data.error:
                    results[source] = SourceResult(
                        source=source,
                        data=None,
                        response_time=response_time,
                        success=False,
                        error=weather_data.error
                    )
                    print(f"   {source.value}: Failed - {weather_data.error[:50]}...")
                else:
                    results[source] = SourceResult(
                        source=source,
                        data=weather_data,
                        response_time=response_time,
                        success=True
                    )
                    print(f"   {source.value}: {weather_data.temperature:.1f}°F ({response_time:.1f}s)")
                    
            except Exception as e:
                response_time = time.time() - start_time
                results[source] = SourceResult(
                    source=source,
                    data=None,
                    response_time=response_time,
                    success=False,
                    error=str(e)[:100]
                )
                print(f"   {source.value}: Error - {str(e)[:50]}...")
        
        return results
    
    def calculate_consensus(self, all_results: Dict[DataSource, SourceResult]) -> ConsensusData:
        """
        Calculate consensus from multiple sources
        
        Args:
            all_results: Results from all sources
            
        Returns:
            ConsensusData with weighted averages and confidence scores
        """
        successful_results = {
            source: result.data 
            for source, result in all_results.items() 
            if result.success and result.data
        }
        
        if not successful_results:
            raise ValueError("No successful weather data sources")
        
        temperatures = []
        feels_likes = []
        humidities = []
        pressures = []
        wind_speeds = []
        wind_directions = []
        descriptions = []
        conditions = []
        
        source_details = {}
        
        for source, data in successful_results.items():
            source_details[source] = data
            
            temperatures.append(data.temperature)
            feels_likes.append(data.feels_like)
            humidities.append(data.humidity)
            pressures.append(data.pressure)
            wind_speeds.append(data.wind_speed)
            wind_directions.append(data.wind_direction)
            descriptions.append(data.description)
            conditions.append(data.conditions)
        
        weighted_temp = self._weighted_average(temperatures, successful_results.keys())
        weighted_feels_like = self._weighted_average(feels_likes, successful_results.keys())
        weighted_humidity = self._weighted_average(humidities, successful_results.keys())
        weighted_pressure = self._weighted_average(pressures, successful_results.keys())
        weighted_wind_speed = self._weighted_average(wind_speeds, successful_results.keys())
        
        weighted_wind_dir = self._circular_mean(wind_directions)
        
        most_common_desc = self._most_common(descriptions)
        most_common_cond = self._most_common(conditions)
        
        confidence = self._calculate_confidence(temperatures, wind_speeds, humidities)
        
        disagreements = self._check_disagreements(successful_results)
        
        primary_source = DataSource.OPENWEATHER if DataSource.OPENWEATHER in successful_results else list(successful_results.keys())[0]
        primary_data = successful_results[primary_source]
        
        return ConsensusData(
            temperature=weighted_temp,
            feels_like=weighted_feels_like,
            temp_min=min(temperatures) if temperatures else 0,
            temp_max=max(temperatures) if temperatures else 0,
            humidity=weighted_humidity,
            pressure=weighted_pressure,
            wind_speed=weighted_wind_speed,
            wind_direction=weighted_wind_dir,
            wind_gust=self._weighted_average(
                [d.wind_gust for d in successful_results.values() if d.wind_gust is not None],
                successful_results.keys()
            ),
            description=most_common_desc,
            conditions=most_common_cond,
            cloud_cover=self._weighted_average(
                [d.cloud_cover for d in successful_results.values() if d.cloud_cover is not None],
                successful_results.keys()
            ),
            precipitation=self._weighted_average(
                [d.precipitation for d in successful_results.values() if d.precipitation is not None],
                successful_results.keys()
            ),
            uv_index=self._weighted_average(
                [d.uv_index for d in successful_results.values() if d.uv_index is not None],
                successful_results.keys()
            ),
            location=primary_data.location,
            country=primary_data.country,
            sources_used=len(successful_results),
            total_sources=len(self.clients),
            confidence=confidence,
            disagreements=disagreements,
            source_details=source_details
        )
    
    def _weighted_average(self, values: List[float], sources: List[DataSource]) -> float:
        """
        Compute weighted average of numeric values based on source reliability.

        Args:
            values (List[float]): List of numeric values.
            sources (List[DataSource]): Corresponding data sources.

        Returns:
            float: Weighted average or simple mean if weights are missing.
        """        
        if not values:
            return 0.0
        
        valid_values = []
        valid_sources = []
        for val, src in zip(values, sources):
            if val is not None:
                valid_values.append(val)
                valid_sources.append(src)
        
        if not valid_values:
            return 0.0
        
        if all(src in self.source_weights for src in valid_sources):
            total = sum(val * self.source_weights[src] for val, src in zip(valid_values, valid_sources))
            weight_sum = sum(self.source_weights[src] for src in valid_sources)
            return total / weight_sum if weight_sum > 0 else statistics.mean(valid_values)
        
        return statistics.mean(valid_values)
    
    def _circular_mean(self, angles: List[float]) -> float:
        """
        Compute mean of circular data (wind directions in degrees).

        Args:
            angles (List[float]): List of angles (0-360°).

        Returns:
            float: Circular mean, normalized to 0-360°.
        """
        if not angles:
            return 0.0
        
        angles_rad = [angle * 3.14159 / 180 for angle in angles]
        
        sin_sum = sum([math.sin(a) for a in angles_rad])
        cos_sum = sum([math.cos(a) for a in angles_rad])
        
        mean_rad = math.atan2(sin_sum / len(angles), cos_sum / len(angles))
        mean_deg = mean_rad * 180 / 3.14159
        
        return mean_deg % 360
    
    def _most_common(self, items: List[str]) -> str:
        """
        Find the most common string in a list.

        Args:
            items (List[str]): List of strings.

        Returns:
            str: Most frequently occurring string, or "Unknown" if empty.
        """
        if not items:
            return "Unknown"
        
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(1)[0][0]
    
    def _calculate_confidence(self, temperatures: List[float], 
                            wind_speeds: List[float], 
                            humidities: List[float]) -> float:
        """
        Compute confidence score based on agreement between sources.

        Args:
            temperatures (List[float]): Temperatures reported by sources.
            wind_speeds (List[float]): Wind speeds reported by sources.
            humidities (List[float]): Humidities reported by sources.

        Returns:
            float: Confidence score (0.0–1.0).
        """
        if len(temperatures) < 2:
            return 0.5  
        
        def cv(values):
            if not values:
                return 0
            mean = statistics.mean(values)
            if mean == 0:
                return 0
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            return stdev / mean
        
        temp_cv = cv(temperatures)
        wind_cv = cv(wind_speeds)
        humidity_cv = cv(humidities)

        temp_confidence = max(0, 1 - temp_cv * 10) 
        wind_confidence = max(0, 1 - wind_cv * 5)   
        humidity_confidence = max(0, 1 - humidity_cv * 2)  

        confidence = (temp_confidence * 0.4 + wind_confidence * 0.3 + humidity_confidence * 0.3)
        
        source_factor = min(1.0, len(temperatures) / 3)  
        confidence *= source_factor
        
        return min(1.0, max(0.0, confidence))
    
    def _check_disagreements(self, successful_results: Dict[DataSource, 'WeatherData']) -> List[str]:
        """
        Detect significant disagreements between sources (temperature and conditions).

        Args:
            successful_results (Dict[DataSource, WeatherData]): Mapping of sources 
        to successful weather data.

        Returns:
            List[str]: List of disagreement messages.
        """
        disagreements = []
        
        if len(successful_results) < 2:
            return disagreements
        
        # Check temperature disagreement (> 5°F difference)
        temps = [data.temperature for data in successful_results.values()]
        if max(temps) - min(temps) > 5:
            temp_str = ", ".join([f"{src.value}: {data.temperature:.1f}°F" 
                                 for src, data in successful_results.items()])
            disagreements.append(f"Temperature disagreement: {temp_str}")
        
        # Check condition disagreement
        conditions = [data.conditions for data in successful_results.values()]
        if len(set(conditions)) > 1:
            cond_str = ", ".join([f"{src.value}: {data.conditions}" 
                                 for src, data in successful_results.items()])
            disagreements.append(f"Condition disagreement: {cond_str}")
        
        return disagreements
    
    def get_weather_summary(self, location: Optional[str] = None) -> dict:
        """
        Get comprehensive weather summary from all sources
        
        Args:
            location: Optional location string
            
        Returns:
            Dictionary with consensus data and source details
        """
        all_results = self.fetch_all_sources(location)
        
        try:
            consensus = self.calculate_consensus(all_results)
            
            source_details = {}
            for source, result in all_results.items():
                if result.success and result.data:
                    source_details[source.value] = {
                        'temperature': f"{result.data.temperature:.1f}°F",
                        'humidity': f"{result.data.humidity}%",
                        'wind_speed': f"{result.data.wind_speed:.1f} mph",
                        'conditions': result.data.conditions,
                        'response_time': f"{result.response_time:.2f}s"
                    }
                else:
                    source_details[source.value] = {
                        'error': result.error,
                        'response_time': f"{result.response_time:.2f}s"
                    }
            
            return {
                'consensus': {
                    'location': consensus.location,
                    'country': consensus.country,
                    'temperature': f"{consensus.temperature:.1f}°F",
                    'feels_like': f"{consensus.feels_like:.1f}°F",
                    'temp_range': f"{consensus.temp_min:.1f}°F - {consensus.temp_max:.1f}°F",
                    'humidity': f"{consensus.humidity:.0f}%",
                    'wind_speed': f"{consensus.wind_speed:.1f} mph",
                    'wind_direction': f"{consensus.wind_direction:.0f}°",
                    'conditions': consensus.conditions,
                    'description': consensus.description,
                    'sources_used': consensus.sources_used,
                    'total_sources': consensus.total_sources,
                    'confidence': f"{consensus.confidence:.0%}",
                    'disagreements': consensus.disagreements
                },
                'source_details': source_details,
                'success': True
            }
            
        except ValueError as e:
            return {
                'success': False,
                'error': str(e),
                'source_details': {
                    source.value: {
                        'error': result.error if result else 'No data',
                        'response_time': result.response_time if result else 0
                    }
                    for source, result in all_results.items()
                }
            }