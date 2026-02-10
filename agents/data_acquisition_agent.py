# agents/agent1_data_collector.py
"""
Agent 1: Weather Data Collection Agent.

This agent is responsible for fetching weather data from all configured
external providers, normalizing the results, and producing a consensus
summary to be stored in the shared workflow state.

The agent is designed to integrate with LangGraph state management.
"""

import asyncio
import statistics
from datetime import datetime
from typing import List, Optional

from langgraph.graph import END, StateGraph

from config.settings import Config
from utils.weather_client import (
    BaseWeatherClient,
    OpenWeatherClient,
    WeatherAPIClient,
    VisualCrossingClient,
    WeatherData,
)
from workflows.state import (
    AgentState,
    WeatherSourceData,
    AgentStatus,
    WeatherConsensus,
)


class DataCollectorAgent:
    """
    Collects weather data from all available external providers.

    Responsibilities:
    - Initialize configured weather API clients
    - Fetch data concurrently from all available sources
    - Normalize raw responses into internal data structures
    - Compute a basic consensus across sources
    - Update the shared agent state
    """

    def __init__(self, config: Config):
        """
        Initialize the data collection agent.

        Args:
            config (Config): Application configuration loaded from environment.
        """
        self.config = config
        self.clients: List[BaseWeatherClient] = []
        self.initialize_clients()

    def initialize_clients(self) -> None:
        """
        Initialize all configured weather API clients based on available API keys.
        """
        weather_config = self.config.weather

        if weather_config.openweather_api_key:
            self.clients.append(
                OpenWeatherClient(weather_config.openweather_api_key)
            )

        if weather_config.weatherapi_key:
            self.clients.append(
                WeatherAPIClient(weather_config.weatherapi_key)
            )

        if weather_config.visual_crossing_api_key:
            self.clients.append(
                VisualCrossingClient(weather_config.visual_crossing_api_key)
            )

        print(
            f"DataCollectorAgent initialized with "
            f"{len(self.clients)} weather clients"
        )

    def get_available_clients(self) -> List[BaseWeatherClient]:
        """
        Return all initialized clients that are currently available.

        Returns:
            List[BaseWeatherClient]: Active and usable weather clients.
        """
        return [client for client in self.clients if client.is_available]

    async def collect_weather_data(self, state: AgentState) -> AgentState:
        """
        Collect weather data from all available providers and update agent state.

        Args:
            state (AgentState): Shared workflow state.

        Returns:
            AgentState: Updated state containing raw data and consensus results.
        """
        location = state["location"]
        print("=" * 60)
        print(f"Agent 1: Collecting weather data for {location}")

        state["agent1_status"] = AgentStatus.COLLECTING
        state["timestamp"] = datetime.now().isoformat()

        available_clients = self.get_available_clients()

        if not available_clients:
            state["errors"].append("No weather API clients available")
            state["agent1_status"] = AgentStatus.FAILED
            return state

        tasks = [
            asyncio.create_task(
                self._fetch_client_data(client, location)
            )
            for client in available_clients
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        raw_data: List[WeatherSourceData] = []
        successful_sources = 0

        for index, result in enumerate(results):
            client = (
                available_clients[index]
                if index < len(available_clients)
                else None
            )

            if isinstance(result, Exception):
                error_msg = (
                    f"Error from "
                    f"{client.source_name.value if client else 'unknown'}: "
                    f"{result}"
                )
                state["errors"].append(error_msg)
                print(error_msg)
                continue

            if result and not result.error:
                source_data = WeatherSourceData(
                    source=result.source.value,
                    temperature=result.temperature,
                    feels_like=result.feels_like,
                    humidity=result.humidity,
                    pressure=result.pressure,
                    wind_speed=result.wind_speed,
                    wind_direction=result.wind_direction,
                    conditions=result.conditions,
                    description=result.description,
                    timestamp=datetime.now().isoformat(),
                )
                raw_data.append(source_data)
                successful_sources += 1

            elif result and result.error:
                error_msg = f"{result.source.value}: {result.error}"
                state["errors"].append(error_msg)
                print(error_msg)

        state["raw_weather_data"] = raw_data

        if successful_sources > 0:
            state["weather_consensus"] = self._calculate_consensus(raw_data)
            state["agent1_status"] = AgentStatus.COMPLETED
            print(
                f"Agent 1 completed: \n"
                f"  - {successful_sources}/{len(available_clients)} "
                f"sources successful"
            )
        else:
            state["agent1_status"] = AgentStatus.FAILED
            print("Agent 1 failed: no successful data sources")

        return state

    async def _fetch_client_data(
        self, client: BaseWeatherClient, location: str
    ) -> Optional[WeatherData]:
        """
        Fetch weather data from a single client with a timeout guard.

        Args:
            client (BaseWeatherClient): Weather API client.
            location (str): Location query string.

        Returns:
            Optional[WeatherData]: Weather data or error-filled result.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(client.get_weather, location),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            return WeatherData(
                source=client.source_name,
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
                error=f"Timeout fetching data from {client.source_name.value}",
            )
        except Exception as exc:
            return WeatherData(
                source=client.source_name,
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
                error=f"Error: {str(exc)}",
            )

    def _calculate_consensus(
        self, data_list: List[WeatherSourceData]
    ) -> Optional[WeatherConsensus]:
        """
        Compute a consensus summary from multiple weather sources.

        Args:
            data_list (List[WeatherSourceData]): Normalized source data.

        Returns:
            WeatherConsensus | None: Aggregated consensus results.
        """
        if not data_list:
            return None

        temperatures = [d.temperature for d in data_list]
        feels_like = [d.feels_like for d in data_list]
        humidities = [d.humidity for d in data_list]
        pressures = [d.pressure for d in data_list]
        wind_speeds = [d.wind_speed for d in data_list]

        condition_counts = {}
        for data in data_list:
            condition_counts[data.conditions] = (
                condition_counts.get(data.conditions, 0) + 1
            )

        most_common_condition = (
            max(condition_counts, key=condition_counts.get)
            if condition_counts
            else "Unknown"
        )
        confidence = (
            max(condition_counts.values()) / len(data_list)
            if condition_counts
            else 0.0
        )

        return WeatherConsensus(
            temperature_avg=statistics.mean(temperatures),
            temperature_min=min(temperatures),
            temperature_max=max(temperatures),
            feels_like_avg=statistics.mean(feels_like),
            humidity_avg=statistics.mean(humidities),
            pressure_avg=statistics.mean(pressures),
            wind_speed_avg=statistics.mean(wind_speeds),
            conditions_consensus=most_common_condition,
            confidence_score=confidence,
            sources_count=len(data_list),
        )
