# agents/orchestrator.py
"""
Main orchestrator for the Weather Forecast Multi-Agent System (MAS).

This module coordinates agent execution using LangGraph, manages shared
state initialization, executes the workflow, and formats final results
for external consumption.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import uuid

from langgraph.graph import StateGraph

from workflows.state import AgentState, AgentStatus
from workflows.weather_workflow import WeatherForecastWorkflow
from config.settings import Config


class WeatherMASOrchestrator:
    """
    Coordinates execution of the Weather Forecast MAS.

    Responsibilities:
    - Initialize the LangGraph workflow
    - Create and manage the initial agent state
    - Execute the multi-agent workflow
    - Format final results and error responses
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the orchestrator.

        Args:
            config (Optional[Config]): Application configuration.
                                       Loaded from environment if omitted.
        """
        self.config = config or Config.from_env()
        self.workflow_builder = WeatherForecastWorkflow(self.config)
        self.graph: Optional[StateGraph] = None

    async def initialize(self) -> None:
        """
        Build and initialize the LangGraph workflow.
        """
        print("Initializing Weather Forecast MAS workflow...")
        self.graph = self.workflow_builder._build_graph()
        print("Workflow initialized")

    async def get_weather_forecast(self, location: str) -> Dict[str, Any]:
        """
        Execute the complete weather forecast workflow for a location.

        Args:
            location (str): Location query string.

        Returns:
            Dict[str, Any]: Formatted weather forecast results or error response.
        """
        if not self.graph:
            await self.initialize()

        initial_state = self._create_initial_state(location)

        print(f"\nStarting weather forecast for: {location}")
        print(f"Request ID: {initial_state['request_id']}")

        start_time = datetime.now()

        try:
            print("\nExecuting LangGraph workflow...")
            print("-" * 40)

            final_state = await self.graph.ainvoke(initial_state)

            execution_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            final_state["execution_time_ms"] = int(execution_time)

            results = self._format_results(final_state)

            consensus = final_state.get("weather_consensus")
            sources_count = consensus.sources_count if consensus else 0
            alerts_count = len(final_state.get("weather_alerts", []))

            print("=" * 40)
            print(f"Workflow completed in {execution_time:.0f} ms")
            print(f"Sources used: {sources_count}")
            print(f"Alerts generated: {alerts_count}")

            return results

        except Exception as exc:
            print(f"\nWorkflow execution failed: {exc}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(initial_state, str(exc))

    def _create_initial_state(self, location: str) -> AgentState:
        """
        Create the initial workflow state.

        Args:
            location (str): Location query string.

        Returns:
            AgentState: Initialized agent state dictionary.
        """
        return {
            "location": location,
            "request_id": str(uuid.uuid4())[:8],
            "agent1_status": AgentStatus.PENDING,
            "agent2_status": AgentStatus.PENDING,
            "raw_weather_data": [],
            "weather_consensus": None,
            "weather_alerts": [],
            "analysis_summary": None,
            "detailed_insights": None,
            "timestamp": datetime.now().isoformat(),
            "errors": [],
            "execution_time_ms": None,
            "llm_analysis": None,
            "config": {
                "temperature_unit": self.config.weather.temp_unit,
                "alert_thresholds": {
                    "temp_high": self.config.weather.temp_alert_high,
                    "temp_low": self.config.weather.temp_alert_low,
                    "wind": self.config.weather.wind_alert,
                    "precip": self.config.weather.precip_alert,
                },
            },
        }

    def _format_results(self, state: AgentState) -> Dict[str, Any]:
        """
        Convert final workflow state into a structured response.

        Args:
            state (AgentState): Final workflow state.

        Returns:
            Dict[str, Any]: API-ready response payload.
        """
        agent1_success = state.get("agent1_status") == AgentStatus.COMPLETED

        if not agent1_success or not state.get("raw_weather_data"):
            return {
                "success": False,
                "request_id": state["request_id"],
                "errors": state["errors"],
                "agent1_status": state.get("agent1_status", "unknown"),
                "agent2_status": state.get("agent2_status", "unknown"),
                "message": "Failed to collect weather data",
            }

        raw_data = state.get("raw_weather_data", [])
        insights = state.get("detailed_insights", {})
        alerts = state.get("weather_alerts", [])
        consensus = state.get("weather_consensus")

        sources = {}
        for data in raw_data:
            if data.source not in sources:
                sources[data.source] = {
                    "temperature": data.temperature,
                    "conditions": data.conditions,
                    "confidence": (
                        "high" if not getattr(data, "error", None) else "low"
                    ),
                }

        if insights:
            weather_summary = {
                "temperature": insights.get("temperature", {}),
                "humidity": insights.get("humidity", {}),
                "wind": insights.get("wind", {}),
                "conditions": insights.get("conditions", {}),
                "comfort": insights.get("comfort", {}),
                "consistency": insights.get("consistency", {}),
                "severity": insights.get("severity", "normal"),
            }
        elif consensus:
            weather_summary = {
                "temperature": {
                    "avg": consensus.temperature_avg,
                    "min": consensus.temperature_min,
                    "max": consensus.temperature_max,
                    "unit": "°F",
                },
                "humidity": {
                    "avg": consensus.humidity_avg,
                    "unit": "%",
                },
                "wind": {
                    "avg_speed": consensus.wind_speed_avg,
                    "unit": "mph",
                },
                "conditions": {
                    "primary": consensus.conditions_consensus,
                    "confidence": f"{consensus.confidence_score:.0%}",
                },
            }
        else:
            weather_summary = {}

        return {
            "success": True,
            "workflow_status": {
                "agent1": state.get("agent1_status", "unknown"),
                "agent2": state.get("agent2_status", "unknown"),
                "execution_time_ms": state.get("execution_time_ms"),
            },
            "request_id": state["request_id"],
            "location": state["location"],
            "timestamp": state["timestamp"],
            "data_sources": {
                "used": len(raw_data),
                "available": len(
                    self.workflow_builder.agent1.get_available_clients()
                ),
                "details": list(sources.keys()),
            },
            "weather_summary": weather_summary,
            "alerts": alerts,
            "analysis": state.get("analysis_summary"),
            "raw_data_preview": [
                {
                    "source": d.source,
                    "temperature": d.temperature,
                    "conditions": d.conditions,
                    "humidity": d.humidity,
                    "wind_speed": d.wind_speed,
                }
                for d in raw_data[:3]
            ],
            "errors": state["errors"] or None,
        }

    def _create_error_response(
        self, state: AgentState, error: str
    ) -> Dict[str, Any]:
        """
        Build a standardized error response.

        Args:
            state (AgentState): Current workflow state.
            error (str): Error message.

        Returns:
            Dict[str, Any]: Error response payload.
        """
        return {
            "success": False,
            "request_id": state["request_id"],
            "location": state["location"],
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "errors": state["errors"] + [error],
            "workflow_status": {
                "agent1": state.get("agent1_status", "unknown"),
                "agent2": state.get("agent2_status", "unknown"),
            },
        }
