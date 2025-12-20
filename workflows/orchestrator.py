"""
WeatherMASOrchestrator

Coordinates execution of the 3-agent Weather Forecast workflow using LangGraph.
Handles:
- Workflow initialization
- State creation
- Execution timing
- Rerun tracking
- Final response formatting
"""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from langgraph.graph import StateGraph

from workflows.state import AgentState, AgentStatus
from workflows.weather_workflow import WeatherForecastWorkflow
from config.settings import Config


class WeatherMASOrchestrator:
    """
    Main orchestrator for the Weather Forecast Multi-Agent System.

    Responsible for:
    - Building the LangGraph workflow
    - Executing agent pipelines
    - Managing reruns
    - Formatting final output
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self.workflow_builder = WeatherForecastWorkflow(self.config)
        self.graph: Optional[StateGraph] = None

    async def initialize(self) -> None:
        """
        Initialize the LangGraph workflow graph.
        """
        self.graph = self.workflow_builder.build_workflow()

    async def get_weather_forecast(
        self,
        location: str,
        user_question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the full weather forecast workflow.

        Args:
            location (str): Location to analyze
            user_question (Optional[str]): Optional user question for LLM

        Returns:
            Dict[str, Any]: Formatted workflow response
        """
        if not self.graph:
            await self.initialize()

        state = self._create_initial_state(location, user_question)
        start_time = datetime.now()

        try:
            final_state = await self.graph.ainvoke(state)

            execution_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            final_state["execution_time_ms"] = int(execution_time)

            return self._format_results(final_state)

        except Exception as exc:
            return self._create_error_response(state, str(exc))

    def _create_initial_state(
        self,
        location: str,
        user_question: Optional[str]
    ) -> AgentState:
        """
        Construct the initial workflow state.

        Returns:
            AgentState: Initial state dictionary
        """
        return {
            "location": location,
            "request_id": str(uuid.uuid4())[:8],
            "user_question": (
                user_question
                or "Provide a comprehensive weather analysis and recommendations"
            ),
            "agent1_status": AgentStatus.PENDING,
            "agent2_status": AgentStatus.PENDING,
            "agent3_status": AgentStatus.PENDING,
            "raw_weather_data": [],
            "weather_consensus": None,
            "weather_alerts": [],
            "analysis_summary": None,
            "detailed_insights": None,
            "weather_report": None,
            "llm_response": None,
            "_rerun_count": 0,
            "_is_rerun": False,
            "timestamp": datetime.now().isoformat(),
            "errors": [],
            "execution_time_ms": None,
            "config": {
                "temperature_unit": self.config.weather.temp_unit,
                "alert_thresholds": {
                    "temp_high": self.config.weather.temp_alert_high,
                    "temp_low": self.config.weather.temp_alert_low,
                    "wind": self.config.weather.wind_alert,
                    "precip": self.config.weather.precip_alert,
                },
                "llm_available": self.config.llm.is_any_provider_available(),
            },
        }

    def _format_results(self, state: AgentState) -> Dict[str, Any]:
        """
        Convert the final workflow state into a user-facing response.
        """
        if (
            state.get("agent1_status") != AgentStatus.COMPLETED
            or not state.get("raw_weather_data")
        ):
            return {
                "success": False,
                "request_id": state["request_id"],
                "errors": state["errors"],
                "reruns": state.get("_rerun_count", 0),
                "message": "Failed to collect weather data",
            }

        raw_data = state["raw_weather_data"]
        insights = state.get("detailed_insights") or {}
        alerts = state.get("weather_alerts") or []

        sources = {d.source for d in raw_data}

        return {
            "success": True,
            "request_id": state["request_id"],
            "location": state["location"],
            "user_question": state.get("user_question"),
            "timestamp": state["timestamp"],
            "workflow_status": {
                "agent1": state.get("agent1_status"),
                "agent2": state.get("agent2_status"),
                "agent3": state.get("agent3_status"),
                "execution_time_ms": state.get("execution_time_ms"),
                "reruns": state.get("_rerun_count", 0),
            },
            "data_sources": {
                "used": len(raw_data),
                "available": len(
                    self.workflow_builder.agent1.get_available_clients()
                ),
                "details": list(sources),
            },
            "weather_summary": insights,
            "alerts": [
                a.to_dict() if hasattr(a, "to_dict") else a for a in alerts
            ],
            "analysis": state.get("analysis_summary"),
            "llm_response": (
                state["llm_response"].to_dict()
                if hasattr(state.get("llm_response"), "to_dict")
                else state.get("llm_response")
            ),
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
        self,
        state: AgentState,
        error: str
    ) -> Dict[str, Any]:
        """
        Create a standardized error response.
        """
        return {
            "success": False,
            "request_id": state["request_id"],
            "location": state["location"],
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "errors": state["errors"] + [error],
            "workflow_status": {
                "agent1": state.get("agent1_status"),
                "agent2": state.get("agent2_status"),
                "agent3": state.get("agent3_status"),
                "reruns": state.get("_rerun_count", 0),
            },
        }
