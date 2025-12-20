"""
WeatherForecastWorkflow

Builds the LangGraph workflow for the 3-agent Weather Forecast MAS with rerun capability.
Agents:
- Agent 1: DataCollectorAgent
- Agent 2: AnalysisAgent
- Agent 3: LLMAgent
Supports conditional reruns based on LLM feedback for different locations.
"""

from langgraph.graph import StateGraph, END
from typing import Callable, Optional
from datetime import datetime

from workflows.state import AgentState, AgentStatus
from agents.weather_data_report import ReportAgent
from agents.llm_chat import LLMAgent
from agents.data_acquisition import DataCollectorAgent
from config.settings import Config


class WeatherForecastWorkflow:
    """
    Builds a LangGraph workflow with 3 agents and rerun capability.
    """

    def __init__(self, config: Config):
        self.config = config
        self.agent1 = DataCollectorAgent(config)
        self.agent2 = ReportAgent(config)
        self.agent3 = LLMAgent(config)
        self.graph: Optional[StateGraph] = None
        self.max_reruns = 3

    def build_workflow(self) -> StateGraph:
        """
        Construct the full 3-agent workflow with routing and rerun logic.
        """
        workflow = StateGraph(state_schema=AgentState)

        # Add agent nodes
        workflow.add_node("agent1_data_collector", self.create_agent1_node())
        workflow.add_node("agent2_analyzer", self.create_agent2_node())
        workflow.add_node("agent3_llm", self.create_agent3_node())

        # Define entry point
        workflow.set_entry_point("agent1_data_collector")

        # Conditional routing
        workflow.add_conditional_edges(
            "agent1_data_collector",
            self.route_after_agent1,
            {"to_agent2": "agent2_analyzer", "end": END, "error": END},
        )
        workflow.add_edge("agent2_analyzer", "agent3_llm")

        
        workflow.add_edge("agent3_llm", END)
        self.graph = workflow.compile()
        return self.graph

    def create_agent1_node(self) -> Callable:
        """Agent 1 node for collecting weather data."""

        async def agent1_node(state: AgentState) -> AgentState:
            try:
                state["agent1_status"] = AgentStatus.COLLECTING
                state["timestamp"] = datetime.now().isoformat()
                return await self.agent1.collect_weather_data(state)
            except Exception as e:
                state["agent1_status"] = AgentStatus.FAILED
                state["errors"].append(f"Agent 1 node failed: {str(e)}")
                return state

        return agent1_node

    def create_agent2_node(self) -> Callable:
        """Agent 2 node for analysis."""

        async def agent2_node(state: AgentState) -> AgentState:
            try:
                return await self.agent2.generate_weather_data_report(state)
            except Exception as e:
                state["agent2_status"] = AgentStatus.FAILED
                state["errors"].append(f"Agent 2 node failed: {str(e)}")
                return state

        return agent2_node

    def create_agent3_node(self) -> Callable:
        """Agent 3 node for LLM analysis."""

        async def agent3_node(state: AgentState) -> AgentState:
            try:
                return await self.agent3.analyze_with_llm(state)
            except Exception as e:
                state["agent3_status"] = AgentStatus.FAILED
                state["errors"].append(f"Agent 3 node failed: {str(e)}")
                return state

        return agent3_node

    def route_after_agent1(self, state: AgentState) -> str:
        """
        Routing logic after Agent 1.
        Ends workflow if data collection fails or returns no data.
        """
        if state.get("agent1_status") == AgentStatus.FAILED:
            return "end"
        if not state.get("raw_weather_data"):
            return "end"
        return "to_agent2"

