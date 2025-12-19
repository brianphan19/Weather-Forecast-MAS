# agents/workflow.py
"""
Main LangGraph workflow builder for the Weather Forecast MAS
"""
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Callable
from datetime import datetime
import uuid

from workflows.state import AgentState, AgentStatus
from agents.data_acquisition import DataCollectorAgent
from config.settings import Config


class WeatherForecastWorkflow:
    """
    Main workflow builder using LangGraph
    Orchestrates the flow between Agent 1 (Data Collector) and Agent 2 (Analyzer)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.agent1 = DataCollectorAgent(config)
        self.agent2 = None  # Will be initialized later
        self.graph = None
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with two agents
        """
        # Initialize the graph
        workflow = StateGraph(AgentState)
        
        # Define nodes (agents)
        workflow.add_node("agent1_data_collector", self._agent1_node)
        workflow.add_node("agent2_analyzer", self._agent2_node)
        
        # Define edges (flow)
        workflow.set_entry_point("agent1_data_collector")
        workflow.add_edge("agent1_data_collector", "agent2_analyzer")
        workflow.add_edge("agent2_analyzer", END)
        
        # Compile the graph
        self.graph = workflow.compile()
        return self.graph
    
    async def _agent1_node(self, state: AgentState) -> AgentState:
        """
        Node for Agent 1: Data Collection
        """
        try:
            # Execute Agent 1's collection logic
            return await self.agent1.collect_weather_data(state)
        except Exception as e:
            state["agent1_status"] = AgentStatus.FAILED
            state["errors"].append(f"Agent 1 failed: {str(e)}")
            return state
    
    async def _agent2_node(self, state: AgentState) -> AgentState:
        """
        Node for Agent 2: Analysis (to be implemented)
        For now, this is a placeholder that will be replaced with actual Agent 2
        """
        try:
            state["agent2_status"] = AgentStatus.ANALYZING
            
            # Check if we have data from Agent 1
            if not state.get("raw_weather_data") or not state.get("weather_consensus"):
                state["agent2_status"] = AgentStatus.FAILED
                state["errors"].append("No weather data available for analysis")
                return state
            
            # Placeholder analysis - will be replaced with actual Agent 2
            consensus = state["weather_consensus"]
            
            # Generate basic analysis
            state["analysis_summary"] = self._generate_basic_analysis(consensus)
            state["weather_alerts"] = self._generate_basic_alerts(consensus)
            
            state["agent2_status"] = AgentStatus.COMPLETED
            print("Agent 2: Analysis completed (placeholder)")
            
            return state
        except Exception as e:
            state["agent2_status"] = AgentStatus.FAILED
            state["errors"].append(f"Agent 2 failed: {str(e)}")
            return state
    
    def _generate_basic_analysis(self, consensus) -> str:
        """Generate basic analysis summary (placeholder for Agent 2)"""
        return (
            f"Weather analysis based on {consensus.sources_count} sources. "
            f"Current temperature: {consensus.temperature_avg:.1f}°F "
            f"(feels like {consensus.feels_like_avg:.1f}°F). "
            f"Conditions: {consensus.conditions_consensus} with {consensus.confidence_score:.0%} confidence."
        )
    
    def _generate_basic_alerts(self, consensus) -> list:
        """Generate basic weather alerts (placeholder for Agent 2)"""
        alerts = []
        weather_config = self.config.weather
        
        # Temperature high alert
        if consensus.temperature_avg > weather_config.temp_alert_high:
            alerts.append({
                "severity": "high",
                "type": "temperature_high",
                "message": f"High temperature alert: {consensus.temperature_avg:.1f}°F exceeds threshold of {weather_config.temp_alert_high}°F",
                "threshold": weather_config.temp_alert_high,
                "current_value": consensus.temperature_avg
            })
        
        # Temperature low alert
        if consensus.temperature_avg < weather_config.temp_alert_low:
            alerts.append({
                "severity": "high",
                "type": "temperature_low",
                "message": f"Low temperature alert: {consensus.temperature_avg:.1f}°F below threshold of {weather_config.temp_alert_low}°F",
                "threshold": weather_config.temp_alert_low,
                "current_value": consensus.temperature_avg
            })
        
        # Wind alert
        if consensus.wind_speed_avg > weather_config.wind_alert:
            alerts.append({
                "severity": "medium",
                "type": "wind",
                "message": f"High wind alert: {consensus.wind_speed_avg:.1f} mph exceeds threshold of {weather_config.wind_alert} mph",
                "threshold": weather_config.wind_alert,
                "current_value": consensus.wind_speed_avg
            })
        
        return alerts