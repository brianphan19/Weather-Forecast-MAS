# agents/state.py
"""
State management for the Weather Forecast MAS
Defines the shared state that flows through all agents
"""
from typing import Dict, List, Optional, Any, TypedDict, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum


class AgentStatus(str, Enum):
    """Status of agent execution"""
    PENDING = "pending"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WeatherSourceData:
    """Individual weather source data"""
    source: str
    temperature: float
    feels_like: float
    humidity: int
    pressure: int
    wind_speed: float
    conditions: str
    description: str
    wind_direction: int
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return asdict(self)


@dataclass
class WeatherConsensus:
    """Consensus data from multiple sources"""
    temperature_avg: float
    temperature_min: float
    temperature_max: float
    feels_like_avg: float
    humidity_avg: float
    pressure_avg: float
    wind_speed_avg: float
    conditions_consensus: str
    confidence_score: float
    sources_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return {
            "temperature_avg": self.temperature_avg,
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "feels_like_avg": self.feels_like_avg,
            "humidity_avg": self.humidity_avg,
            "pressure_avg": self.pressure_avg,
            "wind_speed_avg": self.wind_speed_avg,
            "conditions_consensus": self.conditions_consensus,
            "confidence_score": self.confidence_score,
            "sources_count": self.sources_count
        }


@dataclass
class WeatherAlert:
    """Weather alert structure"""
    severity: str  # "low", "medium", "high", "critical"
    type: str  # "temperature_high", "temperature_low", "wind", "precipitation", "storm"
    message: str
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return asdict(self)

@dataclass
class WeatherReport:
    """Comprehensive weather report for Agent 3 (LLM)"""
    location: str
    timestamp: str
    executive_summary: str
    detailed_analysis: Dict[str, Any]
    statistical_insights: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    data_quality: Dict[str, Any]
    alerts_summary: Dict[str, Any]
    comparative_analysis: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM consumption"""
        return asdict(self)

class AgentState(TypedDict):
    """
    Main state object that flows through the LangGraph workflow
    This is the shared memory between all agents
    """
    # User Input
    location: str
    request_id: str
    
    # Agent Status
    agent1_status: AgentStatus
    agent2_status: AgentStatus
    
    
    # Data Storage
    raw_weather_data: List[WeatherSourceData]
    weather_consensus: Optional[WeatherConsensus]
    
    # Analysis Results
    weather_alerts: List[WeatherAlert]
    analysis_summary: Optional[str]
    detailed_insights: Optional[Dict[str, Any]]
    weather_report: Optional[WeatherReport] 

    # Metadata
    timestamp: str
    errors: List[str]
    execution_time_ms: Optional[int]
    
    # LLM Context (if needed)
    llm_analysis: Optional[str]
    
    # Configuration
    config: Optional[Dict[str, Any]]