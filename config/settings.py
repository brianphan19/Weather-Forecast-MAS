from dataclasses import dataclass
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers - uses any available API key"""
    
    # API Keys - only check if they exist
    openai_api_key: str
    openai_model:str
    groq_api_key: str
    groq_model:str
    gemini_api_key: str
    gemini_model:str
    
    # LLM Settings
    temperature: float
    max_tokens: int
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load LLM configuration from environment variables"""
        return cls(
            # API Keys - just get them, empty string if not set
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL").strip(),

            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL").strip(),

            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL").strip(),            
            # Settings
            temperature=cls._safe_float(os.getenv("LLM_TEMPERATURE", 0.3)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 1000))
        )
    
    @staticmethod
    def _safe_float(value: str) -> float:
        """Safely convert string to float"""
        try:
            return max(0.0, min(2.0, float(value)))
        except (ValueError, TypeError):
            return 0.3
    
    def get_available_providers(self) -> list:
        """Return list of available providers based on API keys"""
        providers = []
        
        if self.openai_api_key:
            providers.append("openai")
        if self.groq_api_key:
            providers.append("groq")
        if self.gemini_api_key:
            providers.append("gemini")
        
        return providers
    
    def is_any_provider_available(self) -> bool:
        """Check if at least one LLM provider is available"""
        return bool(self.get_available_providers())


@dataclass(frozen=True)
class WeatherConfig:
    """Configuration for weather APIs"""
    
    # Weather API Keys
    openweather_api_key: str
    weatherapi_key: str
    visual_crossing_api_key: str
    accuweather_api_key: str
    noaa_user_agent: str

    # Alert Thresholds
    temp_alert_high: float
    temp_alert_low: float
    wind_alert: float
    precip_alert: float
    temp_unit: str

    # Default Location
    default_location: str

    @classmethod
    def from_env(cls) -> 'WeatherConfig':
        # Required keys
        openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
        if not openweather_api_key:
            raise ValueError("OPENWEATHER_API_KEY is required")

        return cls(
            # Weather APIs
            openweather_api_key=openweather_api_key,
            weatherapi_key=os.getenv("WEATHERAPI_KEY", "").strip(),
            visual_crossing_api_key=os.getenv("VISUAL_CROSSING_API_KEY", "").strip(),
            accuweather_api_key=os.getenv("ACCUWEATHER_API_KEY", "").strip(),
            noaa_user_agent=os.getenv("NOAA_USER_AGENT", "").strip(),

            # Alerts
            temp_alert_high=float(os.getenv("TEMP_ALERT_THRESHOLD_HIGH", 100)),
            temp_alert_low=float(os.getenv("TEMP_ALERT_THRESHOLD_LOW", 0)),
            wind_alert=float(os.getenv("WIND_ALERT_THRESHOLD", 50)),
            precip_alert=float(os.getenv("PRECIPITATION_ALERT_THRESHOLD", 2.0)),
            temp_unit=os.getenv("TEMP_UNITS", 'imperial'),

            # Default location
            default_location=os.getenv("DEFAULT_LOCATION", "New York, NY, USA")
        )


@dataclass(frozen=True)
class Config:
    """Main configuration combining LLM and Weather configs"""
    llm: LLMConfig
    weather: WeatherConfig
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            llm=LLMConfig.from_env(),
            weather=WeatherConfig.from_env()
        )