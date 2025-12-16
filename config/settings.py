from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Weather API Keys
    openweather_api_key: str
    weatherapi_key: str
    visual_crossing_api_key: str
    accuweather_api_key: str
    noaa_user_agent: str

    # LLM Configuration
    llm_api_key: str
    llm_provider: str
    llm_model: str
    llm_temperature: float

    # Alert Thresholds
    temp_alert_high: float
    temp_alert_low: float
    wind_alert: float
    precip_alert: float

    # Default Location
    default_location: str

    @classmethod
    def from_env(cls) -> 'Config':
        # Required keys
        openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
        if not openweather_api_key:
            raise ValueError("OPENWEATHER_API_KEY is required")

        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        llm_temp = float(os.getenv("LLM_TEMPERATURE", 0.3))

        return cls(
            # Weather APIs
            openweather_api_key=openweather_api_key,
            weatherapi_key=os.getenv("WEATHERAPI_KEY", "").strip(),
            visual_crossing_api_key=os.getenv("VISUAL_CROSSING_API_KEY", "").strip(),
            accuweather_api_key=os.getenv("ACCUWEATHER_API_KEY", "").strip(),
            noaa_user_agent=os.getenv("NOAA_USER_AGENT", "").strip(),

            # LLM
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=llm_temp,

            # Alerts
            temp_alert_high=float(os.getenv("TEMP_ALERT_THRESHOLD_HIGH", 100)),
            temp_alert_low=float(os.getenv("TEMP_ALERT_THRESHOLD_LOW", 0)),
            wind_alert=float(os.getenv("WIND_ALERT_THRESHOLD", 50)),
            precip_alert=float(os.getenv("PRECIPITATION_ALERT_THRESHOLD", 2.0)),

            # Default location
            default_location=os.getenv("DEFAULT_LOCATION", "New York, NY, USA")
        )