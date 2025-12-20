# agents/agent3_llm.py
"""
Agent 3: LLM Analysis Agent
Uses multi-provider LLM system to analyze weather reports and answer questions.
"""
from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime

from workflows.state import AgentState, AgentStatus, LLMResponse, WeatherReport
from config.settings import Config
from utils.llm_client.multi_provider import MultiProviderLLM


class LLMAgent:
    """
    Agent 3: LLM-based analysis and question answering.
    Processes weather reports and provides insights and recommendations.
    """

    def __init__(self, config: Config):
        self.config = config
        self.llm_config = config.llm
        self.llm_client = None
        self._initialize_llm_client()

    def _initialize_llm_client(self):
        """Initialize the multi-provider LLM client."""
        try:
            self.llm_client = MultiProviderLLM(self.llm_config)
            print(f"LLMAgent initialized with {self.llm_client.current_provider}")
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

    async def analyze_with_llm(self, state: AgentState) -> AgentState:
        """
        Analyze weather report using LLM and update the state.
        
        """
        print(f"Agent 3: Collecting weather data for {state["location"]}")

        try:
            state["agent3_status"] = AgentStatus.PROCESSING

            if not state.get("weather_report"):
                state["agent3_status"] = AgentStatus.FAILED
                state["errors"].append("No weather report available for LLM analysis")
                return state

            report = state["weather_report"]
            user_question = state.get("user_question", "Provide a comprehensive weather analysis and recommendations")
            location = state["location"]

            llm_response = await self._generate_llm_response(report, user_question, location)

            state["llm_response"] = llm_response
            state["llm_analysis"] = llm_response.analysis
            state["agent3_status"] = AgentStatus.COMPLETED

            print(f"Agent 3: Analysis complete using {self.llm_client.current_provider}")
            return state

        except Exception as e:
            state["agent3_status"] = AgentStatus.FAILED
            state["errors"].append(f"Agent 3 LLM analysis failed: {str(e)}")
            return state

    async def _generate_llm_response(self, report: WeatherReport, user_question: str, location: str) -> LLMResponse:
        """
        Generate LLM response based on weather report and user question.
        """
        formatted_report = self._format_report_for_llm(report)
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(formatted_report, user_question, location)

        try:
            if self.llm_client and self.llm_config.is_any_provider_available():
                response_text = await self._call_llm_with_retry(system_prompt, user_prompt)
                return self._parse_llm_response(response_text, report, user_question, location)
            else:
                print("No LLM API keys available, using fallback analysis")
                return self._generate_fallback_response(report, user_question)
        except Exception as e:
            print(f"LLM API call failed: {e}, using fallback")
            return self._generate_fallback_response(report, user_question)

    async def _call_llm_with_retry(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> str:
        """Call LLM with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.llm_client.generate_response,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.4,
                    max_tokens=2000
                )
                return response
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(1)
                else:
                    raise

    def _create_system_prompt(self) -> str:
        """Create system prompt for the LLM."""
        return """You are a professional weather analyst. Provide insights, recommendations, and answer user questions in JSON format.
RESPONSE FORMAT:
{
  "analysis": "Comprehensive weather analysis...",
  "recommendations": ["Recommendation 1", "Recommendation 2", ...],
  "answers": {"question_key": "Answer..."},
  "follow_up_questions": ["Question 1", "Question 2", ...],
  "confidence": 0.95
}"""

    def _create_user_prompt(self, formatted_report: str, user_question: str, location: str) -> str:
        """Create user prompt for the LLM."""
        return f"""WEATHER REPORT DATA:
{formatted_report}

USER QUESTION: {user_question}
LOCATION: {location}

Respond in the JSON format specified by the system prompt."""

    def _format_report_for_llm(self, report: WeatherReport) -> str:
        """Format weather report for LLM consumption."""
        lines = [
            "=" * 60,
            f"WEATHER ANALYSIS REPORT - {report.location}",
            "=" * 60,
            report.executive_summary,
            "",
            "KEY METRICS:"
        ]
        if report.detailed_analysis:
            temp_info = report.detailed_analysis.get('temperature', {})
            lines.append(f"- Temperature: {temp_info.get('avg', 'N/A')}°F")
            conditions = report.detailed_analysis.get('conditions', {})
            lines.append(f"- Conditions: {conditions.get('primary', 'N/A')}")
        alerts = report.alerts_summary
        lines.append(f"- Total Alerts: {alerts.get('total', 0)}")
        if report.recommendations:
            lines.append("\nRECOMMENDATIONS:")
            for rec in report.recommendations[:5]:
                lines.append(f"- {rec}")
        return "\n".join(lines)

    def _parse_llm_response(self, response_text: str, report: WeatherReport,
                            user_question: str, location: str) -> LLMResponse:
        """Parse LLM response into structured LLMResponse object."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                response_data = json.loads(response_text[json_start:json_end])
                return LLMResponse(
                    analysis=response_data.get('analysis', ''),
                    recommendations=response_data.get('recommendations', []),
                    answers=response_data.get('answers', {}),
                    follow_up_questions=response_data.get('follow_up_questions', []),
                    confidence=response_data.get('confidence', 0.7)
                )
        except json.JSONDecodeError:
            pass
        # fallback
        return self._generate_fallback_response(report, user_question)

    def _generate_fallback_response(self, report: WeatherReport, user_question: str) -> LLMResponse:
        """Generate fallback response when LLM is unavailable."""
        analysis = f"Weather Analysis for {report.location}:\n{report.executive_summary}"
        recommendations = report.recommendations[:5] if report.recommendations else []
        answers = {}
        if "cold" in user_question.lower():
            answers["temperature_concern"] = "Temperatures are low. Dress warmly."
        if "rain" in user_question.lower():
            answers["precipitation_concern"] = "Precipitation expected. Carry waterproof gear."
        return LLMResponse(
            analysis=analysis,
            recommendations=recommendations,
            answers=answers,
            follow_up_questions=[],
            confidence=0.7
        )
