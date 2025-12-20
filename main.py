# main.py
"""
Main entry point for the 3-Agent Weather Forecast Multi-Agent System (MAS).

This script handles user interaction, triggers the orchestrator workflow,
and renders results from all three agents, including LLM-generated insights
and recommendations.
"""

import asyncio
from typing import Dict

from workflows.orchestrator import WeatherMASOrchestrator
from config.settings import Config


def display_results(results: Dict):
    """
    Display workflow results in a structured, readable console format.

    Args:
        results (dict): Final response returned by the orchestrator.
    """
    print("\n" + "=" * 60)
    print("WEATHER FORECAST RESULTS (3-Agent System)")
    print("=" * 60)

    if results.get("success"):
        print(f"\nLocation: {results.get('location')}")
        print(f"Request ID: {results.get('request_id')}")

        if results.get("user_question"):
            print(f"User Question: {results['user_question']}")

        workflow_status = results.get("workflow_status", {})
        execution_time = workflow_status.get("execution_time_ms")
        if execution_time is not None:
            print(f"Execution time: {execution_time} ms")

        reruns = workflow_status.get("reruns", 0)
        if reruns > 0:
            print(f"Reruns performed: {reruns}")

        print("\nAgent Statuses:")
        print(f"  - Agent 1 (Data): {workflow_status.get('agent1', 'unknown')}")
        print(f"  - Agent 2 (Analysis): {workflow_status.get('agent2', 'unknown')}")
        print(f"  - Agent 3 (LLM): {workflow_status.get('agent3', 'unknown')}")

        data_sources = results.get("data_sources", {})
        print(
            f"Sources used: "
            f"{data_sources.get('used', 0)}/{data_sources.get('available', 0)}"
        )

        weather_summary = results.get("weather_summary", {})

        # Temperature
        temperature = weather_summary.get("temperature")
        if isinstance(temperature, dict):
            print("\nTemperature:")
            if "avg" in temperature:
                print(f"  - Current: {temperature.get('avg', 0):.1f} °F")
            if "feels_like" in temperature:
                print(
                    f"  - Feels like: "
                    f"{temperature.get('feels_like', 0):.1f} °F"
                )
            elif "feels_like_avg" in temperature:
                print(
                    f"  - Feels like: "
                    f"{temperature.get('feels_like_avg', 0):.1f} °F"
                )
            if "min" in temperature and "max" in temperature:
                print(
                    f"  - Range: "
                    f"{temperature.get('min', 0):.1f} °F - "
                    f"{temperature.get('max', 0):.1f} °F"
                )

        # Conditions
        conditions = weather_summary.get("conditions")
        if isinstance(conditions, dict):
            print("\nConditions:")
            if "primary" in conditions:
                print(f"  - {conditions.get('primary', 'N/A')}")
            if "confidence" in conditions:
                confidence = conditions.get("confidence")
                if isinstance(confidence, (int, float)):
                    print(f"  - Confidence: {confidence:.0%}")
                elif isinstance(confidence, dict):
                    print(f"  - Confidence: {confidence.get('level', 'N/A')}")

        # Alerts
        alerts = results.get("alerts", [])
        if alerts:
            print(f"\nActive Alerts ({len(alerts)}):")
            for alert in alerts[:3]:
                print(
                    f"  [{alert.get('severity', 'low').upper()}] "
                    f"{alert.get('message', '')}"
                )
                if alert.get("recommendation"):
                    print(f"    Recommendation: {alert['recommendation']}")
            if len(alerts) > 3:
                print(f"  ... and {len(alerts) - 3} more alerts")

        # LLM output
        llm_response = results.get("llm_response")
        if llm_response:
            print("\n" + "=" * 40)
            print("LLM ANALYSIS")
            print("=" * 40)

            analysis = llm_response.get("analysis")
            if analysis:
                print(
                    analysis[:500] + "..."
                    if len(analysis) > 500
                    else analysis
                )

            recommendations = llm_response.get("recommendations")
            if recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"  {i}. {rec}")

            answers = llm_response.get("answers")
            if answers:
                print("\nAnswers:")
                for question, answer in answers.items():
                    display_answer = (
                        answer if len(answer) < 150 else answer[:150] + "..."
                    )
                    print(f"  - {question}: {display_answer}")

            follow_ups = llm_response.get("follow_up_questions")
            if follow_ups:
                print("\nSuggested Follow-Up Questions:")
                for i, question in enumerate(follow_ups[:3], 1):
                    print(f"  {i}. {question}")

            if llm_response.get("requires_rerun"):
                print(
                    f"\nLLM suggested re-analysis for: "
                    f"{llm_response.get('rerun_location')}"
                )

            if llm_response.get("confidence") is not None:
                print(
                    f"\nLLM Confidence: "
                    f"{llm_response['confidence']:.0%}"
                )

        # Raw data preview
        raw_preview = results.get("raw_data_preview")
        if raw_preview:
            print("\nData Sources Preview:")
            for data in raw_preview[:3]:
                print(
                    f"  - {data.get('source', 'Unknown')}: "
                    f"{data.get('temperature', 0):.1f} °F, "
                    f"{data.get('conditions', 'Unknown')}"
                )

    else:
        print("\nFailed to retrieve weather forecast")
        print(f"Error: {results.get('error', 'Unknown error')}")
        if results.get("errors"):
            print("Detailed Errors:")
            for error in results["errors"][:3]:
                print(f"  - {error}")

    print("\n" + "=" * 60)


async def main():
    """
    Interactive application entry point.

    Prompts the user for location and optional questions, executes the
    3-agent workflow, and displays the results.
    """
    print("=" * 60)
    print("Weather Forecast Multi-Agent System (3 Agents)")
    print("=" * 60)

    config = Config.from_env()

    location = input("Location: ").strip() or config.weather.default_location
    user_question = input("Questions: ").strip() or "Should I go out today?"

    print(f"\nUsing location: {location}")


    try:
        orchestrator = WeatherMASOrchestrator()
        results = await orchestrator.get_weather_forecast(
            location, user_question
        )
        display_results(results)

    except KeyboardInterrupt:
        print("\nExiting application.")
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


