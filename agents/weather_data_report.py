# agents/agent2_analysis.py
"""
Agent 2: Weather Data Report Agent
Combine weather data and generates comprehensive reports for Agent 3
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import statistics
from dataclasses import dataclass, asdict

from workflows.state import AgentState, AgentStatus, WeatherAlert, WeatherSourceData
from config.settings import Config


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


class ReportAgent:
    """
    Agent 2: Combine weather data and report generation
    Creates structured reports for Agent 3 (LLM) to process
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.weather_config = config.weather
        print("ReportAgent initialized")
        
    async def generate_weather_data_report(self, state: AgentState) -> AgentState:
        """
        Main analysis function for Agent 2
        Performs comprehensive analysis and generates report for Agent 3
        """
        try:
            state["agent2_status"] = AgentStatus.ANALYZING
            print(f"Agent 2: Combine weather data and generating report...")
            
            # Validate input data
            if not state.get("raw_weather_data"):
                state["agent2_status"] = AgentStatus.FAILED
                state["errors"].append("No weather data to analyze")
                return state
            
            raw_data = state["raw_weather_data"]
            location = state["location"]
            
            # Perform comprehensive analysis
            detailed_insights = self._analyze_comprehensive(raw_data)
            alerts = self._generate_alerts(raw_data, detailed_insights)
            report = self._generate_report(location, raw_data, detailed_insights, alerts)
            analysis_summary = self._generate_executive_summary(report)
            
            # Update state with analysis results
            state["detailed_insights"] = detailed_insights
            state["weather_alerts"] = alerts
            state["analysis_summary"] = analysis_summary
            state["weather_report"] = report  # This will be used by Agent 3
            state["agent2_status"] = AgentStatus.COMPLETED
            
            print(f"Agent 2: Analysis complete")
            print(f"  - Generated report with {len(alerts)} alerts")
            print(f"  - Data from {len(raw_data)} sources")
            
            return state
            
        except Exception as e:
            state["agent2_status"] = AgentStatus.FAILED
            state["errors"].append(f"Agent 2 analysis failed: {str(e)}")
            return state
    
    def _analyze_comprehensive(self, raw_data: List[WeatherSourceData]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on weather data
        """
        # Extract all metrics
        temps = [d.temperature for d in raw_data]
        feels_like = [d.feels_like for d in raw_data]
        humidities = [d.humidity for d in raw_data]
        pressures = [d.pressure for d in raw_data]
        wind_speeds = [d.wind_speed for d in raw_data]
        wind_directions = [d.wind_direction for d in raw_data if d.wind_direction > 0]
        
        # Calculate statistics
        temp_stats = self._calculate_statistics(temps, "temperature", "°F")
        humidity_stats = self._calculate_statistics(humidities, "humidity", "%")
        pressure_stats = self._calculate_statistics(pressures, "pressure", "hPa")
        wind_stats = self._calculate_statistics(wind_speeds, "wind_speed", "mph")
        
        # Calculate consistency scores
        consistency = self._calculate_consistency(raw_data)
        
        # Analyze conditions
        conditions_analysis = self._analyze_conditions(raw_data)
        
        # Calculate comfort index
        comfort_index, comfort_level = self._calculate_comfort_index(
            temp_stats["avg"],
            humidity_stats["avg"],
            wind_stats["avg"]
        )
        
        # Determine weather severity
        severity, severity_reason = self._determine_severity(temp_stats, conditions_analysis, wind_stats)
        
        # Analyze trends
        trends = self._analyze_trends(raw_data)
        
        # Calculate risk factors
        risk_factors = self._assess_risk_factors(temp_stats, conditions_analysis, wind_stats)
        
        return {
            "temperature": temp_stats,
            "feels_like": self._calculate_statistics(feels_like, "feels_like", "°F"),
            "humidity": humidity_stats,
            "pressure": pressure_stats,
            "wind": {
                **wind_stats,
                "primary_direction": self._calculate_wind_direction(wind_directions),
                "direction_variability": self._calculate_wind_variability(wind_directions)
            },
            "conditions": conditions_analysis,
            "comfort": {
                "index": comfort_index,
                "level": comfort_level,
                "factors": self._analyze_comfort_factors(temp_stats["avg"], humidity_stats["avg"], wind_stats["avg"])
            },
            "consistency": consistency,
            "severity": {
                "level": severity,
                "reason": severity_reason,
                "score": self._calculate_severity_score(severity, conditions_analysis, temp_stats)
            },
            "trends": trends,
            "risk_factors": risk_factors,
            "data_quality": {
                "source_agreement": consistency["conditions_agreement"],
                "temperature_variance": temp_stats["std"],
                "outliers": self._detect_outliers(raw_data),
                "missing_data": self._check_missing_data(raw_data)
            }
        }
    
    def _calculate_statistics(self, values: List[float], metric_name: str, unit: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a metric"""
        if not values:
            return {"avg": 0, "min": 0, "max": 0, "std": 0, "unit": unit}
        
        return {
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "range": max(values) - min(values),
            "median": statistics.median(values),
            "unit": unit
        }
    
    def _calculate_consistency(self, raw_data: List[WeatherSourceData]) -> Dict[str, Any]:
        """Calculate consistency across different sources"""
        if len(raw_data) < 2:
            return {
                "conditions_agreement": 1.0,
                "temperature_variance": 0.0,
                "reliable_sources": len(raw_data)
            }
        
        # Check condition agreement
        conditions = [d.conditions for d in raw_data]
        most_common = max(set(conditions), key=conditions.count)
        agreement = conditions.count(most_common) / len(conditions)
        
        # Calculate temperature variance
        temps = [d.temperature for d in raw_data]
        variance = statistics.variance(temps) if len(temps) > 1 else 0
        
        # Calculate overall confidence
        confidence_scores = []
        for data in raw_data:
            # Simple confidence scoring
            confidence = 1.0
            if data.error:
                confidence *= 0.5
            if abs(data.temperature - statistics.mean(temps)) > 10:  # Temperature outlier
                confidence *= 0.7
            confidence_scores.append(confidence)
        
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        return {
            "conditions_agreement": agreement,
            "temperature_variance": variance,
            "reliable_sources": len([d for d in raw_data if not d.error]),
            "average_confidence": avg_confidence,
            "assessment": "high" if agreement > 0.7 and variance < 5 else "medium" if agreement > 0.5 else "low"
        }
    
    def _analyze_conditions(self, raw_data: List[WeatherSourceData]) -> Dict[str, Any]:
        """Analyze weather conditions across sources"""
        conditions = {}
        descriptions = []
        
        for data in raw_data:
            condition = data.conditions
            conditions[condition] = conditions.get(condition, 0) + 1
            description = data.description
            if description:
                descriptions.append(description)
        
        most_common = max(conditions.items(), key=lambda x: x[1]) if conditions else ("Unknown", 0)
        
        return {
            "primary": most_common[0],
            "confidence": most_common[1] / len(raw_data),
            "distribution": conditions,
            "descriptions": list(set(descriptions))[:5],  # Top 5 unique descriptions
            "assessment": self._assess_conditions_severity(most_common[0])
        }
    
    def _calculate_comfort_index(self, temp: float, humidity: float, wind_speed: float) -> Tuple[float, str]:
        """
        Calculate comfort index (0-100, higher is more comfortable)
        Based on temperature, humidity, and wind chill
        """
        # Temperature comfort (ideal 68-72°F)
        if 68 <= temp <= 72:
            temp_score = 100
        elif temp < 32:
            temp_score = max(0, 100 - (32 - temp) * 3)
        elif temp > 90:
            temp_score = max(0, 100 - (temp - 90) * 2)
        else:
            temp_score = 100 - abs(temp - 70) * 2
        
        # Humidity comfort (ideal 40-60%)
        if 40 <= humidity <= 60:
            humidity_score = 100
        else:
            humidity_score = 100 - abs(humidity - 50) * 1.5
        
        # Wind comfort (ideal 5-15 mph)
        if 5 <= wind_speed <= 15:
            wind_score = 100
        elif wind_speed > 30:
            wind_score = max(0, 100 - (wind_speed - 30) * 3)
        else:
            wind_score = 100 - abs(wind_speed - 10) * 5
        
        # Weighted average
        comfort = (temp_score * 0.5 + humidity_score * 0.3 + wind_score * 0.2)
        comfort = max(0, min(100, comfort))
        
        # Comfort level
        if comfort >= 80:
            level = "Very Comfortable"
        elif comfort >= 60:
            level = "Comfortable"
        elif comfort >= 40:
            level = "Moderate"
        elif comfort >= 20:
            level = "Uncomfortable"
        else:
            level = "Very Uncomfortable"
        
        return comfort, level
    
    def _analyze_comfort_factors(self, temp: float, humidity: float, wind_speed: float) -> List[str]:
        """Analyze factors affecting comfort"""
        factors = []
        
        if temp < 32:
            factors.append("Extreme cold - risk of hypothermia")
        elif temp > 90:
            factors.append("Extreme heat - risk of heat exhaustion")
        
        if humidity > 80:
            factors.append("High humidity - feels muggier")
        elif humidity < 30:
            factors.append("Low humidity - dry air")
        
        if wind_speed > 20:
            factors.append("High winds - wind chill effect")
        
        return factors
    
    def _calculate_wind_direction(self, directions: List[int]) -> str:
        """Calculate predominant wind direction"""
        if not directions:
            return "Variable/Calm"
        
        # Convert to cardinal directions
        cardinal_map = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
            "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
            "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
            "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
        }
        
        # Find closest cardinal direction for each degree
        cardinals = []
        for deg in directions:
            min_diff = 360
            closest = "N"
            for card, card_deg in cardinal_map.items():
                diff = min(abs(deg - card_deg), 360 - abs(deg - card_deg))
                if diff < min_diff:
                    min_diff = diff
                    closest = card
            cardinals.append(closest)
        
        # Find most common
        most_common = max(set(cardinals), key=cardinals.count)
        return most_common
    
    def _calculate_wind_variability(self, directions: List[int]) -> str:
        """Calculate wind direction variability"""
        if len(directions) < 2:
            return "Low"
        
        # Calculate circular variance
        import math
        
        # Convert to radians
        rads = [math.radians(d) for d in directions]
        
        # Calculate mean vector
        x = sum(math.cos(r) for r in rads) / len(rads)
        y = sum(math.sin(r) for r in rads) / len(rads)
        
        # Resultant vector length
        r = math.sqrt(x**2 + y**2)
        
        if r > 0.9:
            return "Very Low"
        elif r > 0.7:
            return "Low"
        elif r > 0.5:
            return "Moderate"
        elif r > 0.3:
            return "High"
        else:
            return "Very High"
    
    def _determine_severity(self, temp_stats: Dict, conditions: Dict, wind_stats: Dict) -> Tuple[str, str]:
        """Determine overall weather severity"""
        primary_condition = conditions.get("primary", "Unknown")
        temp_avg = temp_stats.get("avg", 0)
        wind_max = wind_stats.get("max", 0)
        
        severity_reasons = []
        
        # Check for extreme conditions
        if primary_condition in ["Thunderstorm", "Blizzard", "Hurricane"]:
            return "severe", "Extreme weather conditions detected"
        
        if primary_condition in ["Heavy Rain", "Heavy Snow", "Ice Storm"]:
            severity_reasons.append(f"{primary_condition} conditions")
        
        if temp_avg > self.weather_config.temp_alert_high:
            severity_reasons.append(f"High temperature ({temp_avg:.1f}°F)")
        
        if temp_avg < self.weather_config.temp_alert_low:
            severity_reasons.append(f"Low temperature ({temp_avg:.1f}°F)")
        
        if wind_max > self.weather_config.wind_alert:
            severity_reasons.append(f"High winds ({wind_max:.1f} mph)")
        
        if conditions.get("confidence", 0) < 0.3:
            severity_reasons.append("Low data confidence")
        
        if severity_reasons:
            return "high", ", ".join(severity_reasons)
        elif conditions.get("assessment") == "moderate":
            return "moderate", "Moderate weather conditions"
        else:
            return "normal", "Normal weather conditions"
    
    def _calculate_severity_score(self, severity: str, conditions: Dict, temp_stats: Dict) -> float:
        """Calculate numerical severity score (0-100)"""
        base_scores = {
            "severe": 80,
            "high": 60,
            "moderate": 40,
            "normal": 20
        }
        
        score = base_scores.get(severity, 20)
        
        # Adjust based on conditions
        if conditions.get("primary") in ["Thunderstorm", "Blizzard"]:
            score += 20
        elif conditions.get("primary") in ["Heavy Rain", "Heavy Snow"]:
            score += 10
        
        # Adjust based on temperature extremes
        temp_range = temp_stats.get("range", 0)
        if temp_range > 15:
            score += 5
        
        return min(100, score)
    
    def _assess_conditions_severity(self, condition: str) -> str:
        """Assess severity of weather conditions"""
        severe_conditions = ["Thunderstorm", "Blizzard", "Hurricane", "Tornado"]
        high_conditions = ["Heavy Rain", "Heavy Snow", "Ice Storm", "Freezing Rain"]
        moderate_conditions = ["Rain", "Snow", "Fog", "Windy"]
        
        if condition in severe_conditions:
            return "severe"
        elif condition in high_conditions:
            return "high"
        elif condition in moderate_conditions:
            return "moderate"
        else:
            return "low"
    
    def _analyze_trends(self, raw_data: List[WeatherSourceData]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        if len(raw_data) < 3:
            return {"available": False, "reason": "Insufficient data points"}
        
        # Check for temperature trends
        temps = [d.temperature for d in raw_data]
        pressures = [d.pressure for d in raw_data]
        
        # Simple trend detection
        temp_trend = "stable"
        if len(temps) >= 3:
            if all(temps[i] < temps[i+1] for i in range(len(temps)-1)):
                temp_trend = "rising"
            elif all(temps[i] > temps[i+1] for i in range(len(temps)-1)):
                temp_trend = "falling"
        
        pressure_trend = "stable"
        if len(pressures) >= 3:
            avg_pressure = statistics.mean(pressures)
            if avg_pressure < 1000:
                pressure_trend = "low"
            elif avg_pressure > 1020:
                pressure_trend = "high"
        
        return {
            "available": True,
            "temperature": temp_trend,
            "pressure": pressure_trend,
            "implications": self._interpret_trends(temp_trend, pressure_trend)
        }
    
    def _interpret_trends(self, temp_trend: str, pressure_trend: str) -> List[str]:
        """Interpret weather trends"""
        implications = []
        
        if pressure_trend == "low" and temp_trend == "rising":
            implications.append("Potential for stormy weather")
        elif pressure_trend == "high" and temp_trend == "falling":
            implications.append("Weather likely to clear")
        elif pressure_trend == "low":
            implications.append("Possible precipitation")
        
        return implications
    
    def _assess_risk_factors(self, temp_stats: Dict, conditions: Dict, wind_stats: Dict) -> Dict[str, Any]:
        """Assess various risk factors"""
        risks = []
        
        primary_condition = conditions.get("primary", "")
        temp_avg = temp_stats.get("avg", 0)
        wind_max = wind_stats.get("max", 0)
        
        # Temperature risks
        if temp_avg > 95:
            risks.append({"type": "heat", "level": "high", "description": "Extreme heat risk"})
        elif temp_avg > 85:
            risks.append({"type": "heat", "level": "medium", "description": "High heat risk"})
        elif temp_avg < 20:
            risks.append({"type": "cold", "level": "high", "description": "Extreme cold risk"})
        elif temp_avg < 32:
            risks.append({"type": "cold", "level": "medium", "description": "Freezing risk"})
        
        # Condition risks
        condition_risks = {
            "Thunderstorm": {"type": "lightning", "level": "high"},
            "Heavy Rain": {"type": "flooding", "level": "medium"},
            "Snow": {"type": "slippery", "level": "medium"},
            "Fog": {"type": "visibility", "level": "low"}
        }
        
        if primary_condition in condition_risks:
            risks.append({
                **condition_risks[primary_condition],
                "description": f"{primary_condition} conditions"
            })
        
        # Wind risks
        if wind_max > 40:
            risks.append({"type": "wind", "level": "high", "description": "Dangerous winds"})
        elif wind_max > 25:
            risks.append({"type": "wind", "level": "medium", "description": "Strong winds"})
        
        # Calculate overall risk score
        risk_levels = {"high": 3, "medium": 2, "low": 1}
        total_risk = sum(risk_levels.get(r["level"], 0) for r in risks)
        avg_risk = total_risk / len(risks) if risks else 0
        
        return {
            "factors": risks,
            "count": len(risks),
            "overall_level": "high" if avg_risk > 2 else "medium" if avg_risk > 1 else "low",
            "score": avg_risk
        }
    
    def _detect_outliers(self, raw_data: List[WeatherSourceData]) -> List[Dict[str, Any]]:
        """Detect outliers in the data"""
        if len(raw_data) < 3:
            return []
        
        temps = [d.temperature for d in raw_data]
        mean_temp = statistics.mean(temps)
        std_temp = statistics.stdev(temps) if len(temps) > 1 else 0
        
        outliers = []
        for i, data in enumerate(raw_data):
            if std_temp > 0 and abs(data.temperature - mean_temp) > 2 * std_temp:
                outliers.append({
                    "source": data.source,
                    "metric": "temperature",
                    "value": data.temperature,
                    "deviation": data.temperature - mean_temp
                })
        
        return outliers
    
    def _check_missing_data(self, raw_data: List[WeatherSourceData]) -> List[str]:
        """Check for missing or incomplete data"""
        missing = []
        
        for data in raw_data:
            if data.temperature == 0 and data.error:
                missing.append(f"{data.source}: Invalid temperature data")
            if not data.conditions or data.conditions == "Unknown":
                missing.append(f"{data.source}: Missing conditions")
        
        return missing
    
    def _generate_alerts(self, raw_data: List[WeatherSourceData], insights: Dict) -> List[WeatherAlert]:
        """Generate comprehensive weather alerts"""
        alerts = []
        
        # Get thresholds from config
        temp_high = self.weather_config.temp_alert_high
        temp_low = self.weather_config.temp_alert_low
        wind_threshold = self.weather_config.wind_alert
        precip_threshold = self.weather_config.precip_alert
        
        # Extract metrics
        temp_stats = insights.get("temperature", {})
        wind_stats = insights.get("wind", {})
        conditions = insights.get("conditions", {})
        risk_factors = insights.get("risk_factors", {})
        
        avg_temp = temp_stats.get("avg", 0)
        max_temp = temp_stats.get("max", 0)
        min_temp = temp_stats.get("min", 0)
        max_wind = wind_stats.get("max", 0)
        primary_condition = conditions.get("primary", "Unknown")
        
        # Temperature alerts
        if avg_temp > temp_high:
            severity = "critical" if avg_temp > temp_high + 10 else "high"
            alerts.append(WeatherAlert(
                severity=severity,
                type="heat",
                message=f"Heat alert: Temperature is {avg_temp:.1f}°F (threshold: {temp_high}°F)",
                threshold=temp_high,
                current_value=avg_temp,
                recommendation="Stay hydrated, avoid outdoor activities during peak heat, check on vulnerable individuals"
            ))
        
        if avg_temp < temp_low:
            severity = "critical" if avg_temp < temp_low - 10 else "high"
            alerts.append(WeatherAlert(
                severity=severity,
                type="cold",
                message=f"Cold alert: Temperature is {avg_temp:.1f}°F (threshold: {temp_low}°F)",
                threshold=temp_low,
                current_value=avg_temp,
                recommendation="Wear layers, limit time outdoors, check heating systems"
            ))
        
        # Extreme temperature range alert
        temp_range = max_temp - min_temp
        if temp_range > 20:
            alerts.append(WeatherAlert(
                severity="medium",
                type="temperature_variation",
                message=f"Large temperature variation: {temp_range:.1f}°F range",
                current_value=temp_range,
                recommendation="Dress in layers to adapt to changing temperatures"
            ))
        
        # Wind alerts
        if max_wind > wind_threshold:
            severity = "critical" if max_wind > wind_threshold + 20 else "high" if max_wind > wind_threshold + 10 else "medium"
            alerts.append(WeatherAlert(
                severity=severity,
                type="wind",
                message=f"Wind alert: Winds up to {max_wind:.1f} mph (threshold: {wind_threshold} mph)",
                threshold=wind_threshold,
                current_value=max_wind,
                recommendation="Secure outdoor objects, be cautious when driving high-profile vehicles"
            ))
        
        # Condition-specific alerts
        condition_alerts = {
            "Thunderstorm": {
                "severity": "critical",
                "type": "lightning",
                "message": "Thunderstorm warning - lightning risk",
                "recommendation": "Seek indoor shelter immediately, avoid open areas and tall objects"
            },
            "Heavy Rain": {
                "severity": "high",
                "type": "flooding",
                "message": "Heavy rainfall - flood risk",
                "recommendation": "Avoid low-lying areas, do not drive through flooded roads"
            },
            "Snow": {
                "severity": "medium",
                "type": "winter",
                "message": "Snow conditions",
                "recommendation": "Drive carefully, watch for icy patches, dress warmly"
            },
            "Fog": {
                "severity": "low",
                "type": "visibility",
                "message": "Reduced visibility due to fog",
                "recommendation": "Use low beam headlights, reduce speed, increase following distance"
            }
        }
        
        if primary_condition in condition_alerts:
            alert_info = condition_alerts[primary_condition]
            alerts.append(WeatherAlert(
                severity=alert_info["severity"],
                type=alert_info["type"],
                message=alert_info["message"],
                recommendation=alert_info["recommendation"]
            ))
        
        # Data quality alerts
        consistency = insights.get("consistency", {})
        if consistency.get("conditions_agreement", 0) < 0.3:
            alerts.append(WeatherAlert(
                severity="low",
                type="data_quality",
                message=f"Low confidence in weather data ({consistency['conditions_agreement']:.0%} agreement)",
                recommendation="Consider checking additional weather sources"
            ))
        
        # Risk-based alerts
        for risk in risk_factors.get("factors", []):
            if risk["level"] == "high":
                alerts.append(WeatherAlert(
                    severity="high",
                    type=f"risk_{risk['type']}",
                    message=f"High risk: {risk['description']}",
                    recommendation="Take appropriate precautions"
                ))
        
        return alerts
    
    def _generate_report(self, location: str, raw_data: List[WeatherSourceData], 
                        insights: Dict, alerts: List[WeatherAlert]) -> WeatherReport:
        """Generate comprehensive weather report for Agent 3"""
        
        # Create executive summary
        executive_summary = self._generate_executive_summary_from_insights(location, insights, alerts)
        
        # Prepare detailed analysis
        detailed_analysis = {
            "location_analysis": self._analyze_location_specifics(location, insights),
            "time_analysis": self._analyze_time_considerations(),
            "source_analysis": self._analyze_data_sources(raw_data),
            "comparative_analysis": self._generate_comparative_analysis(insights)
        }
        
        # Statistical insights
        statistical_insights = {
            "distribution_analysis": self._analyze_distributions(raw_data),
            "correlation_analysis": self._analyze_correlations(raw_data),
            "predictive_insights": self._generate_predictive_insights(insights)
        }
        
        # Risk assessment
        risk_assessment = {
            "immediate_risks": [alert for alert in alerts if alert.severity in ["critical", "high"]],
            "potential_risks": self._identify_potential_risks(insights),
            "mitigation_strategies": self._generate_mitigation_strategies(alerts, insights)
        }
        
        # Recommendations
        recommendations = self._generate_recommendations(alerts, insights)
        
        # Data quality assessment
        data_quality = {
            "source_reliability": self._assess_source_reliability(raw_data),
            "consistency_metrics": insights.get("consistency", {}),
            "completeness_score": self._calculate_completeness_score(raw_data)
        }
        
        # Alerts summary
        alerts_summary = {
            "total": len(alerts),
            "by_severity": self._categorize_alerts_by_severity(alerts),
            "by_type": self._categorize_alerts_by_type(alerts),
            "priority_order": self._prioritize_alerts(alerts)
        }
        
        return WeatherReport(
            location=location,
            timestamp=datetime.now().isoformat(),
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            statistical_insights=statistical_insights,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            data_quality=data_quality,
            alerts_summary=alerts_summary
        )
    
    def _generate_executive_summary(self, report: WeatherReport) -> str:
        """Generate concise executive summary"""
        summary_parts = []
        
        # Basic weather info
        summary_parts.append(f"Current weather in {report.location}:")
        
        # Add key metrics if available in detailed analysis
        if hasattr(report, 'detailed_analysis') and report.detailed_analysis:
            # Try to extract key info
            pass
        
        # Add alert summary
        alerts_summary = report.alerts_summary
        if alerts_summary.get('total', 0) > 0:
            critical = alerts_summary.get('by_severity', {}).get('critical', 0)
            high = alerts_summary.get('by_severity', {}).get('high', 0)
            
            if critical > 0:
                summary_parts.append(f"⚠️ {critical} CRITICAL alert{'s' if critical > 1 else ''} - immediate action required")
            if high > 0:
                summary_parts.append(f"🚨 {high} high-priority alert{'s' if high > 1 else ''}")
        
        return " ".join(summary_parts)
    
    def _generate_executive_summary_from_insights(self, location: str, insights: Dict, alerts: List[WeatherAlert]) -> str:
        """Generate executive summary from insights"""
        temp_stats = insights.get("temperature", {})
        conditions = insights.get("conditions", {})
        comfort = insights.get("comfort", {})
        severity = insights.get("severity", {})
        
        avg_temp = temp_stats.get("avg", 0)
        primary_condition = conditions.get("primary", "Unknown")
        comfort_level = comfort.get("level", "Unknown")
        severity_level = severity.get("level", "normal")
        
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        high_alerts = [a for a in alerts if a.severity == "high"]
        
        summary = [
            f"Weather Report for {location}",
            f"Current Conditions: {primary_condition} at {avg_temp:.1f}°F",
            f"Comfort Level: {comfort_level}",
            f"Severity Assessment: {severity_level.upper()}"
        ]
        
        if critical_alerts:
            summary.append(f"CRITICAL ALERTS: {len(critical_alerts)} active - Immediate attention required")
        
        if high_alerts:
            summary.append(f"High Priority Alerts: {len(high_alerts)} active")
        
        return " | ".join(summary)
    
    def _analyze_location_specifics(self, location: str, insights: Dict) -> Dict[str, Any]:
        """Analyze location-specific considerations"""
        # This could be expanded with geographic data
        return {
            "location_type": self._classify_location_type(location),
            "elevation_considerations": "Assume sea level",  # Could integrate with elevation API
            "urban_heat_island": "Possible in urban areas" if "city" in location.lower() else "Unlikely",
            "microclimate_factors": ["wind patterns", "precipitation variation"]
        }
    
    def _classify_location_type(self, location: str) -> str:
        """Classify location type"""
        location_lower = location.lower()
        
        if any(word in location_lower for word in ["city", "town", "urban"]):
            return "urban"
        elif any(word in location_lower for word in ["coast", "beach", "shore"]):
            return "coastal"
        elif any(word in location_lower for word in ["mountain", "hill", "alps"]):
            return "mountainous"
        elif any(word in location_lower for word in ["desert", "arid", "dry"]):
            return "arid"
        else:
            return "general"
    
    def _analyze_time_considerations(self) -> Dict[str, Any]:
        """Analyze time-related considerations"""
        now = datetime.now()
        hour = now.hour
        
        time_considerations = {
            "current_hour": hour,
            "time_of_day": "night" if 20 <= hour or hour < 6 else "evening" if 18 <= hour else "afternoon" if 12 <= hour else "morning",
            "seasonal_factors": self._determine_seasonal_factors(now),
            "daylight_considerations": "limited visibility" if hour < 7 or hour > 19 else "good visibility"
        }
        
        return time_considerations
    
    def _determine_seasonal_factors(self, dt: datetime) -> List[str]:
        """Determine seasonal factors"""
        month = dt.month
        
        if month in [12, 1, 2]:  # Winter
            return ["cold temperatures possible", "snow potential", "limited daylight"]
        elif month in [3, 4, 5]:  # Spring
            return ["variable conditions", "increasing daylight", "storm potential"]
        elif month in [6, 7, 8]:  # Summer
            return ["warm temperatures", "long daylight", "thunderstorm potential"]
        else:  # Fall
            return ["cooling temperatures", "decreasing daylight", "frost potential"]
    
    def _analyze_data_sources(self, raw_data: List[WeatherSourceData]) -> Dict[str, Any]:
        """Analyze the quality and characteristics of data sources"""
        sources = {}
        
        for data in raw_data:
            source = data.source
            if source not in sources:
                sources[source] = {
                    "count": 0,
                    "avg_temperature": 0,
                    "conditions": [],
                    "reliability_score": 0
                }
            
            sources[source]["count"] += 1
            sources[source]["conditions"].append(data.conditions)
            sources[source]["reliability_score"] += 1 if not data.error else 0
        
        # Calculate averages and scores
        for source in sources.values():
            source["reliability_percentage"] = (source["reliability_score"] / source["count"]) * 100
            source["primary_condition"] = max(set(source["conditions"]), key=source["conditions"].count)
        
        return {
            "total_sources": len(sources),
            "source_details": sources,
            "recommended_source": max(sources.items(), key=lambda x: x[1]["reliability_percentage"])[0] if sources else None
        }
    
    def _generate_comparative_analysis(self, insights: Dict) -> Dict[str, Any]:
        """Generate comparative analysis against norms"""
        temp_avg = insights.get("temperature", {}).get("avg", 0)
        
        return {
            "temperature_vs_normal": self._compare_temperature_to_normal(temp_avg),
            "humidity_vs_normal": self._compare_humidity_to_normal(insights.get("humidity", {}).get("avg", 0)),
            "seasonal_comparison": self._compare_to_seasonal_norms(temp_avg),
            "extremeness_index": self._calculate_extremeness_index(insights)
        }
    
    def _compare_temperature_to_normal(self, temp: float) -> str:
        """Compare temperature to normal range"""
        if temp > 85:
            return "Much warmer than normal"
        elif temp > 75:
            return "Warmer than normal"
        elif temp > 65:
            return "Near normal"
        elif temp > 55:
            return "Cooler than normal"
        else:
            return "Much cooler than normal"
    
    def _compare_humidity_to_normal(self, humidity: float) -> str:
        """Compare humidity to normal range"""
        if humidity > 70:
            return "More humid than normal"
        elif humidity > 50:
            return "Normal humidity"
        else:
            return "Less humid than normal"
    
    def _compare_to_seasonal_norms(self, temp: float) -> Dict[str, Any]:
        """Compare to seasonal norms"""
        now = datetime.now()
        month = now.month
        
        # Very simplified seasonal norms (US average)
        seasonal_norms = {
            1: 32, 2: 35, 3: 45, 4: 55, 5: 65, 6: 75,
            7: 80, 8: 78, 9: 72, 10: 60, 11: 48, 12: 37
        }
        
        normal_temp = seasonal_norms.get(month, 65)
        difference = temp - normal_temp
        
        return {
            "normal_for_season": normal_temp,
            "difference": difference,
            "interpretation": "above normal" if difference > 5 else "below normal" if difference < -5 else "near normal"
        }
    
    def _calculate_extremeness_index(self, insights: Dict) -> float:
        """Calculate how extreme the weather conditions are (0-100)"""
        score = 0
        
        temp_stats = insights.get("temperature", {})
        conditions = insights.get("conditions", {})
        wind_stats = insights.get("wind", {})
        
        # Temperature extremeness
        temp_avg = temp_stats.get("avg", 0)
        if temp_avg > 90 or temp_avg < 20:
            score += 40
        elif temp_avg > 80 or temp_avg < 32:
            score += 20
        
        # Condition extremeness
        primary_condition = conditions.get("primary", "")
        if primary_condition in ["Thunderstorm", "Blizzard", "Hurricane"]:
            score += 40
        elif primary_condition in ["Heavy Rain", "Heavy Snow"]:
            score += 20
        
        # Wind extremeness
        wind_max = wind_stats.get("max", 0)
        if wind_max > 40:
            score += 20
        elif wind_max > 25:
            score += 10
        
        return min(100, score)
    
    def _analyze_distributions(self, raw_data: List[WeatherSourceData]) -> Dict[str, Any]:
        """Analyze statistical distributions of weather data"""
        temps = [d.temperature for d in raw_data]
        
        if len(temps) < 3:
            return {"insufficient_data": True}
        
        return {
            "temperature_distribution": {
                "skewness": self._calculate_skewness(temps),
                "kurtosis": self._calculate_kurtosis(temps),
                "modality": "unimodal" if len(set(temps)) == 1 else "multimodal",
                "outlier_count": len(self._detect_statistical_outliers(temps))
            }
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data distribution"""
        if len(data) < 3:
            return 0
        
        import math
        n = len(data)
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        
        if std == 0:
            return 0
        
        cubed_deviations = sum((x - mean) ** 3 for x in data)
        skewness = (cubed_deviations / n) / (std ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data distribution"""
        if len(data) < 4:
            return 0
        
        import math
        n = len(data)
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        
        if std == 0:
            return 0
        
        fourth_deviations = sum((x - mean) ** 4 for x in data)
        kurtosis = (fourth_deviations / n) / (std ** 4) - 3
        return kurtosis
    
    def _detect_statistical_outliers(self, data: List[float]) -> List[float]:
        """Detect statistical outliers using IQR method"""
        if len(data) < 4:
            return []
        
        q1 = statistics.quantiles(data, n=4)[0]
        q3 = statistics.quantiles(data, n=4)[2]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return [x for x in data if x < lower_bound or x > upper_bound]
    
    def _analyze_correlations(self, raw_data: List[WeatherSourceData]) -> Dict[str, Any]:
        """Analyze correlations between different weather metrics"""
        if len(raw_data) < 3:
            return {"insufficient_data": True}
        
        temps = [d.temperature for d in raw_data]
        humidities = [d.humidity for d in raw_data]
        pressures = [d.pressure for d in raw_data]
        wind_speeds = [d.wind_speed for d in raw_data]
        
        return {
            "temperature_humidity": self._calculate_correlation(temps, humidities),
            "temperature_pressure": self._calculate_correlation(temps, pressures),
            "wind_temperature": self._calculate_correlation(wind_speeds, temps),
            "interpretation": self._interpret_correlations(temps, humidities, pressures)
        }
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        import math
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def _interpret_correlations(self, temps: List[float], humidities: List[float], pressures: List[float]) -> List[str]:
        """Interpret correlation results"""
        interpretations = []
        
        # Simple interpretations based on common weather knowledge
        if len(temps) >= 3:
            temp_humidity_corr = self._calculate_correlation(temps, humidities)
            if temp_humidity_corr < -0.5:
                interpretations.append("Temperature and humidity are inversely related (common in some climates)")
            elif temp_humidity_corr > 0.5:
                interpretations.append("Temperature and humidity are positively correlated")
        
        return interpretations
    
    def _generate_predictive_insights(self, insights: Dict) -> Dict[str, Any]:
        """Generate simple predictive insights"""
        trends = insights.get("trends", {})
        conditions = insights.get("conditions", {})
        temp_stats = insights.get("temperature", {})
        
        predictions = []
        
        if trends.get("available", False):
            if trends.get("pressure") == "low" and trends.get("temperature") == "rising":
                predictions.append("Conditions may deteriorate with potential for precipitation")
            elif trends.get("pressure") == "high":
                predictions.append("Stable or improving weather likely")
        
        primary_condition = conditions.get("primary", "")
        if primary_condition == "Rain" and temp_stats.get("avg", 0) < 35:
            predictions.append("Potential for freezing rain or snow if temperature drops")
        
        return {
            "short_term_outlook": predictions[:3],  # Limit to 3 predictions
            "confidence": "medium" if len(predictions) > 0 else "low",
            "timeframe": "next 6-12 hours"
        }
    
    def _identify_potential_risks(self, insights: Dict) -> List[Dict[str, Any]]:
        """Identify potential future risks"""
        risks = []
        temp_stats = insights.get("temperature", {})
        conditions = insights.get("conditions", {})
        
        avg_temp = temp_stats.get("avg", 0)
        primary_condition = conditions.get("primary", "")
        
        # Temperature-based risks
        if avg_temp < 32 and primary_condition == "Rain":
            risks.append({
                "type": "freezing_rain",
                "likelihood": "medium",
                "impact": "high",
                "description": "Potential for freezing rain if temperature drops further"
            })
        
        if avg_temp > 85 and conditions.get("humidity", {}).get("avg", 0) > 70:
            risks.append({
                "type": "heat_index",
                "likelihood": "high",
                "impact": "medium",
                "description": "High heat index risk due to combination of heat and humidity"
            })
        
        return risks
    
    def _generate_mitigation_strategies(self, alerts: List[WeatherAlert], insights: Dict) -> List[Dict[str, Any]]:
        """Generate mitigation strategies for identified risks"""
        strategies = []
        
        for alert in alerts:
            if alert.severity in ["critical", "high"]:
                strategies.append({
                    "for_alert": alert.type,
                    "strategy": alert.recommendation or "Take appropriate precautions",
                    "priority": "high",
                    "timeline": "immediate"
                })
        
        # Add general strategies based on conditions
        conditions = insights.get("conditions", {})
        primary_condition = conditions.get("primary", "")
        
        if primary_condition == "Rain":
            strategies.append({
                "for_condition": "rain",
                "strategy": "Carry waterproof gear, allow extra travel time",
                "priority": "medium",
                "timeline": "ongoing"
            })
        
        return strategies
    
    def _generate_recommendations(self, alerts: List[WeatherAlert], insights: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Alert-based recommendations
        for alert in alerts:
            if alert.severity == "critical" and alert.recommendation:
                recommendations.append(f"CRITICAL: {alert.recommendation}")
            elif alert.severity == "high" and alert.recommendation:
                recommendations.append(f"High Priority: {alert.recommendation}")
        
        # Condition-based recommendations
        conditions = insights.get("conditions", {})
        comfort = insights.get("comfort", {})
        temp_stats = insights.get("temperature", {})
        
        primary_condition = conditions.get("primary", "")
        comfort_level = comfort.get("level", "")
        avg_temp = temp_stats.get("avg", 0)
        
        # Clothing recommendations
        if avg_temp < 32:
            recommendations.append("Wear warm layers, gloves, and a hat")
        elif avg_temp < 50:
            recommendations.append("Wear a jacket or sweater")
        elif avg_temp > 80:
            recommendations.append("Wear light, breathable clothing")
        
        # Activity recommendations
        if primary_condition in ["Rain", "Snow"]:
            recommendations.append("Consider indoor activities or postpone outdoor plans")
        elif comfort_level in ["Very Comfortable", "Comfortable"]:
            recommendations.append("Good conditions for outdoor activities")
        
        # Health recommendations
        if avg_temp > 85:
            recommendations.append("Stay hydrated and take breaks in shade or air conditioning")
        if avg_temp < 20:
            recommendations.append("Limit outdoor exposure to prevent frostbite")
        
        # Limit to most important recommendations
        return recommendations[:5]
    
    def _assess_source_reliability(self, raw_data: List[WeatherSourceData]) -> Dict[str, Any]:
        """Assess reliability of different data sources"""
        source_stats = {}
        
        for data in raw_data:
            source = data.source
            if source not in source_stats:
                source_stats[source] = {
                    "count": 0,
                    "errors": 0,
                    "temperatures": [],
                    "conditions": []
                }
            
            source_stats[source]["count"] += 1
            source_stats[source]["errors"] += 1 if data.error else 0
            source_stats[source]["temperatures"].append(data.temperature)
            source_stats[source]["conditions"].append(data.conditions)
        
        # Calculate reliability metrics
        for source, stats in source_stats.items():
            stats["error_rate"] = stats["errors"] / stats["count"]
            stats["temperature_consistency"] = statistics.stdev(stats["temperatures"]) if len(stats["temperatures"]) > 1 else 0
            stats["condition_consistency"] = len(set(stats["conditions"])) / len(stats["conditions"])
            stats["reliability_score"] = 100 * (1 - stats["error_rate"]) * (1 - min(1, stats["temperature_consistency"] / 10))
        
        return {
            "by_source": source_stats,
            "most_reliable": max(source_stats.items(), key=lambda x: x[1]["reliability_score"])[0] if source_stats else None,
            "average_reliability": statistics.mean([s["reliability_score"] for s in source_stats.values()]) if source_stats else 0
        }
    
    def _calculate_completeness_score(self, raw_data: List[WeatherSourceData]) -> float:
        """Calculate data completeness score (0-100)"""
        if not raw_data:
            return 0
        
        completeness_factors = []
        
        for data in raw_data:
            factor = 1.0
            
            # Check for missing critical data
            if data.temperature == 0:
                factor *= 0.7
            if not data.conditions or data.conditions == "Unknown":
                factor *= 0.8
            if data.humidity == 0:
                factor *= 0.9
            if data.wind_speed == 0:
                factor *= 0.9
            
            completeness_factors.append(factor)
        
        avg_completeness = statistics.mean(completeness_factors) if completeness_factors else 0
        return avg_completeness * 100
    
    def _categorize_alerts_by_severity(self, alerts: List[WeatherAlert]) -> Dict[str, int]:
        """Categorize alerts by severity level"""
        categories = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for alert in alerts:
            if alert.severity in categories:
                categories[alert.severity] += 1
        
        return categories
    
    def _categorize_alerts_by_type(self, alerts: List[WeatherAlert]) -> Dict[str, int]:
        """Categorize alerts by type"""
        categories = {}
        
        for alert in alerts:
            alert_type = alert.type
            categories[alert_type] = categories.get(alert_type, 0) + 1
        
        return categories
    
    def _prioritize_alerts(self, alerts: List[WeatherAlert]) -> List[Dict[str, Any]]:
        """Prioritize alerts for attention"""
        severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        
        prioritized = []
        for alert in alerts:
            priority_score = severity_weights.get(alert.severity, 1)
            prioritized.append({
                "alert": alert,
                "priority_score": priority_score,
                "attention_required": alert.severity in ["critical", "high"]
            })
        
        # Sort by priority score (descending)
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
        return prioritized[:5]  # Return top 5