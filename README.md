# Weather Forecast Multi-Agent System (MAS)

A production-style **3-agent weather forecasting system** built using **LangGraph**, designed to collect weather data from multiple APIs, reconcile inconsistencies, generate alerts, and provide **LLM-powered analysis and recommendations**.
---

## Key Features

- Multi-source weather data acquisition  
- Deterministic statistical analysis and consensus building  
- LLM-powered reasoning, Q&A, and recommendations  
- Structured, explainable outputs  
- Modular and extensible agent architecture  

---

## High-Level Architecture

The system is organized around 4 cooperating agents coordinated via a LangGraph state machine.

![image](workflow_graph.png)
---

## Set up

1. **Clone the repo**:
```bash
git clone https://github.com/brianphan19/Weather-Forecast-MAS
cd <repo-folder>
```

2. **Create and activate a virtual environment**:
```bash
python -m venv venv  
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```


4. **Create a `.env` file** in the root folder:

```dotenv
# Content directory
CONTENT_DIR=./data

# OpenAI API Configuration
OPENAI_API_KEY=
OPENAI_MODEL=gpt-3.5-turbo


# Groq API Configuration (alternative to OpenAI)
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

# Google Gemini API Configuration (alternative to OpenAI/Groq)
GOOGLE_API_KEY=
GOOGLE_MODEL=gemini-pro

# Embedding Configuration
# Optional: HuggingFace model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Database Configuration
# Optional: ChromaDB collection name (default: rag_documents)
CHROMA_COLLECTION_NAME=rag_documents
```
## Usage

1. **Configure API Key**:
```ini
OPENWEATHER_API_KEY=your_key
WEATHERAPI_KEY=your_key
VISUALCROSSING_API_KEY=your_key
OPENAI_API_KEY=your_key
```

2. **Run the multi-agent system**:
```python
import asyncio
from main import run_forecast   # adjust import if needed

async def main():
    while True:
        location = input("Location: ").strip() or config.weather.default_location
        
        user_question = input("Questions: ").strip() or "weather?"

        print(f"\nUsing location: {location}")

        orchestrator = WeatherMASOrchestrator()
        results = await orchestrator.get_weather_forecast(
            location, user_question
        )
        display_results(results)

        rerun_forecast = input("Quit? (Y/n): ").strip()
        if rerun_forecast.lower() == "y" or None:
            break
```

3. **Example interaction**:
```yaml
Location (or 'quit'): Ha Noi 
Questions: Will it rain tomorrow?

...

Answers:
  - Will it rain tomorrow?: There is a slight chance of scattered showers, but it's not expected to be a heavy downpour.
```

## Agent Responsibilities

### Agent 1 – Data Acquisition Agent

**Purpose**  
Collect reliable weather data from multiple external providers and prepare it for downstream analysis.

**Responsibilities**
- Initialize all available weather API clients from configuration
- Fetch weather data concurrently from multiple sources
- Normalize heterogeneous API responses into a unified internal schema
- Handle provider failures and timeouts without stopping execution
- Track source-level errors and availability
- Compute a consensus summary across providers
- Update shared workflow state for downstream agents

**Outputs**
- `raw_weather_data[]` – normalized weather data from each source
- `weather_consensus` – aggregated statistics across providers
- Source success/failure metadata
- Updated Agent 1 execution status

---

### Agent 2 – Weather Data Report

**Purpose**
Transform raw multi-source weather data into a comprehensive, structured report that downstream agents and LLM components can easily interpret.

**Responsibilities**
- Validate and consume normalized weather data from Agent 1
- Compute:
  - temperature ranges and averages
  - humidity and wind statistics
  - consistency and variance
- Detect anomalies and severe conditions
- Generate alerts with severity levels

**Outputs**
- `weather_report` – comprehensive structured report
- `analysis_summary` – executive summary for downstream use
- `detailed_insights` – computed analytical metrics
- `weather_alerts[]` – generated alerts and risk warnings
- Updated Agent 2 execution status
  
---

### Agent 3 – Query Enhancer Agent

**Purpose**  
Improve user weather queries before LLM analysis so downstream agents receive clearer, more informative questions without changing the original intent.

**Responsibilities**
- Evaluate whether a user query needs improvement
- Detect vague, short, or generic weather questions
- Rewrite queries using an LLM for clarity and specificity
- Preserve original user intent while adding useful context
- Fall back safely to the original query if enhancement fails
- Update workflow state with the enhanced query

**Enhancement Logic**
- Queries are enhanced when they are:
  - Very short
  - Generic (e.g., "weather?", "forecast", "temperature")
- Queries that are already descriptive are passed through unchanged

**LLM Usage**
- Uses a multi-provider LLM client
- Applies retry logic for reliability
- Returns only the rewritten query
- Falls back to original query if providers are unavailable

**Outputs**
- Updated `user_question` in workflow state
- Agent execution status updates
- Optional enhanced query logging
- Updated Agent 3 execution status

---

### Agent 4 – LLM Analysis Agent

**Purpose**  
Transform structured weather analysis into human-friendly insights, recommendations, and direct answers to user questions using a Large Language Model.

**Responsibilities**
- Consume structured weather reports produced by previous agents
- Generate natural-language analysis and summaries
- Answer user weather-related questions
- Produce actionable recommendations
- Provide follow-up questions and confidence scores
- Safely fall back to rule-based summaries if LLM access fails
- Update workflow state with structured LLM outputs

**LLM Processing Flow**
- Formats structured weather reports into LLM-friendly prompts
- Sends prompts using a multi-provider LLM client
- Applies retry logic for reliability
- Requires responses in structured JSON format
- Parses and validates LLM outputs before storing results

**Fallback Behavior**
- Automatically generates simplified analysis if:
  - No LLM provider is available
  - API calls fail
  - Output parsing fails
- Ensures workflow continuity even without LLM services

**Outputs**
- `llm_response`
- `llm_analysis`
- User-facing analysis, recommendations, and answers
- Updated Agent 4 execution status

---

## Shared State Model

All agents operate on a shared immutable `AgentState`.

**Key fields**
- `location`, `request_id`, `user_question`
- `agent1_status`, `agent2_status`, `agent3_status`, `agent4_status`
- `raw_weather_data[]`
- `weather_consensus`
- `weather_alerts[]`
- `analysis_summary`
- `detailed_insights`
- `llm_response`
- `execution_time_ms`
- `errors[]`

This ensures:
- Full traceability
- Deterministic debugging
- Clean agent boundaries

---

## Execution Flow

1. User provides location (default: New York, NY, USA) and optional question (default: Should I go out?)
2. Orchestrator initializes LangGraph  
3. Initial `AgentState` is created  
4. Agent 1 fetches and normalizes data  
5. Agent 2 combine and builds data report  
6. Agent 3 improve query
7. Agent 4 reasons, answers, and evaluates confidence  
8. Final response formatted and returned  

---

## Output Structure

Final responses include:
- Workflow status and execution timing
- Agent completion states
- Weather summary
- Alerts with severity and recommendations
- LLM-generated insights and answers
- Raw data preview

---

## Project Structure

```
.
├── agents/
│   ├── __init__.py
│   ├── data_acquition.py
│   ├── llm_chat.py
│   └── weather_data_report.py   
│   
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── utils/
│   ├── __init__.py
│   ├── llm_client
│   │   ├── __init__.py
│   │   ├── base_client.py
│   │   ├── gemini_client.py
│   │   ├── groq_client.py
│   │   ├── openai_client.py
│   │   └── multi_provider.py
│   │ 
│   └── weather_clients
│       ├── __init__.py
│       ├── base_client.py
│       ├── openweather_client.py
│       ├── visualcrossing_client.py
│       └── weatherapi_client.py
│
├── workflows/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── weather_workflow.py
│   └── state.py
│
├── main.py
│
├── LICENSE
├── requirement.txt
├── .env
└── README.md
```

---

## Configuration

Environment-based configuration via `Config.from_env()`.

### Weather APIs
```
OPENWEATHER_API_KEY=...
WEATHERAPI_KEY=...
VISUALCROSSING_API_KEY=...
```

### LLM Providers
```
OPENAI_API_KEY=...
GROQ_API_KEY=...
GEMINI_API_KEY=...
```

At least one LLM provider must be available.

---

## Running the System

```
python app/main.py
```

**Example**
```
python app/main.py "London" "Should I bring an umbrella?"
```

---

## Design Principles

- Clear separation of concerns between agents
- Deterministic analysis before LLM reasoning
- Confidence-aware decision making
- Easy extensibility (add agents or tools)
