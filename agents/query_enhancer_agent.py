import asyncio
from typing import Dict, Any
from config.settings import Config
from utils.llm_client.multi_provider import MultiProviderLLM
from workflows.state import AgentStatus, AgentState

class QueryEnhancerAgent:
    def __init__(self, config: Config):
        self.config = config
        self.llm_config = config.llm
        self.llm_client = None
        self._initialize_llm_client()

    def _initialize_llm_client(self):
        """Initialize the multi-provider LLM client."""
        try:
            self.llm_client = MultiProviderLLM(self.llm_config)
            print(f"QueryEnhancerAgent initialized with {self.llm_client.current_provider}")
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

    async def _generate_llm_response(self,user_question: str, location: str) -> str:
        """
        Generate LLM response based on weather report and user question.
        """
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(user_question, location)

        try:
            if self.llm_client and self.llm_config.is_any_provider_available():
                response_text = await self._call_llm_with_retry(system_prompt, user_prompt)
                return response_text
            else:
                print("No LLM API keys available, using original query")
                return user_question
        except Exception as e:
            print(f"LLM API call failed: {e}, using original query")
            return user_question

    async def _call_llm_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2
    ) -> str:
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.llm_client.generate_response,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=300
                )
                return response.strip()

            except Exception:
                if attempt < max_retries:
                    await asyncio.sleep(1)
                else:
                    raise

    def _create_system_prompt(self) -> str:
        return """You improve weather-related user queries.

Rewrite queries to be clearer and more detailed while keeping
the original meaning unchanged.

Return ONLY the improved query.
"""

    def _create_user_prompt(self, query: str, location: str) -> str:
        return f"""Location: {location}
User Query: {query}

Rewrite this query for better weather analysis.
"""

    async def enhance_query(self, state:AgentState) -> AgentState:
        print(f"Agent 3: Enhancing Query: {state['user_question']}")

        try:
            state["agent3_status"] = AgentStatus.PROCESSING

            enhanced_query = await self._generate_llm_response(
                user_question=state["user_question"],
                location=state["location"],
            )

            state["user_question"] = enhanced_query
            print("  - Enhanced Query:", enhanced_query)
            state["agent3_status"] = AgentStatus.COMPLETED

        except Exception as e:      
            state["agent3_status"] = AgentStatus.FAILED
            state["errors"].append(f"Agent 3 query enhancer failed: {str(e)}")
        return state


