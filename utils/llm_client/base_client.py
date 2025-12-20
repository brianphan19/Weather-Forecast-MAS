from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time


class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients"""
    
    def __init__(self, api_key: str, temperature: float = 0.3, max_tokens: int = 1000):
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = 30
        
        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.start_time = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'openai', 'groq', 'gemini')"""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this provider"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the client is available (has API key and dependencies)"""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a response from the LLM"""
        pass
    
    def _format_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Format messages for chat-based APIs"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _start_timer(self):
        """Start performance timer"""
        self.start_time = time.time()
    
    def _stop_timer(self, tokens_used: int = 0):
        """Stop timer and update statistics"""
        if self.start_time:
            response_time = time.time() - self.start_time
            self.total_calls += 1
            self.total_tokens += tokens_used
            self.total_time += response_time
            self.start_time = None
            return response_time
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'provider': self.provider_name,
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'total_time': self.total_time,
            'avg_time_per_call': self.total_time / self.total_calls if self.total_calls > 0 else 0,
            'tokens_per_call': self.total_tokens / self.total_calls if self.total_calls > 0 else 0
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.provider_name.title()}Client(available={self.is_available()})"