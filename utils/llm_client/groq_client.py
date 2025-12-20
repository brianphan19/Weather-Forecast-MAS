import time
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

from .base_client import BaseLLMClient, LLMError


class GroqClient(BaseLLMClient):
    """Client for Groq models"""
    
    def __init__(self, api_key: str, temperature: float = 0.3, max_tokens: int = 1000, model: Optional[str] = None):
        super().__init__(api_key, temperature, max_tokens)
        self.model = model or "mixtral-8x7b-32768"
        self.client = None
        self._initialize_client()
    
    @property
    def provider_name(self) -> str:
        return "groq"
    
    @property
    def default_model(self) -> str:
        return "mixtral-8x7b-32768"
    
    def _initialize_client(self):
        """Initialize Groq client if available"""
        try:
            import groq
            self.client = groq.Groq(api_key=self.api_key)
        except ImportError:
            self.client = None
        except Exception as e:
            print(f"⚠️  Failed to initialize Groq client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Groq client is available"""
        return bool(self.api_key and self.client)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,))
    )
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Groq API"""
        if not self.is_available():
            raise LLMError("Groq client is not available (missing API key or library)")
        
        self._start_timer()
        
        try:
            messages = self._format_messages(prompt, system_prompt)
            
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                timeout=self.timeout
            )
            
            content = response.choices[0].message.content
            
            # Groq provides token usage
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else len(content.split()) * 1.3
            self._stop_timer(int(tokens_used))
            
            return content
            
        except ImportError:
            raise LLMError("Groq library not installed. Run: pip install groq")
        except Exception as e:
            self._stop_timer()
            raise LLMError(f"Groq API error: {e}")
    
    def __str__(self) -> str:
        return f"GroqClient(model={self.model}, available={self.is_available()})"