import time
from typing import Optional, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from .base_client import BaseLLMClient, LLMError


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models"""
    
    def __init__(self, api_key: str, temperature: float = 0.3, max_tokens: int = 1000, model: Optional[str] = None):
        super().__init__(api_key, temperature, max_tokens)
        self.model = model or "gemini-1.5-pro"
        self.genai = None
        self.safety_settings = None
        self._initialize_client()
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    @property
    def default_model(self) -> str:
        return "gemini-1.5-pro"
    
    def _initialize_client(self):
        """Initialize Gemini client if available"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            
            # Safety settings
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
        except ImportError:
            self.genai = None
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini client: {e}")
            self.genai = None
    
    def is_available(self) -> bool:
        """Check if Gemini client is available"""
        return bool(self.api_key and self.genai)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,))
    )
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Gemini API"""
        if not self.is_available():
            raise LLMError("Gemini client is not available (missing API key or library)")
        
        self._start_timer()
        
        try:
            # Combine system prompt with user prompt for Gemini
            full_prompt = ""
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Initialize model
            model = self.genai.GenerativeModel(
                model_name=kwargs.get('model', self.model),
                safety_settings=self.safety_settings
            )
            
            # Generate response
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": kwargs.get('temperature', self.temperature),
                    "max_output_tokens": kwargs.get('max_tokens', self.max_tokens),
                }
            )
            
            # Check for blocked responses
            if not response.parts:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    raise LLMError(f"Content blocked: {response.prompt_feedback.block_reason}")
                else:
                    raise LLMError("Empty response from Gemini")
            
            content = response.text
            
            # Estimate tokens
            tokens_used = len(content.split()) * 1.3
            self._stop_timer(int(tokens_used))
            
            return content
            
        except ImportError:
            raise LLMError("Google Generative AI library not installed. Run: pip install google-generativeai")
        except Exception as e:
            self._stop_timer()
            raise LLMError(f"Gemini API error: {e}")
    
    def __str__(self) -> str:
        return f"GeminiClient(model={self.model}, available={self.is_available()})"