from typing import Dict, Any, Optional, List
import time

from .base_client import BaseLLMClient, LLMError
from .openai_client import OpenAIClient
from .groq_client import GroqClient
from .gemini_client import GeminiClient
from config.settings import LLMConfig


class LLMClientFactory:
    """Factory for creating and managing LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.clients: Dict[str, BaseLLMClient] = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all available clients based on configuration"""
        # OpenAI
        if self.config.openai_api_key:
            self.clients["openai"] = OpenAIClient(
                api_key=self.config.openai_api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                model=self.config.openai_model
            )
        
        # Groq
        if self.config.groq_api_key:
            self.clients["groq"] = GroqClient(
                api_key=self.config.groq_api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                model=self.config.groq_model
            )
        
        # Gemini
        if self.config.gemini_api_key:
            self.clients["gemini"] = GeminiClient(
                api_key=self.config.gemini_api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                model=self.config.gemini_model
            )
        
    
    def get_client(self, provider: str = "auto") -> BaseLLMClient:
        """
        Get a client for the specified provider
        
        Args:
            provider: "openai", "groq", "gemini", "fallback", or "auto" for best available
            
        Returns:
            BaseLLMClient instance
        """
        if provider == "auto":
            return self.get_best_available_client()
        
        if provider in self.clients:
            return self.clients[provider]
        
        raise ValueError(f"Unknown provider: {provider}. Available: {list(self.clients.keys())}")
    
    def get_best_available_client(self) -> BaseLLMClient:
        """
        Get the best available LLM client
        Priority: openai > groq > gemini > fallback
        """
        # Check in priority order
        for provider in ["openai", "groq", "gemini"]:
            if provider in self.clients:
                client = self.clients[provider]
                if client.is_available():
                    return client
        

    
    def list_available_clients(self) -> List[Dict[str, Any]]:
        """List all clients with their status"""
        clients_info = []
        
        for provider in ["openai", "groq", "gemini", "fallback"]:
            if provider in self.clients:
                client = self.clients[provider]
                clients_info.append({
                    "provider": provider,
                    "available": client.is_available(),
                    "model": getattr(client, 'model', 'rule-based'),
                    "temperature": client.temperature,
                    "calls": client.total_calls,
                    "tokens": client.total_tokens
                })
        
        return clients_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all clients"""
        stats = {}
        for provider, client in self.clients.items():
            stats[provider] = client.get_stats()
        return stats


class MultiProviderLLM:
    """
    High-level LLM interface that automatically uses the best available provider
    Can also manually switch between providers
    """
    
    def __init__(self, config: LLMConfig):
        self.factory = LLMClientFactory(config)
        self.config = config
        self.client = self.factory.get_best_available_client()
        self.current_provider = self.client.provider_name
        
        # Performance tracking
        self.provider_performance: Dict[str, Dict] = {}
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                         provider: str = "auto", **kwargs) -> str:
        """
        Generate a response using the specified or best available provider
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            provider: "auto" or specific provider name
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Generated response
        """
        start_time = time.time()
        
        # Get the appropriate client
        if provider == "auto":
            client = self.factory.get_best_available_client()
        else:
            client = self.factory.get_client(provider)
        
        # Update current client if different
        if client != self.client:
            self.client = client
            self.current_provider = client.provider_name
        
        try:
            response = self.client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                **{**self._get_default_kwargs(), **kwargs}
            )
            
            response_time = time.time() - start_time
            
            # Track performance
            self._track_performance(self.current_provider, response_time, True)
            
            return response
            
        except LLMError as e:
            # Track failure
            self._track_performance(self.current_provider, time.time() - start_time, False)
            
            # Try to fallback to another provider if current failed
            if self.current_provider != "fallback":
                print(f"❌ {self.current_provider} failed: {e}")
                print("🔄 Attempting to use another provider...")
                
                # Get next best available (excluding current)
                fallback_client = self._get_next_best_provider(exclude=self.current_provider)
                
                if fallback_client and fallback_client != self.client:
                    self.client = fallback_client
                    self.current_provider = fallback_client.provider_name
                    print(f"✅ Switched to {self.current_provider}")
                    
                    # Retry with fallback
                    return self.generate_response(prompt, system_prompt, **kwargs)
            
            # If all else fails, re-raise the error
            raise
    
    def _get_default_kwargs(self) -> Dict[str, Any]:
        """Get default kwargs for LLM calls"""
        return {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
    
    def _get_next_best_provider(self, exclude: str) -> Optional[BaseLLMClient]:
        """Get the next best provider excluding the specified one"""
        priority_order = ["openai", "groq", "gemini", "fallback"]
        
        # Remove excluded provider
        available_providers = [p for p in priority_order if p != exclude]
        
        # Try each in order
        for provider in available_providers:
            try:
                client = self.factory.get_client(provider)
                if client.is_available():
                    return client
            except:
                continue
        
        return None
    
    def _track_performance(self, provider: str, response_time: float, success: bool):
        """Track provider performance"""
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "avg_response_time": 0.0
            }
        
        stats = self.provider_performance[provider]
        stats["total_calls"] += 1
        stats["total_time"] += response_time
        
        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
        
        stats["avg_response_time"] = stats["total_time"] / stats["total_calls"]
    
    def switch_provider(self, provider: str) -> bool:
        """Switch to a specific provider"""
        try:
            client = self.factory.get_client(provider)
            if client.is_available() or provider == "fallback":
                self.client = client
                self.current_provider = client.provider_name
                return True
            else:
                print(f"⚠️  Provider '{provider}' is not available")
                return False
        except Exception as e:
            print(f"❌ Error switching provider: {e}")
            return False
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available providers"""
        return self.factory.list_available_clients()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "current_provider": self.current_provider,
            "provider_stats": self.factory.get_stats(),
            "performance_stats": self.provider_performance
        }
    
    def __str__(self) -> str:
        return f"MultiProviderLLM(current={self.current_provider}, available={self.get_available_providers()})"