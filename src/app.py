#!/usr/bin/env python3
"""
LLM Provider Abstraction Demo
Demonstrates Adapter and Factory patterns for swappable LLM providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum
import json
import requests
from dataclasses import dataclass


# Common data structures
@dataclass
class Message:
    role: str  # 'system', 'user', 'assistant'
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None


class ProviderType(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


# Abstract base class defining the common interface
class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, messages: List[Message], model: str = None, **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models"""
        pass


# Adapter for OpenAI API
class OpenAIAdapter(LLMProvider):
    """Adapter for OpenAI API"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = "gpt-3.5-turbo"
    
    def generate_response(self, messages: List[Message], model: str = None, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API format"""
        model = model or self.default_model
        
        # Convert our Message format to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Note: In a real implementation, you'd make an actual API call
            # response = requests.post(f"{self.base_url}/chat/completions", 
            #                         headers=headers, json=payload)
            
            # Simulated response for demo purposes
            simulated_response = {
                "choices": [{
                    "message": {
                        "content": f"[OpenAI {model}] This is a simulated response to: {messages[-1].content}"
                    }
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25
                }
            }
            
            return LLMResponse(
                content=simulated_response["choices"][0]["message"]["content"],
                model=model,
                provider="openai",
                usage=simulated_response["usage"]
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def list_models(self) -> List[str]:
        """List available OpenAI models"""
        return ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]


# Adapter for Ollama API
class OllamaAdapter(LLMProvider):
    """Adapter for Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.default_model = "llama2"
    
    def generate_response(self, messages: List[Message], model: str = None, **kwargs) -> LLMResponse:
        """Generate response using Ollama API format"""
        model = model or self.default_model
        
        # Ollama expects a different format - convert messages to a prompt
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000)
            }
        }
        
        try:
            # Note: In a real implementation, you'd make an actual API call
            # response = requests.post(f"{self.base_url}/api/generate", json=payload)
            
            # Simulated response for demo purposes
            simulated_response = {
                "response": f"[Ollama {model}] This is a simulated response to: {messages[-1].content}",
                "done": True,
                "context": [1, 2, 3, 4, 5]
            }
            
            return LLMResponse(
                content=simulated_response["response"],
                model=model,
                provider="ollama",
                usage={"context_length": len(simulated_response.get("context", []))}
            )
            
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a single prompt string for Ollama"""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(prompt_parts) + "\nAssistant: "
    
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        return ["llama2", "codellama", "mistral", "neural-chat"]


# Factory for creating LLM providers
class LLMProviderFactory:
    """Factory for creating LLM provider instances"""
    
    @staticmethod
    def create_provider(provider_type: ProviderType, **config) -> LLMProvider:
        """Create an LLM provider instance based on type"""
        
        if provider_type == ProviderType.OPENAI:
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError("OpenAI provider requires 'api_key' in config")
            
            return OpenAIAdapter(
                api_key=api_key,
                base_url=config.get("base_url", "https://api.openai.com/v1")
            )
        
        elif provider_type == ProviderType.OLLAMA:
            return OllamaAdapter(
                base_url=config.get("base_url", "http://localhost:11434")
            )
        
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_supported_providers() -> List[str]:
        """Get list of supported provider types"""
        return [provider.value for provider in ProviderType]


# Client class that uses the abstraction
class LLMClient:
    """High-level client for interacting with LLMs"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
    
    def chat(self, message: str, conversation_history: List[Message] = None, **kwargs) -> str:
        """Simple chat interface"""
        messages = conversation_history or []
        messages.append(Message(role="user", content=message))
        
        response = self.provider.generate_response(messages, **kwargs)
        return response.content
    
    def chat_with_system_prompt(self, user_message: str, system_prompt: str, **kwargs) -> str:
        """Chat with a system prompt"""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message)
        ]
        
        response = self.provider.generate_response(messages, **kwargs)
        return response.content
    
    def get_available_models(self) -> List[str]:
        """Get available models from the provider"""
        return self.provider.list_models()


# Configuration management
class ConfigManager:
    """Manages configuration for different providers"""
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load configuration (in a real app, this might read from file/env)"""
        return {
            "default_provider": "ollama",
            "providers": {
                "openai": {
                    "api_key": "your-openai-api-key-here",  # Set via environment variable
                    "default_model": "gpt-3.5-turbo"
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "default_model": "llama2"
                }
            }
        }


# Demo application
def main():
    """Demo application showing the patterns in action"""
    print("🚀 LLM Provider Abstraction Demo\n")
    
    # Load configuration
    config = ConfigManager.load_config()
    
    # Demo 1: Using Factory to create different providers
    print("📦 Demo 1: Factory Pattern in Action")
    print("-" * 40)
    
    providers = {}
    
    # Create Ollama provider
    try:
        providers["ollama"] = LLMProviderFactory.create_provider(
            ProviderType.OLLAMA,
            **config["providers"]["ollama"]
        )
        print("✅ Ollama provider created successfully")
    except Exception as e:
        print(f"❌ Failed to create Ollama provider: {e}")
    
    # Create OpenAI provider
    try:
        providers["openai"] = LLMProviderFactory.create_provider(
            ProviderType.OPENAI,
            **config["providers"]["openai"]
        )
        print("✅ OpenAI provider created successfully")
    except Exception as e:
        print(f"❌ Failed to create OpenAI provider: {e}")
    
    print(f"\n📋 Supported providers: {LLMProviderFactory.get_supported_providers()}")
    
    # Demo 2: Using the same interface with different providers
    print("\n🔄 Demo 2: Adapter Pattern in Action")
    print("-" * 40)
    
    test_message = "What is the capital of France?"
    system_prompt = "You are a helpful geography assistant."
    
    for name, provider in providers.items():
        if provider:
            print(f"\n🤖 Using {name.upper()} provider:")
            client = LLMClient(provider)
            
            # Show available models
            models = client.get_available_models()
            print(f"Available models: {', '.join(models)}")
            
            # Test basic chat
            try:
                response = client.chat_with_system_prompt(test_message, system_prompt)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
    
    # Demo 3: Runtime provider switching
    print("\n🔀 Demo 3: Runtime Provider Switching")
    print("-" * 40)
    
    # Simulate switching providers based on user preference or load balancing
    current_provider = "ollama"
    
    for i in range(3):
        # Alternate between providers
        current_provider = "openai" if current_provider == "ollama" else "ollama"
        
        if current_provider in providers and providers[current_provider]:
            client = LLMClient(providers[current_provider])
            response = client.chat(f"Request #{i+1}: Tell me a fun fact!")
            print(f"Provider {current_provider}: {response}")
    
    print("\n✨ Demo completed! The patterns allow for:")
    print("  • Easy swapping between LLM providers")
    print("  • Consistent interface regardless of underlying API")
    print("  • Simple addition of new providers")
    print("  • Runtime configuration and switching")


if __name__ == "__main__":
    main()