# LLM Provider Abstraction

A demonstration of using **Adapter** and **Factory** design patterns to create a flexible, swappable LLM provider system. This allows applications to easily switch between different LLM providers (OpenAI, Ollama, etc.) without changing business logic.

## Problem Statement

Different LLM providers have different APIs, data formats, and integration patterns:

- **OpenAI**: Uses `messages` array with structured chat format
- **Ollama**: Uses simple prompt strings and different response structure  
- **Anthropic**: Has different parameter names and response formats
- **Local models**: May have completely different interfaces

This creates tight coupling between your application and specific providers, making it hard to:
- Switch providers for cost optimization
- A/B test different models
- Implement fallback mechanisms
- Support multiple providers simultaneously

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LLMClient     │───▶│  LLMProvider     │◀───│ ProviderFactory │
│ (Application)   │    │  (Interface)     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               ▲                         │
                               │                         │
                    ┌──────────┴──────────┐              │
                    │                     │              │
            ┌───────▼────────┐    ┌──────▼──────────┐    │
            │ OpenAIAdapter  │    │  OllamaAdapter  │◀───┘
            │   (Adapter)    │    │   (Adapter)     │
            └────────────────┘    └─────────────────┘
                    │                      │
            ┌───────▼────────┐    ┌──────▼──────────┐
            │  OpenAI API    │    │   Ollama API    │
            └────────────────┘    └─────────────────┘
```

## Design Patterns Used

### 1. Adapter Pattern
**Purpose**: Convert the interface of different LLM providers into a common interface.

**Why it's perfect here**:
- Each provider has a different API structure
- We need to normalize responses into a consistent format
- Legacy APIs need to work with modern application interfaces

**Implementation**:
```python
class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Message], model: str = None, **kwargs) -> LLMResponse:
        pass

class OpenAIAdapter(LLMProvider):
    def generate_response(self, messages, model=None, **kwargs):
        # Convert our format to OpenAI's format
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        # Make API call and convert response back
        return LLMResponse(...)

class OllamaAdapter(LLMProvider):
    def generate_response(self, messages, model=None, **kwargs):
        # Convert messages to Ollama's prompt format
        prompt = self._messages_to_prompt(messages)
        # Make API call and convert response back
        return LLMResponse(...)
```

### 2. Factory Pattern
**Purpose**: Create provider instances without exposing the instantiation logic.

**Why it's useful here**:
- Centralizes provider creation logic
- Handles different configuration requirements per provider
- Makes adding new providers easier
- Enables runtime provider selection

**Implementation**:
```python
class LLMProviderFactory:
    @staticmethod
    def create_provider(provider_type: ProviderType, **config) -> LLMProvider:
        if provider_type == ProviderType.OPENAI:
            return OpenAIAdapter(api_key=config["api_key"])
        elif provider_type == ProviderType.OLLAMA:
            return OllamaAdapter(base_url=config["base_url"])
        # Easy to add new providers here
```

## Usage Examples

### Basic Usage
```python
# Create provider using factory
provider = LLMProviderFactory.create_provider(
    ProviderType.OLLAMA,
    base_url="http://localhost:11434"
)

# Use the same interface regardless of provider
client = LLMClient(provider)
response = client.chat("What is machine learning?")
print(response)
```

### Runtime Provider Switching
```python
# Switch providers at runtime
config = load_config()
provider_name = config.get("preferred_provider", "ollama")

provider = LLMProviderFactory.create_provider(
    ProviderType(provider_name),
    **config["providers"][provider_name]
)

client = LLMClient(provider)
```

### Configuration-Driven Setup
```python
def create_client_from_config():
    config = {
        "provider": "openai",
        "openai_config": {"api_key": "sk-..."},
        "ollama_config": {"base_url": "http://localhost:11434"}
    }
    
    provider = LLMProviderFactory.create_provider(
        ProviderType(config["provider"]),
        **config[f"{config['provider']}_config"]
    )
    
    return LLMClient(provider)
```

## Alternative Design Patterns

While Adapter + Factory works well, here are other patterns to consider:

### Strategy Pattern
**When to use**: If you need runtime algorithm selection or behavior switching.

```python
class LLMStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: pass

class OpenAIStrategy(LLMStrategy): ...
class OllamaStrategy(LLMStrategy): ...

class LLMContext:
    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: LLMStrategy):
        self.strategy = strategy
```

**Pros**: 
- Runtime strategy switching
- Clean separation of algorithms
- Easy to add new strategies

**Cons**: 
- Client needs to know about concrete strategies
- May be overkill for simple cases

### Abstract Factory Pattern
**When to use**: When you need families of related objects (chat models, embedding models, image models).

```python
class LLMAbstractFactory(ABC):
    @abstractmethod
    def create_chat_model(self) -> ChatModel: pass
    
    @abstractmethod
    def create_embedding_model(self) -> EmbeddingModel: pass

class OpenAIFactory(LLMAbstractFactory):
    def create_chat_model(self) -> ChatModel:
        return OpenAIChatModel()
    
    def create_embedding_model(self) -> EmbeddingModel:
        return OpenAIEmbeddingModel()
```

**Pros**: 
- Creates families of related objects
- Ensures compatibility between related objects
- Easy to swap entire provider families

**Cons**: 
- More complex than simple Factory
- Overkill if you only need one type of object

### Bridge Pattern
**When to use**: When you want to separate abstraction from implementation and both can vary independently.

```python
class LLMAbstraction:
    def __init__(self, implementation: LLMImplementation):
        self.implementation = implementation

class ChatBot(LLMAbstraction):
    def chat(self, message: str) -> str:
        return self.implementation.process(message)

class CodeAssistant(LLMAbstraction):
    def generate_code(self, prompt: str) -> str:
        return self.implementation.process(f"Generate code: {prompt}")
```

**Pros**: 
- Separates interface from implementation
- Both can evolve independently
- Great for complex hierarchies

**Cons**: 
- Increased complexity
- May be overkill for simple use cases

### Command Pattern
**When to use**: When you want to parameterize requests, queue them, or support undo operations.

```python
class LLMCommand(ABC):
    @abstractmethod
    def execute(self) -> LLMResponse: pass

class ChatCommand(LLMCommand):
    def __init__(self, provider: LLMProvider, message: str):
        self.provider = provider
        self.message = message
    
    def execute(self) -> LLMResponse:
        return self.provider.generate_response([Message("user", self.message)])
```

**Pros**: 
- Decouples request from execution
- Supports queuing, logging, undo
- Great for complex request processing

**Cons**: 
- Adds indirection
- May be overkill for simple cases

## Pattern Selection Guide

| Pattern | Use When | Complexity | Flexibility |
|---------|----------|------------|-------------|
| **Adapter + Factory** | Different APIs need common interface | Low-Medium | Medium |
| **Strategy** | Runtime algorithm switching needed | Low | High |
| **Abstract Factory** | Need families of related objects | Medium | Medium-High |
| **Bridge** | Both abstraction and implementation vary | High | Very High |
| **Command** | Need request queuing, logging, undo | Medium | High |

## Running the Demo

1. **Install dependencies**:
   ```bash
   pip install requests  # For API calls (if using real APIs)
   ```

2. **Run the demo**:
   ```bash
   python llm_provider_demo.py
   ```

3. **For real API usage**:
   - Set up Ollama: `ollama run llama2`
   - Set OpenAI API key: `export OPENAI_API_KEY=your_key`
   - Uncomment the actual API calls in the adapter classes

## Configuration

Create a `config.json` file:

```json
{
  "default_provider": "ollama",
  "providers": {
    "openai": {
      "api_key": "${OPENAI_API_KEY}",
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-3.5-turbo"
    },
    "ollama": {
      "base_url": "http://localhost:11434",
      "default_model": "llama2"
    }
  }
}
```

## Adding New Providers

Adding a new provider is straightforward:

1. **Create an Adapter**:
   ```python
   class AnthropicAdapter(LLMProvider):
       def generate_response(self, messages, model=None, **kwargs):
           # Implement Anthropic's API format
           pass
   ```

2. **Add to Factory**:
   ```python
   # Add to ProviderType enum
   ANTHROPIC = "anthropic"
   
   # Add to factory
   elif provider_type == ProviderType.ANTHROPIC:
       return AnthropicAdapter(api_key=config["api_key"])
   ```

3. **Update Configuration**:
   ```json
   "anthropic": {
     "api_key": "${ANTHROPIC_API_KEY}",
     "default_model": "claude-3-sonnet"
   }
   ```

## Benefits

- **Loose Coupling**: Application logic independent of specific providers  
- **Easy Testing**: Mock providers for unit tests  
- **Cost Optimization**: Switch providers based on cost/performance  
- **Fault Tolerance**: Implement fallback mechanisms  
- **Future-Proof**: Easy to add new providers as they emerge  
- **Consistent Interface**: Same code works with any provider  
- **Configuration-Driven**: Change providers without code changes


## Further Reading

- [Adapter Pattern - Gang of Four](https://en.wikipedia.org/wiki/Adapter_pattern)
- [Factory Pattern - Gang of Four](https://en.wikipedia.org/wiki/Factory_method_pattern)
- [When to Use Which Design Pattern](https://refactoring.guru/design-patterns/catalog)
