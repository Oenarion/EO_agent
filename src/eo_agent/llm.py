"""Model factory. All supported providers speak the OpenAI-compatible API,
so switching provider is a config change (LLM_PROVIDER, LLM_MODEL, ...)."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from eo_agent.config import Settings, get_settings

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    if s.llm_provider not in DEFAULT_BASE_URLS:
        raise ValueError(f"Unknown LLM_PROVIDER '{s.llm_provider}'. Use one of {list(DEFAULT_BASE_URLS)}.")
    if not s.llm_model:
        raise ValueError("LLM_MODEL is not set. Copy .env.example to .env and fill it in.")
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url or DEFAULT_BASE_URLS[s.llm_provider],
        api_key=s.llm_api_key or "not-needed",  # Ollama and vLLM ignore the key
        temperature=0,
        timeout=60,
        max_retries=1,
    )
