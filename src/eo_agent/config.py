"""Settings read from environment variables (and a local .env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

USER_AGENT = "eo-scene-agent/0.1 (take-home assignment)"


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    mcp_url: str = os.getenv("MCP_URL", "http://127.0.0.1:8001/mcp")
    max_tool_chars: int = _int("MAX_TOOL_CHARS", 4000)
    max_window: int = _int("MAX_WINDOW", 8)
    summary_trigger: int = _int("SUMMARY_TRIGGER", 12)
    summary_max_chars: int = _int("SUMMARY_MAX_CHARS", 1200)
    max_steps: int = _int("MAX_STEPS", 6)


def get_settings() -> Settings:
    return Settings()
