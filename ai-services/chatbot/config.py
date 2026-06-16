"""Application configuration — loaded from environment / .env via pydantic-settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (OpenRouter) ─────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Model id on OpenRouter. Change freely to any model your key can access.
    openrouter_model: str = "openai/gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200
    openrouter_site_url: str = ""
    openrouter_app_title: str = ""

    # ── Web search (Tavily) ──────────────────────────────────────────
    tavily_api_key: str = ""
    tavily_base_url: str = "https://api.tavily.com/search"
    search_max_results: int = 6
    # "basic" (faster/cheaper) or "advanced" (deeper) search depth.
    search_depth: str = "advanced"

    # ── Server ───────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 5000
    log_level: str = "info"

    @property
    def llm_ready(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def search_ready(self) -> bool:
        return bool(self.tavily_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached settings — parsed once per process."""
    return Settings()
