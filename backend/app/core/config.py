"""
Runtime configuration loaded once at startup.

- Secrets / connection strings come from `.env` (validated by pydantic-settings).
- Tuning knobs (chunk sizes, model task map, etc.) come from
  `config/config.yaml` and `config/models.yaml`.
- Both layers are merged into a single `Settings` instance accessible via
  `get_settings()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


class Settings(BaseSettings):
    """Environment-driven secrets and connection strings."""

    # --- Secrets / connection strings (from .env) ---
    database_url: str = Field(..., description="postgresql+asyncpg://...")
    redis_url: str = Field("redis://localhost:6379/0")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None

    render_api_key: str | None = None

    storage_dir: Path = Path("./uploads")
    bearer_token: str = Field(..., description="Single shared API auth token")

    log_level: str = "INFO"
    env: str = "development"

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Layered YAML config (loaded lazily) ---
    @property
    def app_config(self) -> dict[str, Any]:
        return _load_yaml(CONFIG_DIR / "config.yaml")

    @property
    def models_config(self) -> dict[str, Any]:
        return _load_yaml(CONFIG_DIR / "models.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Did you `cp config/{path.stem}.example.yaml config/{path.name}`?"
        )
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
