"""
Runtime configuration loaded once at startup.

- Secrets / connection strings come from `.env` (validated by pydantic-settings).
- Tuning knobs (chunk sizes, model task map, etc.) come from
  `config/config.yaml` and `config/models.yaml`.
- Both layers are merged into a single `Settings` instance accessible via
  `get_settings()`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


def cache_root() -> Path:
    """Root of the on-disk caches (summaries, tables, corpus profiles).

    `GRAPHRAG_CACHE_ROOT` relocates it -- used to give each benchmark arm its
    own cache, since the table cache is keyed on document bytes only and would
    otherwise hand one LLM mode's extractions to another.
    """
    override = os.environ.get("GRAPHRAG_CACHE_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "your-end-to-end-graphrag-implementation"


class Settings(BaseSettings):
    """Environment-driven secrets and connection strings."""

    # --- Secrets / connection strings (from .env) ---
    database_url: str = Field(
        ...,
        description=(
            "Postgres DSN. Accepts the bare Supabase format "
            "(postgresql://postgres:PASSWORD@db.xxx.supabase.co:5432/postgres); "
            "the validator below normalizes the scheme to asyncpg and appends "
            "?ssl=require for *.supabase.co hosts."
        ),
    )
    redis_url: str = Field("redis://localhost:6379/0")

    openai_api_key: str | None = None
    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    deepinfra_api_key: str | None = None

    render_api_key: str | None = None

    storage_dir: Path = Path("./uploads")
    bearer_token: str = Field(..., description="Single shared API auth token")

    frontend_origin: str = Field(
        "http://localhost:5173",
        description=(
            "CORS allow-origin(s) for the React UI. Comma-separate to allow "
            "multiple (e.g. dev + prod): "
            "'http://localhost:5173,https://your-frontend.onrender.com'."
        ),
    )

    @property
    def frontend_origins(self) -> list[str]:
        """Parse `frontend_origin` as a CSV into a list of allowed origins."""
        return [o.strip() for o in self.frontend_origin.split(",") if o.strip()]

    log_level: str = "INFO"
    env: str = "development"

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("database_url", mode="after")
    @classmethod
    def _normalize_database_url(cls, raw: str) -> str:
        return _normalize_postgres_dsn(raw)

    # --- Layered YAML config (loaded lazily) ---
    @property
    def app_config(self) -> dict[str, Any]:
        # APP_CONFIG_PATH / MODELS_CONFIG_PATH point a run at an alternate
        # file (e.g. one benchmark arm) without swapping config/*.yaml.
        cfg = _load_yaml(Path(os.environ.get("APP_CONFIG_PATH") or CONFIG_DIR / "config.yaml"))
        return apply_embedding_profile(cfg, self.models_config)

    @property
    def models_config(self) -> dict[str, Any]:
        return _load_yaml(Path(os.environ.get("MODELS_CONFIG_PATH") or CONFIG_DIR / "models.yaml"))


_SUPABASE_HOST_SUFFIXES = (".supabase.co", ".supabase.com")


def _normalize_postgres_dsn(raw: str) -> str:
    """Accept bare `postgresql://` DSNs (e.g. Supabase) and produce an
    asyncpg-driver DSN with SSL forced on for managed hosts.

    Rules:
      1. `postgresql://` (or `postgres://`) → `postgresql+asyncpg://`.
      2. For any Supabase host (.supabase.co direct, OR .supabase.com pooler):
         force `sslmode=require` if no `ssl`/`sslmode` param is present.
         asyncpg's default is `prefer`, which silently falls back to plaintext
         if the server allows it — Supabase's pooler does allow plaintext, so
         omitting this means passwords + queries travel unencrypted.
      3. Already-normalized DSNs pass through unchanged.
    """
    parts = urlsplit(raw)

    scheme = parts.scheme
    if scheme in ("postgresql", "postgres"):
        scheme = "postgresql+asyncpg"

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    query_keys = {k.lower() for k, _ in query_pairs}

    hostname = (parts.hostname or "").lower()
    is_supabase = any(hostname.endswith(suffix) for suffix in _SUPABASE_HOST_SUFFIXES)
    if is_supabase and "ssl" not in query_keys and "sslmode" not in query_keys:
        query_pairs.append(("sslmode", "require"))

    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query_pairs), parts.fragment))


def _load_yaml(path: Path) -> dict[str, Any]:
    # Prefer the user's local .yaml (gitignored, may be tuned). Fall back
    # to the tracked .example.yaml so deployed images (which only ship the
    # examples, never the gitignored real files) boot cleanly. The two
    # files are functionally identical -- they differ only in comments.
    if not path.exists():
        example = path.with_name(f"{path.stem}.example{path.suffix}")
        if example.exists():
            path = example
        else:
            raise FileNotFoundError(
                f"Missing {path.name} and {example.name}. Did you `cp "
                f"config/{path.stem}.example.yaml config/{path.name}`?"
            )
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def apply_embedding_profile(app_cfg: dict[str, Any], models_cfg: dict[str, Any]) -> dict[str, Any]:
    """Overlay the active embedding model's calibrated thresholds.

    Similarity thresholds (probe distance floors, the relationship hop floor,
    ...) are measured on ONE embedding model's distance distribution. An
    optional block

        embedding_profiles:
          Qwen/Qwen3-Embedding-8B:
            qa.relationship_similarity: 0.62

    replaces those dotted keys when that model is the configured embedder.
    No block, or no entry for the active model -> config returned unchanged.
    """
    profiles = app_cfg.get("embedding_profiles") or {}
    spec = ((models_cfg or {}).get("tasks") or {}).get("embeddings") or {}
    overrides = profiles.get(spec.get("model", "text-embedding-3-small")) or {}
    if not overrides:
        return app_cfg
    import copy

    out = copy.deepcopy(app_cfg)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = str(dotted).split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
