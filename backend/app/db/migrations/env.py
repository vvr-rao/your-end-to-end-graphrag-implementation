"""Alembic async environment.

Drives migrations using the same async engine the application uses
(scheme already normalized + sslmode=require by the core/config
validator). Targets the `graphrag` schema.

`alembic upgrade head` is safe to run repeatedly; migrations are
idempotent within Alembic's revision tracking.
"""
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from backend.app.core.config import get_settings
from backend.app.db.base import GRAPHRAG_SCHEMA, Base
import backend.app.db.models  # noqa: F401  -- registers all models with Base.metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the live DSN from app settings, overriding alembic.ini.
settings = get_settings()
config.set_main_option("sqlalchemy.url", str(settings.database_url))

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    """Restrict autogenerate to the graphrag schema."""
    if type_ == "table" and getattr(obj, "schema", None) != GRAPHRAG_SCHEMA:
        return False
    return True


def do_run_migrations(connection: Connection) -> None:
    # Ensure the schema exists BEFORE Alembic looks up its version
    # tracking table -- the alembic_version table lives inside
    # `graphrag`, so the schema has to be there first.
    connection.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {GRAPHRAG_SCHEMA}")

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        include_object=include_object,
        version_table_schema=GRAPHRAG_SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    # Reuse the app's engine -- already strips libpq-only `sslmode` and
    # attaches the right SSL context for Supabase.
    from backend.app.db.engine import get_engine

    connectable = get_engine()
    # `connectable.begin()` wraps in an async transaction that commits
    # on successful exit -- needed for DDL to persist through the
    # sync-bridge of `run_sync`.
    async with connectable.begin() as connection:
        await connection.run_sync(do_run_migrations)
    # Don't dispose -- the engine is process-scoped + cached.


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        include_schemas=True,
        include_object=include_object,
        version_table_schema=GRAPHRAG_SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
