#!/usr/bin/env python3
"""Throwaway local Postgres+pgvector for benchmark arms.

    uv run python scripts/bench/bench_db.py up            # start container (port 55433)
    uv run python scripts/bench/bench_db.py create <db>   # empty DB + extensions -> prints DSN
    uv run python scripts/bench/bench_db.py dump <db> <file> / restore <file> <db>
    uv run python scripts/bench/bench_db.py down          # remove container (data gone)

One database per (arm, corpus): entity dedup and graph walks must not see
another arm's rows, and the embedding-model guard refuses a database whose
vectors came from a different embedder anyway. No named volume -- nothing
here is meant to outlive the benchmark, and nothing touches the
docker-compose database or Supabase.
"""
from __future__ import annotations

import subprocess
import sys
import time

NAME = "graphrag-bench-pg"
PORT = 55433
USER = PASSWORD = "bench"
IMAGE = "pgvector/pgvector:pg16"


def dsn(db: str) -> str:
    return f"postgresql://{USER}:{PASSWORD}@localhost:{PORT}/{db}"


def _psql(sql: str, db: str = "postgres") -> None:
    subprocess.run(["docker", "exec", NAME, "psql", "-U", USER, "-d", db, "-q", "-c", sql], check=True)


def up() -> None:
    running = subprocess.run(["docker", "ps", "-q", "-f", f"name={NAME}"], capture_output=True, text=True)
    if not running.stdout.strip():
        subprocess.run(["docker", "run", "-d", "--rm", "--name", NAME, "-e", f"POSTGRES_USER={USER}",
                        "-e", f"POSTGRES_PASSWORD={PASSWORD}", "-p", f"{PORT}:5432",
                        # Small VM (2.7 GB): keep Postgres modest.
                        IMAGE, "-c", "shared_buffers=128MB", "-c", "max_connections=60"],
                       check=True, capture_output=True)
    for _ in range(60):
        if subprocess.run(["docker", "exec", NAME, "pg_isready", "-U", USER, "-q"]).returncode == 0:
            print(f"up: {NAME} on :{PORT}")
            return
        time.sleep(1)
    raise SystemExit("postgres did not become ready")


def create(db: str) -> str:
    exists = subprocess.run(
        ["docker", "exec", NAME, "psql", "-U", USER, "-d", "postgres", "-tAc",
         f"SELECT 1 FROM pg_database WHERE datname='{db}'"], capture_output=True, text=True)
    if exists.stdout.strip() != "1":
        _psql(f'CREATE DATABASE "{db}"')
    _psql("CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;", db)
    return dsn(db)


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "up":
        up()
    elif cmd == "create" and len(argv) == 2:
        print(create(argv[1]))
    elif cmd == "dump" and len(argv) == 3:
        with open(argv[2], "wb") as f:
            subprocess.run(["docker", "exec", NAME, "pg_dump", "-U", USER, "-Fc", argv[1]], check=True, stdout=f)
    elif cmd == "restore" and len(argv) == 3:
        create(argv[2])
        with open(argv[1], "rb") as f:
            subprocess.run(["docker", "exec", "-i", NAME, "pg_restore", "-U", USER, "-d", argv[2],
                            "--no-owner"], check=True, stdin=f)
    elif cmd == "down":
        subprocess.run(["docker", "rm", "-f", NAME], check=False, capture_output=True)
        print("down")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
