# =============================================================================
# Makefile — shortcuts for common dev + ops tasks.
#
# Every recipe wraps an explicit `uv run python -m backend.app.cli ...` or
# `uv run pytest ...` so you can always copy-paste the underlying command
# if you want flags this file doesn't expose.
#
# Discover all targets with: make help
# =============================================================================

# Use bash + fail on error in multi-line recipes.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Use the repo's .env (gitignored) for any target that needs creds.
ifneq (,$(wildcard .env))
  include .env
  export
endif

CLI := uv run python -m backend.app.cli

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Help — `make` or `make help` prints every target with its docstring.
# A target's docstring is the line directly after `## ` in the .PHONY block.
# -----------------------------------------------------------------------------

.PHONY: help
help: ## Show this help.
	@printf "Usage: make <target>\n\n"
	@printf "Available targets:\n"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------
# Install + setup
# -----------------------------------------------------------------------------

.PHONY: install
install: ## Install Python + frontend deps (uv sync + npm install).
	uv sync
	cd frontend && npm install

.PHONY: install-py
install-py: ## Python deps only (uv sync).
	uv sync

.PHONY: install-fe
install-fe: ## Frontend deps only (npm install).
	cd frontend && npm install

# -----------------------------------------------------------------------------
# Tests + quality
# -----------------------------------------------------------------------------

.PHONY: test
test: ## Run the fast unit-test suite.
	uv run pytest backend/tests/unit -q

.PHONY: test-int
test-int: ## Run the integration suite (DB + LLM mocked).
	uv run pytest backend/tests/integration -q

.PHONY: test-all
test-all: ## Run unit + integration suites.
	uv run pytest backend/tests -q

.PHONY: test-slow
test-slow: ## Run only slow tests (deselected by default).
	uv run pytest backend/tests -q -m slow

.PHONY: coverage
coverage: ## Run unit tests with coverage report.
	uv run pytest backend/tests/unit --cov=backend/app --cov-report=term-missing

.PHONY: coverage-html
coverage-html: ## Generate an HTML coverage report at htmlcov/.
	uv run pytest backend/tests/unit --cov=backend/app --cov-report=html
	@echo "Open htmlcov/index.html in a browser."

.PHONY: lint
lint: ## Lint the Python code (ruff).
	uv run ruff check backend/

.PHONY: lint-fix
lint-fix: ## Auto-fix lint errors (ruff --fix).
	uv run ruff check --fix backend/

.PHONY: format
format: ## Format Python code (ruff format).
	uv run ruff format backend/

.PHONY: typecheck
typecheck: ## Run mypy.
	uv run mypy backend/app

.PHONY: check
check: lint typecheck test ## All quality checks: lint + typecheck + tests.

# -----------------------------------------------------------------------------
# Local dev servers
# -----------------------------------------------------------------------------

.PHONY: dev
dev: ## Run the FastAPI + MCP backend locally with auto-reload (port 8000).
	uv run uvicorn backend.app.main:app --reload --port 8000

.PHONY: frontend-dev
frontend-dev: ## Run the React UI locally with Vite hot-reload (port 5173).
	cd frontend && npm run dev

.PHONY: frontend-build
frontend-build: ## Build the React UI for production (frontend/dist).
	cd frontend && npm run build

# -----------------------------------------------------------------------------
# Database lifecycle (run against DATABASE_URL from .env)
# -----------------------------------------------------------------------------

.PHONY: db-status
db-status: ## Report alembic revision + per-table row counts.
	$(CLI) db-status

.PHONY: db-size
db-size: ## Per-table + total Postgres size (500 MB Supabase Free-tier cap).
	$(CLI) db-size

.PHONY: db-migrate
db-migrate: ## Apply Alembic migrations (alembic upgrade head).
	$(CLI) db-migrate

# Convenience for `make db-init INPUT=output_ontologies/v...-prune-expand`
.PHONY: db-init
db-init: ## Replace-load ontology into Postgres. Set INPUT=<merge folder>.
	@if [ -z "$(INPUT)" ]; then echo "ERROR: pass INPUT=<merge-folder>"; exit 1; fi
	$(CLI) db-init --input $(INPUT) --mode replace --yes

# -----------------------------------------------------------------------------
# Render deploy + lifecycle (needs RENDER_API_KEY in .env)
# -----------------------------------------------------------------------------

.PHONY: deploy
deploy: ## First-time deploy (create backend + frontend services from render.yaml + .env).
	$(CLI) render-init

.PHONY: redeploy
redeploy: ## Force a fresh build + redeploy of the backend service.
	$(CLI) render-deploy --service backend --wait

.PHONY: status
status: ## Show backend + frontend deploy state.
	$(CLI) render-status

.PHONY: logs
logs: ## Tail recent backend logs (override SINCE=30m to widen window).
	$(CLI) render-logs --service backend --since $${SINCE:-10m}

.PHONY: suspend
suspend: ## Suspend backend + frontend (stop billing immediately).
	$(CLI) render-suspend --all

.PHONY: resume
resume: ## Resume backend + frontend after suspend.
	$(CLI) render-resume --all

.PHONY: takedown
takedown: ## Soft takedown (same as suspend --all).
	$(CLI) render-takedown --yes

.PHONY: terminate
terminate: ## DESTROY backend + frontend services (irreversible — URLs reassigned next deploy).
	@echo "This will DELETE both Render services. URLs will be reassigned next deploy."
	@read -p "Type 'yes' to confirm: " ans && [ "$$ans" = "yes" ] || (echo "aborted"; exit 1)
	$(CLI) render-takedown --hard --yes

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove Python caches + frontend build output.
	find . -type d -name '__pycache__' -prune -exec rm -rf {} \;
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} \;
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} \;
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} \;
	rm -rf htmlcov .coverage frontend/dist frontend/.vite
	@echo "Cleaned."
