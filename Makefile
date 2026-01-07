.PHONY: up down logs reset seed seed-csv train health shell inspect

# Default target
up:
	docker compose up --build -d

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

# Factory reset: remove containers + data volume + rebuild
reset:
	docker compose down --remove-orphans
	@echo "Removing ./data directory..."
	@if exist data rmdir /s /q data 2>nul || rm -rf ./data 2>/dev/null || true
	@mkdir data 2>nul || mkdir -p ./data 2>/dev/null || true
	docker compose up --build -d

# Fetch news + prices from online sources
seed:
	docker compose exec backend python -m app.data_manager --fetch

# Import CSV files from /data/seed/ directory
seed-csv:
	docker compose exec backend python -m app.seed_db

# Run FinBERT scoring + DailySentiment + XGBoost training
train:
	docker compose exec backend python -m app.ai_engine --run-all --target-symbol HG=F

# Health check
health:
	docker compose exec backend python -c "import urllib.request, json; r=urllib.request.urlopen('http://localhost:8000/api/health'); print(json.dumps(json.load(r), indent=2))"

# Open shell in backend container
shell:
	docker compose exec backend /bin/bash

# Quick DB inspection
inspect:
	docker compose exec backend python -m app.inspect_db

# Run tests
test:
	docker compose exec backend pytest -v

# Windows-specific reset (PowerShell)
reset-win:
	docker compose down --remove-orphans
	powershell -Command "if (Test-Path data) { Remove-Item -Recurse -Force data }"
	powershell -Command "New-Item -ItemType Directory -Force -Path data"
	docker compose up --build -d

