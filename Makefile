.PHONY: install data benchmark train app test lint typecheck check

install:
	uv sync --extra dev --extra app --extra report

data:
	uv run vfm prepare

benchmark:
	uv run vfm benchmark --protocol walk-forward
	uv run vfm benchmark --protocol holdout

train:
	uv run vfm train --model physics

app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest --cov=vfm --cov-report=term-missing

lint:
	uv run ruff check src tests app
	uv run ruff format --check src tests app

typecheck:
	uv run mypy

check: lint typecheck test
