.PHONY: install data test eval eval-gen serve

install:
	python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:
	python -m govrag.ingest

test:
	ruff check src evals tests && ruff format --check src evals tests && pytest -q

eval:
	python evals/run_retrieval.py --pool-sweep 10 20 30 50

eval-gen:
	python evals/run_generation.py

serve:
	uvicorn govrag.api:app --reload
