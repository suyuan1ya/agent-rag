.PHONY: install lint format typecheck test test-cov serve clean docker-build docker-up docker-down

install:
	pip install -r requirements.txt

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/ rag/ tests/ cli.py

test:
	pytest -v

test-cov:
	pytest -v --cov=src --cov=rag --cov-report=term-missing

serve:
	python cli.py serve

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/**/__pycache__ rag/**/__pycache__ tests/**/__pycache__
	rm -f *.bm25.json

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
