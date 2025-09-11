# Makefile for IFC Semantic Agent

.PHONY: help install install-dev test test-cov lint format clean build docs serve-docs docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install the package"
	@echo "  install-dev  - Install the package in development mode with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting (flake8, mypy)"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build the package"
	@echo "  docs         - Build documentation"
	@echo "  serve-docs   - Serve documentation locally"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy . --ignore-missing-imports

format:
	black . --line-length=100
	isort . --profile black

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python setup.py sdist bdist_wheel

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Docker
docker-build:
	docker build -t ifc-semantic-agent .

docker-run:
	docker run -it --rm -p 8000:8000 ifc-semantic-agent

# Development helpers
run-example:
	python run_agent.py --example basic

run-interactive:
	python run_agent.py --interactive

run-tests-watch:
	pytest-watch tests/

# Environment setup
setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

freeze-deps:
	pip freeze > requirements-frozen.txt

# Data and model management
download-models:
	@echo "Downloading required models..."
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
	@echo "Models downloaded successfully"

# Performance testing
perf-test:
	python -m pytest tests/test_performance.py -v

benchmark:
	python examples/benchmark.py

# Security
security-check:
	bandit -r . -f json -o security-report.json
	safety check

# Release
release-test:
	python setup.py sdist bdist_wheel
	twine check dist/*

release:
	python setup.py sdist bdist_wheel
	twine upload dist/*