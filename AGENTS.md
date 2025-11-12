# Repository Guidelines

## Project Structure & Module Organization
- `alignment_agent/`: core package for IFC semantic alignment; key code lives under `core/`, `llm/`, and `utils/`, with configuration in `config/` and fixtures/tests in `tests/`.
- `semantic_alignment_agent/` and `regu_agent/`: experimental agents that share preprocessing patterns with the main package; mirror their configs in the local `config.yaml` files before running.
- `boolean/`: geometric helpers (e.g., wall extraction) that are imported by `multi_agent.py`.
- Root-level specs (`user_interface_specification.md`, `multi_agent_framework_user_flow.md`, etc.) document workflows; keep them synchronized with code changes.

## Build, Test, and Development Commands
- `cd alignment_agent && make install-dev`: install in editable mode with dev tools and pre-commit hooks.
- `make test` / `make test-cov`: run the pytest suite, optionally with coverage reports under `htmlcov/`.
- `make lint` and `make format`: enforce linting (flake8, mypy) and apply `black`/`isort` formatting.
- `python run_agent.py --example basic`: smoke-test the alignment agent locally; pass `--interactive` for manual prompts.

## Coding Style & Naming Conventions
- Python 3.8+ with 4-space indentation, max line length 100 (`black` config). Use type hints and docstrings for public functions.
- Module names stay snake_case; classes in CapWords; constants upper snake case. Prefer descriptive directory names when adding subpackages.
- Keep config keys lowercase with hyphen-separated YAML entries, matching existing `config.yaml` patterns.

## Testing Guidelines
- Tests live in `alignment_agent/tests/`; follow `test_<module>.py` naming. Use pytest markers defined in `setup.cfg` (`unit`, `integration`, `slow`, `performance`).
- New logic needs unit coverage plus integration checks when touching IFC parsing or LLM toolchains. Gate large IFC fixtures under the `slow` marker.
- Run `make test` locally before pushing; publish coverage deltas when significant behaviour changes.

## Commit & Pull Request Guidelines
- Commit messages should be concise, present-tense, and scoped (`alignment: add facade parser`). Group unrelated changes into separate commits.
- PRs must describe intent, list functional/test impacts, and reference issue IDs. Attach screenshots or CLI traces for agent UX updates.
- Ensure CI-ready state: lint, tests, and sample `run_agent.py --example basic` output validated. Tag reviewers responsible for impacted submodules.
