# Repository Guidelines

## Project Structure & Module Organization
- `src/brain/` holds the core simulation code (entry point: `neuron.py`).
- `tests/` contains pytest-based tests (current coverage focuses on neuron behavior).
- `README.md` provides a short project overview and setup notes.
- Tooling and metadata live in `pyproject.toml` and `.pre-commit-config.yaml`.

## Build, Test, and Development Commands
- `uv sync --dev` installs runtime and dev dependencies.
- `uv run brain --steps 500` runs a short simulation in the terminal.
- `uv run brain --server --auto-scale` starts the local visualization server (dashboard controls, scenarios, circuit builder).
- `uv run pytest` runs the test suite.
- `uv run pre-commit run --all-files` runs formatting and lint checks.

## Coding Style & Naming Conventions
- Python code is formatted with `black` (4-space indentation, standard Black line breaks).
- Linting is handled by `flake8` with a generous `max-line-length = 1000`.
- Use descriptive, sentence-case function and variable names (e.g., `simulate_step`, `energy_loss`).
- Prefer small, focused functions in `src/brain/neuron.py` when extending behavior.

## Testing Guidelines
- Tests use `pytest`; place new tests under `tests/` with filenames like `test_<topic>.py`.
- Use explicit assertions for neuron/brain state changes.
- No explicit coverage threshold is configured; add tests for new logic or bug fixes.

## Commit & Pull Request Guidelines
- Commit messages are simple, imperative sentences (e.g., "Refactor Neuron functions for int only").
- Keep commits scoped; avoid mixing refactors with behavior changes when possible.
- PRs should include a short summary, test command results, and any relevant context or issues.
- For user-visible changes, describe expected behavior and include examples when helpful.

## Configuration Notes
- Python version target: `3.12` (see `pyproject.toml`).
- If you add new tools, update `pyproject.toml` and `.pre-commit-config.yaml` together.
