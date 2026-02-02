# LLM Project Skeleton

## Quickstart (Poetry)
1) Install Poetry (one time):
   - Homebrew: `brew install poetry`
   - Or pipx: `pipx install poetry`
2) From project root:
   ```bash
   cd /Users/shivaninagpal/Documents/Amit/Project/vision/llm
   # If you want Poetry to use your existing venv:
   poetry env use .venv/bin/python
   # Install everything (runtime + dev):
   poetry install --with dev
   ```
3) Add code under `src/` and run with:
   ```bash
   poetry run python -m src.your_module_here
   ```

## Tooling
- Format: `poetry run black .`
- Lint: `poetry run ruff check .` (auto-fix with `--fix`)
- Types: `poetry run mypy .`
- Tests: `poetry run pytest`

## Notes
- `pyproject.toml` now uses Poetry for dependency management; dev deps live in the `dev` group.
- `.gitignore` excludes venv, caches, data. Add project data under `data/` (ignored by default).
- See detailed notes in [`src/report.md`](src/report.md).

