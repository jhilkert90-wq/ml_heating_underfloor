# Project Guidelines — ML Heating Underfloor

## Post-Implementation Documentation (MANDATORY)

After completing any code change — feature, fix, or refactoring — you **must** update these files before considering the task done:

1. **`CHANGELOG.md`** — Add entries to the `[Unreleased]` section following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format. See `memory-bank/changelogStandards.md` for rules (one section per type, standard headings only, no empty sections).
2. **`memory-bank/progress.md`** — Add a new milestone entry at the top with date, status, implementation summary, and files changed.
3. **`memory-bank/activeContext.md`** — Add a context entry at the top describing what changed, why, and which files were modified.

Do not skip these updates. Do not wait for the user to ask. This applies to every session.

## Commit Messages

Use Conventional Commits: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- See `memory-bank/changelogStandards.md` for details

## Development Workflow

- **TDD**: All features/fixes start with tests; 100% pass rate required
- **Test structure**: `tests/unit/` and `tests/integration/`, pytest with mocked external deps
- **Release model**: Alpha/stable dual-channel — see `memory-bank/developmentWorkflow.md`
- **Run tests**: `python -m pytest tests/ -q --tb=short`

## Architecture

- `src/` — Core system (main.py, model_wrapper.py, thermal_equilibrium_model.py, ha_client.py, prediction_metrics.py)
- `dashboard/` — Streamlit dashboard
- `memory-bank/` — Project context and standards documentation
- `Logs/unified_thermal_state.json` — Persistent thermal state
