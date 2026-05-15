# Contributing

Thanks for your interest in medsight-ai.

## Development setup

```bash
git clone https://github.com/alfianmushh-eng/medsight-ai.git
cd medsight-ai
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests and lint

```bash
pytest
ruff check src tests
```

All new features should ship with tests. Aim for the existing style: small
dataclasses, pure functions where possible, and numpy arrays at the boundary.

## Commit style

Short, declarative subject lines. Prefer one logical change per commit. If a
change touches a public surface, mention the module in the subject.

## Reporting issues

Please include:

- Python version
- A minimal reproducer (synthetic data is fine)
- The full traceback
