# Contributing

Bug reports, fixes and new generic research tools are welcome.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

Guidelines:
- Every function gets a docstring that says what question it answers, and a test that checks a
  known value or a property (for example "more trials lowers the Deflated Sharpe Ratio").
- Keep dependencies to numpy, pandas and matplotlib.
- Examples and tests use synthetic or public data only. Please do not add trading signals,
  strategy parameters, broker code or personal trade data.
- Add a line to `CHANGELOG.md`.
