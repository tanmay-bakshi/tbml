To run or test code, use the venv at `.venv`.
If you need to add dependencies, add them to the pyproject.toml file and run `uv sync --all-extras` to update the virtual environment.

When performing a `git diff`, make sure to pass the `--no-ext-diff` flag.