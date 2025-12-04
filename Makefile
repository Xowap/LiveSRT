.PHONY: clean prettier format lint types

clean: prettier format lint types

prettier:
	pnpx prettier -w README.md src

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

types:
	uv run mypy src
