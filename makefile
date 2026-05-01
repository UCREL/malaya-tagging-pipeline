VERSION_CMD = "uv run scripts/get_version.py ./pyproject.toml"

.PHONY: lint
lint:
	@echo "Linting with Ruff:"
	@uv run ruff check --fix-only src tests 
	@uv run ruff check src tests
	@echo "Type checking with Ty"
	@uv run ty check src tests

.PHONY: test
test:
	@echo "Testing with coverage"
	@uv run coverage run
	@uv run coverage report

.PHONY: build-python-package
build-python-package:
	@uv lock --check
	@rm -rf ./dist
	@uv build

.PHONY: release-notes
release-notes: build-python-package
	@uv run --no-project --script \
	--with dist/malaya_tagging_pipeline-$$("${VERSION_CMD}")-py3-none-any.whl \
	./scripts/release_notes.py