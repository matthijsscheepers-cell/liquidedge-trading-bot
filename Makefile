# LIQUIDEDGE Trading Bot - Makefile
# Quick commands for common tasks

PYTHON := ./venv/bin/python3
PIP := ./venv/bin/pip

.PHONY: help
help:
	@echo "LIQUIDEDGE Trading Bot - Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install all dependencies"
	@echo "  make test-connection  Test Capital.com API connection"
	@echo ""
	@echo "TODO Management:"
	@echo "  make todos            List all todos"
	@echo "  make todos-week W=3   List todos for specific week"
	@echo "  make todo-add         Add new todo interactively"
	@echo "  make todo-done ID=1   Mark todo as complete"
	@echo ""
	@echo "Utilities:"
	@echo "  make instruments      List available trading instruments"
	@echo "  make clean            Remove cache files"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run test suite"
	@echo "  make lint             Run linting checks"
	@echo ""

.PHONY: install
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

.PHONY: test-connection
test-connection:
	$(PYTHON) scripts/hello_world.py

.PHONY: todos
todos:
	$(PYTHON) scripts/todo_tracker.py list

.PHONY: todos-week
todos-week:
	@if [ -z "$(W)" ]; then \
		echo "Error: Week number required. Usage: make todos-week W=3"; \
		exit 1; \
	fi
	$(PYTHON) scripts/todo_tracker.py list --week $(W)

.PHONY: todo-add
todo-add:
	$(PYTHON) scripts/todo_tracker.py add

.PHONY: todo-done
todo-done:
	@if [ -z "$(ID)" ]; then \
		echo "Error: Todo ID required. Usage: make todo-done ID=1"; \
		exit 1; \
	fi
	$(PYTHON) scripts/todo_tracker.py complete $(ID)

.PHONY: instruments
instruments:
	$(PYTHON) scripts/list_instruments.py

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cache files cleaned"

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v --cov=src

.PHONY: lint
lint:
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m mypy src/
