PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RESTORE_ROOT ?=

.PHONY: help check check-python paths test-python test-runtime preservation-verify

help:
	@printf '%s\n' \
	  'make check                 Compile Python, run foundation tests, and run MCU golden gate' \
	  'make paths                 Print resolved workspace paths' \
	  'make test-python           Run dependency-free package tests' \
	  'make test-runtime          Build and run the portable C runtime golden gate' \
	  'make preservation-verify RESTORE_ROOT=/tmp/recovery'

check: check-python test-python test-runtime

check-python:
	PYTHONPATH=src $(PYTHON) -m compileall -q src tools

paths:
	PYTHONPATH=src $(PYTHON) -m saanotts.workspace

test-python:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test-runtime:
	$(MAKE) -C mcu clean test

preservation-verify:
	@test -n "$(RESTORE_ROOT)" || (printf '%s\n' 'RESTORE_ROOT is required' >&2; exit 2)
	$(PYTHON) tools/preserve_local_assets.py verify --root "$(RESTORE_ROOT)"
