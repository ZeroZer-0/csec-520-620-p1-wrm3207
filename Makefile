# One-command reproducibility. Run `make help` to see targets.
# Uses a plain Python venv + pip (no conda).
.PHONY: help setup data reproduce test lint clean

VENV := .venv
PY := $(VENV)/bin/python

help:
	@echo "make setup      - create .venv and install requirements"
	@echo "make data       - fetch data/dataset.csv if missing, and verify it against data/dataset.sha256"
	@echo "make reproduce  - train + evaluate, writing results/ (THE grading command)"
	@echo "make test       - run the test suite"
	@echo "make lint       - byte-compile check (stdlib only)"
	@echo "make clean      - remove generated results"

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) data/download_dataset.py

# The single command a grader runs. Must recreate your reported results.
reproduce: data
	$(PY) -m src.train --config config.yaml

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m compileall -q src tests tools data/build_dataset.py data/download_dataset.py

clean:
	rm -f results/*.png results/*.json results/*.pt results/*.npz
