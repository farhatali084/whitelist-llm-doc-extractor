PYTHON ?= python

.PHONY: install eval eval-live record test

install:
	$(PYTHON) -m pip install -r requirements.txt

# Real pipeline on recorded real model answers (no Ollama needed)
eval:
	$(PYTHON) eval.py

# Live local model (needs Ollama running + `ollama pull qwen2.5:3b`)
eval-live:
	$(PYTHON) eval.py --backend ollama

# Live run that also refreshes data/recordings/
record:
	$(PYTHON) eval.py --backend ollama --record

test:
	$(PYTHON) -m pytest -q
