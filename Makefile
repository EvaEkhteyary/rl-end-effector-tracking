# --- RL End-Effector Tracking ------------------------------------------------
# Override the interpreter if `python3` is older than 3.10:
#     make setup PYTHON=python3.12
PYTHON ?= python3
PY = .venv/bin/python
PIP = .venv/bin/pip

.PHONY: setup smoke train eval video demo all clean

setup:                 ## create venv + install dependencies
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

smoke:                 ## quick 20k-step sanity run
	$(PY) -m src.train --timesteps 20000 --n-envs 8 --out models --logdir runs

train:                 ## full training run (~2M steps, ~12 min on 8 cores)
	$(PY) -m src.train --timesteps 2000000 --n-envs 8 --out models --logdir runs

eval:                  ## evaluate + generate plots and metrics
	$(PY) -m src.evaluate

video:                 ## render tracking videos (MP4 + GIF)
	$(PY) -m src.render

demo: eval video       ## evaluate + render using the shipped model

all: train eval video  ## full pipeline from scratch

clean:                 ## remove training logs + caches (keeps models/ + results/)
	rm -rf runs __pycache__ src/__pycache__ tools/__pycache__
