# Hyperbolic SimCSE

Contrastive sentence embeddings on the Lorentz model of hyperbolic space,
tested on WordNet noun hypernymy. See `docs/superpowers/plans/` for the plan
and the experiment spec.

## Setup
    uv venv && source .venv/bin/activate   # or python -m venv
    uv pip install -e ".[dev]"
    python scripts/download_data.py        # WordNet + STS-B

## Run tests
    pytest -q

## Run an experiment
    python experiments/run.py --config experiments/configs/E1_graph.yaml --seed 0
