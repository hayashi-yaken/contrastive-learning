# Hyperbolic SimCSE

Contrastive sentence embeddings on the Lorentz model of hyperbolic space,
tested on WordNet noun hypernymy. See `docs/superpowers/plans/` for the plan
and the experiment spec.

> 日本語ドキュメント: [README.ja.md](README.ja.md) ／ [docs/ja/設計と実験ガイド.md](docs/ja/設計と実験ガイド.md)

## Setup
    uv venv && source .venv/bin/activate   # or python -m venv
    uv pip install -e ".[dev]"
    python scripts/download_data.py        # WordNet + STS-B

## Run tests
    pytest -q

## Run an experiment
    python experiments/run.py --config experiments/configs/E1_graph.yaml --seed 0

## Reproducing the experiments

All commands assume `python scripts/download_data.py` has run. Full scale uses
all ~82k noun synsets (`max_synsets: 0`). On CPU/MPS this is slow; reduce with
`--max-synsets 20000` for quick checks.

### E1 — main ablation (H2): score comparison at d=64
    python experiments/run_matrix.py --config experiments/configs/E1_graph.yaml \
        --outdir results/E1_graph --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2
    python experiments/run_matrix.py --config experiments/configs/E1_encoder.yaml \
        --outdir results/E1_encoder --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2

### E2 — efficiency / dimension sweep (H1)
    python experiments/run_matrix.py --config experiments/configs/E2_sweep.yaml \
        --outdir results/E2 --dims 2 5 10 16 32 64 128 256 \
        --scores EUC HYP-inner --seeds 0 1 2

### E3 — collapse / temperature sweep (H3)
    python experiments/run_matrix.py --config experiments/configs/E3_collapse.yaml \
        --outdir results/E3 --taus 0.01 0.02 0.05 0.1 0.2 0.5 \
        --scores EUC HYP-inner --seeds 0 1 2

### E4 — curvature (fixed vs learnable)
    python experiments/run.py --config experiments/configs/E4_curvature.yaml --seed 0 --out results/E4_fixed_s0.json
    # then set learnable_c: true in the YAML (or a copy) and rerun -> results/E4_learn_s0.json

### E5 — encoder full (inductive, STS, norm-depth, H4) with root-anchor on
    python experiments/run.py --config experiments/configs/E5_encoder_full.yaml \
        --seed 0 --full-hierarchy --out results/E5_s0.json

### E6 — dropout lower bound
    python experiments/run.py --config experiments/configs/E6_dropout.yaml --seed 0 --out results/E6_s0.json

### Figures
    python -c "import sys; sys.path.insert(0,'experiments'); import plots; \
        plots.plot_metric_vs_dim('results/E2','figures/fig1_metric_vs_dim.png'); \
        plots.plot_condition_vs_tau('results/E3','figures/fig3_condition_vs_tau.png'); \
        plots.plot_norm_vs_depth('results/E5_s0.json','figures/fig2_norm_vs_depth.png')"

## Experiment ↔ hypothesis map

| Exp | Track | Factor | Tests |
|-----|-------|--------|-------|
| E1  | graph + encoder | score: EUC/HYP-dist/dist2/inner | H2 (score) |
| E2  | graph | dimension sweep | H1 (efficiency) |
| E3  | graph | temperature sweep + Fisher | H3 (collapse) |
| E4  | encoder | curvature fixed vs learnable | curvature × scale |
| E5  | encoder | full metrics + root-anchor | H4 (uniformity replacement) |
| E6  | dropout | unsup lower bound | supervised-edge contribution |

See `docs/superpowers/plans/2026-07-02-hyperbolic-simcse.md` for the full plan.
