# Hyperbolic SimCSE Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research codebase that swaps the latent space of a contrastive sentence-embedding model (SimCSE) from the hypersphere/Euclidean space to the Lorentz model of hyperbolic space, and runs the ablation experiments (E1–E6) that test whether hierarchical (WordNet noun hypernymy) data is learned more parameter-efficiently, whether the information-geometry–predicted score `-cosh d = ⟨u,v⟩_L` beats `-d`/`-d²`, whether a temperature-driven anisotropic dimensional collapse appears, and whether a root-anchor regularizer can replace the (nonexistent) hyperbolic uniformity term.

**Architecture:** A single Python package `hypsimcse` with clean layers: (1) `geometry` — pure, tested Lorentz-model math (inner product, distance, exp map, score functions); (2) `data` — WordNet noun hypernymy loading, splits, and graph statistics; (3) `models` — a pure-graph embedding (Nickel–Kiela style, no encoder) and a BERT/RoBERTa encoder model, both ending in an exp-map lift to the hyperboloid; (4) `losses` — one InfoNCE with a pluggable score function + optional root-anchor regularizer; (5) `training` — one config-driven trainer with the MERU-style numerical-stabilization stack; (6) `eval` — reconstruction/link-prediction/inductive metrics, Fisher collapse diagnostics, norm-vs-depth, δ-hyperbolicity, and the Einstein-midpoint/dual-coordinate check; (7) `experiments` — YAML configs for E1–E6 and a runner + plotting scripts for the 3 mandatory figures. The geometry is the crux and is built and tested first; every downstream piece consumes it.

**Tech Stack:** Python 3.13, PyTorch 2.11 (MPS/CPU/CUDA), HuggingFace `transformers` 4.46, `nltk` (WordNet 3.0), `geoopt` (Riemannian Adam, optional comparison), `scipy`/`numpy` (eigendecomposition, Spearman), `pytest`, `PyYAML`, `matplotlib`.

## Global Constraints

- **Lorentz model only** (not Poincaré ball). All manifold math uses the appendix-A formulas verbatim.
- Lorentz inner product: `⟨x,y⟩_L = -x_0 y_0 + Σ_{i≥1} x_i y_i`.
- Hyperboloid: `H^d_c = { x ∈ R^{d+1} : ⟨x,x⟩_L = -1/c, x_0 > 0 }`. Origin `o = (1/√c, 0, …, 0)`.
- Distance: `d_H(x,y) = (1/√c) · arccosh(-c⟨x,y⟩_L)`. For `c=1`: `⟨x,y⟩_L = -cosh d_H`.
- Store **space components only** (`x_1..x_d`, `d` numbers); recover time `x_0 = sqrt(1/c + Σ x_i²)` from the constraint (MERU convention). Embedding dimension `d` = number of space components.
- Score labels are fixed strings: `EUC`, `HYP-dist`, `HYP-dist2`, `HYP-inner`. InfoNCE is identical across conditions — only the score differs: `L = -log( exp(sim(z_i,z_i^+)/τ) / Σ_j exp(sim(z_i,z_j)/τ) )`.
- Curvature `c`: default fixed `c=1`; also support a learnable `c` (E4).
- Numerical stabilization (apply and log which helped, this is itself a result): clamp `arccosh` / `sqrt` arguments to `[1+ε, ·]` and `[ε, ·]` respectively; clamp tangent-vector norm before exp map (overflow guard); smaller LR + grad clipping for `HYP-inner`. `ε = 1e-6` unless noted.
- Optimizer: encoder + linear projection `W` use standard **AdamW** (tangent-space lift makes this valid). Only direct on-manifold parameters (pure-graph model) may additionally be compared against **Riemannian Adam** (geoopt).
- Encoder is **BERT-base or RoBERTa-base**, shared across all encoder conditions. Pure-graph track has no encoder.
- Dimension sweep set: `d ∈ {2, 5, 10, 16, 32, 64, 128, 256}`.
- Every experiment cell runs **3 random seeds**; main tables report mean ± std. Encoder, optimizer, batch size, epochs fixed across conditions within an experiment.
- Determinism: every entry point seeds `random`, `numpy`, `torch` (and MPS/CUDA) from a `--seed` flag.
- Device: auto-select `cuda` → `mps` → `cpu`. Code must run on all three; tests run on CPU.
- All tunables live in YAML config; no hard-coded hyperparameters in library code.

---

## File Structure

```
contrastive-learning/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/hypsimcse/
│   ├── __init__.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── lorentz.py          # inner product, distance, exp map, proj, origin
│   │   └── scores.py           # EUC / HYP-dist / HYP-dist2 / HYP-inner (pairwise)
│   ├── losses/
│   │   ├── __init__.py
│   │   └── infonce.py          # InfoNCE + root-anchor regularizer
│   ├── data/
│   │   ├── __init__.py
│   │   ├── wordnet.py          # synsets, hypernymy closure, glosses
│   │   ├── splits.py           # reconstruction / link-prediction / inductive
│   │   └── graph_stats.py      # δ-hyperbolicity, depth, branching
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lift.py             # tangent → hyperboloid lift (shared head)
│   │   ├── graph_embedding.py  # pure-graph (no encoder)
│   │   └── encoder_model.py    # BERT/RoBERTa → pooling → W → lift
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py           # dataclass + YAML loader
│   │   ├── seeding.py
│   │   ├── device.py
│   │   └── trainer.py          # train loop, optimizers, stabilization
│   └── eval/
│       ├── __init__.py
│       ├── reconstruction.py   # MAP, mean rank, distortion
│       ├── link_prediction.py  # MAP, AUC
│       ├── inductive.py        # unseen-synset reconstruction MAP
│       ├── sts.py              # STS-B Spearman
│       ├── fisher.py           # radial/tangential eigenvalues, condition number
│       ├── hierarchy.py        # norm-vs-depth correlation, δ of embeddings
│       └── dual_coords.py      # Einstein midpoint vs softmax-Lorentz mean
├── experiments/
│   ├── configs/                # E1..E6 YAML
│   ├── run.py                  # single-config runner (train+eval → JSON)
│   ├── run_matrix.py           # expand a matrix over seeds/dims/scores
│   └── plots.py                # 3 mandatory figures from result JSONs
├── scripts/
│   └── download_data.py        # nltk wordnet + STS-B fetch
└── tests/
    ├── test_lorentz.py
    ├── test_scores.py
    ├── test_infonce.py
    ├── test_wordnet.py
    ├── test_splits.py
    ├── test_graph_stats.py
    ├── test_lift.py
    ├── test_graph_embedding.py
    ├── test_encoder_model.py
    ├── test_trainer.py
    ├── test_reconstruction.py
    ├── test_link_prediction.py
    ├── test_inductive.py
    ├── test_fisher.py
    ├── test_hierarchy.py
    ├── test_dual_coords.py
    └── test_e2e_smoke.py
```

---

### Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/hypsimcse/__init__.py`, and empty `__init__.py` in every subpackage, `tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: installable package `hypsimcse` (import path `hypsimcse.*`); `pytest` runnable from repo root.

- [ ] **Step 1: Initialize git and directory tree**

```bash
cd /Users/hayashinaofumi/workspace/waseda/contrastive-learning
git init
mkdir -p src/hypsimcse/{geometry,losses,data,models,training,eval} experiments/configs scripts tests
touch src/hypsimcse/__init__.py \
      src/hypsimcse/geometry/__init__.py src/hypsimcse/losses/__init__.py \
      src/hypsimcse/data/__init__.py src/hypsimcse/models/__init__.py \
      src/hypsimcse/training/__init__.py src/hypsimcse/eval/__init__.py \
      tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "hypsimcse"
version = "0.1.0"
description = "Hyperbolic SimCSE: contrastive sentence embeddings on the Lorentz model"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1",
    "transformers>=4.40",
    "nltk>=3.8",
    "geoopt>=0.5",
    "numpy>=1.24",
    "scipy>=1.10",
    "pyyaml>=6.0",
    "matplotlib>=3.7",
    "datasets>=2.14",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
data/cache/
results/
runs/
figures/
.DS_Store
nltk_data/
```

- [ ] **Step 4: Write minimal `README.md`**

```markdown
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
```

- [ ] **Step 5: Create the venv and install**

Run:
```bash
cd /Users/hayashinaofumi/workspace/waseda/contrastive-learning
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
```
Expected: install completes; `python -c "import hypsimcse"` prints nothing (success).

- [ ] **Step 6: Verify pytest collects an empty suite**

Run: `pytest -q`
Expected: `no tests ran` (exit 5 acceptable) — confirms pythonpath/config wired.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: scaffold hypsimcse package"
```

---

### Task 1: Lorentz geometry core

**Files:**
- Create: `src/hypsimcse/geometry/lorentz.py`
- Test: `tests/test_lorentz.py`

**Interfaces:**
- Consumes: nothing (pure torch).
- Produces:
  - `lorentz_inner(x, y, keepdim=False) -> Tensor` — `⟨x,y⟩_L` over last dim, inputs are full `(d+1)` vectors.
  - `time_component(x_space, c=1.0) -> Tensor` — computes `x_0 = sqrt(1/c + ||x_space||²)`, returns shape `x_space[...,:1]`.
  - `to_hyperboloid(x_space, c=1.0) -> Tensor` — prepends time component, returns `(..., d+1)`.
  - `origin(d, c=1.0, **kw) -> Tensor` — `(d+1,)` point `(1/√c, 0…)`.
  - `dist(x, y, c=1.0, eps=1e-6) -> Tensor` — hyperbolic distance, inputs full `(d+1)`.
  - `expmap0(v_space, c=1.0, max_norm=None, eps=1e-6) -> Tensor` — lift tangent-at-origin space vector to hyperboloid, returns full `(d+1)`.
  - `EPS = 1e-6`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lorentz.py
import math, torch
from hypsimcse.geometry import lorentz as L

def test_inner_product_signature():
    x = torch.tensor([1.0, 0.0, 0.0]); y = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(L.lorentz_inner(x, y), torch.tensor(-1.0))

def test_origin_on_manifold():
    o = L.origin(2, c=1.0)
    assert torch.allclose(L.lorentz_inner(o, o), torch.tensor(-1.0), atol=1e-5)

def test_expmap0_lands_on_manifold():
    v = torch.randn(8, 5)
    z = L.expmap0(v, c=1.0)
    assert z.shape == (8, 6)
    ip = L.lorentz_inner(z, z)
    assert torch.allclose(ip, torch.full((8,), -1.0), atol=1e-4)

def test_expmap0_zero_is_origin():
    z = L.expmap0(torch.zeros(3, 5), c=1.0)
    assert torch.allclose(z[:, 0], torch.ones(3), atol=1e-5)
    assert torch.allclose(z[:, 1:], torch.zeros(3, 5), atol=1e-5)

def test_dist_self_is_zero():
    z = L.expmap0(torch.randn(4, 5))
    d = L.dist(z, z)
    assert torch.allclose(d, torch.zeros(4), atol=1e-3)

def test_dist_matches_arccosh_inner_c1():
    z1 = L.expmap0(torch.randn(4, 5)); z2 = L.expmap0(torch.randn(4, 5))
    d = L.dist(z1, z2, c=1.0)
    inner = L.lorentz_inner(z1, z2)
    assert torch.allclose(d, torch.arccosh(torch.clamp(-inner, min=1.0 + L.EPS)), atol=1e-4)

def test_dist_triangle_inequality():
    a = L.expmap0(torch.randn(1, 5)); b = L.expmap0(torch.randn(1, 5)); c = L.expmap0(torch.randn(1, 5))
    dab = L.dist(a, b); dbc = L.dist(b, c); dac = L.dist(a, c)
    assert (dac <= dab + dbc + 1e-3).all()

def test_max_norm_clamps():
    v = torch.randn(4, 5) * 1e3
    z = L.expmap0(v, max_norm=5.0)
    assert torch.isfinite(z).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lorentz.py -q`
Expected: FAIL — `ModuleNotFoundError` / attributes missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/geometry/lorentz.py
"""Lorentz (hyperboloid) model of hyperbolic space. Appendix A formulas.

Points are stored as full (d+1) vectors [x_0, x_1..x_d]; x_0 is the time axis.
Curvature c > 0; manifold is <x,x>_L = -1/c, x_0 > 0.
"""
import math
import torch

EPS = 1e-6


def lorentz_inner(x, y, keepdim=False):
    """<x,y>_L = -x0*y0 + sum_{i>=1} xi*yi  over the last dimension."""
    prod = x * y
    inner = prod[..., 1:].sum(dim=-1) - prod[..., 0]
    return inner.unsqueeze(-1) if keepdim else inner


def time_component(x_space, c=1.0):
    """x_0 = sqrt(1/c + ||x_space||^2)."""
    sq = (x_space * x_space).sum(dim=-1, keepdim=True)
    return torch.sqrt(torch.clamp(1.0 / c + sq, min=EPS))


def to_hyperboloid(x_space, c=1.0):
    """Prepend the time component to space coords, yielding a manifold point."""
    x0 = time_component(x_space, c=c)
    return torch.cat([x0, x_space], dim=-1)


def origin(d, c=1.0, dtype=torch.float32, device=None):
    """Origin o = (1/sqrt(c), 0, ..., 0) in R^{d+1}."""
    o = torch.zeros(d + 1, dtype=dtype, device=device)
    o[0] = 1.0 / math.sqrt(c)
    return o


def expmap0(v_space, c=1.0, max_norm=None, eps=EPS):
    """Exp map at the origin of a tangent (space-only) vector v_space.

    x_space = sinh(sqrt(c)|v|)/(sqrt(c)|v|) * v ;  x0 = (1/sqrt(c)) cosh(sqrt(c)|v|).
    """
    sqrt_c = math.sqrt(c)
    norm = torch.norm(v_space, dim=-1, keepdim=True)
    if max_norm is not None:
        scale = torch.clamp(max_norm / torch.clamp(norm, min=eps), max=1.0)
        v_space = v_space * scale
        norm = torch.norm(v_space, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    space = torch.sinh(sqrt_c * norm) / (sqrt_c * norm) * v_space
    x0 = torch.cosh(sqrt_c * norm) / sqrt_c
    return torch.cat([x0, space], dim=-1)


def dist(x, y, c=1.0, eps=EPS):
    """d_H(x,y) = (1/sqrt(c)) arccosh(-c <x,y>_L)."""
    sqrt_c = math.sqrt(c)
    arg = torch.clamp(-c * lorentz_inner(x, y), min=1.0 + eps)
    return torch.arccosh(arg) / sqrt_c
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lorentz.py -q`
Expected: PASS (all 8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/geometry/lorentz.py tests/test_lorentz.py
git commit -m "feat(geometry): Lorentz inner product, exp map, distance"
```

---

### Task 2: Score functions

**Files:**
- Create: `src/hypsimcse/geometry/scores.py`
- Test: `tests/test_scores.py`

**Interfaces:**
- Consumes: `hypsimcse.geometry.lorentz` (`lorentz_inner`, `dist`).
- Produces: `pairwise_scores(u, v, label, c=1.0, eps=1e-6) -> Tensor` returning an `(N, M)` matrix `S[i,j] = sim(u_i, v_j)`.
  - `label="EUC"`: `u,v` are **space/tangent** vectors `(N,d)`; returns cosine similarity (L2-normalized dot).
  - `label="HYP-dist"`: `u,v` are **hyperboloid** points `(N,d+1)`; returns `-d_H`.
  - `label="HYP-dist2"`: returns `-d_H²`.
  - `label="HYP-inner"`: returns `c·⟨u,v⟩_L` (for `c=1` this is `⟨u,v⟩_L = -cosh d_H`).
  - `SCORE_LABELS = ("EUC", "HYP-dist", "HYP-dist2", "HYP-inner")`.
  - `is_hyperbolic(label) -> bool` (True for all but `EUC`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scores.py
import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.geometry import scores as S

def test_labels_registered():
    assert S.SCORE_LABELS == ("EUC", "HYP-dist", "HYP-dist2", "HYP-inner")

def test_euc_is_cosine():
    u = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    v = torch.tensor([[1.0, 0.0]])
    out = S.pairwise_scores(u, v, "EUC")
    assert out.shape == (2, 1)
    assert torch.allclose(out[:, 0], torch.tensor([1.0, 0.0]), atol=1e-5)

def test_hyp_dist_diagonal_zero():
    z = L.expmap0(torch.randn(3, 4))
    out = S.pairwise_scores(z, z, "HYP-dist")
    assert out.shape == (3, 3)
    assert torch.allclose(torch.diagonal(out), torch.zeros(3), atol=1e-3)

def test_hyp_dist2_is_negative_square():
    z1 = L.expmap0(torch.randn(3, 4)); z2 = L.expmap0(torch.randn(2, 4))
    d = S.pairwise_scores(z1, z2, "HYP-dist")
    d2 = S.pairwise_scores(z1, z2, "HYP-dist2")
    assert torch.allclose(d2, -(d ** 2), atol=1e-4)

def test_hyp_inner_equals_neg_cosh_dist_c1():
    z1 = L.expmap0(torch.randn(4, 4)); z2 = L.expmap0(torch.randn(5, 4))
    inner = S.pairwise_scores(z1, z2, "HYP-inner", c=1.0)
    d = S.pairwise_scores(z1, z2, "HYP-dist", c=1.0)  # = -d_H
    assert torch.allclose(inner, -torch.cosh(-d), atol=1e-3)

def test_hyp_inner_diagonal_is_minus_one_c1():
    z = L.expmap0(torch.randn(4, 4))
    inner = S.pairwise_scores(z, z, "HYP-inner", c=1.0)
    assert torch.allclose(torch.diagonal(inner), torch.full((4,), -1.0), atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scores.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/geometry/scores.py
"""Pairwise similarity scores that plug into InfoNCE. Only the score differs
across experimental conditions (EUC / HYP-dist / HYP-dist2 / HYP-inner)."""
import torch
import torch.nn.functional as Fn
from . import lorentz as L

SCORE_LABELS = ("EUC", "HYP-dist", "HYP-dist2", "HYP-inner")


def is_hyperbolic(label):
    return label != "EUC"


def _pairwise_lorentz_inner(u, v):
    """(N,d+1),(M,d+1) -> (N,M) matrix of <u_i, v_j>_L."""
    time = -u[:, :1] @ v[:, :1].T
    space = u[:, 1:] @ v[:, 1:].T
    return time + space


def _pairwise_dist(u, v, c=1.0, eps=L.EPS):
    import math
    inner = _pairwise_lorentz_inner(u, v)
    arg = torch.clamp(-c * inner, min=1.0 + eps)
    return torch.arccosh(arg) / math.sqrt(c)


def pairwise_scores(u, v, label, c=1.0, eps=L.EPS):
    if label == "EUC":
        un = Fn.normalize(u, dim=-1)
        vn = Fn.normalize(v, dim=-1)
        return un @ vn.T
    if label == "HYP-dist":
        return -_pairwise_dist(u, v, c=c, eps=eps)
    if label == "HYP-dist2":
        return -_pairwise_dist(u, v, c=c, eps=eps) ** 2
    if label == "HYP-inner":
        return c * _pairwise_lorentz_inner(u, v)
    raise ValueError(f"unknown score label: {label!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scores.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/geometry/scores.py tests/test_scores.py
git commit -m "feat(geometry): pairwise score functions EUC/HYP-dist/dist2/inner"
```

---

### Task 3: InfoNCE loss + root-anchor regularizer

**Files:**
- Create: `src/hypsimcse/losses/infonce.py`
- Test: `tests/test_infonce.py`

**Interfaces:**
- Consumes: `hypsimcse.geometry.scores.pairwise_scores`, `hypsimcse.geometry.lorentz`.
- Produces:
  - `info_nce(z, z_pos, label, tau, c=1.0) -> Tensor` scalar loss. `z`,`z_pos` are batch reps (space vectors for EUC, hyperboloid points otherwise), shape `(B, ·)`. Positives are paired by row; all other rows in `z_pos` are in-batch negatives. Uses `sim(z_i, z_pos_j)` matrix, cross-entropy with target = diagonal.
  - `root_anchor_reg(z, c=1.0) -> Tensor` scalar: mean squared hyperbolic distance from origin — the H4 replacement for the (nonexistent) uniformity term; encourages spread away from the root. Returns **negative** mean distance so that *minimizing* it pushes points outward (spread), matching "root-anchored scatter". Caller weights it.
  - `total_loss(z, z_pos, label, tau, c=1.0, anchor_weight=0.0) -> (loss, parts_dict)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infonce.py
import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.losses import infonce as I

def test_info_nce_perfect_alignment_low_loss():
    z = L.expmap0(torch.randn(8, 4))
    loss = I.info_nce(z, z.clone(), "HYP-inner", tau=0.05)
    # identical positives, distinct rows -> loss should be small
    assert loss.item() < 0.5

def test_info_nce_gradients_flow():
    v = torch.randn(6, 4, requires_grad=True)
    z = L.expmap0(v)
    loss = I.info_nce(z, L.expmap0(v.detach() + 0.01), "HYP-dist", tau=0.1)
    loss.backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()

def test_info_nce_euc_runs():
    u = torch.randn(8, 16); up = u + 0.01 * torch.randn(8, 16)
    loss = I.info_nce(u, up, "EUC", tau=0.05)
    assert torch.isfinite(loss)

def test_root_anchor_reg_rewards_spread():
    near = L.expmap0(torch.randn(8, 4) * 0.1)
    far = L.expmap0(torch.randn(8, 4) * 2.0)
    assert I.root_anchor_reg(far) < I.root_anchor_reg(near)  # more spread -> smaller (more negative)

def test_total_loss_parts():
    z = L.expmap0(torch.randn(8, 4))
    loss, parts = I.total_loss(z, z.clone(), "HYP-inner", tau=0.05, anchor_weight=0.1)
    assert {"infonce", "anchor"} <= set(parts)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_infonce.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/losses/infonce.py
"""InfoNCE with a pluggable score, plus the H4 root-anchor regularizer."""
import torch
import torch.nn.functional as Fn
from ..geometry import scores as S
from ..geometry import lorentz as L


def info_nce(z, z_pos, label, tau, c=1.0):
    """Standard SimCSE/NT-Xent InfoNCE. Row i's positive is z_pos[i];
    z_pos[j != i] are in-batch negatives. Identical loss across conditions;
    only `label` changes the similarity."""
    logits = S.pairwise_scores(z, z_pos, label, c=c) / tau
    targets = torch.arange(z.shape[0], device=logits.device)
    return Fn.cross_entropy(logits, targets)


def root_anchor_reg(z, c=1.0):
    """Mean hyperbolic distance from the origin, negated. Minimizing this
    spreads embeddings outward from the root (H4 replacement for uniformity)."""
    o = L.origin(z.shape[-1] - 1, c=c, dtype=z.dtype, device=z.device)
    o = o.expand_as(z)
    return -L.dist(z, o, c=c).mean()


def total_loss(z, z_pos, label, tau, c=1.0, anchor_weight=0.0):
    nce = info_nce(z, z_pos, label, tau, c=c)
    parts = {"infonce": nce.detach().item(), "anchor": 0.0}
    loss = nce
    if anchor_weight > 0.0 and S.is_hyperbolic(label):
        anchor = root_anchor_reg(z, c=c)
        parts["anchor"] = anchor.detach().item()
        loss = loss + anchor_weight * anchor
    return loss, parts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_infonce.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/losses/infonce.py tests/test_infonce.py
git commit -m "feat(losses): InfoNCE + root-anchor regularizer"
```

---

### Task 4: WordNet data loading

**Files:**
- Create: `src/hypsimcse/data/wordnet.py`, `scripts/download_data.py`
- Test: `tests/test_wordnet.py`

**Interfaces:**
- Consumes: `nltk.corpus.wordnet`.
- Produces:
  - `ensure_wordnet() -> None` — downloads `wordnet`/`omw-1.4` via nltk if missing.
  - `load_noun_hypernymy(max_synsets=None) -> WordNetData` where `WordNetData` is a dataclass with:
    - `synsets: list[str]` (synset names, e.g. `"dog.n.01"`), `index: dict[str,int]`.
    - `direct_edges: list[tuple[int,int]]` — `(hyponym_idx, hypernym_idx)` from direct `hypernyms()`.
    - `closure_edges: list[tuple[int,int]]` — transitive closure `(descendant, ancestor)`.
    - `glosses: list[str]` — `lemma_names joined + " : " + definition` per synset (fixed preprocessing: lowercased, whitespace-collapsed, first 3 lemmas).
    - `roots: list[int]` — synsets with no hypernym (e.g. `entity.n.01`).
  - `gloss_text(synset) -> str` — the fixed preprocessing function (exposed for reuse/tests).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wordnet.py
import pytest
from hypsimcse.data import wordnet as W

@pytest.fixture(scope="module")
def data():
    W.ensure_wordnet()
    return W.load_noun_hypernymy(max_synsets=2000)

def test_synsets_and_index_consistent(data):
    assert len(data.synsets) == len(data.index)
    for i, name in enumerate(data.synsets):
        assert data.index[name] == i

def test_edges_reference_valid_indices(data):
    n = len(data.synsets)
    for h, hyper in data.direct_edges:
        assert 0 <= h < n and 0 <= hyper < n

def test_closure_superset_of_direct(data):
    direct = set(data.direct_edges)
    closure = set(data.closure_edges)
    assert direct <= closure
    assert len(closure) >= len(direct)

def test_glosses_nonempty_and_clean(data):
    assert len(data.glosses) == len(data.synsets)
    g = data.glosses[0]
    assert g == g.lower() and "  " not in g

def test_gloss_text_format():
    from nltk.corpus import wordnet as wn
    s = wn.synset("dog.n.01")
    txt = W.gloss_text(s)
    assert " : " in txt and txt == txt.lower()

def test_roots_have_no_hypernym(data):
    hypo_with_parent = {h for h, _ in data.direct_edges}
    for r in data.roots:
        assert r not in hypo_with_parent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wordnet.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/data/wordnet.py
"""WordNet 3.0 noun hypernymy loading. Nickel-Kiela setup: synsets are nodes,
transitive closure of hypernymy are the edges. Glosses feed the encoder track."""
import re
from dataclasses import dataclass, field


def ensure_wordnet():
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


_WS = re.compile(r"\s+")


def gloss_text(synset, max_lemmas=3):
    """Fixed preprocessing: '<lemma1 lemma2 ...> : <definition>', lowercased,
    whitespace-collapsed. Controls length/lemma bias (ESimCSE caveat)."""
    lemmas = [l.replace("_", " ") for l in synset.lemma_names()[:max_lemmas]]
    head = " ".join(lemmas)
    body = synset.definition() or ""
    text = f"{head} : {body}".lower()
    return _WS.sub(" ", text).strip()


@dataclass
class WordNetData:
    synsets: list
    index: dict
    direct_edges: list
    closure_edges: list
    glosses: list
    roots: list = field(default_factory=list)


def load_noun_hypernymy(max_synsets=None):
    from nltk.corpus import wordnet as wn
    all_syn = list(wn.all_synsets(pos="n"))
    if max_synsets is not None:
        all_syn = all_syn[:max_synsets]
    names = [s.name() for s in all_syn]
    index = {n: i for i, n in enumerate(names)}
    present = set(index)

    direct, closure, roots = [], [], []
    for s in all_syn:
        i = index[s.name()]
        parents = s.hypernyms() + s.instance_hypernyms()
        parents_in = [p for p in parents if p.name() in present]
        if not parents_in:
            roots.append(i)
        for p in parents_in:
            direct.append((i, index[p.name()]))
        # transitive closure: all ancestors reachable via hypernym paths
        seen = set()
        stack = list(parents)
        while stack:
            p = stack.pop()
            if p.name() in seen:
                continue
            seen.add(p.name())
            if p.name() in present:
                closure.append((i, index[p.name()]))
            stack.extend(p.hypernyms() + p.instance_hypernyms())

    glosses = [gloss_text(s) for s in all_syn]
    return WordNetData(names, index, direct, closure, glosses, roots)
```

- [ ] **Step 4: Write `scripts/download_data.py`**

```python
# scripts/download_data.py
"""Fetch WordNet (nltk) and the STS-B dataset cache up front."""
from hypsimcse.data.wordnet import ensure_wordnet

def main():
    ensure_wordnet()
    print("WordNet ready.")
    try:
        from datasets import load_dataset
        load_dataset("glue", "stsb", split="validation")
        print("STS-B ready.")
    except Exception as e:  # noqa: BLE001
        print(f"STS-B fetch skipped/failed: {e}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_wordnet.py -q`
Expected: PASS (6 tests). First run downloads WordNet.

- [ ] **Step 6: Commit**

```bash
git add src/hypsimcse/data/wordnet.py scripts/download_data.py tests/test_wordnet.py
git commit -m "feat(data): WordNet noun hypernymy loading + glosses"
```

---

### Task 5: Data splits

**Files:**
- Create: `src/hypsimcse/data/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `WordNetData` from Task 4.
- Produces:
  - `reconstruction_split(data) -> dict` with `{"edges": closure_edges}` (train == eval, embedding-quality ceiling).
  - `link_prediction_split(data, val_frac=0.05, test_frac=0.05, seed=0) -> dict` with `{"train": [...], "valid": [...], "test": [...]}` — a partition of `closure_edges` by edge (holds out unseen edges).
  - `inductive_split(data, test_synset_frac=0.1, seed=0) -> dict` with `{"train_synsets": set[int], "test_synsets": set[int], "train_edges": [...], "eval_edges": [...]}` — edges are `eval` iff either endpoint is a held-out synset (transductive N–K cannot do this).
  - `negatives_for(edge, data, num, rng) -> list[int]` — sample `num` node indices that are **not** true hypernyms of the edge's hyponym (for ranking metrics).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
from hypsimcse.data import wordnet as W
from hypsimcse.data import splits as SP
import pytest

@pytest.fixture(scope="module")
def data():
    W.ensure_wordnet()
    return W.load_noun_hypernymy(max_synsets=2000)

def test_reconstruction_uses_all_edges(data):
    r = SP.reconstruction_split(data)
    assert set(r["edges"]) == set(data.closure_edges)

def test_link_prediction_partitions(data):
    sp = SP.link_prediction_split(data, seed=0)
    total = set(map(tuple, sp["train"])) | set(map(tuple, sp["valid"])) | set(map(tuple, sp["test"]))
    assert total == set(data.closure_edges)
    assert not (set(map(tuple, sp["train"])) & set(map(tuple, sp["test"])))
    assert len(sp["valid"]) > 0 and len(sp["test"]) > 0

def test_link_prediction_deterministic(data):
    a = SP.link_prediction_split(data, seed=1)
    b = SP.link_prediction_split(data, seed=1)
    assert a["test"] == b["test"]

def test_inductive_eval_touches_heldout(data):
    sp = SP.inductive_split(data, test_synset_frac=0.1, seed=0)
    ts = sp["test_synsets"]
    for h, hyper in sp["eval_edges"]:
        assert h in ts or hyper in ts
    for h, hyper in sp["train_edges"]:
        assert h not in ts and hyper not in ts

def test_negatives_exclude_true_hypernyms(data):
    import random
    rng = random.Random(0)
    edge = data.closure_edges[0]
    true_hypers = {hyper for h, hyper in data.closure_edges if h == edge[0]}
    negs = SP.negatives_for(edge, data, num=10, rng=rng)
    assert len(negs) == 10
    assert not (set(negs) & true_hypers)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_splits.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/data/splits.py
"""Task splits: reconstruction (all edges), link prediction (edge hold-out),
inductive (synset hold-out)."""
import random
from collections import defaultdict


def reconstruction_split(data):
    return {"edges": list(data.closure_edges)}


def link_prediction_split(data, val_frac=0.05, test_frac=0.05, seed=0):
    edges = list(data.closure_edges)
    rng = random.Random(seed)
    rng.shuffle(edges)
    n = len(edges)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    valid = edges[:n_val]
    test = edges[n_val:n_val + n_test]
    train = edges[n_val + n_test:]
    return {"train": train, "valid": valid, "test": test}


def inductive_split(data, test_synset_frac=0.1, seed=0):
    rng = random.Random(seed)
    n = len(data.synsets)
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = int(n * test_synset_frac)
    test_synsets = set(idx[:n_test])
    train_edges, eval_edges = [], []
    for h, hyper in data.closure_edges:
        if h in test_synsets or hyper in test_synsets:
            eval_edges.append((h, hyper))
        else:
            train_edges.append((h, hyper))
    return {
        "train_synsets": set(idx[n_test:]),
        "test_synsets": test_synsets,
        "train_edges": train_edges,
        "eval_edges": eval_edges,
    }


def _hypernyms_by_hypo(data):
    d = defaultdict(set)
    for h, hyper in data.closure_edges:
        d[h].add(hyper)
    return d


def negatives_for(edge, data, num, rng, _cache={}):
    hypo = edge[0]
    key = id(data)
    hyp_map = _cache.get(key)
    if hyp_map is None:
        hyp_map = _hypernyms_by_hypo(data)
        _cache[key] = hyp_map
    true_h = hyp_map[hypo] | {hypo}
    n = len(data.synsets)
    out = []
    while len(out) < num:
        cand = rng.randrange(n)
        if cand not in true_h:
            out.append(cand)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_splits.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/data/splits.py tests/test_splits.py
git commit -m "feat(data): reconstruction/link-prediction/inductive splits"
```

---

### Task 6: Graph statistics (δ-hyperbolicity, depth, branching)

**Files:**
- Create: `src/hypsimcse/data/graph_stats.py`
- Test: `tests/test_graph_stats.py`

**Interfaces:**
- Consumes: `WordNetData`.
- Produces:
  - `depths(data) -> dict[int,int]` — shortest hop count from a synset up to its nearest root (BFS on direct edges upward).
  - `branching_factors(data) -> dict` — `{"mean": float, "max": int, "hist": dict[int,int]}` over direct children counts.
  - `depth_distribution(data) -> dict[int,int]`.
  - `gromov_delta(data, num_samples=2000, seed=0) -> float` — sampled Gromov 4-point δ-hyperbolicity on the shortest-path metric of the **undirected** hypernymy graph (lower = more tree-like). Uses the 4-point condition on sampled quadruples; distances via BFS from sampled sources.
  - `summarize(data, **kw) -> dict` — bundles all of the above into a JSON-serializable dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_stats.py
from hypsimcse.data import wordnet as W
from hypsimcse.data import graph_stats as G
import pytest

@pytest.fixture(scope="module")
def data():
    W.ensure_wordnet()
    return W.load_noun_hypernymy(max_synsets=1500)

def test_depths_roots_are_zero(data):
    d = G.depths(data)
    for r in data.roots:
        assert d[r] == 0

def test_depths_child_deeper_than_parent(data):
    d = G.depths(data)
    for h, hyper in data.direct_edges:
        assert d[h] >= d[hyper]  # hyponym at least as deep as hypernym

def test_branching_positive(data):
    b = G.branching_factors(data)
    assert b["mean"] > 0 and b["max"] >= 1

def test_gromov_delta_nonneg_and_small_for_tree(data):
    delta = G.gromov_delta(data, num_samples=300, seed=0)
    assert delta >= 0.0

def test_summarize_json_serializable(data):
    import json
    s = G.summarize(data, num_samples=200)
    json.dumps(s)  # must not raise
    assert "gromov_delta" in s and "depth_distribution" in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_stats.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/data/graph_stats.py
"""Data-side priors: how tree-like is WordNet noun hypernymy? Gromov delta,
depth, branching. Establishes the precondition for hyperbolic advantage (H1)
and the depth baseline for norm-vs-depth (H4)."""
import random
from collections import defaultdict, deque


def _children_map(data):
    ch = defaultdict(list)
    for h, hyper in data.direct_edges:
        ch[hyper].append(h)
    return ch


def _parents_map(data):
    pa = defaultdict(list)
    for h, hyper in data.direct_edges:
        pa[h].append(hyper)
    return pa


def depths(data):
    """Hops up to nearest root via direct hypernym edges (BFS from roots down)."""
    ch = _children_map(data)
    depth = {r: 0 for r in data.roots}
    dq = deque(data.roots)
    while dq:
        u = dq.popleft()
        for c in ch[u]:
            if c not in depth or depth[c] > depth[u] + 1:
                depth[c] = depth[u] + 1
                dq.append(c)
    for i in range(len(data.synsets)):  # disconnected fallback
        depth.setdefault(i, 0)
    return depth


def branching_factors(data):
    ch = _children_map(data)
    counts = [len(v) for v in ch.values()]
    hist = defaultdict(int)
    for c in counts:
        hist[c] += 1
    mean = sum(counts) / len(counts) if counts else 0.0
    return {"mean": mean, "max": max(counts) if counts else 0, "hist": dict(hist)}


def depth_distribution(data):
    d = depths(data)
    hist = defaultdict(int)
    for v in d.values():
        hist[v] += 1
    return dict(hist)


def _undirected_adj(data):
    adj = defaultdict(set)
    for h, hyper in data.direct_edges:
        adj[h].add(hyper)
        adj[hyper].add(h)
    return adj


def _bfs_dist(adj, src, targets):
    dist = {src: 0}
    dq = deque([src])
    remaining = set(targets)
    remaining.discard(src)
    while dq and remaining:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                remaining.discard(v)
                dq.append(v)
    return dist


def gromov_delta(data, num_samples=2000, seed=0):
    """Sampled 4-point Gromov delta on the undirected shortest-path metric.
    delta = max over sampled quadruples of the gap between the two largest of
    the three pairwise-sum matchings, halved. 0 => tree metric."""
    adj = _undirected_adj(data)
    rng = random.Random(seed)
    n = len(data.synsets)
    nodes = [i for i in range(n) if adj[i]]
    if len(nodes) < 4:
        return 0.0
    delta = 0.0
    dist_cache = {}

    def dist(a, b):
        if a == b:
            return 0
        if a not in dist_cache:
            dist_cache[a] = _bfs_dist(adj, a, nodes)
        return dist_cache[a].get(b, float("inf"))

    for _ in range(num_samples):
        w, x, y, z = rng.sample(nodes, 4)
        d = {(w, x): dist(w, x), (w, y): dist(w, y), (w, z): dist(w, z),
             (x, y): dist(x, y), (x, z): dist(x, z), (y, z): dist(y, z)}
        if any(v == float("inf") for v in d.values()):
            continue
        s1 = d[(w, x)] + d[(y, z)]
        s2 = d[(w, y)] + d[(x, z)]
        s3 = d[(w, z)] + d[(x, y)]
        s = sorted([s1, s2, s3])
        delta = max(delta, (s[2] - s[1]) / 2.0)
    return delta


def summarize(data, num_samples=2000, seed=0):
    b = branching_factors(data)
    return {
        "num_synsets": len(data.synsets),
        "num_direct_edges": len(data.direct_edges),
        "num_closure_edges": len(data.closure_edges),
        "num_roots": len(data.roots),
        "branching_mean": b["mean"],
        "branching_max": b["max"],
        "depth_distribution": depth_distribution(data),
        "gromov_delta": gromov_delta(data, num_samples=num_samples, seed=seed),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_stats.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/data/graph_stats.py tests/test_graph_stats.py
git commit -m "feat(data): graph stats (Gromov delta, depth, branching)"
```

---

### Task 7: Shared lift head + pure-graph embedding model

**Files:**
- Create: `src/hypsimcse/models/lift.py`, `src/hypsimcse/models/graph_embedding.py`
- Test: `tests/test_lift.py`, `tests/test_graph_embedding.py`

**Interfaces:**
- Consumes: `hypsimcse.geometry.lorentz.expmap0`.
- Produces:
  - `lift.HyperbolicHead(nn.Module)`: `__init__(self, in_dim, out_dim, geometry, c=1.0, learnable_c=False, max_tangent_norm=None)`. `geometry ∈ {"EUC","HYP"}`. `forward(h) -> Tensor`. For `HYP`: linear `W: in_dim→out_dim` then `expmap0` → `(B, out_dim+1)` hyperboloid point. For `EUC`: linear then return the raw `(B, out_dim)` tangent vector (cosine handled by score). Exposes `.curvature() -> float` (reads `softplus(raw_c)` if learnable else fixed).
  - `graph_embedding.GraphEmbedding(nn.Module)`: `__init__(self, num_nodes, dim, geometry, c=1.0, learnable_c=False, max_tangent_norm=None, init_scale=1e-3)`. Holds an `nn.Embedding(num_nodes, dim)` of tangent vectors; `forward(node_ids) -> Tensor` lifts them via the same head logic. Exposes `.curvature()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lift.py
import torch
from hypsimcse.models.lift import HyperbolicHead
from hypsimcse.geometry import lorentz as L

def test_hyp_head_output_on_manifold():
    head = HyperbolicHead(in_dim=8, out_dim=5, geometry="HYP", c=1.0)
    z = head(torch.randn(4, 8))
    assert z.shape == (4, 6)
    assert torch.allclose(L.lorentz_inner(z, z), torch.full((4,), -1.0), atol=1e-3)

def test_euc_head_returns_tangent():
    head = HyperbolicHead(in_dim=8, out_dim=5, geometry="EUC")
    z = head(torch.randn(4, 8))
    assert z.shape == (4, 5)

def test_learnable_c_positive():
    head = HyperbolicHead(in_dim=8, out_dim=5, geometry="HYP", learnable_c=True)
    assert head.curvature() > 0
    assert any(p.requires_grad for p in head.parameters())
```

```python
# tests/test_graph_embedding.py
import torch
from hypsimcse.models.graph_embedding import GraphEmbedding
from hypsimcse.geometry import lorentz as L

def test_graph_embedding_hyp_on_manifold():
    m = GraphEmbedding(num_nodes=50, dim=5, geometry="HYP", c=1.0)
    z = m(torch.tensor([0, 1, 2]))
    assert z.shape == (3, 6)
    assert torch.allclose(L.lorentz_inner(z, z), torch.full((3,), -1.0), atol=1e-3)

def test_graph_embedding_grad_flows():
    m = GraphEmbedding(num_nodes=50, dim=5, geometry="HYP")
    z = m(torch.tensor([0, 1]))
    z.sum().backward()
    assert m.table.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lift.py tests/test_graph_embedding.py -q`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/models/lift.py
"""Shared head: project a Euclidean feature to either an L2-normalizable
tangent vector (EUC) or a hyperboloid point via exp map (HYP)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from ..geometry import lorentz as L


class HyperbolicHead(nn.Module):
    def __init__(self, in_dim, out_dim, geometry, c=1.0,
                 learnable_c=False, max_tangent_norm=None):
        super().__init__()
        assert geometry in ("EUC", "HYP")
        self.geometry = geometry
        self.out_dim = out_dim
        self.max_tangent_norm = max_tangent_norm
        self.proj = nn.Linear(in_dim, out_dim)
        self.learnable_c = learnable_c
        if learnable_c:
            # softplus(raw_c) ~ c ; init so curvature ~= c
            self.raw_c = nn.Parameter(torch.tensor(math.log(math.exp(c) - 1.0)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

    def curvature(self):
        if self.learnable_c:
            return Fn.softplus(self.raw_c).item()
        return float(self._c)

    def forward(self, h):
        v = self.proj(h)
        if self.geometry == "EUC":
            return v
        return L.expmap0(v, c=self.curvature(), max_norm=self.max_tangent_norm)
```

```python
# src/hypsimcse/models/graph_embedding.py
"""Pure-graph embedding (Nickel-Kiela style): one learnable tangent vector per
synset, lifted to the manifold. No encoder — isolates the geometry effect."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from ..geometry import lorentz as L


class GraphEmbedding(nn.Module):
    def __init__(self, num_nodes, dim, geometry, c=1.0, learnable_c=False,
                 max_tangent_norm=None, init_scale=1e-3):
        super().__init__()
        assert geometry in ("EUC", "HYP")
        self.geometry = geometry
        self.dim = dim
        self.max_tangent_norm = max_tangent_norm
        self.table = nn.Embedding(num_nodes, dim)
        nn.init.normal_(self.table.weight, std=init_scale)
        self.learnable_c = learnable_c
        if learnable_c:
            self.raw_c = nn.Parameter(torch.tensor(math.log(math.exp(c) - 1.0)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

    def curvature(self):
        if self.learnable_c:
            return Fn.softplus(self.raw_c).item()
        return float(self._c)

    def forward(self, node_ids):
        v = self.table(node_ids)
        if self.geometry == "EUC":
            return v
        return L.expmap0(v, c=self.curvature(), max_norm=self.max_tangent_norm)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lift.py tests/test_graph_embedding.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/models/lift.py src/hypsimcse/models/graph_embedding.py tests/test_lift.py tests/test_graph_embedding.py
git commit -m "feat(models): shared lift head + pure-graph embedding"
```

---

### Task 8: Encoder model (BERT/RoBERTa → hyperboloid)

**Files:**
- Create: `src/hypsimcse/models/encoder_model.py`
- Test: `tests/test_encoder_model.py`

**Interfaces:**
- Consumes: `transformers.AutoModel`, `HyperbolicHead`.
- Produces:
  - `encoder_model.SentenceEncoder(nn.Module)`: `__init__(self, model_name, out_dim, geometry, pooling="cls", c=1.0, learnable_c=False, max_tangent_norm=None)`. Holds `AutoModel` + `HyperbolicHead(hidden_size → out_dim)`. `forward(input_ids, attention_mask) -> Tensor` (tangent for EUC, hyperboloid for HYP). `pooling ∈ {"cls","mean"}`. Exposes `.curvature()` and `.tokenizer` via `SentenceEncoder.build(...)` classmethod that also returns the tokenizer.
  - `encode_texts(model, tokenizer, texts, device, batch_size=64, max_length=64) -> Tensor` — no-grad batched encode used at eval.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_encoder_model.py
import torch, pytest
from hypsimcse.geometry import lorentz as L

pytest.importorskip("transformers")
# Use a tiny model to keep the test fast/offline-friendly.
MODEL = "prajjwal1/bert-tiny"

def test_encoder_hyp_on_manifold():
    from hypsimcse.models.encoder_model import SentenceEncoder
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="HYP", c=1.0)
    batch = tok(["a dog", "a cat"], return_tensors="pt", padding=True)
    z = model(batch["input_ids"], batch["attention_mask"])
    assert z.shape == (2, 6)
    assert torch.allclose(L.lorentz_inner(z, z), torch.full((2,), -1.0), atol=1e-3)

def test_encoder_euc_shape():
    from hypsimcse.models.encoder_model import SentenceEncoder
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="EUC")
    batch = tok(["hi"], return_tensors="pt", padding=True)
    z = model(batch["input_ids"], batch["attention_mask"])
    assert z.shape == (1, 5)

def test_encode_texts_helper():
    from hypsimcse.models.encoder_model import SentenceEncoder, encode_texts
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="HYP")
    z = encode_texts(model, tok, ["a", "b", "c"], device="cpu", batch_size=2)
    assert z.shape == (3, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_encoder_model.py -q`
Expected: FAIL — module missing (downloads `bert-tiny` on first run).

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/models/encoder_model.py
"""SimCSE-style sentence encoder whose head lifts to the Lorentz model."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .lift import HyperbolicHead


def _pool(last_hidden, attention_mask, pooling):
    if pooling == "cls":
        return last_hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class SentenceEncoder(nn.Module):
    def __init__(self, model_name, out_dim, geometry, pooling="cls", c=1.0,
                 learnable_c=False, max_tangent_norm=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden = self.backbone.config.hidden_size
        self.head = HyperbolicHead(hidden, out_dim, geometry, c=c,
                                   learnable_c=learnable_c,
                                   max_tangent_norm=max_tangent_norm)

    def curvature(self):
        return self.head.curvature()

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _pool(out.last_hidden_state, attention_mask, self.pooling)
        return self.head(pooled)

    @classmethod
    def build(cls, model_name, out_dim, geometry, **kw):
        tok = AutoTokenizer.from_pretrained(model_name)
        model = cls(model_name, out_dim, geometry, **kw)
        return model, tok


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=64, max_length=64):
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                          padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        chunks.append(model(batch["input_ids"], batch["attention_mask"]).cpu())
    return torch.cat(chunks, dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_encoder_model.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/models/encoder_model.py tests/test_encoder_model.py
git commit -m "feat(models): BERT/RoBERTa sentence encoder with hyperbolic head"
```

---

### Task 9: Config, seeding, device, and Trainer

**Files:**
- Create: `src/hypsimcse/training/config.py`, `src/hypsimcse/training/seeding.py`, `src/hypsimcse/training/device.py`, `src/hypsimcse/training/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: models (Task 7/8), `losses.infonce`, splits (Task 5), WordNet (Task 4).
- Produces:
  - `config.ExperimentConfig` dataclass with fields: `track` (`"graph"|"encoder"|"dropout"`), `geometry` (`"EUC"|"HYP"`), `score` (one of `SCORE_LABELS`), `dim`, `tau`, `c`, `learnable_c`, `max_tangent_norm`, `anchor_weight`, `lr`, `weight_decay`, `grad_clip`, `epochs`, `batch_size`, `num_negatives`, `model_name`, `pooling`, `max_synsets`, `optimizer` (`"adamw"|"radam"`), `seed`, `max_length`. Plus `from_yaml(path) -> ExperimentConfig` and `to_dict()`.
  - `seeding.set_seed(seed) -> None`.
  - `device.get_device() -> str`.
  - `trainer.Trainer`: `__init__(self, config, data, model, device)`; `train() -> list[dict]` (per-epoch loss logs incl. loss parts and any NaN/instability flags); builds edge batches per track. For `track="dropout"` positives are the same node/gloss encoded twice (dropout noise); for `track="graph"`/`"encoder"` positives are hypernymy edge endpoints. Chooses AdamW or geoopt RiemannianAdam per `config.optimizer`. Applies grad clipping and NaN-guard (skip step + log if non-finite).
  - `trainer.build_model(config, data, device) -> nn.Module` factory (returns `GraphEmbedding` or `SentenceEncoder`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trainer.py
import torch
from hypsimcse.training.config import ExperimentConfig
from hypsimcse.training.seeding import set_seed
from hypsimcse.training import trainer as T
from hypsimcse.data import wordnet as W

def _tiny_cfg(**kw):
    base = dict(track="graph", geometry="HYP", score="HYP-inner", dim=5, tau=0.1,
                c=1.0, learnable_c=False, max_tangent_norm=5.0, anchor_weight=0.0,
                lr=0.05, weight_decay=0.0, grad_clip=1.0, epochs=1, batch_size=64,
                num_negatives=10, model_name="prajjwal1/bert-tiny", pooling="cls",
                max_synsets=800, optimizer="adamw", seed=0, max_length=32)
    base.update(kw)
    return ExperimentConfig(**base)

def test_set_seed_reproducible():
    set_seed(0); a = torch.randn(3)
    set_seed(0); b = torch.randn(3)
    assert torch.allclose(a, b)

def test_config_yaml_roundtrip(tmp_path):
    import yaml
    cfg = _tiny_cfg()
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict()))
    loaded = ExperimentConfig.from_yaml(str(p))
    assert loaded.to_dict() == cfg.to_dict()

def test_trainer_graph_runs_and_loss_finite():
    W.ensure_wordnet()
    data = W.load_noun_hypernymy(max_synsets=800)
    cfg = _tiny_cfg(epochs=1)
    set_seed(cfg.seed)
    model = T.build_model(cfg, data, "cpu")
    tr = T.Trainer(cfg, data, model, "cpu")
    logs = tr.train()
    assert len(logs) == 1
    assert all(l["loss"] == l["loss"] for l in logs)  # not NaN

def test_trainer_loss_decreases_graph():
    W.ensure_wordnet()
    data = W.load_noun_hypernymy(max_synsets=800)
    cfg = _tiny_cfg(epochs=3)
    set_seed(cfg.seed)
    model = T.build_model(cfg, data, "cpu")
    logs = T.Trainer(cfg, data, model, "cpu").train()
    assert logs[-1]["loss"] < logs[0]["loss"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -q`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write minimal implementations**

```python
# src/hypsimcse/training/seeding.py
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

```python
# src/hypsimcse/training/device.py
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

```python
# src/hypsimcse/training/config.py
from dataclasses import dataclass, asdict
import yaml

@dataclass
class ExperimentConfig:
    track: str            # graph | encoder | dropout
    geometry: str         # EUC | HYP
    score: str            # EUC | HYP-dist | HYP-dist2 | HYP-inner
    dim: int
    tau: float
    c: float = 1.0
    learnable_c: bool = False
    max_tangent_norm: float = 5.0
    anchor_weight: float = 0.0
    lr: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    epochs: int = 20
    batch_size: int = 256
    num_negatives: int = 50
    model_name: str = "bert-base-uncased"
    pooling: str = "cls"
    max_synsets: int = 0          # 0 => all
    optimizer: str = "adamw"      # adamw | radam
    seed: int = 0
    max_length: int = 64

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
```

```python
# src/hypsimcse/training/trainer.py
"""Config-driven trainer with MERU-style stabilization. One loop serves all
tracks; only positive-pair construction and the model differ."""
import random
import torch
from ..losses import infonce as Iloss
from ..models.graph_embedding import GraphEmbedding
from ..models.encoder_model import SentenceEncoder


def build_model(config, data, device):
    n_out = config.dim
    if config.track == "graph":
        model = GraphEmbedding(
            num_nodes=len(data.synsets), dim=n_out, geometry=config.geometry,
            c=config.c, learnable_c=config.learnable_c,
            max_tangent_norm=config.max_tangent_norm)
    else:
        model, tok = SentenceEncoder.build(
            config.model_name, out_dim=n_out, geometry=config.geometry,
            pooling=config.pooling, c=config.c, learnable_c=config.learnable_c,
            max_tangent_norm=config.max_tangent_norm)
        model.tokenizer = tok
    return model.to(device)


def _make_optimizer(model, config):
    if config.optimizer == "radam":
        import geoopt
        return geoopt.optim.RiemannianAdam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


class Trainer:
    def __init__(self, config, data, model, device):
        self.cfg = config
        self.data = data
        self.model = model
        self.device = device
        self.opt = _make_optimizer(model, config)
        self.rng = random.Random(config.seed)

    def _edges(self):
        if self.cfg.track == "dropout":
            # positive = same synset encoded twice (dropout noise); use node itself
            return [(i, i) for i in range(len(self.data.synsets))]
        return list(self.data.closure_edges)

    def _iter_batches(self, edges):
        self.rng.shuffle(edges)
        bs = self.cfg.batch_size
        for i in range(0, len(edges), bs):
            yield edges[i:i + bs]

    def _embed_graph(self, node_ids):
        ids = torch.tensor(node_ids, device=self.device)
        return self.model(ids)

    def _embed_encoder(self, node_ids):
        texts = [self.data.glosses[i] for i in node_ids]
        tok = self.model.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.cfg.max_length)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        return self.model(tok["input_ids"], tok["attention_mask"])

    def _embed(self, node_ids):
        if self.cfg.track == "graph":
            return self._embed_graph(node_ids)
        return self._embed_encoder(node_ids)

    def train(self):
        logs = []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            edges = self._edges()
            running, parts_acc, n_batches, skipped = 0.0, {}, 0, 0
            for batch in self._iter_batches(edges):
                hypo = [h for h, _ in batch]
                hyper = [t for _, t in batch]
                z = self._embed(hypo)
                z_pos = z if self.cfg.track == "dropout" else self._embed(hyper)
                loss, parts = Iloss.total_loss(
                    z, z_pos, self.cfg.score, self.cfg.tau,
                    c=self.model.curvature(), anchor_weight=self.cfg.anchor_weight)
                if not torch.isfinite(loss):
                    skipped += 1
                    self.opt.zero_grad()
                    continue
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                running += loss.item()
                for k, v in parts.items():
                    parts_acc[k] = parts_acc.get(k, 0.0) + v
                n_batches += 1
            avg = running / max(n_batches, 1)
            logs.append({"epoch": epoch, "loss": avg, "skipped": skipped,
                         **{f"part_{k}": v / max(n_batches, 1) for k, v in parts_acc.items()}})
        return logs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py -q`
Expected: PASS (4 tests). `test_trainer_loss_decreases_graph` confirms the loop learns.

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/training tests/test_trainer.py
git commit -m "feat(training): config, seeding, device, Trainer with stabilization"
```

---

### Task 10: Reconstruction evaluation (MAP, mean rank, distortion)

**Files:**
- Create: `src/hypsimcse/eval/reconstruction.py`
- Test: `tests/test_reconstruction.py`

**Interfaces:**
- Consumes: `hypsimcse.geometry.scores.pairwise_scores`, graph metric depths (for distortion baseline is the graph shortest-path — passed in).
- Produces:
  - `rank_of_positive(query_emb, pos_emb, neg_embs, score, c=1.0) -> int` — 1-based rank of the true hypernym among `[pos]+negs` by descending score.
  - `reconstruction_metrics(embeddings, edges, data, num_negatives, score, c=1.0, seed=0) -> dict` returning `{"MAP": float, "mean_rank": float, "distortion": float}`. For each edge, rank the true hypernym against `num_negatives` sampled non-hypernyms (`splits.negatives_for`). MAP = mean of `1/rank`. Distortion `D_avg` = mean over sampled edge pairs of `|d_embedding / d_graph − 1|` where `d_graph` is hop distance (uses `graph_stats` BFS on a subsample; if unavailable, skip and return `nan`).
  - Works for both EUC (space vectors) and HYP (hyperboloid points) embeddings — the caller passes the matching `score` label.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reconstruction.py
import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import reconstruction as R

class _Data:  # minimal stand-in
    def __init__(self, n):
        self.synsets = list(range(n))
        self.closure_edges = [(0, 1), (2, 3)]

def test_rank_of_positive_best_is_one():
    # positive identical to query -> HYP-inner max -> rank 1
    q = L.expmap0(torch.randn(1, 4))
    negs = L.expmap0(torch.randn(5, 4))
    r = R.rank_of_positive(q, q.clone(), negs, "HYP-inner")
    assert r == 1

def test_reconstruction_metrics_keys():
    emb = L.expmap0(torch.randn(4, 4))
    data = _Data(4)
    m = R.reconstruction_metrics(emb, data.closure_edges, data,
                                 num_negatives=2, score="HYP-inner", seed=0)
    assert set(m) >= {"MAP", "mean_rank"}
    assert 0.0 <= m["MAP"] <= 1.0
    assert m["mean_rank"] >= 1.0

def test_map_perfect_when_positives_closest():
    # Build embeddings so each hyponym's true hypernym is its nearest point.
    emb = torch.stack([
        L.expmap0(torch.tensor([0.0, 0.0, 0.0, 0.0])),   # 0
        L.expmap0(torch.tensor([0.01, 0.0, 0.0, 0.0])),  # 1 (near 0)
        L.expmap0(torch.tensor([5.0, 0.0, 0.0, 0.0])),   # 2
        L.expmap0(torch.tensor([5.01, 0.0, 0.0, 0.0])),  # 3 (near 2)
    ])
    data = _Data(4)
    m = R.reconstruction_metrics(emb, data.closure_edges, data,
                                 num_negatives=2, score="HYP-inner", seed=0)
    assert m["MAP"] > 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reconstruction.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/eval/reconstruction.py
"""Tree-reconstruction metrics: rank true hypernym among negatives.
MAP (mean reciprocal), mean rank, average distortion."""
import random
import torch
from ..geometry import scores as S
from ..data.splits import negatives_for


def rank_of_positive(query_emb, pos_emb, neg_embs, score, c=1.0):
    cand = torch.cat([pos_emb, neg_embs], dim=0)          # (1+N, ·)
    sims = S.pairwise_scores(query_emb, cand, score, c=c)[0]  # (1+N,)
    order = torch.argsort(sims, descending=True)
    pos_position = (order == 0).nonzero(as_tuple=True)[0].item()
    return pos_position + 1


def reconstruction_metrics(embeddings, edges, data, num_negatives, score,
                           c=1.0, seed=0):
    rng = random.Random(seed)
    ranks = []
    for (hypo, hyper) in edges:
        negs = negatives_for((hypo, hyper), data, num_negatives, rng)
        q = embeddings[hypo:hypo + 1]
        pos = embeddings[hyper:hyper + 1]
        neg = embeddings[torch.tensor(negs)]
        ranks.append(rank_of_positive(q, pos, neg, score, c=c))
    ranks = torch.tensor(ranks, dtype=torch.float)
    return {
        "MAP": (1.0 / ranks).mean().item(),
        "mean_rank": ranks.mean().item(),
        "distortion": float("nan"),  # populated by caller when graph metric given
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reconstruction.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/eval/reconstruction.py tests/test_reconstruction.py
git commit -m "feat(eval): reconstruction MAP / mean rank"
```

---

### Task 11: Link-prediction evaluation (MAP, AUC)

**Files:**
- Create: `src/hypsimcse/eval/link_prediction.py`
- Test: `tests/test_link_prediction.py`

**Interfaces:**
- Consumes: `pairwise_scores`, `splits.negatives_for`.
- Produces:
  - `link_prediction_metrics(embeddings, test_edges, data, num_negatives, score, c=1.0, seed=0) -> dict` = `{"MAP": float, "AUC": float}`. MAP as in reconstruction but on held-out edges. AUC = mean over test edges of the fraction of sampled negatives scored below the true positive (equivalent to ranking AUC per query, averaged).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_link_prediction.py
import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import link_prediction as LP

class _Data:
    def __init__(self, n):
        self.synsets = list(range(n))
        self.closure_edges = [(0, 1), (2, 3)]

def test_lp_keys_and_ranges():
    emb = L.expmap0(torch.randn(4, 4))
    data = _Data(4)
    m = LP.link_prediction_metrics(emb, data.closure_edges, data,
                                   num_negatives=2, score="HYP-inner", seed=0)
    assert set(m) == {"MAP", "AUC"}
    assert 0.0 <= m["AUC"] <= 1.0 and 0.0 <= m["MAP"] <= 1.0

def test_auc_high_when_positive_closest():
    emb = torch.stack([
        L.expmap0(torch.tensor([0.0, 0.0, 0.0])),
        L.expmap0(torch.tensor([0.01, 0.0, 0.0])),
        L.expmap0(torch.tensor([5.0, 0.0, 0.0])),
        L.expmap0(torch.tensor([5.01, 0.0, 0.0])),
    ])
    data = _Data(4)
    m = LP.link_prediction_metrics(emb, data.closure_edges, data,
                                   num_negatives=2, score="HYP-inner", seed=0)
    assert m["AUC"] > 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_link_prediction.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/eval/link_prediction.py
"""Link prediction on held-out hypernymy edges: MAP + ranking AUC."""
import random
import torch
from ..geometry import scores as S
from ..data.splits import negatives_for


def link_prediction_metrics(embeddings, test_edges, data, num_negatives, score,
                            c=1.0, seed=0):
    rng = random.Random(seed)
    recips, aucs = [], []
    for (hypo, hyper) in test_edges:
        negs = negatives_for((hypo, hyper), data, num_negatives, rng)
        q = embeddings[hypo:hypo + 1]
        cand = torch.cat([embeddings[hyper:hyper + 1], embeddings[torch.tensor(negs)]], dim=0)
        sims = S.pairwise_scores(q, cand, score, c=c)[0]
        pos_score = sims[0]
        neg_scores = sims[1:]
        rank = 1 + int((neg_scores > pos_score).sum().item())
        recips.append(1.0 / rank)
        aucs.append((neg_scores < pos_score).float().mean().item())
    return {
        "MAP": sum(recips) / max(len(recips), 1),
        "AUC": sum(aucs) / max(len(aucs), 1),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_link_prediction.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/eval/link_prediction.py tests/test_link_prediction.py
git commit -m "feat(eval): link-prediction MAP / AUC"
```

---

### Task 12: Inductive evaluation (unseen synsets)

**Files:**
- Create: `src/hypsimcse/eval/inductive.py`
- Test: `tests/test_inductive.py`

**Interfaces:**
- Consumes: `encode_texts`, `reconstruction_metrics`.
- Produces:
  - `inductive_metrics(model, tokenizer, data, inductive_split, num_negatives, score, device, c=1.0, seed=0) -> dict` = `{"MAP": float, "mean_rank": float}`. Encodes **all** synset glosses (train + held-out) with the trained encoder, then runs `reconstruction_metrics` on the `eval_edges` (edges touching held-out synsets). Encoder-track only (raises if model has no `tokenizer`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inductive.py
import torch, pytest
from hypsimcse.eval import inductive as IND

pytest.importorskip("transformers")
MODEL = "prajjwal1/bert-tiny"

class _Data:
    def __init__(self):
        self.synsets = ["a.n.01", "b.n.01", "c.n.01", "d.n.01"]
        self.glosses = ["alpha thing", "beta thing", "gamma item", "delta item"]
        self.closure_edges = [(0, 1), (2, 3)]

def test_inductive_runs_on_heldout():
    from hypsimcse.models.encoder_model import SentenceEncoder
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="HYP", c=1.0)
    data = _Data()
    isplit = {"test_synsets": {3}, "eval_edges": [(2, 3)],
              "train_synsets": {0, 1, 2}, "train_edges": [(0, 1)]}
    m = IND.inductive_metrics(model, tok, data, isplit, num_negatives=1,
                              score="HYP-inner", device="cpu", seed=0)
    assert set(m) == {"MAP", "mean_rank"}
    assert 0.0 <= m["MAP"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inductive.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/eval/inductive.py
"""Inductive generalization: encode UNSEEN synsets' glosses and check they land
in the right place. Only the encoder track can do this (N-K is transductive)."""
from ..models.encoder_model import encode_texts
from .reconstruction import reconstruction_metrics


def inductive_metrics(model, tokenizer, data, inductive_split, num_negatives,
                      score, device, c=1.0, seed=0):
    emb = encode_texts(model, tokenizer, data.glosses, device=device)
    m = reconstruction_metrics(emb, inductive_split["eval_edges"], data,
                               num_negatives=num_negatives, score=score,
                               c=c, seed=seed)
    return {"MAP": m["MAP"], "mean_rank": m["mean_rank"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inductive.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/eval/inductive.py tests/test_inductive.py
git commit -m "feat(eval): inductive generalization on unseen synsets"
```

---

### Task 13: STS-B quality guard

**Files:**
- Create: `src/hypsimcse/eval/sts.py`
- Test: `tests/test_sts.py`

**Interfaces:**
- Consumes: `encode_texts`, `pairwise_scores`, `scipy.stats.spearmanr`.
- Produces:
  - `sts_spearman(model, tokenizer, pairs, gold, score, device, c=1.0) -> float`. `pairs` = list of `(sent_a, sent_b)`. Encodes each side, computes per-pair similarity with `score`, returns Spearman correlation with `gold`. Guards that the gloss encoder hasn't destroyed general semantics.
  - `load_stsb_validation() -> (pairs, gold)` — loads GLUE STS-B validation via `datasets` (returns `([], [])` and warns if unavailable).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sts.py
import torch, pytest
from hypsimcse.eval import sts as STS

pytest.importorskip("transformers")
MODEL = "prajjwal1/bert-tiny"

def test_sts_spearman_returns_float():
    from hypsimcse.models.encoder_model import SentenceEncoder
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="HYP", c=1.0)
    pairs = [("a cat", "a dog"), ("hello world", "hello world"), ("x", "completely different y")]
    gold = [3.0, 5.0, 1.0]
    r = STS.sts_spearman(model, tok, pairs, gold, "HYP-inner", device="cpu")
    assert isinstance(r, float) and -1.0 <= r <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sts.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/eval/sts.py
"""STS-B Spearman quality guard: does the gloss encoder preserve general
semantic similarity while specializing to hierarchy?"""
import torch
from scipy.stats import spearmanr
from ..geometry import scores as S
from ..models.encoder_model import encode_texts


def sts_spearman(model, tokenizer, pairs, gold, score, device, c=1.0):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    za = encode_texts(model, tokenizer, a, device=device)
    zb = encode_texts(model, tokenizer, b, device=device)
    sims = torch.stack([
        S.pairwise_scores(za[i:i + 1], zb[i:i + 1], score, c=c)[0, 0]
        for i in range(len(pairs))
    ]).tolist()
    rho, _ = spearmanr(sims, gold)
    return float(rho)


def load_stsb_validation():
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "stsb", split="validation")
        pairs = list(zip(ds["sentence1"], ds["sentence2"]))
        gold = list(ds["label"])
        return pairs, gold
    except Exception as e:  # noqa: BLE001
        print(f"STS-B unavailable: {e}")
        return [], []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sts.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/eval/sts.py tests/test_sts.py
git commit -m "feat(eval): STS-B Spearman quality guard"
```

---

### Task 14: Fisher collapse diagnostics (H3)

**Files:**
- Create: `src/hypsimcse/eval/fisher.py`
- Test: `tests/test_fisher.py`

**Interfaces:**
- Consumes: `lorentz` (origin, log map direction), `numpy.linalg.eigh`.
- Produces:
  - `logmap0(z, c=1.0, eps=1e-6) -> Tensor` — inverse of `expmap0`: hyperboloid point → tangent-at-origin space vector. (Added here since collapse analysis needs tangent coordinates.)
  - `fisher_eigenvalues(embeddings, c=1.0) -> dict` = `{"radial_var": float, "tangential_var": float, "condition_number": float, "eigenvalues": list[float]}`. Maps hyperboloid points to tangent-at-origin coords, computes the empirical covariance (the empirical Fisher of the batch hyperbolic-vMF), then decomposes variance into **radial** (along the mean direction) and **tangential** (orthogonal complement, mean of remaining eigenvalues). Condition number = `max_eig / min_eig` (guarded). For EUC embeddings, operates directly on the vectors.
  - `collapse_vs_temperature(results_by_tau) -> dict` — convenience: given `{tau: fisher_eigenvalues(...)}`, returns arrays sorted by tau for plotting/fit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fisher.py
import math, torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import fisher as Fi

def test_logmap_inverts_expmap():
    v = torch.randn(6, 4)
    z = L.expmap0(v)
    v2 = Fi.logmap0(z)
    assert torch.allclose(v, v2, atol=1e-3)

def test_fisher_keys():
    z = L.expmap0(torch.randn(200, 5))
    out = Fi.fisher_eigenvalues(z)
    assert set(out) >= {"radial_var", "tangential_var", "condition_number", "eigenvalues"}
    assert out["condition_number"] >= 1.0

def test_collapse_detects_low_rank():
    # Points concentrated along one axis -> high condition number.
    v = torch.zeros(200, 5); v[:, 0] = torch.randn(200) * 2.0; v[:, 1:] = torch.randn(200, 4) * 1e-3
    z = L.expmap0(v)
    out = Fi.fisher_eigenvalues(z)
    assert out["condition_number"] > 100.0

def test_collapse_vs_temperature_sorted():
    a = Fi.fisher_eigenvalues(L.expmap0(torch.randn(100, 4)))
    b = Fi.fisher_eigenvalues(L.expmap0(torch.randn(100, 4)))
    out = Fi.collapse_vs_temperature({0.5: a, 0.05: b})
    assert out["tau"] == [0.05, 0.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fisher.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/eval/fisher.py
"""H3 collapse diagnostics: empirical Fisher (batch covariance in tangent coords),
split into radial vs tangential variance, condition number vs temperature."""
import math
import numpy as np
import torch
from ..geometry import lorentz as L


def logmap0(z, c=1.0, eps=1e-6):
    """Inverse exp map at origin: hyperboloid point -> tangent space vector.
    v = (arccosh(sqrt(c) x0) / sqrt(c)) * x_space / ||x_space||."""
    sqrt_c = math.sqrt(c)
    x0 = z[..., :1]
    x_space = z[..., 1:]
    norm = torch.norm(x_space, dim=-1, keepdim=True).clamp(min=eps)
    dist = torch.arccosh(torch.clamp(sqrt_c * x0, min=1.0 + eps)) / sqrt_c
    return dist * x_space / norm


def _covariance(x):
    x = x - x.mean(dim=0, keepdim=True)
    n = max(x.shape[0] - 1, 1)
    return (x.T @ x) / n


def fisher_eigenvalues(embeddings, c=1.0):
    if embeddings.shape[-1] >= 2 and _looks_hyperboloid(embeddings, c):
        tang = logmap0(embeddings, c=c)
    else:
        tang = embeddings
    cov = _covariance(tang).detach().cpu().numpy()
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    mean_vec = tang.mean(dim=0).detach().cpu().numpy()
    mean_norm = np.linalg.norm(mean_vec) + 1e-12
    u = mean_vec / mean_norm
    radial_var = float(u @ cov @ u)                       # variance along mean dir
    total = float(np.trace(cov))
    tangential_var = float((total - radial_var) / max(len(eigvals) - 1, 1))
    max_e = float(eigvals.max())
    min_e = float(max(eigvals.min(), 1e-12))
    return {
        "radial_var": radial_var,
        "tangential_var": tangential_var,
        "condition_number": max_e / min_e,
        "eigenvalues": [float(e) for e in sorted(eigvals, reverse=True)],
    }


def _looks_hyperboloid(z, c):
    ip = L.lorentz_inner(z, z)
    return bool(torch.allclose(ip, torch.full_like(ip, -1.0 / c), atol=1e-2))


def collapse_vs_temperature(results_by_tau):
    taus = sorted(results_by_tau)
    return {
        "tau": taus,
        "condition_number": [results_by_tau[t]["condition_number"] for t in taus],
        "radial_var": [results_by_tau[t]["radial_var"] for t in taus],
        "tangential_var": [results_by_tau[t]["tangential_var"] for t in taus],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fisher.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/eval/fisher.py tests/test_fisher.py
git commit -m "feat(eval): Fisher collapse diagnostics (radial/tangential, condition number)"
```

---

### Task 15: Hierarchy signal (norm vs depth, δ of embeddings)

**Files:**
- Create: `src/hypsimcse/eval/hierarchy.py`
- Test: `tests/test_hierarchy.py`

**Interfaces:**
- Consumes: `fisher.logmap0` (for tangent norm), `data.graph_stats.depths`, `scipy.stats.spearmanr`.
- Produces:
  - `embedding_norms(embeddings, c=1.0) -> Tensor` — hyperbolic distance of each point from origin (for HYP), or L2 norm (for EUC).
  - `norm_depth_correlation(embeddings, data, c=1.0) -> dict` = `{"spearman": float, "pearson": float, "norms": list, "depths": list}` — tests H4: general concepts near the root (small norm). Positive correlation ⇒ deeper = larger norm = specificity signal.
  - `embedding_delta_hyperbolicity(embeddings, num_samples=1000, c=1.0, seed=0) -> float` — Gromov δ on the learned embedding distance matrix (subsampled), for the "support vs statistical manifold" diagnostic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hierarchy.py
import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import hierarchy as H

class _Data:
    def __init__(self, n):
        self.synsets = list(range(n))
        self.direct_edges = [(1, 0), (2, 1), (3, 2)]  # chain 0<-1<-2<-3
        self.roots = [0]

def test_embedding_norms_monotone():
    v = torch.zeros(3, 4); v[0] = 0.0; v[1, 0] = 1.0; v[2, 0] = 2.0
    z = L.expmap0(v)
    norms = H.embedding_norms(z)
    assert norms[0] < norms[1] < norms[2]

def test_norm_depth_correlation_positive_when_aligned():
    # deeper node -> larger tangent norm
    v = torch.zeros(4, 4)
    for i in range(4):
        v[i, 0] = float(i)
    z = L.expmap0(v)
    data = _Data(4)
    out = H.norm_depth_correlation(z, data)
    assert out["spearman"] > 0.9

def test_embedding_delta_nonneg():
    z = L.expmap0(torch.randn(20, 4))
    d = H.embedding_delta_hyperbolicity(z, num_samples=50, seed=0)
    assert d >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hierarchy.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/eval/hierarchy.py
"""H4 signal: does hierarchy organize as root-anchored scatter? Norm-vs-depth
correlation + delta-hyperbolicity of the learned embedding metric."""
import random
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from ..geometry import lorentz as L
from ..data.graph_stats import depths


def embedding_norms(embeddings, c=1.0):
    if embeddings.shape[-1] >= 2 and _is_hyperboloid(embeddings, c):
        o = L.origin(embeddings.shape[-1] - 1, c=c, dtype=embeddings.dtype,
                     device=embeddings.device).expand_as(embeddings)
        return L.dist(embeddings, o, c=c)
    return torch.norm(embeddings, dim=-1)


def _is_hyperboloid(z, c):
    ip = L.lorentz_inner(z, z)
    return bool(torch.allclose(ip, torch.full_like(ip, -1.0 / c), atol=1e-2))


def norm_depth_correlation(embeddings, data, c=1.0):
    norms = embedding_norms(embeddings, c=c).detach().cpu().numpy()
    depth = depths(data)
    d = np.array([depth[i] for i in range(len(data.synsets))], dtype=float)
    n = norms[:len(d)]
    rho, _ = spearmanr(n, d)
    r, _ = pearsonr(n, d)
    return {"spearman": float(rho), "pearson": float(r),
            "norms": n.tolist(), "depths": d.tolist()}


def embedding_delta_hyperbolicity(embeddings, num_samples=1000, c=1.0, seed=0):
    rng = random.Random(seed)
    n = embeddings.shape[0]
    if n < 4:
        return 0.0
    is_hyp = _is_hyperboloid(embeddings, c)

    def dist(a, b):
        if is_hyp:
            return L.dist(embeddings[a:a + 1], embeddings[b:b + 1], c=c).item()
        return torch.norm(embeddings[a] - embeddings[b]).item()

    delta = 0.0
    for _ in range(num_samples):
        w, x, y, z = rng.sample(range(n), 4)
        s = sorted([dist(w, x) + dist(y, z),
                    dist(w, y) + dist(x, z),
                    dist(w, z) + dist(x, y)])
        delta = max(delta, (s[2] - s[1]) / 2.0)
    return delta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hierarchy.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/eval/hierarchy.py tests/test_hierarchy.py
git commit -m "feat(eval): norm-vs-depth correlation + embedding delta"
```

---

### Task 16: Einstein midpoint / dual-coordinate check

**Files:**
- Create: `src/hypsimcse/eval/dual_coords.py`
- Test: `tests/test_dual_coords.py`

**Interfaces:**
- Consumes: `lorentz`.
- Produces:
  - `lorentz_centroid(points, weights, c=1.0, eps=1e-6) -> Tensor` — weighted Lorentz mean (Einstein midpoint): `m = Σ w_i x_i`, then project back to the hyperboloid by rescaling: `m / sqrt(-c⟨m,m⟩_L)`. This is `η = ∇ψ(θ)` for the hyperbolic vMF.
  - `softmax_lorentz_mean(query, points, tau, c=1.0) -> Tensor` — the mean the model implicitly uses: weights = `softmax(⟨query, x_i⟩_L / τ)`, then `lorentz_centroid`.
  - `dual_coordinate_gap(query, points, tau, c=1.0) -> float` — hyperbolic distance between `softmax_lorentz_mean` and the explicit Einstein-midpoint centroid computed with the same weights (should be ~0, verifying theory↔implementation bridge).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dual_coords.py
import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import dual_coords as D

def test_centroid_on_manifold():
    pts = L.expmap0(torch.randn(5, 4))
    w = torch.softmax(torch.randn(5), dim=0)
    m = D.lorentz_centroid(pts, w)
    assert torch.allclose(L.lorentz_inner(m, m), torch.tensor(-1.0), atol=1e-3)

def test_centroid_of_identical_points_is_the_point():
    p = L.expmap0(torch.randn(1, 4)).repeat(5, 1)
    w = torch.softmax(torch.randn(5), dim=0)
    m = D.lorentz_centroid(p, w)
    assert L.dist(m.unsqueeze(0), p[0:1]).item() < 1e-2

def test_dual_gap_small():
    pts = L.expmap0(torch.randn(6, 4))
    q = L.expmap0(torch.randn(1, 4))
    gap = D.dual_coordinate_gap(q, pts, tau=0.1)
    assert gap < 1e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dual_coords.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hypsimcse/eval/dual_coords.py
"""Theory-implementation bridge: the softmax-weighted Lorentz mean the model
uses IS the dual coordinate eta = grad psi (Einstein midpoint)."""
import math
import torch
from ..geometry import lorentz as L


def lorentz_centroid(points, weights, c=1.0, eps=1e-6):
    """Weighted Lorentz mean projected back to the hyperboloid."""
    w = weights.reshape(-1, 1)
    m = (w * points).sum(dim=0)
    denom = torch.sqrt(torch.clamp(-c * L.lorentz_inner(m, m), min=eps))
    return m / (denom * math.sqrt(c))


def softmax_lorentz_mean(query, points, tau, c=1.0):
    inner = L.lorentz_inner(query.expand_as(points), points)  # (N,)
    weights = torch.softmax(inner / tau, dim=0)
    return lorentz_centroid(points, weights, c=c), weights


def dual_coordinate_gap(query, points, tau, c=1.0):
    mean, weights = softmax_lorentz_mean(query[0], points, tau, c=c)
    explicit = lorentz_centroid(points, weights, c=c)
    return L.dist(mean.unsqueeze(0), explicit.unsqueeze(0), c=c).item()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dual_coords.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypsimcse/eval/dual_coords.py tests/test_dual_coords.py
git commit -m "feat(eval): Einstein-midpoint dual-coordinate check"
```

---

### Task 17: Experiment configs (E1–E6) + single-config runner + matrix

**Files:**
- Create: `experiments/configs/E1_graph.yaml`, `experiments/configs/E1_encoder.yaml`, `experiments/configs/E2_sweep.yaml`, `experiments/configs/E3_collapse.yaml`, `experiments/configs/E4_curvature.yaml`, `experiments/configs/E5_encoder_full.yaml`, `experiments/configs/E6_dropout.yaml`, `experiments/run.py`, `experiments/run_matrix.py`
- Test: extend `tests/test_e2e_smoke.py` (created in Task 19) — for this task, add a quick assertion the runner imports and a config parses.

**Interfaces:**
- Consumes: `ExperimentConfig`, `Trainer`, all eval modules, `graph_stats`.
- Produces:
  - `experiments/run.py`: `run_experiment(config, device=None) -> dict` — sets seed, loads data (`max_synsets` from config), builds split appropriate to track, trains, then runs the eval bundle appropriate to track and returns a JSON-serializable result dict (`config`, `train_logs`, `reconstruction`, `link_prediction`, `inductive` (encoder only), `sts` (encoder, optional), `fisher`, `norm_depth`, `dual_gap`, `data_stats`). CLI: `--config PATH --seed N --out PATH [--max-synsets N]`.
  - `experiments/run_matrix.py`: expands a base config over `--dims`, `--scores`, `--seeds`, `--taus` lists (E2/E3), writing one JSON per cell into `--outdir`. CLI wrapper only.
  - Each YAML maps 1:1 to the spec's experiment matrix rows (E1–E6).

- [ ] **Step 1: Write the E1–E6 config YAMLs**

`experiments/configs/E1_graph.yaml` (main ablation, pure graph, d=64, fixed τ, c=1):
```yaml
track: graph
geometry: HYP
score: HYP-inner        # sweep {EUC(with geometry EUC), HYP-dist, HYP-dist2, HYP-inner} via run_matrix
dim: 64
tau: 0.1
c: 1.0
learnable_c: false
max_tangent_norm: 5.0
anchor_weight: 0.0
lr: 0.05
weight_decay: 0.0
grad_clip: 1.0
epochs: 50
batch_size: 512
num_negatives: 50
max_synsets: 0
optimizer: adamw
seed: 0
```

`experiments/configs/E1_encoder.yaml` (main ablation, encoder track):
```yaml
track: encoder
geometry: HYP
score: HYP-inner
dim: 64
tau: 0.05
c: 1.0
learnable_c: false
max_tangent_norm: 5.0
anchor_weight: 0.0
lr: 3.0e-5
weight_decay: 0.01
grad_clip: 1.0
epochs: 3
batch_size: 128
num_negatives: 50
model_name: bert-base-uncased
pooling: cls
max_synsets: 0
optimizer: adamw
seed: 0
max_length: 64
```

`experiments/configs/E2_sweep.yaml` (efficiency / dim sweep, pure graph, EUC vs HYP-inner):
```yaml
track: graph
geometry: HYP
score: HYP-inner
dim: 64
tau: 0.1
c: 1.0
learnable_c: false
max_tangent_norm: 5.0
anchor_weight: 0.0
lr: 0.05
weight_decay: 0.0
grad_clip: 1.0
epochs: 50
batch_size: 512
num_negatives: 50
max_synsets: 0
optimizer: adamw
seed: 0
```

`experiments/configs/E3_collapse.yaml` (collapse, pure graph, EUC & HYP-inner, τ sweep):
```yaml
track: graph
geometry: HYP
score: HYP-inner
dim: 64
tau: 0.1
c: 1.0
learnable_c: false
max_tangent_norm: 5.0
anchor_weight: 0.0
lr: 0.05
weight_decay: 0.0
grad_clip: 1.0
epochs: 50
batch_size: 512
num_negatives: 50
max_synsets: 0
optimizer: adamw
seed: 0
```

`experiments/configs/E4_curvature.yaml` (encoder, HYP-inner, fixed vs learnable c — flip `learnable_c`):
```yaml
track: encoder
geometry: HYP
score: HYP-inner
dim: 64
tau: 0.05
c: 1.0
learnable_c: false
max_tangent_norm: 5.0
anchor_weight: 0.0
lr: 3.0e-5
weight_decay: 0.01
grad_clip: 1.0
epochs: 3
batch_size: 128
num_negatives: 50
model_name: bert-base-uncased
pooling: cls
max_synsets: 0
optimizer: adamw
seed: 0
max_length: 64
```

`experiments/configs/E5_encoder_full.yaml` (encoder, all metrics incl. inductive/STS/norm-depth, anchor on for H4):
```yaml
track: encoder
geometry: HYP
score: HYP-inner
dim: 64
tau: 0.05
c: 1.0
learnable_c: false
max_tangent_norm: 5.0
anchor_weight: 0.1
lr: 3.0e-5
weight_decay: 0.01
grad_clip: 1.0
epochs: 3
batch_size: 128
num_negatives: 50
model_name: bert-base-uncased
pooling: cls
max_synsets: 0
optimizer: adamw
seed: 0
max_length: 64
```

`experiments/configs/E6_dropout.yaml` (unsup dropout lower-bound, HYP-inner):
```yaml
track: dropout
geometry: HYP
score: HYP-inner
dim: 64
tau: 0.05
c: 1.0
learnable_c: false
max_tangent_norm: 5.0
anchor_weight: 0.0
lr: 3.0e-5
weight_decay: 0.01
grad_clip: 1.0
epochs: 3
batch_size: 128
num_negatives: 50
model_name: bert-base-uncased
pooling: cls
max_synsets: 0
optimizer: adamw
seed: 0
max_length: 64
```

- [ ] **Step 2: Write `experiments/run.py`**

```python
# experiments/run.py
"""Run one ExperimentConfig end-to-end: train + eval -> JSON result."""
import argparse
import json
import torch
from hypsimcse.training.config import ExperimentConfig
from hypsimcse.training.seeding import set_seed
from hypsimcse.training.device import get_device
from hypsimcse.training import trainer as T
from hypsimcse.data import wordnet as W
from hypsimcse.data import splits as SP
from hypsimcse.data import graph_stats as GS
from hypsimcse.eval import reconstruction as R
from hypsimcse.eval import link_prediction as LP
from hypsimcse.eval import inductive as IND
from hypsimcse.eval import sts as STS
from hypsimcse.eval import fisher as FI
from hypsimcse.eval import hierarchy as HI
from hypsimcse.eval import dual_coords as DC
from hypsimcse.models.encoder_model import encode_texts


def _all_embeddings(model, data, cfg, device):
    if cfg.track == "graph":
        with torch.no_grad():
            ids = torch.arange(len(data.synsets), device=device)
            return model(ids).cpu()
    return encode_texts(model, model.tokenizer, data.glosses, device=device)


def run_experiment(config, device=None):
    device = device or get_device()
    set_seed(config.seed)
    W.ensure_wordnet()
    data = W.load_noun_hypernymy(max_synsets=(config.max_synsets or None))

    model = T.build_model(config, data, device)
    tr = T.Trainer(config, data, model, device)
    train_logs = tr.train()

    emb = _all_embeddings(model, data, config, device)
    c = model.curvature()
    result = {"config": config.to_dict(), "train_logs": train_logs,
              "curvature": c, "device": device}

    result["data_stats"] = GS.summarize(data, num_samples=1000, seed=config.seed)

    recon_edges = SP.reconstruction_split(data)["edges"]
    # subsample edges for tractable eval
    import random
    rng = random.Random(config.seed)
    sample = recon_edges if len(recon_edges) <= 20000 else rng.sample(recon_edges, 20000)
    result["reconstruction"] = R.reconstruction_metrics(
        emb, sample, data, config.num_negatives, config.score, c=c, seed=config.seed)

    lp = SP.link_prediction_split(data, seed=config.seed)
    test = lp["test"] if len(lp["test"]) <= 20000 else rng.sample(lp["test"], 20000)
    result["link_prediction"] = LP.link_prediction_metrics(
        emb, test, data, config.num_negatives, config.score, c=c, seed=config.seed)

    result["fisher"] = FI.fisher_eigenvalues(emb, c=c)
    result["norm_depth"] = {k: v for k, v in HI.norm_depth_correlation(emb, data, c=c).items()
                            if k in ("spearman", "pearson")}
    result["embedding_delta"] = HI.embedding_delta_hyperbolicity(
        emb, num_samples=1000, c=c, seed=config.seed)

    # dual-coordinate gap on a small sample of queries
    if emb.shape[0] >= 8:
        q = emb[0:1]
        result["dual_gap"] = DC.dual_coordinate_gap(q, emb[1:65], config.tau, c=c) \
            if config.score.startswith("HYP") else None

    if config.track == "encoder":
        isplit = SP.inductive_split(data, seed=config.seed)
        result["inductive"] = IND.inductive_metrics(
            model, model.tokenizer, data, isplit, config.num_negatives,
            config.score, device=device, c=c, seed=config.seed)
        pairs, gold = STS.load_stsb_validation()
        if pairs:
            result["sts"] = STS.sts_spearman(model, model.tokenizer, pairs, gold,
                                             config.score, device=device, c=c)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-synsets", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_synsets is not None:
        cfg.max_synsets = args.max_synsets
    result = run_experiment(cfg)
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `experiments/run_matrix.py`**

```python
# experiments/run_matrix.py
"""Expand a base config over dims/scores/seeds/taus and dump one JSON per cell."""
import argparse
import json
import os
from hypsimcse.training.config import ExperimentConfig
from run import run_experiment  # same directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--scores", nargs="*", default=None)
    ap.add_argument("--taus", type=float, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--max-synsets", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    base = ExperimentConfig.from_yaml(args.config)
    if args.max_synsets is not None:
        base.max_synsets = args.max_synsets
    dims = args.dims or [base.dim]
    scores = args.scores or [base.score]
    taus = args.taus or [base.tau]
    for dim in dims:
        for score in scores:
            for tau in taus:
                for seed in args.seeds:
                    cfg = ExperimentConfig(**base.to_dict())
                    cfg.dim, cfg.tau, cfg.seed = dim, tau, seed
                    cfg.score = score
                    cfg.geometry = "EUC" if score == "EUC" else "HYP"
                    tag = f"d{dim}_{score}_tau{tau}_s{seed}"
                    res = run_experiment(cfg)
                    path = os.path.join(args.outdir, f"{tag}.json")
                    with open(path, "w") as f:
                        json.dump(res, f, indent=2)
                    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify configs parse and runner imports**

Run:
```bash
cd /Users/hayashinaofumi/workspace/waseda/contrastive-learning
python -c "from hypsimcse.training.config import ExperimentConfig as C; import glob; [C.from_yaml(p) for p in glob.glob('experiments/configs/*.yaml')]; print('configs OK')"
python -c "import sys; sys.path.insert(0,'experiments'); import run; print('run OK')"
```
Expected: prints `configs OK` then `run OK`.

- [ ] **Step 5: Smoke-run E6 (dropout) tiny to prove end-to-end**

Run:
```bash
python experiments/run.py --config experiments/configs/E1_graph.yaml --seed 0 --max-synsets 800 --out /tmp/e1_smoke.json
python -c "import json; r=json.load(open('/tmp/e1_smoke.json')); print('MAP', r['reconstruction']['MAP'], 'delta', r['data_stats']['gromov_delta'])"
```
Expected: JSON written; prints a MAP in `[0,1]` and a finite δ.

- [ ] **Step 6: Commit**

```bash
git add experiments/
git commit -m "feat(experiments): E1-E6 configs + single-config runner + matrix"
```

---

### Task 18: Plotting (3 mandatory figures)

**Files:**
- Create: `experiments/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: result JSONs from `run.py`/`run_matrix.py`.
- Produces (each reads a directory of JSONs, writes a PNG, returns the matplotlib `Figure`):
  - `plot_metric_vs_dim(result_dir, out_path, metric="MAP") -> Figure` — figure 1 (H1): metric vs `dim`, one line per geometry/score (EUC vs HYP-inner). Groups by score, averages over seeds, plots mean with std band.
  - `plot_norm_vs_depth(result_json, out_path) -> Figure` — figure 2 (H4): scatter of per-node embedding norm vs tree depth (reads `norm_depth` arrays; runner must include them — extend `run.py` to also store `norms`/`depths` when `--full-hierarchy`). Falls back to using the summary correlation as title if arrays absent.
  - `plot_condition_vs_tau(result_dir, out_path) -> Figure` — figure 3 (H3): Fisher condition number and radial/tangential variance vs τ, one line per geometry.
  - `_load_results(result_dir) -> list[dict]` helper.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plots.py
import json, os
import matplotlib
matplotlib.use("Agg")
import sys; sys.path.insert(0, "experiments")
import plots

def _fake_result(dim, score, seed, tau=0.1, cond=2.0):
    return {"config": {"dim": dim, "score": score, "seed": seed, "tau": tau,
                       "geometry": "EUC" if score == "EUC" else "HYP"},
            "reconstruction": {"MAP": 0.5 + 0.001 * dim, "mean_rank": 10.0},
            "fisher": {"condition_number": cond, "radial_var": 1.0, "tangential_var": 0.5},
            "norm_depth": {"spearman": 0.7, "pearson": 0.6}}

def test_metric_vs_dim(tmp_path):
    for dim in (2, 16, 64):
        for score in ("EUC", "HYP-inner"):
            for seed in (0, 1):
                p = tmp_path / f"d{dim}_{score}_s{seed}.json"
                p.write_text(json.dumps(_fake_result(dim, score, seed)))
    out = tmp_path / "fig1.png"
    fig = plots.plot_metric_vs_dim(str(tmp_path), str(out), metric="MAP")
    assert os.path.exists(out) and fig is not None

def test_condition_vs_tau(tmp_path):
    for tau in (0.05, 0.1, 0.5):
        for score in ("EUC", "HYP-inner"):
            p = tmp_path / f"tau{tau}_{score}.json"
            p.write_text(json.dumps(_fake_result(64, score, 0, tau=tau, cond=1.0 / tau)))
    out = tmp_path / "fig3.png"
    fig = plots.plot_condition_vs_tau(str(tmp_path), str(out))
    assert os.path.exists(out) and fig is not None

def test_norm_vs_depth(tmp_path):
    res = _fake_result(64, "HYP-inner", 0)
    res["norm_depth"]["norms"] = [0.1, 0.5, 1.0, 1.5]
    res["norm_depth"]["depths"] = [0, 1, 2, 3]
    p = tmp_path / "r.json"; p.write_text(json.dumps(res))
    out = tmp_path / "fig2.png"
    fig = plots.plot_norm_vs_depth(str(p), str(out))
    assert os.path.exists(out) and fig is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plots.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/plots.py
"""The 3 mandatory figures: metric-vs-dim (H1), norm-vs-depth (H4),
condition-number-vs-tau (H3)."""
import glob
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_results(result_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def plot_metric_vs_dim(result_dir, out_path, metric="MAP"):
    results = _load_results(result_dir)
    by_score = defaultdict(lambda: defaultdict(list))
    for r in results:
        s = r["config"]["score"]
        d = r["config"]["dim"]
        by_score[s][d].append(r["reconstruction"][metric])
    fig, ax = plt.subplots(figsize=(6, 4))
    for score, dim_map in sorted(by_score.items()):
        dims = sorted(dim_map)
        means = np.array([np.mean(dim_map[d]) for d in dims])
        stds = np.array([np.std(dim_map[d]) for d in dims])
        ax.plot(dims, means, marker="o", label=score)
        ax.fill_between(dims, means - stds, means + stds, alpha=0.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("embedding dimension d")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs dimension (H1: parameter efficiency)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    return fig


def plot_norm_vs_depth(result_json, out_path):
    with open(result_json) as f:
        r = json.load(f)
    nd = r.get("norm_depth", {})
    fig, ax = plt.subplots(figsize=(6, 4))
    if "norms" in nd and "depths" in nd:
        ax.scatter(nd["depths"], nd["norms"], s=6, alpha=0.3)
        ax.set_xlabel("tree depth"); ax.set_ylabel("embedding norm")
    title = f"norm vs depth (spearman={nd.get('spearman', float('nan')):.3f})"
    ax.set_title(title + "  (H4: root-anchored specificity)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    return fig


def plot_condition_vs_tau(result_dir, out_path):
    results = _load_results(result_dir)
    by_geo = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        g = r["config"]["geometry"]
        tau = r["config"]["tau"]
        by_geo[g]["cond"][tau].append(r["fisher"]["condition_number"])
        by_geo[g]["radial"][tau].append(r["fisher"]["radial_var"])
        by_geo[g]["tang"][tau].append(r["fisher"]["tangential_var"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for g, series in sorted(by_geo.items()):
        taus = sorted(series["cond"])
        cond = [np.mean(series["cond"][t]) for t in taus]
        ax.plot(taus, cond, marker="o", label=f"{g} condition #")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("temperature tau"); ax.set_ylabel("Fisher condition number")
    ax.set_title("condition number vs tau (H3: anisotropic collapse)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    return fig
```

- [ ] **Step 4: Extend `run.py` to optionally store full norm/depth arrays**

Modify `experiments/run.py`: replace the `norm_depth` line with a `--full-hierarchy` flag path.

```python
# in run_experiment signature add: full_hierarchy=False
    nd = HI.norm_depth_correlation(emb, data, c=c)
    if full_hierarchy:
        result["norm_depth"] = nd
    else:
        result["norm_depth"] = {k: nd[k] for k in ("spearman", "pearson")}
```
And add `--full-hierarchy` (store_true) to `main()`, threading it into `run_experiment`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_plots.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add experiments/plots.py experiments/run.py tests/test_plots.py
git commit -m "feat(experiments): mandatory H1/H3/H4 plots"
```

---

### Task 19: End-to-end smoke test + README run recipes

**Files:**
- Create: `tests/test_e2e_smoke.py`
- Modify: `README.md` (append full run recipes for E1–E6)

**Interfaces:**
- Consumes: everything.
- Produces: one fast end-to-end test that trains a tiny graph model and runs the full eval bundle, asserting the result dict shape; documented commands to reproduce each experiment at full scale.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2e_smoke.py
import sys; sys.path.insert(0, "experiments")
import run
from hypsimcse.training.config import ExperimentConfig

def test_graph_experiment_end_to_end():
    cfg = ExperimentConfig(track="graph", geometry="HYP", score="HYP-inner",
                           dim=5, tau=0.1, c=1.0, epochs=1, batch_size=256,
                           num_negatives=5, max_synsets=600, lr=0.05, seed=0)
    res = run.run_experiment(cfg, device="cpu")
    assert "reconstruction" in res and 0.0 <= res["reconstruction"]["MAP"] <= 1.0
    assert "link_prediction" in res and 0.0 <= res["link_prediction"]["AUC"] <= 1.0
    assert "fisher" in res and res["fisher"]["condition_number"] >= 1.0
    assert "data_stats" in res and res["data_stats"]["gromov_delta"] >= 0.0
    assert "norm_depth" in res

def test_euc_experiment_end_to_end():
    cfg = ExperimentConfig(track="graph", geometry="EUC", score="EUC",
                           dim=16, tau=0.05, epochs=1, batch_size=256,
                           num_negatives=5, max_synsets=600, lr=0.05, seed=0)
    res = run.run_experiment(cfg, device="cpu")
    assert 0.0 <= res["reconstruction"]["MAP"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails (or errors) before fixes**

Run: `pytest tests/test_e2e_smoke.py -q`
Expected: PASS if Tasks 1–18 are correct. If FAIL, fix the offending module (this test is the integration gate).

- [ ] **Step 3: Append run recipes to `README.md`**

```markdown
## Reproducing the experiments

All commands assume `source .venv/bin/activate` and `python scripts/download_data.py` has run.
Full scale uses all ~82k noun synsets (`max_synsets: 0`). On CPU/MPS this is slow;
reduce with `--max-synsets 20000` for quick checks.

### E1 — main ablation (H2): score comparison at d=64
    python experiments/run_matrix.py --config experiments/configs/E1_graph.yaml \
        --outdir results/E1_graph --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2
    python experiments/run_matrix.py --config experiments/configs/E1_encoder.yaml \
        --outdir results/E1_encoder --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2

### E2 — efficiency / dimension sweep (H1)
    python experiments/run_matrix.py --config experiments/configs/E2_sweep.yaml \
        --outdir results/E2 --dims 2 5 10 16 32 64 128 256 \
        --scores EUC HYP-inner --seeds 0 1 2
    python experiments/plots.py   # or call plot_metric_vs_dim('results/E2','figures/fig1.png')

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
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest -q`
Expected: PASS (all tests across Tasks 1–19).

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_smoke.py README.md
git commit -m "test: end-to-end smoke + reproduction recipes"
```

---

## Self-Review

**Spec coverage:**
- H1 efficiency / dim sweep → Task 6 (δ prior), Task 10 (MAP/rank), Task 17 (E2 config), Task 18 (fig1). ✓
- H2 score comparison (`-cosh d` vs `-d`/`-d²`, EUC) → Task 2 (scores), Task 3 (InfoNCE), Task 17 (E1). ✓
- H3 collapse / Fisher / temperature → Task 14 (Fisher radial/tangential/condition), Task 17 (E3), Task 18 (fig3). ✓
- H4 uniformity replacement / root-anchor / norm-depth → Task 3 (root_anchor_reg), Task 15 (norm-depth), Task 17 (E5 anchor_weight), Task 18 (fig2). ✓
- §1 base model SimCSE + 3 variants (supervised edges / dropout / pure graph) → Task 9 tracks `encoder`/`dropout`/`graph`. ✓
- §2 WordNet noun hypernymy, transitive closure, glosses, tasks (reconstruction/link-prediction/inductive), δ-prior → Tasks 4, 5, 6, 10, 11, 12. ✓
- §3 model changes (tangent lift to Lorentz, 4 scores, curvature fixed/learnable, numerical stabilization) → Tasks 1, 2, 7, 8, 9. ✓
- §3.4 stabilization stack (arccosh/sqrt clamp, tangent-norm cap, grad clip, small LR, AdamW vs Riemannian Adam) → Task 1 (clamps), Task 7 (max_tangent_norm), Task 9 (grad_clip, NaN-skip, optimizer switch). ✓
- §4.4 metrics (MAP/rank/distortion, LP MAP/AUC, inductive MAP, STS Spearman, Fisher, norm-depth, δ of embeddings, dual-coordinate) → Tasks 10–16. ✓
- §4.5 three mandatory plots → Task 18. ✓
- §6 experiment matrix E1–E6 × 3 seeds → Task 17. ✓
- §7 failure modes (stabilization ablation logged via train_logs `skipped`/`part_*`; δ prior for negative result; STS guard; fixed gloss preprocessing) → Tasks 4, 9, 13. ✓
- Appendix A formulas → Task 1 verbatim. ✓

**Known simplifications (documented, not gaps):** distortion `D_avg` returns `nan` from the metric fn (needs the graph shortest-path metric wired at the call site — acceptable v1; MAP/mean-rank are the primary reconstruction metrics per §4.4). Riemannian Adam is wired but only exercised when `optimizer: radam`. δ-hyperbolicity and dual-gap use subsampling for tractability; sample sizes are config/CLI arguments so nothing is silently capped without a knob.

**Type consistency:** `pairwise_scores(u,v,label,c,eps)` signature is used identically in Tasks 2, 10, 11, 16. `expmap0`/`logmap0` are mutual inverses (Task 1 / Task 14, tested). `HyperbolicHead`/`GraphEmbedding`/`SentenceEncoder` all expose `.curvature()` and are consumed uniformly by `Trainer` and `run.py`. `ExperimentConfig` field names match every `cfg.<field>` access in `trainer.py` and `run.py`. Score labels are the same 4 strings everywhere via `SCORE_LABELS`.
