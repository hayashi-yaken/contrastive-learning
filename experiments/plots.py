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
