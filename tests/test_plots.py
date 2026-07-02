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
