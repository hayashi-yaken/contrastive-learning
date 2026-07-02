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
