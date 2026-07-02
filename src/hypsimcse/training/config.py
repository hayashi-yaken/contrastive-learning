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
