"""Config-driven trainer with MERU-style stabilization. One loop serves all
tracks; only positive-pair construction and the model differ."""
import math
import random
import torch
from ..losses import infonce as Iloss
from ..models.graph_embedding import GraphEmbedding
from ..models.encoder_model import SentenceEncoder
from .logutil import log


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
        n_edges = len(self._edges())
        total_batches = max(1, math.ceil(n_edges / self.cfg.batch_size))
        log_every = getattr(self.cfg, "log_every", 50)
        log(f"train start: track={self.cfg.track} geometry={self.cfg.geometry} "
            f"score={self.cfg.score} dim={self.cfg.dim} tau={self.cfg.tau} "
            f"epochs={self.cfg.epochs} batches/epoch={total_batches}")
        for epoch in range(self.cfg.epochs):
            self.model.train()
            edges = self._edges()
            running, parts_acc, n_batches, skipped = 0.0, {}, 0, 0
            for batch in self._iter_batches(edges):
                hypo = [h for h, _ in batch]
                hyper = [t for _, t in batch]
                z = self._embed(hypo)
                # dropout track: re-encode the SAME items so independent dropout
                # masks give a distinct positive view (unsup SimCSE). Other
                # tracks: the positive is the hypernym endpoint of the edge.
                z_pos = self._embed(hypo) if self.cfg.track == "dropout" else self._embed(hyper)
                loss, parts = Iloss.total_loss(
                    z, z_pos, self.cfg.score, self.cfg.tau,
                    c=self.model.curvature_tensor(), anchor_weight=self.cfg.anchor_weight)
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
                if log_every and n_batches % log_every == 0:
                    log(f"  epoch {epoch + 1}/{self.cfg.epochs} "
                        f"batch {n_batches}/{total_batches} "
                        f"loss={running / n_batches:.4f}"
                        + (f" skipped={skipped}" if skipped else ""))
            avg = running / max(n_batches, 1)
            log(f"epoch {epoch + 1}/{self.cfg.epochs} done: loss={avg:.4f} "
                f"skipped={skipped}")
            logs.append({"epoch": epoch, "loss": avg, "skipped": skipped,
                         **{f"part_{k}": v / max(n_batches, 1) for k, v in parts_acc.items()}})
        return logs
