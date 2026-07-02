# 双曲 SimCSE（Hyperbolic SimCSE）

対照学習の潜在空間を超球面／ユークリッド空間から**双曲空間（Lorentz 模型）**へ差し替え、
WordNet 名詞の上位下位（hypernymy）関係という階層データを、より少ないパラメータで
効率よく学習できるかを検証する研究用コードベースです。

> English version: [README.md](README.md) ／ 詳細な設計・実験ガイド: [docs/ja/設計と実験ガイド.md](docs/ja/設計と実験ガイド.md)

## 何を検証するのか（仮説）

- **H1（効率）** 階層データでは、双曲空間にすると低次元でユークリッド／超球面と同等以上の階層再現性が得られる。
- **H2（スコア関数・本研究の核）** 情報幾何が予言するスコア `sim(u,v) = -cosh d = ⟨u,v⟩_L`（負の Lorentz 内積）は、既存の `-d` や `-d²` を上回る。
- **H3（崩壊現象）** 低温で Fisher 計量がランク落ち（次元崩壊）し、動径方向が接方向より速く退化する異方的な悪条件化が温度の関数として観測される。
- **H4（一様性の消失）** 双曲空間には正規化可能な一様分布が存在しないため、Wang–Isola の uniformity 項の代わりに「根アンカー型の散らばり」正則化を用いても階層タスクで性能を損なわない。

## セットアップ

    uv venv && source .venv/bin/activate   # もしくは python -m venv
    uv pip install -e ".[dev]"
    python scripts/download_data.py        # WordNet + STS-B の取得

※ 本リポジトリはすでに system Python（pyenv 3.13, torch 2.11 / transformers 4.46）へ
editable install 済みで動作確認されています。新規に環境を作る場合のみ上記を実行してください。

## テスト

    pytest -q

（全 69 テストが通過します。CPU で完結します。）

## 実験を 1 本走らせる

    python experiments/run.py --config experiments/configs/E1_graph.yaml --seed 0

## 実験の再現

いずれのコマンドも `python scripts/download_data.py` 実行済みを前提とします。
フル規模は全 ~8.2 万名詞 synset（`max_synsets: 0`）を使います。CPU/MPS では重いため、
動作確認は `--max-synsets 20000` などで縮小してください。

### E1 — 主 ablation（H2）: d=64 でのスコア比較
    python experiments/run_matrix.py --config experiments/configs/E1_graph.yaml \
        --outdir results/E1_graph --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2
    python experiments/run_matrix.py --config experiments/configs/E1_encoder.yaml \
        --outdir results/E1_encoder --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2

### E2 — 効率／次元スイープ（H1）
    python experiments/run_matrix.py --config experiments/configs/E2_sweep.yaml \
        --outdir results/E2 --dims 2 5 10 16 32 64 128 256 \
        --scores EUC HYP-inner --seeds 0 1 2

### E3 — 崩壊／温度スイープ（H3）
    python experiments/run_matrix.py --config experiments/configs/E3_collapse.yaml \
        --outdir results/E3 --taus 0.01 0.02 0.05 0.1 0.2 0.5 \
        --scores EUC HYP-inner --seeds 0 1 2

### E4 — 曲率（固定 vs 学習）
    python experiments/run.py --config experiments/configs/E4_curvature.yaml --seed 0 --out results/E4_fixed_s0.json
    # 続いて YAML（またはコピー）で learnable_c: true にして再実行 -> results/E4_learn_s0.json

### E5 — エンコーダ フル（帰納的汎化・STS・ノルム-深さ, H4）＋ 根アンカー正則化 ON
    python experiments/run.py --config experiments/configs/E5_encoder_full.yaml \
        --seed 0 --full-hierarchy --out results/E5_s0.json

### E6 — dropout 下限ベースライン
    python experiments/run.py --config experiments/configs/E6_dropout.yaml --seed 0 --out results/E6_s0.json

### 図（必須 3 プロット）
    python -c "import sys; sys.path.insert(0,'experiments'); import plots; \
        plots.plot_metric_vs_dim('results/E2','figures/fig1_metric_vs_dim.png'); \
        plots.plot_condition_vs_tau('results/E3','figures/fig3_condition_vs_tau.png'); \
        plots.plot_norm_vs_depth('results/E5_s0.json','figures/fig2_norm_vs_depth.png')"

## 実験 ↔ 仮説 対応表

| 実験 | トラック | 主因子 | 検証対象 |
|------|----------|--------|----------|
| E1  | 純グラフ + エンコーダ | スコア: EUC/HYP-dist/dist2/inner | H2（スコア） |
| E2  | 純グラフ | 次元スイープ | H1（効率） |
| E3  | 純グラフ | 温度スイープ + Fisher | H3（崩壊） |
| E4  | エンコーダ | 曲率 固定 vs 学習 | 曲率 × 規模 |
| E5  | エンコーダ | 全指標 + 根アンカー | H4（一様性の置換） |
| E6  | dropout | 教師なし下限 | 教師あり辺注入の寄与 |

## ディレクトリ構成

```
src/hypsimcse/
  geometry/   Lorentz 模型の数式（内積・距離・指数写像）とスコア関数
  losses/     InfoNCE + 根アンカー正則化
  data/       WordNet 読み込み・分割・グラフ統計（δ双曲性ほか）
  models/     共有 lift ヘッド・純グラフ埋め込み・文エンコーダ
  training/   config・seed・device・Trainer
  eval/       reconstruction / link prediction / inductive / STS /
              Fisher 崩壊診断 / ノルム-深さ / 双対座標
experiments/  E1–E6 の設定・runner・プロット
docs/         実装計画（英）と日本語ガイド
tests/        各モジュールの単体テスト + end-to-end スモーク
```

詳細は [docs/ja/設計と実験ガイド.md](docs/ja/設計と実験ガイド.md) と、
実装計画 [docs/superpowers/plans/2026-07-02-hyperbolic-simcse.md](docs/superpowers/plans/2026-07-02-hyperbolic-simcse.md) を参照してください。
