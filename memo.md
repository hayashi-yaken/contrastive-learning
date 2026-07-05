# 実験実行メモ（nohup 版）

全実験を nohup + バックグラウンドで回すためのコマンド集。
- 実験結果(JSON) → `--out` / `--outdir`（`results/` 以下）
- 実行ログ → `logs/` 以下の `.out`（`2>&1` で標準エラーもまとめる）

基本は **1本ずつ順番に**回す（1枚のGPUを取り合うと遅くなる/OOM）。

---

## 0. 準備（毎回）

```bash
conda activate hypsimcse
cd ~/workspace/contrastive-learning
mkdir -p logs results figures
python scripts/download_data.py     # 初回のみ（WordNet + STS-B）
python -c "import torch; print('cuda', torch.cuda.is_available())"   # True を確認
```

---

## 1. 動作確認（小規模スモーク）

```bash
nohup python experiments/run.py \
    --config experiments/configs/E1_graph.yaml --seed 0 --max-synsets 2000 \
    --out results/smoke.json \
    > logs/smoke.out 2>&1 &
echo "PID = $!"
```

確認:
```bash
tail -f logs/smoke.out
python -c "import json; r=json.load(open('results/smoke.json')); print('MAP', r['reconstruction']['MAP'], 'AUC', r['link_prediction']['AUC'])"
```

---

## 2. 純グラフ系（H1/H2/H3・比較的速い）

### E1_graph — 主ablation（4スコア比較, d=64, 3シード）※ H2
```bash
nohup python experiments/run_matrix.py \
    --config experiments/configs/E1_graph.yaml \
    --outdir results/E1_graph \
    --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2 \
    > logs/E1_graph.out 2>&1 &
echo "PID = $!"
```

### E2 — 次元スイープ ※ H1（本丸）
```bash
nohup python experiments/run_matrix.py \
    --config experiments/configs/E2_sweep.yaml \
    --outdir results/E2 \
    --dims 2 5 10 16 32 64 128 256 \
    --scores EUC HYP-inner --seeds 0 1 2 \
    > logs/E2.out 2>&1 &
echo "PID = $!"
```

### E3 — 温度スイープ（崩壊診断）※ H3
```bash
nohup python experiments/run_matrix.py \
    --config experiments/configs/E3_collapse.yaml \
    --outdir results/E3 \
    --taus 0.01 0.02 0.05 0.1 0.2 0.5 \
    --scores EUC HYP-inner --seeds 0 1 2 \
    > logs/E3.out 2>&1 &
echo "PID = $!"
```

---

## 3. エンコーダ系（BERT-base・重い）

### E1_encoder — 主ablation（4スコア比較）
```bash
nohup python experiments/run_matrix.py \
    --config experiments/configs/E1_encoder.yaml \
    --outdir results/E1_encoder \
    --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2 \
    > logs/E1_encoder.out 2>&1 &
echo "PID = $!"
```

### E5 — エンコーダ フル（帰納的汎化・STS・ノルム-深さ）※ H4
`--full-hierarchy` で図2用の生データ（norm/depth 配列）も保存。3シード分。
```bash
for s in 0 1 2; do
nohup python experiments/run.py \
    --config experiments/configs/E5_encoder_full.yaml \
    --seed $s --full-hierarchy \
    --out results/E5_s${s}.json \
    > logs/E5_s${s}.out 2>&1 &
echo "E5 seed $s PID = $!"
done
```
（順番に回したい場合は `for` を使わず1本ずつ実行）

### E6 — dropout 下限ベースライン
```bash
for s in 0 1 2; do
nohup python experiments/run.py \
    --config experiments/configs/E6_dropout.yaml \
    --seed $s \
    --out results/E6_s${s}.json \
    > logs/E6_s${s}.out 2>&1 &
echo "E6 seed $s PID = $!"
done
```

---

## 4. E4 — 曲率（固定 vs 学習）

固定版:
```bash
nohup python experiments/run.py \
    --config experiments/configs/E4_curvature.yaml --seed 0 \
    --out results/E4_fixed_s0.json \
    > logs/E4_fixed_s0.out 2>&1 &
echo "PID = $!"
```

学習版（YAML をコピーして learnable_c を true に）:
```bash
cp experiments/configs/E4_curvature.yaml /tmp/E4_learn.yaml
sed -i 's/learnable_c: false/learnable_c: true/' /tmp/E4_learn.yaml
nohup python experiments/run.py \
    --config /tmp/E4_learn.yaml --seed 0 \
    --out results/E4_learn_s0.json \
    > logs/E4_learn_s0.out 2>&1 &
echo "PID = $!"
```

---

## 5. 図（必須3プロット）※ 実験完了後に実行

```bash
nohup python -c "import sys; sys.path.insert(0,'experiments'); import plots; \
    plots.plot_metric_vs_dim('results/E2','figures/fig1_metric_vs_dim.png'); \
    plots.plot_condition_vs_tau('results/E3','figures/fig3_condition_vs_tau.png'); \
    plots.plot_norm_vs_depth('results/E5_s0.json','figures/fig2_norm_vs_depth.png')" \
    > logs/plots.out 2>&1 &
echo "PID = $!"
```

---

## 6. 順番に全部流す（連結版・GPU 1枚で安全）

`&&` で直列。前が失敗したら止まる。
```bash
nohup bash -c '
  python experiments/run_matrix.py --config experiments/configs/E1_graph.yaml --outdir results/E1_graph --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2 &&
  python experiments/run_matrix.py --config experiments/configs/E2_sweep.yaml --outdir results/E2 --dims 2 5 10 16 32 64 128 256 --scores EUC HYP-inner --seeds 0 1 2 &&
  python experiments/run_matrix.py --config experiments/configs/E3_collapse.yaml --outdir results/E3 --taus 0.01 0.02 0.05 0.1 0.2 0.5 --scores EUC HYP-inner --seeds 0 1 2 &&
  python experiments/run_matrix.py --config experiments/configs/E1_encoder.yaml --outdir results/E1_encoder --scores EUC HYP-dist HYP-dist2 HYP-inner --seeds 0 1 2 &&
  python experiments/run.py --config experiments/configs/E5_encoder_full.yaml --seed 0 --full-hierarchy --out results/E5_s0.json &&
  python experiments/run.py --config experiments/configs/E6_dropout.yaml --seed 0 --out results/E6_s0.json &&
  python experiments/run.py --config experiments/configs/E4_curvature.yaml --seed 0 --out results/E4_fixed_s0.json
' > logs/all.out 2>&1 &
echo "ALL PID = $!"
```

---

## 監視・停止

```bash
tail -f logs/<name>.out        # リアルタイム表示（抜けるのは Ctrl-c）
tail -n 30 logs/<name>.out     # 末尾だけ
ps aux | grep -E 'run.py|run_matrix.py'   # 実行中プロセス
nvidia-smi                     # GPU使用状況
ls results/E2/                 # 出力JSONが増えているか

kill <PID>                     # 停止（PID は echo $! で控えた番号）
pkill -f run_matrix.py         # 名前で一括停止
```

ログ末尾に `wrote results/...` が並び、`ps` に出なくなれば完了。

---

## 縮小して試すとき

各コマンドに `--max-synsets 20000` を付けると速く終わる（フル規模は付けない＝全82k）。
```bash
# 例
nohup python experiments/run.py --config experiments/configs/E1_graph.yaml \
    --seed 0 --max-synsets 20000 --out results/E1_graph_small_s0.json \
    > logs/E1_graph_small.out 2>&1 &
```
