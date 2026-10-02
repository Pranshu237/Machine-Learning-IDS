# Network Intrusion Detection on CICIDS-2017

A machine learning pipeline that classifies network flows from the
[CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset as benign
traffic or one of 14 attack types. It compares four tree ensembles with a graph
neural network, and tests how well a detector catches attack types it was never
trained on.

## What it does

- **Preprocessing** (`preprocessing/preprocess.py`): merges the daily CSV files,
  fixes the dataset's corrupted label encodings, removes infinite and missing
  values and exact duplicate flows, and makes a stratified 80/20 split. The
  scaler is fitted on the training set only.
- **Models** (`models/`): Random Forest, XGBoost, LightGBM, CatBoost and a
  two-layer GCN built with PyTorch Geometric.
- **Benchmark** (`train.py`): trains the tree models on the full training set
  (about 2M flows). Because the GNN's graph does not scale to millions of
  flows, all five models are also trained and scored on the same small subset,
  so the GNN is compared on equal terms.
- **Unseen attacks** (`evaluate_unseen.py`): for each attack family, trains a
  benign-vs-attack Random Forest with that family left out, then measures how
  much of it is still flagged. An Isolation Forest trained only on benign
  traffic is evaluated alongside it.

Results are written to `results/` as Markdown and JSON.

## Results

See [`results/benchmark.md`](results/benchmark.md) and
[`results/unseen_attacks.md`](results/unseen_attacks.md).

Accuracy is not a useful headline here: about 83% of flows are benign, so a
model can score over 99% while missing rare attacks. The results report macro
precision, recall and F1, which weight every class equally, together with
per-class recall.

## How the graph is built

The GNN connects each flow to its 5 most similar flows in feature space (a
k-nearest-neighbour graph). This is a similarity graph, not the real network
structure: the version of the dataset used here has no IP addresses, so flows
cannot be linked by the hosts they connect.

## Limitations

- A random split puts flows from the same attack session in both training and
  test sets, which flatters the benchmark. The unseen-attack evaluation is the
  more realistic test.
- Some classes are tiny (11 Heartbleed flows, 21 SQL injection, 36
  Infiltration), so their per-class scores rest on a handful of examples.
- CICIDS-2017 has known labelling and flow-construction issues
  (Engelen et al., 2021), which apply to these results too.

## How to run

1. Download the CICIDS-2017 `MachineLearningCSV` files and place them in
   `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Merge the files: `python merge_data.py`
4. Run the benchmark, in two stages:
   `python train.py --stage full`, then `python train.py --stage subset`
   (the stages run separately because PyTorch and XGBoost/LightGBM can
   conflict in one process on macOS). To retrain only some models and keep
   the other saved results, add e.g. `--models xgboost`.
5. Run the unseen-attack evaluation: `python evaluate_unseen.py`
   (add `--benign-train 500000` for a faster run on a laptop)

Tests use small synthetic data and need no download: `python -m pytest`

*The dataset is not included in this repository because of its size.*
