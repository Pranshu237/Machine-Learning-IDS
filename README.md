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

Full details: [`results/benchmark.md`](results/benchmark.md) and
[`results/unseen_attacks.md`](results/unseen_attacks.md).

Accuracy is not a useful headline here: about 83% of flows are benign, so a
model can score over 99% while missing rare attacks. The tables use macro F1,
which weights every class equally.

**Full test set** (504,160 flows, 15 classes)

| Model | Macro F1 | Accuracy |
|---|---|---|
| XGBoost | 0.879 | 0.9989 |
| CatBoost | 0.875 | 0.9988 |
| LightGBM | 0.865 | 0.9988 |
| Random Forest | 0.852 | 0.9987 |

The four ensembles are close; the remaining errors are concentrated in a few
classes (Bot, XSS and SQL injection web attacks, Heartbleed), several of which
have only a handful of test flows.

**Same subset for all five models** (5,000 training / 2,000 test flows)

| Model | Macro F1 |
|---|---|
| XGBoost | 0.840 |
| CatBoost | 0.829 |
| LightGBM | 0.767 |
| Random Forest | 0.763 |
| GNN (GCN) | 0.394 |

The GNN trails the tree models by a wide margin and fits even its own
training data poorly (0.60 macro F1). One likely reason is that its graph
links flows that look alike rather than hosts that communicate, so it adds
little that the features do not already contain (see below).

**Attack types never seen in training**

| Attack family | Detected when in training | Detected when held out | Anomaly detector |
|---|---|---|---|
| DoS | 99.9% | 1.2% | 41.9% |
| DDoS | 100.0% | 63.5% | 5.2% |
| PortScan | 99.6% | 0.3% | 0.0% |
| Brute force (FTP/SSH) | 99.9% | 0.2% | 0.0% |
| Web attacks | 98.8% | 27.2% | 0.0% |
| Botnet | 91.3% | 0.0% | 0.7% |
| Infiltration (36 flows) | 100% | 0.0% | 36.1% |
| Heartbleed (11 flows) | 100% | 0.0% | 100% |

A detector that catches nearly every attack type it was trained on misses
most attack types it was not: for six of the eight families, detection drops
below 2%. The anomaly detector, trained on benign traffic only (1% false
alarms), catches some of what the supervised model misses (DoS,
Infiltration, Heartbleed) but none of the attacks that resemble normal
traffic (port scans, brute force, web attacks).

### Two models that failed to train

A model that scores poorly on its own training data has failed to train,
which is a different problem from failing to generalise, so `train.py`
reports training macro F1 for every model.

- **LightGBM** scored 0.14 macro F1 in an earlier run. The most likely cause
  is its default `min_child_weight` (1e-3): leaves for the rarest classes can
  have almost no hessian, giving huge leaf values. Setting it to 1.0 (XGBoost's default)
  gave 0.865.
- **XGBoost** reached only 0.67 macro F1 on its own training data.
  [`experiments/xgboost_check.py`](experiments/xgboost_check.py) retrained it
  with one change at a time ([results](results/xgboost_check.json)): a fixed
  `base_score` of 0.5 restored it (0.95 training, 0.879 test), while float64
  input changed nothing. `max_delta_step=1` scored higher on the test set
  (0.924), but choosing by test score would tune on the test set, so the fix
  targets the diagnosed cause instead.

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
