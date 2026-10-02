"""Train and score every model on CICIDS-2017.

1. Full benchmark: the four tree ensembles on the full 80/20 split.
2. Subset comparison: all five models (including the GNN) trained on the
   same small training subset and scored on the same small test subset,
   since the GNN's k-nearest-neighbour graph does not scale to 2.5M flows.

Results are printed and saved to results/benchmark.md and results/benchmark.json.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import sklearn

from config import GNN_TEST_SIZE, GNN_TRAIN_SIZE, RANDOM_STATE, RESULTS_DIR
from models.boosting_models import train_catboost, train_lightgbm, train_xgboost
from models.gnn_model import GNNClassifier
from models.rf_model import train_rf
from preprocessing.preprocess import preprocess
from utils.metrics import evaluate

TREE_MODELS = {
    "Random Forest": train_rf,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
    "CatBoost": train_catboost,
}


def train_gnn(X, y):
    return GNNClassifier().fit(X, y)


def run(models, X_train, y_train, X_test, y_test, target_names, labels=None):
    results = {}
    for name, train_fn in models.items():
        print(f"\n--- Training {name} ---")
        start = time.time()
        model = train_fn(X_train, y_train)
        train_seconds = time.time() - start

        # A sanity check: a model that cannot fit its own training data
        # has failed to train, which is different from failing to generalise.
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(y_train), size=min(len(y_train), 200_000), replace=False)
        train_summary = evaluate(y_train[idx], model.predict(X_train[idx]), target_names, show=False)

        print(f"\n--- {name} results ---")
        summary = evaluate(y_test, model.predict(X_test), target_names, labels=labels)
        summary["train_accuracy"] = train_summary["accuracy"]
        summary["train_macro_f1"] = train_summary["macro_f1"]
        summary["train_seconds"] = round(train_seconds, 1)
        results[name] = summary
    return results


def table(results):
    rows = [
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Train macro F1 |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        rows.append(
            f"| {name} | {r['accuracy']:.4f} | {r['macro_precision']:.4f} | "
            f"{r['macro_recall']:.4f} | {r['macro_f1']:.4f} | {r['train_macro_f1']:.4f} |"
        )
    return "\n".join(rows)


def per_class_table(results):
    names = list(next(iter(results.values()))["per_class"])
    rows = ["| Class | Test flows | " + " | ".join(results) + " |",
            "|---|---|" + "---|" * len(results)]
    for n in names:
        support = int(next(iter(results.values()))["per_class"][n]["support"])
        recalls = " | ".join(f"{r['per_class'][n]['recall']:.4f}" for r in results.values())
        rows.append(f"| {n} | {support} | {recalls} |")
    return "\n".join(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, target_names = preprocess()
    counts = np.bincount(np.concatenate([y_train, y_test]), minlength=len(target_names))
    print(f"Flows after cleaning: {len(y_train) + len(y_test)} "
          f"(train {len(y_train)}, test {len(y_test)}), features: {X_train.shape[1]}")

    full = run(TREE_MODELS, X_train, y_train, X_test, y_test, target_names)

    # Same subset for every model. The split is already shuffled, so the
    # first rows are a random sample.
    Xs_tr, ys_tr = X_train[:GNN_TRAIN_SIZE], y_train[:GNN_TRAIN_SIZE]
    Xs_te, ys_te = X_test[:GNN_TEST_SIZE], y_test[:GNN_TEST_SIZE]
    subset_labels = sorted(np.unique(ys_te).tolist())
    subset = run({**TREE_MODELS, "GNN (GCN)": train_gnn},
                 Xs_tr, ys_tr, Xs_te, ys_te, target_names, labels=subset_labels)

    env = {
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
    }
    with open(os.path.join(RESULTS_DIR, "benchmark.json"), "w") as f:
        json.dump({"class_counts": dict(zip(map(str, target_names), counts.tolist())),
                   "full": full, "subset": subset, "environment": env}, f, indent=2)

    lines = [
        "# Benchmark results",
        "",
        f"CICIDS-2017, {len(y_train) + len(y_test):,} flows after cleaning and de-duplication, "
        f"{X_train.shape[1]} features, {len(target_names)} classes. "
        f"Stratified 80/20 split (seed {RANDOM_STATE}); the scaler is fitted on the training set only.",
        "",
        "Macro scores weight every class equally, so they show how well the rare attacks are caught. "
        "Accuracy is dominated by BENIGN traffic (about 83% of flows).",
        "",
        f"## Full test set ({len(y_test):,} flows)",
        "",
        table(full),
        "",
        "### Recall per class",
        "",
        per_class_table(full),
        "",
        "Classes with only a handful of test flows (Heartbleed, Infiltration, SQL injection) "
        "give recall figures that rest on very few examples.",
        "",
        f"## Same subset for all models ({GNN_TRAIN_SIZE:,} training / {GNN_TEST_SIZE:,} test flows)",
        "",
        f"Averaged over the {len(subset_labels)} classes present in the test subset.",
        "",
        table(subset),
        "",
        "The GNN connects each flow to its most similar flows in feature space "
        "(a k-nearest-neighbour graph), not to the hosts it communicated with.",
        "",
        f"Environment: Python {env['python']}, scikit-learn {env['scikit-learn']}, {env['system']} {env['machine']}.",
    ]
    with open(os.path.join(RESULTS_DIR, "benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved results to {RESULTS_DIR}/benchmark.md")


if __name__ == "__main__":
    main()
