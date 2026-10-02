"""Train and score every model on CICIDS-2017.

Run in two stages:

  python train.py --stage full     the four tree ensembles on the full 80/20 split
  python train.py --stage subset   all five models (including the GNN) trained on
                                   the same small training subset and scored on the
                                   same small test subset, since the GNN's
                                   k-nearest-neighbour graph does not scale to 2.5M flows

The stages run as separate processes because on macOS, PyTorch and
XGBoost/LightGBM ship different copies of the OpenMP threading library,
and loading both into one process can crash or freeze it. The full stage
never imports PyTorch; the subset stage limits OpenMP to one thread,
which costs little on 5,000 flows.

Each stage saves results/benchmark_<stage>.json, and results/benchmark.md
is rebuilt from whichever stages have finished.
"""
import argparse
import json
import os
import platform
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["full", "subset"], required=True)
parser.add_argument("--models", default=None,
                    help="Comma-separated models to retrain (rf, xgboost, lightgbm, catboost, gnn); "
                         "their results replace the saved ones and the rest are kept.")
ARGS = parser.parse_args() if __name__ == "__main__" else None
if ARGS is not None and ARGS.stage == "subset":
    # Must be set before NumPy, LightGBM or PyTorch are imported.
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from config import GNN_TEST_SIZE, GNN_TRAIN_SIZE, RANDOM_STATE, RESULTS_DIR  # noqa: E402
from models.boosting_models import train_catboost, train_lightgbm, train_xgboost  # noqa: E402
from models.rf_model import train_rf  # noqa: E402
from preprocessing.preprocess import preprocess  # noqa: E402
from utils.metrics import evaluate  # noqa: E402

SHORT_NAMES = {
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "gnn": "GNN (GCN)",
}

TREE_MODELS = {
    "Random Forest": train_rf,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
    "CatBoost": train_catboost,
}


def train_gnn(X, y):
    from models.gnn_model import GNNClassifier  # imported here so the full stage never loads PyTorch

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


def write_report():
    stages = {}
    for stage in ("full", "subset"):
        path = os.path.join(RESULTS_DIR, f"benchmark_{stage}.json")
        if os.path.exists(path):
            with open(path) as f:
                stages[stage] = json.load(f)
    if not stages:
        return
    info = next(iter(stages.values()))["data"]
    env = next(iter(stages.values()))["environment"]

    lines = [
        "# Benchmark results",
        "",
        f"CICIDS-2017, {info['flows']:,} flows after cleaning and de-duplication, "
        f"{info['features']} features, {info['classes']} classes. "
        f"Stratified 80/20 split (seed {RANDOM_STATE}); the scaler is fitted on the training set only.",
        "",
        "Macro scores weight every class equally, so they show how well the rare attacks are caught. "
        "Accuracy is dominated by BENIGN traffic (about 83% of flows).",
    ]
    if "full" in stages:
        full = stages["full"]["results"]
        lines += [
            "",
            f"## Full test set ({info['test_flows']:,} flows)",
            "",
            table(full),
            "",
            "### Recall per class",
            "",
            per_class_table(full),
            "",
            "Classes with only a handful of test flows (Heartbleed, Infiltration, SQL injection) "
            "give recall figures that rest on very few examples.",
        ]
    if "subset" in stages:
        sub = stages["subset"]
        lines += [
            "",
            f"## Same subset for all models ({GNN_TRAIN_SIZE:,} training / {GNN_TEST_SIZE:,} test flows)",
            "",
            f"Averaged over the {sub['classes_scored']} classes present in the test subset.",
            "",
            table(sub["results"]),
            "",
            "The GNN connects each flow to its most similar flows in feature space "
            "(a k-nearest-neighbour graph), not to the hosts it communicated with.",
        ]
    lines += ["", f"Environment: Python {env['python']}, scikit-learn {env['scikit-learn']}, "
                  f"{env['system']} {env['machine']}."]
    with open(os.path.join(RESULTS_DIR, "benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(stage, only=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{stage}.json")
    previous = {}
    if only:
        unknown = [m for m in only if m not in SHORT_NAMES]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Choose from {list(SHORT_NAMES)}.")
        if not os.path.exists(json_path):
            raise SystemExit(f"--models needs an earlier full run of this stage ({json_path}).")
        with open(json_path) as f:
            previous = json.load(f)["results"]
        keep = {SHORT_NAMES[m] for m in only}

    X_train, X_test, y_train, y_test, target_names = preprocess()
    counts = np.bincount(np.concatenate([y_train, y_test]), minlength=len(target_names))
    print(f"Flows after cleaning: {len(y_train) + len(y_test)} "
          f"(train {len(y_train)}, test {len(y_test)}), features: {X_train.shape[1]}")

    output = {
        "data": {
            "flows": int(len(y_train) + len(y_test)),
            "test_flows": int(len(y_test)),
            "features": int(X_train.shape[1]),
            "classes": int(len(target_names)),
            "class_counts": dict(zip(map(str, target_names), counts.tolist())),
        },
        "environment": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }

    def selected(models):
        return {n: fn for n, fn in models.items() if not only or n in keep}

    if stage == "full":
        output["results"] = run(selected(TREE_MODELS), X_train, y_train, X_test, y_test, target_names)
    else:
        # Same subset for every model. The split is already shuffled, so the
        # first rows are a random sample.
        Xs_tr, ys_tr = X_train[:GNN_TRAIN_SIZE], y_train[:GNN_TRAIN_SIZE]
        Xs_te, ys_te = X_test[:GNN_TEST_SIZE], y_test[:GNN_TEST_SIZE]
        labels = sorted(np.unique(ys_te).tolist())
        output["classes_scored"] = len(labels)
        output["results"] = run(selected({**TREE_MODELS, "GNN (GCN)": train_gnn}),
                                Xs_tr, ys_tr, Xs_te, ys_te, target_names, labels=labels)

    if previous:
        # Keep the original model order, replacing only the retrained ones
        output["results"] = {n: output["results"].get(n, r) for n, r in previous.items()}

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    write_report()
    print(f"\nSaved results to {RESULTS_DIR}/benchmark_{stage}.json and {RESULTS_DIR}/benchmark.md")


if __name__ == "__main__":
    main(ARGS.stage, ARGS.models.split(",") if ARGS.models else None)
