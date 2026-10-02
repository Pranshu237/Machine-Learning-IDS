"""Why does XGBoost underfit the full training set?

On the full data, XGBoost reached only 0.67 macro F1 on its own training
set (the other models reached 0.91-0.99), so it failed to train rather
than failing to generalise. This script retrains it with a few changes,
one at a time, and reports training and test macro F1 for each.

Run from the repository root: python -m experiments.xgboost_check
"""
import json
import os
import time

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from config import RANDOM_STATE, RESULTS_DIR
from preprocessing.preprocess import preprocess

VARIANTS = {
    "default": {},
    "base_score=0.5": {"base_score": 0.5},
    "max_delta_step=1": {"max_delta_step": 1},
    "learning_rate=0.1, 300 trees": {"learning_rate": 0.1, "n_estimators": 300},
}


def main():
    X_train, X_test, y_train, y_test, _ = preprocess()
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_train), size=min(200_000, len(y_train)), replace=False)

    runs = [(name, params, X_train, X_test) for name, params in VARIANTS.items()]
    runs.append(("default, float64 input", {}, X_train.astype(np.float64), X_test.astype(np.float64)))

    results = {}
    for name, params, Xtr, Xte in runs:
        start = time.time()
        model = XGBClassifier(eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE, **params)
        model.fit(Xtr, y_train)
        train_f1 = f1_score(y_train[idx], model.predict(Xtr[idx]), average="macro")
        test_f1 = f1_score(y_test, model.predict(Xte), average="macro")
        results[name] = {"train_macro_f1": train_f1, "test_macro_f1": test_f1,
                         "seconds": round(time.time() - start, 1)}
        print(f"{name}: train macro F1 {train_f1:.4f}, test macro F1 {test_f1:.4f} "
              f"({time.time() - start:.0f}s)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "xgboost_check.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
