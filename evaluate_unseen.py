"""How well are attack types caught when the model has never seen them?

The benchmark in train.py trains and tests on the same attack types, so it
measures recognising known attacks. A real intrusion detector also has to
flag new ones. Here, for each attack family:

  * seen:   a benign-vs-attack Random Forest trained on ALL families,
            scored on that family's test flows;
  * unseen: the same model trained with that family removed entirely,
            scored on every flow of that family;
  * anomaly: an Isolation Forest trained on benign traffic only, with its
            threshold set so that 1% of held-out benign flows are flagged.

Each fold also reports the false-positive rate on benign test traffic.
Results are saved to results/unseen_attacks.md and results/unseen_attacks.json.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from preprocessing.preprocess import load_clean

FAMILIES = {
    "DoS": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"],
    "DDoS": ["DDoS"],
    "PortScan": ["PortScan"],
    "Brute force (FTP/SSH)": ["FTP-Patator", "SSH-Patator"],
    "Web attacks": ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"],
    "Botnet": ["Bot"],
    "Infiltration": ["Infiltration"],
    "Heartbleed": ["Heartbleed"],
}
ANOMALY_BENIGN_FPR = 0.01


def family_of(labels):
    lookup = {label: fam for fam, members in FAMILIES.items() for label in members}
    lookup["BENIGN"] = "BENIGN"
    fam = labels.map(lookup)
    unknown = sorted(labels[fam.isna()].unique())
    if unknown:
        raise ValueError(f"Labels not assigned to a family: {unknown}")
    return fam.to_numpy()


def forest(n_jobs):
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=n_jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign-train", type=int, default=None,
                        help="Use at most this many benign flows for training (faster runs).")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, labels = load_clean()
    X = X.to_numpy(dtype=np.float32)
    fam = family_of(labels)

    # Same stratified 80/20 split as the benchmark (no scaling needed for trees).
    idx_train, idx_test = train_test_split(
        np.arange(len(fam)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    if args.benign_train:
        rng = np.random.default_rng(RANDOM_STATE)
        benign_tr = idx_train[fam[idx_train] == "BENIGN"]
        keep = rng.choice(benign_tr, size=min(args.benign_train, len(benign_tr)), replace=False)
        idx_train = np.concatenate([keep, idx_train[fam[idx_train] != "BENIGN"]])

    benign_test = idx_test[fam[idx_test] == "BENIGN"]
    is_attack = fam != "BENIGN"

    # Seen: one model trained on every family
    print("Training benign-vs-attack model on all families...")
    seen_model = forest(args.n_jobs).fit(X[idx_train], is_attack[idx_train])
    seen_fpr = seen_model.predict(X[benign_test]).mean()

    # Anomaly detector: benign traffic only
    print("Training Isolation Forest on benign traffic...")
    rng = np.random.default_rng(RANDOM_STATE)
    benign_train = idx_train[fam[idx_train] == "BENIGN"]
    benign_fit, benign_val = train_test_split(benign_train, test_size=0.2, random_state=RANDOM_STATE)
    benign_fit = rng.choice(benign_fit, size=min(len(benign_fit), 200_000), replace=False)
    iso = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=args.n_jobs)
    iso.fit(X[benign_fit])
    # Lower score = more anomalous; flag the lowest 1% of held-out benign flows
    threshold = np.quantile(iso.score_samples(X[benign_val]), ANOMALY_BENIGN_FPR)
    iso_fpr = (iso.score_samples(X[benign_test]) < threshold).mean()

    results = {}
    for family in FAMILIES:
        start = time.time()
        fam_test = idx_test[fam[idx_test] == family]
        fam_all = np.where(fam == family)[0]
        if len(fam_all) == 0:
            continue

        train_without = idx_train[fam[idx_train] != family]
        model = forest(args.n_jobs).fit(X[train_without], is_attack[train_without])

        results[family] = {
            "flows": int(len(fam_all)),
            "test_flows": int(len(fam_test)),
            "seen_detection": float(seen_model.predict(X[fam_test]).mean()) if len(fam_test) else None,
            "unseen_detection": float(model.predict(X[fam_all]).mean()),
            "unseen_benign_fpr": float(model.predict(X[benign_test]).mean()),
            "anomaly_detection": float((iso.score_samples(X[fam_all]) < threshold).mean()),
        }
        r = results[family]
        print(f"{family}: seen {r['seen_detection']}, unseen {r['unseen_detection']:.4f}, "
              f"anomaly {r['anomaly_detection']:.4f} ({time.time() - start:.0f}s)")

    with open(os.path.join(RESULTS_DIR, "unseen_attacks.json"), "w") as f:
        json.dump({"families": FAMILIES, "results": results, "seen_benign_fpr": float(seen_fpr),
                   "anomaly_benign_fpr": float(iso_fpr), "benign_train_cap": args.benign_train}, f, indent=2)

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    lines = [
        "# Detecting attack types the model has never seen",
        "",
        "Detection rate = share of that family's flows flagged as attacks. "
        "\"Seen\" is scored on the family's test flows; \"unseen\" and \"anomaly\" on all of its flows, "
        "since neither model trained on them.",
        "",
        "| Attack family | Flows | Seen (RF) | Unseen (RF) | Benign false alarms (unseen RF) | Anomaly detector |",
        "|---|---|---|---|---|---|",
    ]
    for family, r in results.items():
        lines.append(f"| {family} | {r['flows']:,} | {pct(r['seen_detection'])} | {pct(r['unseen_detection'])} | "
                     f"{pct(r['unseen_benign_fpr'])} | {pct(r['anomaly_detection'])} |")
    lines += [
        "",
        f"Benign false-alarm rate: seen RF {pct(seen_fpr)}, anomaly detector {pct(iso_fpr)} "
        f"(threshold set for {pct(ANOMALY_BENIGN_FPR)} on held-out benign flows).",
    ]
    if args.benign_train:
        lines.append(f"Benign training flows capped at {args.benign_train:,}.")
    with open(os.path.join(RESULTS_DIR, "unseen_attacks.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved results to {RESULTS_DIR}/unseen_attacks.md")


if __name__ == "__main__":
    main()
