"""Write a small CSV shaped like the raw CICIDS-2017 files, for tests."""
import numpy as np
import pandas as pd

LABELS = {
    "BENIGN": 600, "DoS Hulk": 60, "DoS GoldenEye": 20, "DoS slowloris": 12, "DoS Slowhttptest": 12,
    "DDoS": 50, "PortScan": 40, "FTP-Patator": 12, "SSH-Patator": 12, "Bot": 10,
    "Web Attack \x96 Brute Force": 10, "Web Attack \x96 XSS": 8, "Web Attack \x96 Sql Injection": 6,
    "Infiltration": 6, "Heartbleed": 6,
}


def write_fake_cicids(path, n_features=20, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for k, (label, n) in enumerate(LABELS.items()):
        centre = rng.normal(0, 3, n_features)
        rows.append(centre + rng.normal(0, 1, (n, n_features)))
        labels += [label] * n
    df = pd.DataFrame(np.vstack(rows), columns=[f" Feature {i}" for i in range(n_features)])
    df[" Fwd Header Length.1"] = 0.0
    df = df.astype(object)
    df.iloc[0, 0] = "Infinity"  # raw files contain these strings
    df[" Label"] = labels
    df.to_csv(path, index=False)
    return df
