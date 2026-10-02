# Detecting attack types the model has never seen

Detection rate = share of that family's flows flagged as attacks. "Seen" is scored on the family's test flows; "unseen" and "anomaly" on all of its flows, since neither model trained on them.

| Attack family | Flows | Seen (RF) | Unseen (RF) | Benign false alarms (unseen RF) | Anomaly detector |
|---|---|---|---|---|---|
| DoS | 193,745 | 99.9% | 1.2% | 0.1% | 41.9% |
| DDoS | 128,014 | 100.0% | 63.5% | 0.1% | 5.2% |
| PortScan | 90,694 | 99.6% | 0.3% | 0.1% | 0.0% |
| Brute force (FTP/SSH) | 9,150 | 99.9% | 0.2% | 0.1% | 0.0% |
| Web attacks | 2,143 | 98.8% | 27.2% | 0.1% | 0.0% |
| Botnet | 1,948 | 91.3% | 0.0% | 0.1% | 0.7% |
| Infiltration | 36 | 100.0% | 0.0% | 0.1% | 36.1% |
| Heartbleed | 11 | 100.0% | 0.0% | 0.1% | 100.0% |

Benign false-alarm rate: seen RF 0.1%, anomaly detector 1.0% (threshold set for 1.0% on held-out benign flows).
Benign training flows capped at 500,000.
