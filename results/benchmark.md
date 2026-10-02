# Benchmark results

CICIDS-2017, 2,520,798 flows after cleaning and de-duplication, 77 features, 15 classes. Stratified 80/20 split (seed 42); the scaler is fitted on the training set only.

Macro scores weight every class equally, so they show how well the rare attacks are caught. Accuracy is dominated by BENIGN traffic (about 83% of flows).

## Full test set (504,160 flows)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Train macro F1 |
|---|---|---|---|---|---|
| Random Forest | 0.9987 | 0.9374 | 0.8316 | 0.8522 | 0.9872 |
| XGBoost | 0.9932 | 0.6603 | 0.6906 | 0.6681 | 0.6723 |
| LightGBM | 0.9988 | 0.9341 | 0.8416 | 0.8654 | 0.9681 |
| CatBoost | 0.9988 | 0.9467 | 0.8668 | 0.8753 | 0.9135 |

### Recall per class

| Class | Test flows | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|---|
| BENIGN | 419012 | 0.9994 | 0.9958 | 0.9993 | 0.9993 |
| Bot | 390 | 0.6744 | 0.4154 | 0.7564 | 0.7256 |
| DDoS | 25603 | 0.9998 | 0.9951 | 0.9999 | 1.0000 |
| DoS GoldenEye | 2057 | 0.9917 | 0.9713 | 0.9937 | 0.9971 |
| DoS Hulk | 34569 | 0.9971 | 0.9826 | 0.9997 | 0.9998 |
| DoS Slowhttptest | 1046 | 0.9962 | 0.9398 | 0.9933 | 0.9952 |
| DoS slowloris | 1077 | 0.9898 | 0.8774 | 0.9879 | 0.9907 |
| FTP-Patator | 1186 | 0.9983 | 0.9089 | 0.9975 | 0.9983 |
| Heartbleed | 2 | 0.5000 | 0.0000 | 0.5000 | 1.0000 |
| Infiltration | 7 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| PortScan | 18139 | 0.9992 | 0.9925 | 0.9994 | 0.9992 |
| SSH-Patator | 644 | 0.9984 | 0.9596 | 1.0000 | 1.0000 |
| Web Attack - Brute Force | 294 | 0.9490 | 0.6939 | 0.8469 | 0.9694 |
| Web Attack - Sql Injection | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| Web Attack - XSS | 130 | 0.1308 | 0.3769 | 0.3000 | 0.0769 |

Classes with only a handful of test flows (Heartbleed, Infiltration, SQL injection) give recall figures that rest on very few examples.

## Same subset for all models (5,000 training / 2,000 test flows)

Averaged over the 11 classes present in the test subset.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Train macro F1 |
|---|---|---|---|---|---|
| Random Forest | 0.9905 | 0.7949 | 0.7445 | 0.7634 | 1.0000 |
| XGBoost | 0.9945 | 0.8882 | 0.8161 | 0.8404 | 1.0000 |
| LightGBM | 0.9930 | 0.7776 | 0.7700 | 0.7673 | 0.8123 |
| CatBoost | 0.9915 | 0.8852 | 0.7982 | 0.8290 | 1.0000 |
| GNN (GCN) | 0.9455 | 0.4325 | 0.3885 | 0.3939 | 0.6006 |

The GNN connects each flow to its most similar flows in feature space (a k-nearest-neighbour graph), not to the hosts it communicated with.

Environment: Python 3.12.15, scikit-learn 1.8.0, Darwin arm64.
