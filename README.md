# Network Intrusion Detection System (NIDS) with CICIDS-2017

This repository contains a machine learning pipeline for network intrusion detection using the CICIDS-2017 dataset.

## Features
- **Data Preprocessing**: Handles missing values, normalizes features, and cleans corrupted labels from the CICIDS dataset.
- **Models**:
  - Tree-based Models: Random Forest, XGBoost, LightGBM, CatBoost
  - Graph Neural Networks (GNN): Built using PyTorch Geometric
- **Evaluation**: Computes precision, recall, f1-score, accuracy, and macro averages.
- **Explainability**: SHAP value integration for model interpretation.

## Structure
- `preprocessing/`: Scripts for data cleaning and feature engineering.
- `models/`: Implementations of tree-based and GNN models.
- `utils/`: Helper functions for metrics and evaluation.
- `explainability/`: Scripts for generating SHAP explanations.
- `train.py`: The main orchestrator script to run the full training pipeline.
- `config.py`: Configuration for paths and hyperparameters.

## How to Run
1. Place the CICIDS-2017 CSV files in the `data/raw/` directory.
2. Run `merge_data.py` to combine the CSV files.
3. Install dependencies: `pip install -r requirements.txt`
4. Execute `train.py` to start the training and evaluation pipeline.

*Note: The dataset is not included in this repository due to size limits.*
