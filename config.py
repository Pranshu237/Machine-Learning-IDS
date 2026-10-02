import os

DATA_PATH = os.path.join("data", "cicids.csv")
RAW_DIR = os.path.join("data", "raw")
RESULTS_DIR = "results"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# CatBoost's default is 1000 rounds; the earlier runs used only 100.
CATBOOST_ITERATIONS = 1000

# The GNN builds a k-nearest-neighbour graph, which does not scale to
# millions of flows, so it is trained and tested on a fixed subset.
# The tree models are also scored on the same subset for a fair comparison.
GNN_TRAIN_SIZE = 5000
GNN_TEST_SIZE = 2000
GNN_NEIGHBOURS = 5
GNN_EPOCHS = 200
GNN_LR = 0.01
