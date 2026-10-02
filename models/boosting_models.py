import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from config import CATBOOST_ITERATIONS, RANDOM_STATE


class _LabelMapped:
    """Wraps a model trained on labels 0..k-1 so it predicts the original labels.

    XGBoost needs contiguous class labels, which a subset may not have.
    """

    def __init__(self, model, classes):
        self.model = model
        self.classes = classes

    def predict(self, X):
        return self.classes[np.asarray(self.model.predict(X)).ravel().astype(int)]


def train_xgboost(X_train, y_train):
    classes, y_mapped = np.unique(y_train, return_inverse=True)
    # XGBoost 3.x estimates each class's starting score (base_score) from
    # the data. On the full training set, with classes as rare as 9
    # Heartbleed flows in 2 million, the default model reached only 0.67
    # macro F1 on its own training data. A fixed starting score of 0.5,
    # the default in older versions, restored it (0.95). See
    # experiments/xgboost_check.py and results/xgboost_check.json.
    model = XGBClassifier(eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE, base_score=0.5)
    model.fit(X_train, y_mapped)
    return _LabelMapped(model, classes)


def train_lightgbm(X_train, y_train):
    # With LightGBM's default min_child_weight (1e-3), leaves for the rarest
    # classes (e.g. 9 Heartbleed flows in training) can have almost no
    # hessian, which gives huge leaf values and the multiclass model
    # diverges: in our first run it scored 0.86 accuracy, worse than
    # always predicting BENIGN. 1.0 is XGBoost's default, so this
    # also puts the two boosting libraries on the same footing.
    model = LGBMClassifier(n_jobs=-1, verbose=-1, min_child_weight=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    model = CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS, verbose=0, thread_count=-1, random_seed=RANDOM_STATE,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    return _CatBoost(model)


class _CatBoost:
    """CatBoost returns predictions as a column vector; flatten them."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel().astype(int)
