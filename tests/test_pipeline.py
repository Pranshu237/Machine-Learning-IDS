import numpy as np
import pytest

from models.boosting_models import train_catboost, train_lightgbm, train_xgboost
from models.gnn_model import GNNClassifier
from models.rf_model import train_rf
from preprocessing.preprocess import load_clean, preprocess
from tests.fake_data import LABELS, write_fake_cicids
from utils.metrics import evaluate


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "cicids.csv"
    write_fake_cicids(path)
    return path


def test_load_clean_fixes_labels_and_drops_bad_rows(data):
    X, y = load_clean(data)
    assert "Fwd Header Length.1" not in X.columns
    assert "Web Attack - XSS" in set(y)
    assert not any("\x96" in label for label in y)
    assert len(X) == sum(LABELS.values()) - 1  # the row with "Infinity" is dropped
    assert np.isfinite(X.to_numpy()).all()


def test_scaler_is_fitted_on_training_set_only(data):
    X_train, X_test, y_train, y_test, names = preprocess(data)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)
    assert not np.allclose(X_test.mean(axis=0), 0, atol=1e-6)
    assert len(names) == 15


@pytest.mark.parametrize("train_fn", [train_rf, train_xgboost, train_lightgbm, train_catboost])
def test_tree_models_learn_separable_data(data, train_fn):
    X_train, X_test, y_train, y_test, names = preprocess(data)
    preds = train_fn(X_train, y_train).predict(X_test)
    assert preds.shape == y_test.shape
    assert evaluate(y_test, preds, names, show=False)["accuracy"] > 0.9


def test_xgboost_handles_missing_classes(data):
    X_train, X_test, y_train, y_test, names = preprocess(data)
    mask = y_train != 8  # drop a class so labels are no longer contiguous
    preds = train_xgboost(X_train[mask], y_train[mask]).predict(X_test)
    assert 8 not in preds
    assert set(preds) <= set(y_train)


def test_gnn_trains_and_predicts(data):
    X_train, X_test, y_train, y_test, names = preprocess(data)
    model = GNNClassifier(epochs=30).fit(X_train, y_train)
    assert model.losses[-1] < model.losses[0]
    assert model.predict(X_test).shape == y_test.shape


def test_macro_average_ignores_classes_absent_from_test():
    names = np.array(["a", "b", "c"])
    summary = evaluate([0, 0, 1, 1], [0, 0, 1, 1], names, show=False)
    assert summary["macro_f1"] == 1.0
    assert summary["classes_scored"] == 2
