from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np

def train_xgboost(X_train, y_train):
    # Ensure XGBoost target labels are contiguous if subsetted
    unique_y = np.unique(y_train)
    if not np.array_equal(unique_y, np.arange(len(unique_y))):
        label_mapping = {old: new for new, old in enumerate(unique_y)}
        y_train_mapped = np.array([label_mapping[val] for val in y_train])
        model = XGBClassifier(eval_metric='mlogloss', n_jobs=-1)
        model.fit(X_train, y_train_mapped)
        model._label_mapping = label_mapping
        model._reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        
        # Override predict to map back to original label indices
        original_predict = model.predict
        def predict_mapped(X):
            preds = original_predict(X)
            return np.array([model._reverse_label_mapping.get(p, p) for p in preds])
        model.predict = predict_mapped
        return model
    else:
        model = XGBClassifier(eval_metric='mlogloss', n_jobs=-1)
        model.fit(X_train, y_train)
        return model

def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train):
    model = CatBoostClassifier(iterations=100, verbose=0, thread_count=-1)
    model.fit(X_train, y_train)
    return model