from sklearn.ensemble import IsolationForest

def train_isolation_forest(X_train):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    return model

def detect_anomalies(model, X):
    preds = model.predict(X)
    return [1 if p == -1 else 0 for p in preds]