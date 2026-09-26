from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model