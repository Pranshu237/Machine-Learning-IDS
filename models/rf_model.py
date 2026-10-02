from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE


def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
