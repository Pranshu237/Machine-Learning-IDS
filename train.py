from preprocessing.preprocess import preprocess
from models.rf_model import train_rf
from models.boosting_models import train_xgboost, train_lightgbm, train_catboost
from models.gnn_model import GNN, build_graph
from utils.metrics import evaluate

import torch


# ---------------- GNN ----------------
def train_gnn(X_train, y_train, X_test, y_test):
    print("\n--- Starting GNN training ---")

    x_train, edge_index = build_graph(X_train)

    num_classes = int(max(y_train)) + 1
    model = GNN(input_dim=x_train.shape[1], num_classes=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    y_train_tensor = torch.tensor(y_train.astype(int), dtype=torch.long)

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()

        out = model(x_train, edge_index)
        loss = loss_fn(out, y_train_tensor)

        loss.backward()
        optimizer.step()

        print(f"GNN Epoch {epoch+1}, Loss: {loss.item()}")

    x_test, edge_test = build_graph(X_test)

    model.eval()
    preds = model(x_test, edge_test).argmax(dim=1)

    print("\n--- GNN Results ---")
    evaluate(y_test, preds.numpy())


# ---------------- MAIN ----------------
def main():
    X_train, X_test, y_train, y_test, target_names = preprocess()

    # -------- Random Forest --------
    print("\n--- Training Random Forest ---")
    rf_model = train_rf(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("\n--- RF Results ---")
    evaluate(y_test, rf_preds, target_names)

    # -------- XGBoost --------
    print("\n--- Training XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    print("\n--- XGBoost Results ---")
    evaluate(y_test, xgb_preds, target_names)

    # -------- LightGBM --------
    print("\n--- Training LightGBM ---")
    lgb_model = train_lightgbm(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)

    print("\n--- LightGBM Results ---")
    evaluate(y_test, lgb_preds, target_names)

    # -------- CatBoost --------
    print("\n--- Training CatBoost ---")
    cat_model = train_catboost(X_train, y_train)
    cat_preds = cat_model.predict(X_test)

    print("\n--- CatBoost Results ---")
    evaluate(y_test, cat_preds, target_names)

    # -------- GNN (small data) --------
    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]
    X_test_small = X_test[:2000]
    y_test_small = y_test[:2000]

    train_gnn(X_train_small, y_train_small, X_test_small, y_test_small)


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()