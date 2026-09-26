from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(y_true, y_pred, target_names=None):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    
    if target_names is not None:
        target_names = [str(name) for name in target_names]
        labels = list(range(len(target_names)))
        print("\nClassification Report:\n", classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        ))
    else:
        print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))