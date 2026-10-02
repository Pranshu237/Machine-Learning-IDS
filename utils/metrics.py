import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(y_true, y_pred, target_names, labels=None, show=True):
    """Score predictions and return a summary dict.

    `labels` fixes which classes are averaged over, so every model is
    scored on exactly the same set of classes. By default that is the
    classes present in y_true (a class with no test examples cannot be
    measured, so it should not count as zero in the macro average).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())
    names = [str(target_names[i]) for i in labels]

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names, zero_division=0, output_dict=True
    )
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {n: report[n] for n in names},
        "classes_scored": len(labels),
    }

    if show:
        print(f"Accuracy: {summary['accuracy']:.4f}")
        print(classification_report(
            y_true, y_pred, labels=labels, target_names=names, zero_division=0, digits=4
        ))
        print("Confusion matrix (rows = true, columns = predicted):")
        print(confusion_matrix(y_true, y_pred, labels=labels))
    return summary
