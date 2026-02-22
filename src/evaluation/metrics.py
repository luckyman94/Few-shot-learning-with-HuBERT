from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred):
    """
    Computes classification metrics given true and predicted labels.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }
