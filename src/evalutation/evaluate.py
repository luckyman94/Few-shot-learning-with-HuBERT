import numpy as np

from .metrics import classification_metrics
from .confusion import plot_confusion_matrix
from .tsne import plot_tsne


def evaluate_and_plot(
    results,
    class_names=None,
    plot_confusion=True,
    plot_tsne_flag=True,
    tsne_perplexity=30,
):

    y_true = results["y_true"]
    y_pred = results["y_pred"]
    embeddings = results["embeddings"]

    metrics = classification_metrics(y_true, y_pred)

    print("\n=== Evaluation metrics ===")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")

    if plot_confusion:
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=class_names,
        )

    if plot_tsne_flag:
        plot_tsne(
            embeddings,
            labels=y_true,
            class_names=class_names,
            perplexity=tsne_perplexity,
            title="t-SNE of query embeddings",
        )

    return metrics
