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
    max_tsne_points=2000,
    tsne_title="t-SNE of query embeddings",
):
    
    """Evaluates classification metrics and plots confusion matrix and t-SNE visualization."""


    y_true = results["all_targets"].numpy()
    y_pred = results["all_preds"].numpy()
    embeddings = results["all_embeddings"].numpy()

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
        if len(embeddings) > max_tsne_points:
            idx = np.random.choice(len(embeddings), max_tsne_points, replace=False)
            embeddings = embeddings[idx]
            y_true = y_true[idx]

        plot_tsne(
            embeddings,
            labels=y_true,
            class_names=class_names,
            perplexity=min(tsne_perplexity, len(embeddings) - 1),
            title=tsne_title,
        )

    return metrics



def evaluate_benchmark(
    results_dict,
    class_names=None,
    plot_confusion=True,
    plot_tsne=True
):
    """
    Evaluate and plot few-shot benchmark results for different k-shot values.
    """

    print("\n===== Few-shot evaluation summary =====")

    for k in sorted(results_dict.keys()):
        res = results_dict[k]

        print(f"\n--- {k}-shot ---")
        print(f"Accuracy : {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}")
        print(f"F1 macro : {res['f1_macro']:.4f}")

        if plot_confusion:
            plot_confusion_matrix(
                res["all_targets"].numpy(),
                res["all_preds"].numpy(),
                class_names=class_names,
                title=f"Confusion matrix for {k}-shot"
            )

        if plot_tsne:
            evaluate_and_plot(
                res,
                class_names=class_names,
                plot_confusion=False,
                plot_tsne_flag=True,
                tsne_title=f"t-SNE of query embeddings ({k}-shot)",
            )

