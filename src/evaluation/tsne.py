import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(
    embeddings,
    labels,
    class_names=None,
    perplexity=30,
    n_iter=1000,
    title="t-SNE projection",
    figsize=(6, 6),
):
    """
    Projects embeddings to 2D using t-SNE and visualizes class separation.
    Optionally displays class names in the legend.
    """
    

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=0,
    )

    z_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=figsize)

    for c in np.unique(labels):
        idx = labels == c
        name = class_names[c] if class_names else f"class {c}"
        plt.scatter(
            z_2d[idx, 0],
            z_2d[idx, 1],
            s=20,
            alpha=0.7,
            label=name,
        )

    plt.legend(markerscale=1.5)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.show()
