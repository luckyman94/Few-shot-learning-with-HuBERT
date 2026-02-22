import torch
import random
from collections import defaultdict
import random
import numpy as np


def sample_episode_from_dataset(
    dataset,
    n_way,
    k_shot,
    n_query,
    device,
):
    """
    Sample one few-shot episode from a PyTorch Dataset.
    Dataset must return (waveform, label).
    """

    # collect indices per class
    label_to_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_to_indices.setdefault(label, []).append(idx)

    classes = list(label_to_indices.keys())
    selected_classes = np.random.choice(classes, n_way, replace=False)

    Xs, ys = [], []
    Xq, yq = [], []

    for c in selected_classes:
        indices = label_to_indices[c]
        assert len(indices) >= k_shot + n_query

        perm = np.random.permutation(indices)
        support_idx = perm[:k_shot]
        query_idx = perm[k_shot : k_shot + n_query]

        for i in support_idx:
            x, _ = dataset[i]
            Xs.append(x)
            ys.append(c)

        for i in query_idx:
            x, _ = dataset[i]
            Xq.append(x)
            yq.append(c)

    Xs = torch.stack(Xs).to(device)
    ys = torch.tensor(ys).to(device)
    Xq = torch.stack(Xq).to(device)
    yq = torch.tensor(yq).to(device)

    return Xs, ys, Xq, yq


def sample_task(
    X, y,
    n_way=5,
    k_shot=1,
    n_query=20
):
    classes = torch.unique(y)
    selected = classes[torch.randperm(len(classes))[:n_way]]

    X_support, y_support = [], []
    X_query, y_query = [], []

    for c in selected:
        idx = torch.where(y == c)[0]
        perm = idx[torch.randperm(len(idx))]

        X_support.append(X[perm[:k_shot]])
        y_support.append(y[perm[:k_shot]])

        X_query.append(X[perm[k_shot:k_shot+n_query]])
        y_query.append(y[perm[k_shot:k_shot+n_query]])

    return (
        torch.cat(X_support),
        torch.cat(y_support),
        torch.cat(X_query),
        torch.cat(y_query),
    )


def sample_k_per_class(
    X: torch.Tensor,
    y: torch.Tensor,
    n_way=5,
    k: int = 1,
):

    classes = torch.unique(y)
    selected = classes[torch.randperm(len(classes))[:n_way]]

    X_support, y_support = [], []
    X_query, y_query = [], []

    for c in classes:
        idx = torch.where(y == c)[0]
        perm = idx[torch.randperm(len(idx))]

        support_idx = perm[:k]
        query_idx = perm[k:]

        X_support.append(X[support_idx])
        y_support.append(y[support_idx])

        X_query.append(X[query_idx])
        y_query.append(y[query_idx])

    return (
        torch.cat(X_support, dim=0),
        torch.cat(y_support, dim=0),
        torch.cat(X_query, dim=0),
        torch.cat(y_query, dim=0),
    )



def sample_n_way_task(X, y, n_way=5, k_shot=1, n_query=20):
   

    classes = torch.unique(y).tolist()
    selected_classes = random.sample(classes, n_way)

    X_support, y_support = [], []
    X_query, y_query = [], []

    for i, c in enumerate(selected_classes):
        idx = torch.where(y == c)[0]
        perm = idx[torch.randperm(len(idx))]

        support_idx = perm[:k_shot]
        query_idx   = perm[k_shot:k_shot + n_query]

        X_support.append(X[support_idx])
        y_support.append(torch.full((k_shot,), i))

        X_query.append(X[query_idx])
        y_query.append(torch.full((len(query_idx),), i))

    return (
        torch.cat(X_support),
        torch.cat(y_support),
        torch.cat(X_query),
        torch.cat(y_query),
    )



def sample_fewshot_task(
    X: torch.Tensor,
    y: torch.Tensor,
    n_way: int,
    k_shot: int,
    n_query: int,
):
    classes = torch.unique(y)

    if n_way > len(classes):
        raise ValueError("n_way larger than number of classes")

    selected_classes = classes[torch.randperm(len(classes))[:n_way]]

    X_support, y_support = [], []
    X_query, y_query = [], []

    for c in selected_classes:
        idx = torch.where(y == c)[0]

        if len(idx) < k_shot + n_query:
            raise ValueError(
                f"Class {c.item()} has only {len(idx)} samples"
            )

        perm = idx[torch.randperm(len(idx))]

        X_support.append(X[perm[:k_shot]])
        y_support.append(y[perm[:k_shot]])

        X_query.append(X[perm[k_shot:k_shot + n_query]])
        y_query.append(y[perm[k_shot:k_shot + n_query]])

    return (
        torch.cat(X_support),
        torch.cat(y_support),
        torch.cat(X_query),
        torch.cat(y_query),
    )

