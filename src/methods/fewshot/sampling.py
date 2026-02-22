import torch
import numpy as np
import torch


def sample_episode_from_dataset(
    dataset,
    n_way,
    k_shot,
    n_query,
    device,
):
    """
    Samples a few-shot episode from a dataset using class level sampling.
    Returns support and query sets for episodic training or evaluation.
    """
    

    label_to_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        label = int(label)
        label_to_indices.setdefault(label, []).append(i)

    available_classes = list(label_to_indices.keys())

    valid_classes = [
        c for c in available_classes
        if len(label_to_indices[c]) >= k_shot + n_query
    ]

    if len(valid_classes) < n_way:
        raise ValueError(
            f"Not enough valid classes ({len(valid_classes)}) "
            f"for n_way={n_way}, k_shot={k_shot}, n_query={n_query}"
        )
    selected_classes = np.random.choice(
        valid_classes, n_way, replace=False
    )

    Xs, ys, Xq, yq = [], [], [], []

    for c in selected_classes:
        indices = label_to_indices[c]
        perm = np.random.permutation(indices)

        support_idx = perm[:k_shot]
        query_idx   = perm[k_shot:k_shot + n_query]

        for i in support_idx:
            x, _ = dataset[i]
            Xs.append(x)
            ys.append(c)

        for i in query_idx:
            x, _ = dataset[i]
            Xq.append(x)
            yq.append(c)

    return (
        torch.stack(Xs).to(device),
        torch.tensor(ys, dtype=torch.long).to(device),
        torch.stack(Xq).to(device),
        torch.tensor(yq, dtype=torch.long).to(device),
    )


def sample_task(
    X, y,
    n_way=5,
    k_shot=1,
    n_query=20
):
    """
    Samples a few-shot task directly from tensors of features and labels.
    Returns support and query tensors.
    """
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




