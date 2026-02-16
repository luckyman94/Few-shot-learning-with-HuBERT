import random
import torch
import numpy as np
from tqdm import trange



def create_episode(dataset, n_way, n_shot, n_query):
    classes = random.sample(list(dataset.keys()), n_way)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for i, cls in enumerate(classes):
        samples = random.sample(dataset[cls], n_shot + n_query)

        support_x += samples[:n_shot]
        support_y += [i] * n_shot

        query_x += samples[n_shot:]
        query_y += [i] * n_query

    support_y = torch.tensor(support_y)
    query_y = torch.tensor(query_y)

    return support_x, support_y, query_x, query_y


def evaluate_fewshot(
    dataset,
    model,
    device,
    n_way=5,
    n_shot=1,
    n_query=5,
    n_episodes=50,
    return_details=False,
):
    accs = []

    all_zq = []
    all_y_true = []
    all_y_pred = []

    for _ in trange(n_episodes, desc=f"{n_way}-way {n_shot}-shot"):
        sx, sy, qx, qy = create_episode(
            dataset,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
        )

        sy = sy.to(device)
        qy = qy.to(device)

        with torch.no_grad():
            z_s = model(sx)
            z_q = model(qx)

        protos = torch.stack([
            z_s[sy == c].mean(0)
            for c in range(n_way)
        ])

        logits = -torch.cdist(z_q, protos)
        preds = logits.argmax(dim=1)

        acc = (preds == qy).float().mean().item()
        accs.append(acc)

        if return_details:
            all_zq.append(z_q.cpu())
            all_y_true.append(qy.cpu())
            all_y_pred.append(preds.cpu())

    mean = np.mean(accs)
    std = np.std(accs)

    if not return_details:
        return mean, std

    return {
        "mean_acc": mean,
        "std_acc": std,
        "embeddings": torch.cat(all_zq).numpy(),
        "y_true": torch.cat(all_y_true).numpy(),
        "y_pred": torch.cat(all_y_pred).numpy(),
    }

