import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
from src.methods.fewshot.sampling import sample_episode_from_dataset


def prototypical_train(
    dataset,
    hubert,
    head,
    optimizer,
    device,
    n_way=5,
    k_shot=1,
    n_query=20,
    n_episodes=1000,
):
    """
    Exact episodic Prototypical Networks training.
    Returns training metrics.
    """

    hubert.eval()        # frozen encoder
    head.train()

    losses = []
    accs = []

    for _ in trange(n_episodes, desc="Prototypical training"):

        # --------------------------------------------------
        # 1) Sample episode
        # --------------------------------------------------
        Xs, ys, Xq, yq = sample_episode_from_dataset(
            dataset,
            n_way,
            k_shot,
            n_query,
            device,
        )

        # --------------------------------------------------
        # 2) Forward (support)
        # --------------------------------------------------
        out_s = hubert(Xs)
        z_s = out_s.last_hidden_state.mean(dim=1)
        z_s = head(z_s)

        # --------------------------------------------------
        # 3) Forward (query)
        # --------------------------------------------------
        out_q = hubert(Xq)
        z_q = out_q.last_hidden_state.mean(dim=1)
        z_q = head(z_q)

        # --------------------------------------------------
        # 4) Prototypes
        # --------------------------------------------------
        classes = torch.unique(ys)

        prototypes = torch.stack([
            z_s[ys == c].mean(dim=0)
            for c in classes
        ])

        # --------------------------------------------------
        # 5) Distances → logits
        # --------------------------------------------------
        dists = torch.cdist(z_q, prototypes)
        logits = -dists

        # --------------------------------------------------
        # 6) Relabel query targets (0 … n_way-1)
        # --------------------------------------------------
        yq_episode = torch.zeros_like(yq)
        for i, c in enumerate(classes):
            yq_episode[yq == c] = i

        # --------------------------------------------------
        # 7) Loss + backward
        # --------------------------------------------------
        loss = F.cross_entropy(logits, yq_episode)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --------------------------------------------------
        # 8) Metrics
        # --------------------------------------------------
        preds = logits.argmax(dim=1)
        acc = (preds == yq_episode).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

    return {
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
    }