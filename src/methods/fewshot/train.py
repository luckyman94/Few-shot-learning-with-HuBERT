import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
from src.methods.fewshot.sampling import sample_episode_from_dataset


def prototypical_train_batched(
    dataset,
    hubert,
    head,
    optimizer,
    device,
    n_way,
    k_shot,
    n_query,
    n_episodes,
    episodes_per_batch=8,
):
    hubert.eval()
    head.train()

    optimizer.zero_grad()
    losses = []

    for episode in trange(n_episodes, desc="Prototypical training (batched)"):

        # -------- sample one episode
        Xs, ys, Xq, yq = sample_episode_from_dataset(
            dataset,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            device=device,
        )

        # -------- embeddings
        z_s = hubert(Xs).last_hidden_state.mean(dim=1)
        z_s = head(z_s)

        z_q = hubert(Xq).last_hidden_state.mean(dim=1)
        z_q = head(z_q)

        # -------- prototypes
        classes = torch.unique(ys)
        prototypes = torch.stack([
            z_s[ys == c].mean(dim=0) for c in classes
        ])

        # -------- distances & loss
        dists = torch.cdist(z_q, prototypes)
        logits = -dists

        yq_ep = torch.zeros_like(yq)
        for i, c in enumerate(classes):
            yq_ep[yq == c] = i

        loss = torch.nn.functional.cross_entropy(logits, yq_ep)
        loss = loss / episodes_per_batch   # 🔑 IMPORTANT
        loss.backward()

        losses.append(loss.item())

        # -------- optimizer step
        if (episode + 1) % episodes_per_batch == 0:
            optimizer.step()
            optimizer.zero_grad()

    # dernier step si besoin
    optimizer.step()
    optimizer.zero_grad()

    return {
        "train_loss_mean": float(np.mean(losses)),
        "train_loss_std": float(np.std(losses)),
    }