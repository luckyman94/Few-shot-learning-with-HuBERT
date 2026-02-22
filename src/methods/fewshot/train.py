import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
from src.methods.fewshot.sampling import sample_episode_from_dataset
from tqdm import tqdm
from torch.utils.data import Dataset


def prototypical_train(
    dataset,
    head,
    optimizer,
    device,
    n_way,
    k_shot,
    n_query,
    n_episodes,
    episodes_per_batch=8,
):
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
        # APRÈS (cache)
        z_s = head(Xs.to(device))
        z_q = head(Xq.to(device))

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





@torch.no_grad()
def build_embedding_cache(
    dataset,
    hubert,
    device,
    batch_size=16,
):
    hubert.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    all_embeddings = []
    all_labels = []

    for x, y in tqdm(loader, desc="Caching HuBERT embeddings"):
        x = x.to(device)

        out = hubert(x)
        z = out.last_hidden_state.mean(dim=1)

        all_embeddings.append(z.cpu())
        all_labels.append(y)

    return (
        torch.cat(all_embeddings),   # [N, D]
        torch.cat(all_labels),       # [N]
    )


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, classes):
        self.embeddings = embeddings
        self.labels = labels
        self.classes = classes  # indices de classes visibles

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]