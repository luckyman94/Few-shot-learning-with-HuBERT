import torch
import numpy as np
from tqdm import trange
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from src.methods.fewshot.sampling import sample_task, sample_episode_from_dataset
from src.methods.fewshot.prototypical import compute_prototypes, classify


def benchmark_fewshot(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    device,
    n_tasks: int = 100,
    k_shot: int = 1,
    n_query: int = 20,
):
    

    accs = []
    all_preds = []
    all_targets = []
    all_embeddings = []


    classes = torch.unique(y)

    for t in trange(n_tasks, desc=f"{k_shot}-shot benchmark"):
        Xs, ys, Xq, yq = sample_task(
            X, y, k_shot=k_shot, n_query=n_query
        )

        prototypes, proto_labels = compute_prototypes(
            model, Xs, ys, device=device
        )

        preds = classify(
            model, Xq, prototypes, proto_labels, device=device
        )

        with torch.no_grad():
            out = model(Xq.to(device))
            h = out.last_hidden_state
            z = h.mean(dim=1)

        all_embeddings.append(z.cpu())


        acc = (preds.cpu() == yq.cpu()).float().mean().item()
        accs.append(acc)

        all_preds.append(preds.cpu())
        all_targets.append(yq.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    acc_mean = np.mean(accs)
    acc_std = np.std(accs)

    f1 = f1_score(
        all_targets.numpy(),
        all_preds.numpy(),
        average="macro"
    )

    cm = confusion_matrix(
        all_targets.numpy(),
        all_preds.numpy(),
        labels=classes.numpy()
    )

    all_embeddings = torch.cat(all_embeddings)

    return {
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "all_embeddings": all_embeddings, 
        "classes": classes,
    }




@torch.no_grad()
def benchmark_fewshot_training(
    dataset,
    head,
    device,
    n_tasks: int = 100,
    n_way: int = None,
    k_shot: int = 1,
    n_query: int = 20,
):
    

    head.eval()

    accs = []
    all_preds = []
    all_targets = []

    for _ in trange(n_tasks, desc=f"{k_shot}-shot benchmark"):

        Xs, ys, Xq, yq = sample_episode_from_dataset(
            dataset,
            n_way=n_way or len(dataset.classes),
            k_shot=k_shot,
            n_query=n_query,
            device=device,
        )

        # ---- embeddings already cached
        z_s = head(Xs.to(device))
        z_q = head(Xq.to(device))

        # ---- prototypes
        classes = torch.unique(ys)
        prototypes = torch.stack([
            z_s[ys == c].mean(dim=0) for c in classes
        ])

        # ---- distances
        dists = torch.cdist(z_q, prototypes)
        preds_idx = dists.argmin(dim=1)

        # ---- relabel targets (episode-wise)
        yq_ep = torch.zeros_like(yq)
        for i, c in enumerate(classes):
            yq_ep[yq == c] = i

        preds = preds_idx.cpu()
        targets = yq_ep.cpu()

        accs.append((preds == targets).float().mean().item())
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro": f1_score(
            all_targets.numpy(),
            all_preds.numpy(),
            average="macro",
        ),
    }