import torch

@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Computes model predictions and ground-truth labels over a dataloader.
    Returns concatenated true labels and predicted labels.
    """
    model.eval()

    all_preds = []
    all_true  = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_true.append(y.cpu())

    return torch.cat(all_true), torch.cat(all_preds)