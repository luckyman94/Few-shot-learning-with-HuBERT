from tqdm import tqdm
import torch

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct / total:.3f}"
        })

    return total_loss / total, correct / total




@torch.no_grad()
def eval_epoch(model, loader,device,split="Val"):
    model.eval()

    correct = 0
    total = 0

    pbar = tqdm(loader, desc=split, leave=False)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

        pbar.set_postfix({
            "acc": f"{correct / total:.3f}"
        })

    return correct / total



def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    n_epochs=10,
    device="cuda",
    ckpt_path="best_lora_model.pt",
    patience_ratio=0.05,
):
    
    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,device
        )

        val_acc = eval_epoch(model, val_loader,device,split="Val")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_acc:.3f} | "
            f"Val acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc - patience_ratio * best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ Saved checkpoint (val acc = {val_acc:.3f})")

    return history


