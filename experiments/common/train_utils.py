# experiments/common/train_utils.py

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    # 10x10 confusion matrix
    confusion = torch.zeros(10, 10, dtype=torch.long)

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        for t, p in zip(labels.cpu(), preds.cpu()):
            confusion[int(t), int(p)] += 1

    avg_loss = total_loss / total
    acc = 100.0 * correct / total

    per_class_acc = {}
    for cls in range(10):
        cls_total = confusion[cls].sum().item()
        cls_correct = confusion[cls, cls].item()
        per_class_acc[cls] = 100.0 * cls_correct / cls_total if cls_total > 0 else 0.0

    return {
        "loss": avg_loss,
        "acc": acc,
        "correct": correct,
        "total": total,
        "confusion": confusion,
        "per_class_acc": per_class_acc,
    }