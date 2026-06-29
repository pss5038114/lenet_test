# experiments/01_train_fp32.py

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from common.dataset import make_loaders, save_split_info
from common.model import LeNet5
from common.train_utils import set_seed, get_device, evaluate


def main():
    dataset_name = "MNIST"
    seed = 42
    calib_size = 1024
    num_epochs = 10
    lr = 1e-3

    set_seed(seed)
    device = get_device()

    print(f"device: {device}")

    train_loader, calib_loader, test_loader, split_info = make_loaders(
        dataset_name=dataset_name,
        root="./data",
        calib_size=calib_size,
        seed=seed,
        train_batch_size=64,
        eval_batch_size=256,
        num_workers=0,
    )

    out_dir = Path("outputs") / dataset_name.lower() / f"seed{seed}_calib{calib_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_split_info(split_info, out_dir / "split_info.json")

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log_rows = []

    print("FP32 training start")

    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        test_metrics = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
        }
        log_rows.append(row)

        print(
            f"[Epoch {epoch:02d}/{num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.2f}%"
        )

    # 최종 평가
    test_metrics = evaluate(model, test_loader, criterion, device)
    calib_metrics = evaluate(model, calib_loader, criterion, device)

    print()
    print("Final FP32 result")
    print(f"calib_acc: {calib_metrics['acc']:.2f}%")
    print(f"test_acc : {test_metrics['acc']:.2f}%")

    # 모델 저장
    ckpt_path = out_dir / "lenet5_fp32.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": dataset_name,
            "seed": seed,
            "calib_size": calib_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "test_acc": test_metrics["acc"],
            "test_loss": test_metrics["loss"],
        },
        ckpt_path,
    )

    print(f"Saved model: {ckpt_path}")

    # 로그 저장
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Saved train log: {log_path}")

    # confusion matrix 저장
    confusion_path = out_dir / "test_confusion.pt"
    torch.save(test_metrics["confusion"], confusion_path)
    print(f"Saved confusion matrix: {confusion_path}")


if __name__ == "__main__":
    main()