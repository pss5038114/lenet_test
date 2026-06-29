# experiments/common/dataset.py

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


def get_32x32_transform():
    """
    MNIST/Fashion-MNIST 28x28 image를 32x32로 zero-padding.
    Resize가 아니라 Pad(2)를 사용한다.
    """
    return transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])


def get_dataset_class(dataset_name: str):
    name = dataset_name.lower()

    if name in ["mnist"]:
        return torchvision.datasets.MNIST

    if name in ["fashionmnist", "fashion-mnist", "fashion_mnist"]:
        return torchvision.datasets.FashionMNIST

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def make_splits(
    dataset_name: str = "MNIST",
    root: str = "./data",
    calib_size: int = 1024,
    seed: int = 42,
):
    """
    Original train 60,000장을 train_fit/calibration으로 분리한다.

    Example:
        calib_size=1024이면
        train_fit: 58,976
        calibration: 1,024
        test: 10,000
    """
    transform = get_32x32_transform()
    dataset_cls = get_dataset_class(dataset_name)

    full_train = dataset_cls(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = dataset_cls(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    n_train = len(full_train)
    if calib_size <= 0 or calib_size >= n_train:
        raise ValueError(f"Invalid calib_size={calib_size}, n_train={n_train}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    perm = torch.randperm(n_train, generator=generator).tolist()

    calib_indices = perm[:calib_size]
    train_indices = perm[calib_size:]

    train_fit_set = Subset(full_train, train_indices)
    calib_set = Subset(full_train, calib_indices)

    split_info = {
        "dataset": dataset_name,
        "seed": seed,
        "full_train_size": n_train,
        "train_fit_size": len(train_fit_set),
        "calibration_size": len(calib_set),
        "test_size": len(test_set),
        "calib_indices": calib_indices,
        "train_indices": train_indices,
    }

    return train_fit_set, calib_set, test_set, split_info


def make_loaders(
    dataset_name: str = "MNIST",
    root: str = "./data",
    calib_size: int = 1024,
    seed: int = 42,
    train_batch_size: int = 64,
    eval_batch_size: int = 256,
    num_workers: int = 0,
):
    train_fit_set, calib_set, test_set, split_info = make_splits(
        dataset_name=dataset_name,
        root=root,
        calib_size=calib_size,
        seed=seed,
    )

    train_loader = DataLoader(
        train_fit_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    calib_loader = DataLoader(
        calib_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, calib_loader, test_loader, split_info


def save_split_info(split_info: Dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)