# experiments/00_check_dataset_split.py

from pathlib import Path

from common.dataset import make_loaders, save_split_info


def main():
    dataset_name = "MNIST"
    calib_size = 1024
    seed = 42

    train_loader, calib_loader, test_loader, split_info = make_loaders(
        dataset_name=dataset_name,
        root="./data",
        calib_size=calib_size,
        seed=seed,
        train_batch_size=64,
        eval_batch_size=256,
        num_workers=0,
    )

    print("Dataset split complete")
    print(f"dataset          : {split_info['dataset']}")
    print(f"seed             : {split_info['seed']}")
    print(f"full_train_size  : {split_info['full_train_size']}")
    print(f"train_fit_size   : {split_info['train_fit_size']}")
    print(f"calibration_size : {split_info['calibration_size']}")
    print(f"test_size        : {split_info['test_size']}")

    # shape 확인
    x, y = next(iter(train_loader))
    print()
    print("One train batch")
    print(f"input shape: {tuple(x.shape)}")
    print(f"label shape: {tuple(y.shape)}")
    print(f"input min/max: {float(x.min()):.4f} / {float(x.max()):.4f}")

    out_dir = Path("outputs") / dataset_name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_path = out_dir / f"split_seed{seed}_calib{calib_size}.json"
    save_split_info(split_info, split_path)

    print()
    print(f"Saved split info: {split_path}")


if __name__ == "__main__":
    main()