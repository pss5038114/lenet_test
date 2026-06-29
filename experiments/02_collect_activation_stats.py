# experiments/02_collect_activation_stats.py

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.dataset import make_loaders
from common.model import LeNet5
from common.train_utils import set_seed, get_device


# ------------------------------------------------------------
# Basic tensor statistics
# ------------------------------------------------------------

def calc_basic_stats(values: np.ndarray, name: str, kind: str):
    """
    values:
        FP32 tensor values flattened to 1D numpy array.

    kind:
        activation / weight / bias 등 구분용.
    """
    values = values.astype(np.float64).reshape(-1)
    abs_values = np.abs(values)

    absmax = float(np.max(abs_values)) if values.size > 0 else 0.0
    p99 = float(np.percentile(abs_values, 99)) if values.size > 0 else 0.0
    p99_9 = float(np.percentile(abs_values, 99.9)) if values.size > 0 else 0.0
    p99_99 = float(np.percentile(abs_values, 99.99)) if values.size > 0 else 0.0

    outlier_ratio = absmax / (p99_9 + 1e-12)

    return {
        "name": name,
        "kind": kind,
        "num_values": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "absmax": absmax,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p99_abs": p99,
        "p99_9_abs": p99_9,
        "p99_99_abs": p99_99,
        "absmax_over_p99_9": outlier_ratio,
    }


def calc_absmax_quant_error(values: np.ndarray):
    """
    참고용 quantization error 계산.

    여기서는 tensor마다 absmax 기준 symmetric INT8 scale을 잡았을 때의
    saturation / MSE / SQNR을 계산한다.

    실제 하드웨어 scale sweep과는 별도다.
    이 값은 '이 tensor가 INT8로 표현하기 쉬운지'를 보는 참고 지표다.
    """
    values = values.astype(np.float64).reshape(-1)
    absmax = float(np.max(np.abs(values)))

    if absmax < 1e-12:
        return {
            "absmax_int8_scale": 1.0,
            "sat_rate_absmax_scale": 0.0,
            "quant_mse_absmax_scale": 0.0,
            "quant_sqnr_db_absmax_scale": 999.0,
        }

    scale = 127.0 / absmax

    q_float = np.round(values * scale)
    sat_mask = (q_float < -128) | (q_float > 127)
    q = np.clip(q_float, -128, 127)
    dq = q / scale

    err = values - dq
    mse = float(np.mean(err ** 2))
    signal_power = float(np.mean(values ** 2))
    sqnr = 10.0 * np.log10((signal_power + 1e-12) / (mse + 1e-12))

    return {
        "absmax_int8_scale": float(scale),
        "sat_rate_absmax_scale": float(np.mean(sat_mask)),
        "quant_mse_absmax_scale": mse,
        "quant_sqnr_db_absmax_scale": float(sqnr),
    }


def make_stat_row(values: np.ndarray, name: str, kind: str):
    row = calc_basic_stats(values, name=name, kind=kind)
    row.update(calc_absmax_quant_error(values))
    return row


# ------------------------------------------------------------
# Activation collection
# ------------------------------------------------------------

@torch.no_grad()
def forward_with_activations(model: LeNet5, x: torch.Tensor):
    """
    PyTorch FP32 모델을 통과시키면서 layer별 activation을 저장한다.

    현재 LeNet5 구조:
      input
      conv1 -> relu -> pool1
      conv2 -> relu -> pool2
      flatten
      fc1 -> relu
      fc2 -> relu
      fc3 logits

    이 구조는 experiments/common/model.py의 LeNet5와 맞춘다.
    """

    acts = {}

    acts["input"] = x.detach().cpu()

    z = model.conv1(x)
    acts["conv1_pre_relu"] = z.detach().cpu()

    z = F.relu(z)
    acts["conv1_post_relu"] = z.detach().cpu()

    z = model.pool1(z)
    acts["pool1"] = z.detach().cpu()

    z = model.conv2(z)
    acts["conv2_pre_relu"] = z.detach().cpu()

    z = F.relu(z)
    acts["conv2_post_relu"] = z.detach().cpu()

    z = model.pool2(z)
    acts["pool2"] = z.detach().cpu()

    z = z.view(z.size(0), 400)
    acts["flatten"] = z.detach().cpu()

    z = model.fc1(z)
    acts["fc1_pre_relu"] = z.detach().cpu()

    z = F.relu(z)
    acts["fc1_post_relu"] = z.detach().cpu()

    z = model.fc2(z)
    acts["fc2_pre_relu"] = z.detach().cpu()

    z = F.relu(z)
    acts["fc2_post_relu"] = z.detach().cpu()

    z = model.fc3(z)
    acts["fc3_logits"] = z.detach().cpu()

    return z, acts


@torch.no_grad()
def collect_activation_arrays(model, loader, device, max_batches=None):
    """
    loader 전체를 통과시키면서 activation을 layer별로 모은다.
    calibration set 1024장 정도는 메모리에 모아도 충분히 작다.
    """
    model.eval()

    buckets = {}

    total_images = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device)

        logits, acts = forward_with_activations(model, inputs)

        batch_size = inputs.size(0)
        total_images += batch_size

        for name, tensor in acts.items():
            arr = tensor.numpy().astype(np.float32).reshape(-1)
            buckets.setdefault(name, []).append(arr)

    merged = {}
    for name, parts in buckets.items():
        merged[name] = np.concatenate(parts, axis=0)

    return merged, total_images


# ------------------------------------------------------------
# Weight / bias collection
# ------------------------------------------------------------

def collect_parameter_stats(model: LeNet5):
    rows = []

    param_map = {
        "conv1_weight": model.conv1.weight.detach().cpu().numpy(),
        "conv1_bias": model.conv1.bias.detach().cpu().numpy(),
        "conv2_weight": model.conv2.weight.detach().cpu().numpy(),
        "conv2_bias": model.conv2.bias.detach().cpu().numpy(),
        "fc1_weight": model.fc1.weight.detach().cpu().numpy(),
        "fc1_bias": model.fc1.bias.detach().cpu().numpy(),
        "fc2_weight": model.fc2.weight.detach().cpu().numpy(),
        "fc2_bias": model.fc2.bias.detach().cpu().numpy(),
        "fc3_weight": model.fc3.weight.detach().cpu().numpy(),
        "fc3_bias": model.fc3.bias.detach().cpu().numpy(),
    }

    for name, values in param_map.items():
        kind = "bias" if name.endswith("_bias") else "weight"
        rows.append(make_stat_row(values, name=name, kind=kind))

    return rows


# ------------------------------------------------------------
# CSV save
# ------------------------------------------------------------

def save_rows_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save")

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    dataset_name = "MNIST"
    seed = 42
    calib_size = 1024
    split_to_analyze = "calibration"  # calibration / test 중 선택 가능

    set_seed(seed)
    device = get_device()

    out_dir = Path("outputs") / dataset_name.lower() / f"seed{seed}_calib{calib_size}"
    ckpt_path = out_dir / "lenet5_fp32.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"먼저 python experiments/01_train_fp32.py 를 실행해야 합니다."
        )

    print(f"device: {device}")
    print(f"checkpoint: {ckpt_path}")

    train_loader, calib_loader, test_loader, split_info = make_loaders(
        dataset_name=dataset_name,
        root="./data",
        calib_size=calib_size,
        seed=seed,
        train_batch_size=64,
        eval_batch_size=256,
        num_workers=0,
    )

    if split_to_analyze == "calibration":
        target_loader = calib_loader
    elif split_to_analyze == "test":
        target_loader = test_loader
    else:
        raise ValueError(f"Unknown split_to_analyze: {split_to_analyze}")

    model = LeNet5().to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print()
    print(f"Collecting activation stats from: {split_to_analyze}")

    activation_arrays, total_images = collect_activation_arrays(
        model=model,
        loader=target_loader,
        device=device,
    )

    activation_rows = []
    for name, values in activation_arrays.items():
        row = make_stat_row(values, name=name, kind="activation")
        row["split"] = split_to_analyze
        row["total_images"] = total_images
        activation_rows.append(row)

    parameter_rows = collect_parameter_stats(model)
    for row in parameter_rows:
        row["split"] = "model_parameter"
        row["total_images"] = 0

    all_rows = activation_rows + parameter_rows

    stats_csv_path = out_dir / f"activation_weight_stats_{split_to_analyze}.csv"
    save_rows_csv(all_rows, stats_csv_path)

    print()
    print(f"Saved stats CSV: {stats_csv_path}")
    print(f"Analyzed images: {total_images}")
    print()

    print("Top activation outlier ratios:")
    act_rows_sorted = sorted(
        activation_rows,
        key=lambda r: r["absmax_over_p99_9"],
        reverse=True,
    )

    for row in act_rows_sorted:
        print(
            f"{row['name']:18s} "
            f"absmax={row['absmax']:.6f} "
            f"p99.9={row['p99_9_abs']:.6f} "
            f"ratio={row['absmax_over_p99_9']:.3f} "
            f"sqnr={row['quant_sqnr_db_absmax_scale']:.2f} dB"
        )


if __name__ == "__main__":
    main()