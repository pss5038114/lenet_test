# experiments/04b_fine_sweep_global_scale.py

import csv
import json
from pathlib import Path

import numpy as np
import torch

from common.dataset import make_loaders
from common.model import LeNet5
from common.train_utils import set_seed, get_device
from common.quant_numpy import (
    quantize_lenet5_params_global_scale_flow,
    lenet5_forward_int8_numpy,
)


@torch.no_grad()
def collect_eval_arrays(model, loader, device):
    model.eval()

    xs = []
    labels = []
    fp32_preds = []

    for inputs, y in loader:
        inputs = inputs.to(device)

        logits = model(inputs)
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.int64)

        xs.append(inputs.detach().cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int64))
        fp32_preds.append(preds)

    x_np = np.concatenate(xs, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    fp32_preds_np = np.concatenate(fp32_preds, axis=0)

    fp32_acc = 100.0 * np.mean(fp32_preds_np == labels_np)

    return x_np, labels_np, fp32_preds_np, fp32_acc


def update_debug_aggregate(debug_global, debug, batch_size):
    for key, value in debug.items():
        if isinstance(value, bool):
            value = bool(value)

        if key.endswith("_min"):
            debug_global[key] = int(value) if key not in debug_global else min(debug_global[key], int(value))

        elif key.endswith("_max"):
            debug_global[key] = int(value) if key not in debug_global else max(debug_global[key], int(value))

        elif key.endswith("_int32_ok"):
            debug_global[key] = bool(value) if key not in debug_global else bool(debug_global[key]) and bool(value)

        else:
            sum_key = key + "_weighted_sum"
            debug_global[sum_key] = debug_global.get(sum_key, 0.0) + float(value) * batch_size


def finalize_debug_aggregate(debug_global, total):
    final = {}

    for key, value in debug_global.items():
        if key.endswith("_weighted_sum"):
            clean_key = key.replace("_weighted_sum", "")
            final[clean_key] = float(value) / max(total, 1)
        else:
            final[key] = value

    return final


def evaluate_one_scale(
    x_np,
    labels_np,
    fp32_preds_np,
    model,
    x_scale,
    w_scale,
    shift_val,
    bias_mode,
    batch_size=256,
):
    qparams = quantize_lenet5_params_global_scale_flow(
        model=model,
        x_scale=x_scale,
        w_scale=w_scale,
        shift_val=shift_val,
        bias_mode=bias_mode,
    )

    total = labels_np.shape[0]
    int8_preds_all = []
    debug_global = {}

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        batch_x = x_np[start:end]
        batch_size_actual = end - start

        scores_int8, debug = lenet5_forward_int8_numpy(
            batch_x,
            qparams=qparams,
            x_scale=x_scale,
            shift_val=shift_val,
            return_debug=True,
        )

        preds = scores_int8.astype(np.int16).argmax(axis=1).astype(np.int64)
        int8_preds_all.append(preds)

        update_debug_aggregate(debug_global, debug, batch_size_actual)

    int8_preds_np = np.concatenate(int8_preds_all, axis=0)

    fp32_acc = 100.0 * np.mean(fp32_preds_np == labels_np)
    int8_acc = 100.0 * np.mean(int8_preds_np == labels_np)
    mismatch_count = int(np.sum(int8_preds_np != fp32_preds_np))
    mismatch_rate = 100.0 * mismatch_count / total

    debug_final = finalize_debug_aggregate(debug_global, total)

    weight_info = qparams.get("_quant_info", {})
    scale_flow = qparams.get("_scale_flow", {})

    row = {
        "x_scale": float(x_scale),
        "w_scale": float(w_scale),
        "bias_mode": bias_mode,
        "shift_val": int(shift_val),

        "bias_scale_conv1": float(scale_flow["conv1_bias"]),
        "bias_scale_conv2": float(scale_flow["conv2_bias"]),
        "bias_scale_fc1": float(scale_flow["fc1_bias"]),
        "bias_scale_fc2": float(scale_flow["fc2_bias"]),
        "bias_scale_fc3": float(scale_flow["fc3_bias"]),

        "fp32_acc": float(fp32_acc),
        "int8_acc": float(int8_acc),
        "acc_drop": float(fp32_acc - int8_acc),
        "mismatch_count": mismatch_count,
        "mismatch_rate_percent": float(mismatch_rate),
        "total": int(total),
    }

    row.update(weight_info)
    row.update(debug_final)

    int32_ok_keys = [k for k in row.keys() if k.endswith("_int32_ok")]
    row["all_int32_ok"] = all(bool(row[k]) for k in int32_ok_keys)

    return row


def save_rows_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    dataset_name = "MNIST"
    seed = 42
    calib_size = 1024

    shift_val = 4
    bias_mode = "round"

    # Coarse sweep 결과상 좋은 영역:
    # X_SCALE 48~80 부근, W_SCALE 8~16 부근
    #
    # 1차 fine sweep:
    #   X는 2 간격
    #   W는 1 간격
    x_scales = list(range(48, 81, 2))
    w_scales = list(range(8, 17, 1))

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
    print(f"fine x_scales: {x_scales}")
    print(f"fine w_scales: {w_scales}")

    train_loader, calib_loader, test_loader, split_info = make_loaders(
        dataset_name=dataset_name,
        root="./data",
        calib_size=calib_size,
        seed=seed,
        train_batch_size=64,
        eval_batch_size=256,
        num_workers=0,
    )

    model = LeNet5().to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print()
    print("Collecting test arrays and FP32 predictions...")
    x_np, labels_np, fp32_preds_np, fp32_acc = collect_eval_arrays(
        model=model,
        loader=test_loader,
        device=device,
    )

    print(f"test samples: {labels_np.shape[0]}")
    print(f"fp32_acc    : {fp32_acc:.2f}%")

    rows = []
    exp_id = 0

    print()
    print("Starting fine global scale sweep...")

    for x_scale in x_scales:
        for w_scale in w_scales:
            exp_id += 1

            row = evaluate_one_scale(
                x_np=x_np,
                labels_np=labels_np,
                fp32_preds_np=fp32_preds_np,
                model=model,
                x_scale=float(x_scale),
                w_scale=float(w_scale),
                shift_val=shift_val,
                bias_mode=bias_mode,
                batch_size=256,
            )

            row["experiment_id"] = exp_id
            row["dataset"] = dataset_name
            row["seed"] = seed
            row["calib_size"] = calib_size
            row["scale_granularity"] = "global_fine"
            row["rtl_compatible"] = True
            row["requires_rtl_change"] = False
            row["pynq_verified"] = False

            rows.append(row)

            print(
                f"[{exp_id:03d}] "
                f"X={x_scale:>3} W={w_scale:>2} "
                f"INT8 acc={row['int8_acc']:.2f}% "
                f"drop={row['acc_drop']:.2f}%p "
                f"mismatch={row['mismatch_rate_percent']:.2f}% "
                f"fc3_sat={row.get('fc3_score_sat_rate', 0.0):.3f}"
            )

    sweep_dir = out_dir / "scale_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sweep_dir / f"global_fine_sweep_shift{shift_val}_{bias_mode}.csv"
    save_rows_csv(rows, csv_path)

    best = max(rows, key=lambda r: r["int8_acc"])
    best_path = sweep_dir / f"best_global_fine_sweep_shift{shift_val}_{bias_mode}.json"

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print()
    print("Fine sweep complete")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved best: {best_path}")
    print()
    print("Best fine config")
    print(f"  X_SCALE : {best['x_scale']}")
    print(f"  W_SCALE : {best['w_scale']}")
    print(f"  INT8 acc: {best['int8_acc']:.2f}%")
    print(f"  drop    : {best['acc_drop']:.2f}%p")
    print(f"  mismatch: {best['mismatch_rate_percent']:.2f}%")
    print(f"  fc3_sat : {best.get('fc3_score_sat_rate', 0.0):.3f}")


if __name__ == "__main__":
    main()