# experiments/03_numpy_fixed_point_forward.py

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.dataset import make_loaders
from common.model import LeNet5
from common.train_utils import set_seed, get_device, evaluate
from common.quant_numpy import (
    quantize_lenet5_params,
    lenet5_forward_int8_numpy,
)


@torch.no_grad()
def evaluate_numpy_fixed_point(
    model,
    loader,
    qparams,
    device,
    x_scale=127.0,
    w_scale=16.0,
    shift_val=4,
    max_batches=None,
    keep_prediction_rows=200,
):
    """
    PyTorch FP32 prediction과 Numpy INT8 fixed-point prediction을 함께 계산한다.

    중요:
      여기서 비교 기준은 아직 PYNQ가 아니라,
      PyTorch FP32 vs Numpy INT8이다.

    다음 단계에서:
      Numpy INT8 score == PYNQ hardware score
    를 확인하게 된다.
    """
    model.eval()

    total = 0

    fp32_correct = 0
    int8_correct = 0
    mismatch_fp32_int8 = 0

    confusion_int8 = np.zeros((10, 10), dtype=np.int64)

    debug_sums = {}
    debug_count = 0

    prediction_rows = []

    for batch_idx, (inputs, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device)
        labels_np = labels.numpy().astype(np.int64)

        # FP32 prediction
        logits = model(inputs)
        fp32_preds = logits.argmax(dim=1).cpu().numpy().astype(np.int64)

        # Numpy INT8 fixed-point prediction
        x_np = inputs.detach().cpu().numpy().astype(np.float32)
        scores_int8, debug = lenet5_forward_int8_numpy(
            x_np,
            qparams=qparams,
            x_scale=x_scale,
            shift_val=shift_val,
            return_debug=True,
        )

        int8_preds = scores_int8.astype(np.int16).argmax(axis=1).astype(np.int64)

        batch_size = labels_np.shape[0]
        total += batch_size

        fp32_correct += int(np.sum(fp32_preds == labels_np))
        int8_correct += int(np.sum(int8_preds == labels_np))
        mismatch_fp32_int8 += int(np.sum(fp32_preds != int8_preds))

        for true_label, pred_label in zip(labels_np, int8_preds):
            confusion_int8[true_label, pred_label] += 1

        for key, value in debug.items():
            if isinstance(value, bool):
                value = float(value)
            debug_sums[key] = debug_sums.get(key, 0.0) + float(value)
        debug_count += 1

        # 앞부분 일부 prediction 저장
        for i in range(batch_size):
            if len(prediction_rows) >= keep_prediction_rows:
                break

            prediction_rows.append({
                "global_index": len(prediction_rows),
                "label": int(labels_np[i]),
                "fp32_pred": int(fp32_preds[i]),
                "int8_pred": int(int8_preds[i]),
                "match_fp32_int8": int(fp32_preds[i] == int8_preds[i]),
                "int8_score_0": int(scores_int8[i, 0]),
                "int8_score_1": int(scores_int8[i, 1]),
                "int8_score_2": int(scores_int8[i, 2]),
                "int8_score_3": int(scores_int8[i, 3]),
                "int8_score_4": int(scores_int8[i, 4]),
                "int8_score_5": int(scores_int8[i, 5]),
                "int8_score_6": int(scores_int8[i, 6]),
                "int8_score_7": int(scores_int8[i, 7]),
                "int8_score_8": int(scores_int8[i, 8]),
                "int8_score_9": int(scores_int8[i, 9]),
            })

    debug_avg = {
        key: value / max(debug_count, 1)
        for key, value in debug_sums.items()
    }

    fp32_acc = 100.0 * fp32_correct / total
    int8_acc = 100.0 * int8_correct / total
    mismatch_rate = 100.0 * mismatch_fp32_int8 / total

    return {
        "total": total,
        "fp32_correct": fp32_correct,
        "int8_correct": int8_correct,
        "fp32_acc": fp32_acc,
        "int8_acc": int8_acc,
        "mismatch_fp32_int8": mismatch_fp32_int8,
        "mismatch_rate_percent": mismatch_rate,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "bias_scale": x_scale * w_scale,
        "shift_val": shift_val,
        "debug_avg": debug_avg,
        "confusion_int8": confusion_int8,
        "prediction_rows": prediction_rows,
    }


def save_json_summary(result, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        key: value
        for key, value in result.items()
        if key not in ["confusion_int8", "prediction_rows"]
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def save_confusion_csv(confusion, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + [str(i) for i in range(10)])

        for true_label in range(10):
            writer.writerow([true_label] + confusion[true_label].tolist())


def save_prediction_rows(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    dataset_name = "MNIST"
    seed = 42
    calib_size = 1024

    # 현재 RTL-compatible baseline
    x_scale = 127.0
    w_scale = 16.0
    shift_val = 4
    bias_mode = "round"

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
    print()
    print("Numpy fixed-point config")
    print(f"  X_SCALE   = {x_scale}")
    print(f"  W_SCALE   = {w_scale}")
    print(f"  B_SCALE   = {x_scale * w_scale}")
    print(f"  SHIFT_VAL = {shift_val}")
    print(f"  bias_mode = {bias_mode}")

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

    # PyTorch baseline sanity check
    criterion = nn.CrossEntropyLoss()
    fp32_test_metrics = evaluate(model, test_loader, criterion, device)

    print()
    print("PyTorch FP32 sanity check")
    print(f"  test_acc  = {fp32_test_metrics['acc']:.2f}%")
    print(f"  test_loss = {fp32_test_metrics['loss']:.4f}")

    # Quantize model parameters
    qparams = quantize_lenet5_params(
        model=model,
        x_scale=x_scale,
        w_scale=w_scale,
        bias_mode=bias_mode,
    )

    print()
    print("Evaluating Numpy INT8 fixed-point on test set...")

    result = evaluate_numpy_fixed_point(
        model=model,
        loader=test_loader,
        qparams=qparams,
        device=device,
        x_scale=x_scale,
        w_scale=w_scale,
        shift_val=shift_val,
        max_batches=None,
        keep_prediction_rows=200,
    )

    print()
    print("Numpy INT8 fixed-point result")
    print(f"  total              : {result['total']}")
    print(f"  fp32_acc           : {result['fp32_acc']:.2f}%")
    print(f"  int8_acc           : {result['int8_acc']:.2f}%")
    print(f"  fp32/int8 mismatch : {result['mismatch_fp32_int8']} "
          f"({result['mismatch_rate_percent']:.2f}%)")

    print()
    print("Average debug stats")
    for key, value in sorted(result["debug_avg"].items()):
        if key.endswith("_int32_ok"):
            print(f"  {key:24s}: {bool(round(value))}")
        else:
            print(f"  {key:24s}: {value:.6f}")

    tag = f"x{int(x_scale)}_w{int(w_scale)}_shift{shift_val}_{bias_mode}"

    summary_path = out_dir / f"numpy_int8_summary_{tag}.json"
    confusion_path = out_dir / f"numpy_int8_confusion_{tag}.csv"
    pred_path = out_dir / f"numpy_int8_predictions_head_{tag}.csv"

    save_json_summary(result, summary_path)
    save_confusion_csv(result["confusion_int8"], confusion_path)
    save_prediction_rows(result["prediction_rows"], pred_path)

    print()
    print("Saved files")
    print(f"  summary    : {summary_path}")
    print(f"  confusion  : {confusion_path}")
    print(f"  predictions: {pred_path}")


if __name__ == "__main__":
    main()