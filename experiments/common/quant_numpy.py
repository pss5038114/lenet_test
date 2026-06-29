# experiments/common/quant_numpy.py

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def quantize_int8_symmetric(x, scale, qmin=-128, qmax=127):
    """
    Symmetric INT8 quantization.
    현재 weight용 기본 함수.
    """
    q_float = np.round(x * scale)
    sat_mask = (q_float < qmin) | (q_float > qmax)
    q = np.clip(q_float, qmin, qmax).astype(np.int8)

    return q, {
        "sat_count": int(np.sum(sat_mask)),
        "sat_rate": float(np.mean(sat_mask)),
        "min_before_clip": float(np.min(q_float)),
        "max_before_clip": float(np.max(q_float)),
    }


def quantize_input_uint_like_int8(x, x_scale=127.0):
    """
    MNIST input은 0~1 범위이므로 0~127로 quantize한다.
    저장 dtype은 int8이지만 값 범위는 0~127이다.
    """
    q_float = np.round(x * x_scale)
    sat_mask = (q_float < 0) | (q_float > 127)
    q = np.clip(q_float, 0, 127).astype(np.int8)

    return q, {
        "sat_count": int(np.sum(sat_mask)),
        "sat_rate": float(np.mean(sat_mask)),
        "min_before_clip": float(np.min(q_float)),
        "max_before_clip": float(np.max(q_float)),
    }


def quantize_bias_int32(bias_fp32, act_scale, weight_scale, mode="round"):
    """
    q_bias = bias * S_A * S_W

    현재 global baseline에서는:
      S_A = 127
      S_W = 16
      S_B = 2032
    """
    scaled = bias_fp32 * act_scale * weight_scale

    if mode == "round":
        q = np.round(scaled)
    elif mode == "floor":
        q = np.floor(scaled)
    elif mode == "ceil":
        q = np.ceil(scaled)
    else:
        raise ValueError(f"Unknown bias quantization mode: {mode}")

    q = np.clip(q, -(2**31), 2**31 - 1).astype(np.int32)
    return q


def requant_relu_shift_clip(acc, shift_val=4):
    """
    RTL의 ReLU + arithmetic right shift + clamp를 흉내낸다.

    Conv, FC1, FC2용:
      negative -> 0
      >>> SHIFT_VAL
      clamp 0~127
    """
    acc = acc.astype(np.int64)
    relu = np.maximum(acc, 0)
    shifted = relu >> shift_val
    q = np.clip(shifted, 0, 127).astype(np.int8)
    return q


def requant_signed_shift_clip(acc, shift_val=4):
    """
    FC3용:
      ReLU 없음
      arithmetic right shift
      clamp -128~127

    Python의 >> 는 signed integer에 대해 arithmetic shift처럼 동작한다.
    """
    acc = acc.astype(np.int64)
    shifted = acc >> shift_val
    q = np.clip(shifted, -128, 127).astype(np.int8)
    return q


def conv2d_valid_nchw_int(x, w, b):
    """
    NCHW valid convolution.

    x: [N, C_in, H, W], int8
    w: [C_out, C_in, KH, KW], int8
    b: [C_out], int32

    return:
      acc: [N, C_out, H-KH+1, W-KW+1], int64
    """
    x64 = x.astype(np.int64)
    w64 = w.astype(np.int64)
    b64 = b.astype(np.int64)

    kh = w.shape[2]
    kw = w.shape[3]

    # windows shape:
    # [N, C_in, H_out, W_out, KH, KW]
    windows = sliding_window_view(x64, (kh, kw), axis=(2, 3))

    acc = np.einsum(
        "nchwkl,ockl->nohw",
        windows,
        w64,
        optimize=True,
    )

    acc = acc + b64.reshape(1, -1, 1, 1)
    return acc


def maxpool2d_2x2_int(x):
    """
    2x2 max pooling, stride 2.
    x: [N, C, H, W], int8
    """
    n, c, h, w = x.shape
    assert h % 2 == 0 and w % 2 == 0

    y = x.reshape(n, c, h // 2, 2, w // 2, 2)
    y = y.max(axis=(3, 5))
    return y.astype(np.int8)


def linear_int(x, w, b):
    """
    x: [N, in_features], int8
    w: [out_features, in_features], int8
    b: [out_features], int32

    return:
      acc: [N, out_features], int64
    """
    x64 = x.astype(np.int64)
    w64 = w.astype(np.int64)
    b64 = b.astype(np.int64)

    return x64 @ w64.T + b64.reshape(1, -1)


def check_int32_range(arr, name):
    """
    Python에서는 int64로 계산하지만,
    실제 RTL accumulator는 INT32이므로 범위 체크를 한다.
    """
    amin = int(arr.min())
    amax = int(arr.max())

    ok = (amin >= -(2**31)) and (amax <= 2**31 - 1)

    return {
        f"{name}_min": amin,
        f"{name}_max": amax,
        f"{name}_int32_ok": bool(ok),
    }


def saturation_rate_int8(q, signed=True):
    q_int = q.astype(np.int16)

    if signed:
        sat = (q_int <= -128) | (q_int >= 127)
    else:
        sat = q_int >= 127

    return float(np.mean(sat))


def quantize_lenet5_params(
    model,
    x_scale=127.0,
    w_scale=16.0,
    bias_mode="round",
):
    """
    PyTorch LeNet5 model에서 weight/bias를 꺼내서 INT8/INT32로 변환한다.

    현재 baseline:
      X_SCALE = 127
      W_SCALE = 16
      B_SCALE = 127 * 16 = 2032

    현재 RTL은 SHIFT_VAL=4, 즉 /16을 하므로,
    W_SCALE=16일 때 activation scale이 layer마다 대체로 유지된다.
    """
    model_cpu = model.cpu()

    q = {}

    q["conv1_w"], _ = quantize_int8_symmetric(
        model_cpu.conv1.weight.detach().numpy(),
        scale=w_scale,
    )
    q["conv2_w"], _ = quantize_int8_symmetric(
        model_cpu.conv2.weight.detach().numpy(),
        scale=w_scale,
    )
    q["fc1_w"], _ = quantize_int8_symmetric(
        model_cpu.fc1.weight.detach().numpy(),
        scale=w_scale,
    )
    q["fc2_w"], _ = quantize_int8_symmetric(
        model_cpu.fc2.weight.detach().numpy(),
        scale=w_scale,
    )
    q["fc3_w"], _ = quantize_int8_symmetric(
        model_cpu.fc3.weight.detach().numpy(),
        scale=w_scale,
    )

    q["conv1_b"] = quantize_bias_int32(
        model_cpu.conv1.bias.detach().numpy(),
        act_scale=x_scale,
        weight_scale=w_scale,
        mode=bias_mode,
    )
    q["conv2_b"] = quantize_bias_int32(
        model_cpu.conv2.bias.detach().numpy(),
        act_scale=x_scale,
        weight_scale=w_scale,
        mode=bias_mode,
    )
    q["fc1_b"] = quantize_bias_int32(
        model_cpu.fc1.bias.detach().numpy(),
        act_scale=x_scale,
        weight_scale=w_scale,
        mode=bias_mode,
    )
    q["fc2_b"] = quantize_bias_int32(
        model_cpu.fc2.bias.detach().numpy(),
        act_scale=x_scale,
        weight_scale=w_scale,
        mode=bias_mode,
    )
    q["fc3_b"] = quantize_bias_int32(
        model_cpu.fc3.bias.detach().numpy(),
        act_scale=x_scale,
        weight_scale=w_scale,
        mode=bias_mode,
    )

    return q


def lenet5_forward_int8_numpy(
    x_fp32_nchw,
    qparams,
    x_scale=127.0,
    shift_val=4,
    return_debug=False,
):
    """
    현재 RTL과 최대한 같은 방식의 LeNet-5 INT8 fixed-point forward.

    x_fp32_nchw:
      [N, 1, 32, 32], float32, 0~1

    return:
      scores_int8: [N, 10], int8
      debug: optional dict
    """
    debug = {}

    x_q, x_info = quantize_input_uint_like_int8(x_fp32_nchw, x_scale=x_scale)
    debug["input_sat_rate"] = x_info["sat_rate"]

    # Conv1 -> ReLU -> shift -> clamp -> Pool
    acc1 = conv2d_valid_nchw_int(x_q, qparams["conv1_w"], qparams["conv1_b"])
    debug.update(check_int32_range(acc1, "conv1_acc"))

    q1 = requant_relu_shift_clip(acc1, shift_val=shift_val)
    debug["conv1_out_sat_rate"] = saturation_rate_int8(q1, signed=False)

    p1 = maxpool2d_2x2_int(q1)

    # Conv2 -> ReLU -> shift -> clamp -> Pool
    acc2 = conv2d_valid_nchw_int(p1, qparams["conv2_w"], qparams["conv2_b"])
    debug.update(check_int32_range(acc2, "conv2_acc"))

    q2 = requant_relu_shift_clip(acc2, shift_val=shift_val)
    debug["conv2_out_sat_rate"] = saturation_rate_int8(q2, signed=False)

    p2 = maxpool2d_2x2_int(q2)

    # Flatten: PyTorch 기준 NCHW flatten
    flat = p2.reshape(p2.shape[0], 400)

    # FC1 -> ReLU -> shift -> clamp
    acc3 = linear_int(flat, qparams["fc1_w"], qparams["fc1_b"])
    debug.update(check_int32_range(acc3, "fc1_acc"))

    q3 = requant_relu_shift_clip(acc3, shift_val=shift_val)
    debug["fc1_out_sat_rate"] = saturation_rate_int8(q3, signed=False)

    # FC2 -> ReLU -> shift -> clamp
    acc4 = linear_int(q3, qparams["fc2_w"], qparams["fc2_b"])
    debug.update(check_int32_range(acc4, "fc2_acc"))

    q4 = requant_relu_shift_clip(acc4, shift_val=shift_val)
    debug["fc2_out_sat_rate"] = saturation_rate_int8(q4, signed=False)

    # FC3 -> no ReLU -> shift -> signed clamp
    acc5 = linear_int(q4, qparams["fc3_w"], qparams["fc3_b"])
    debug.update(check_int32_range(acc5, "fc3_acc"))

    scores = requant_signed_shift_clip(acc5, shift_val=shift_val)
    debug["fc3_score_sat_rate"] = saturation_rate_int8(scores, signed=True)

    if return_debug:
        return scores, debug

    return scores

def compute_global_scale_flow(x_scale=127.0, w_scale=16.0, shift_val=4):
    """
    현재 RTL처럼 layer마다 accumulator를 오른쪽 shift해서 다시 INT8 activation으로
    만드는 구조에서, global W_SCALE을 쓸 때 각 layer의 activation scale 흐름을 계산한다.

    핵심:
      S_A,out = S_A,in * S_W / 2^SHIFT_VAL

    Pooling은 max만 하므로 scale이 변하지 않는다고 본다.
    """
    div = float(2 ** shift_val)

    scales = {}

    scales["input"] = float(x_scale)

    scales["conv1_in"] = scales["input"]
    scales["conv1_bias"] = scales["conv1_in"] * float(w_scale)
    scales["conv1_out"] = scales["conv1_bias"] / div

    scales["conv2_in"] = scales["conv1_out"]
    scales["conv2_bias"] = scales["conv2_in"] * float(w_scale)
    scales["conv2_out"] = scales["conv2_bias"] / div

    scales["fc1_in"] = scales["conv2_out"]
    scales["fc1_bias"] = scales["fc1_in"] * float(w_scale)
    scales["fc1_out"] = scales["fc1_bias"] / div

    scales["fc2_in"] = scales["fc1_out"]
    scales["fc2_bias"] = scales["fc2_in"] * float(w_scale)
    scales["fc2_out"] = scales["fc2_bias"] / div

    scales["fc3_in"] = scales["fc2_out"]
    scales["fc3_bias"] = scales["fc3_in"] * float(w_scale)
    scales["fc3_out"] = scales["fc3_bias"] / div

    return scales


def quantize_lenet5_params_global_scale_flow(
    model,
    x_scale=127.0,
    w_scale=16.0,
    shift_val=4,
    bias_mode="round",
):
    """
    다양한 W_SCALE sweep을 위한 parameter quantization.

    기존 quantize_lenet5_params()는 모든 layer bias에 같은 x_scale을 썼다.
    이 함수는 현재 RTL의 SHIFT_VAL을 고려해서 layer별 activation scale을 추적하고,
    그에 맞춰 layer별 bias scale을 다르게 적용한다.
    """
    model_cpu = model.cpu()
    scales = compute_global_scale_flow(
        x_scale=x_scale,
        w_scale=w_scale,
        shift_val=shift_val,
    )

    q = {}

    q["conv1_w"], conv1_w_info = quantize_int8_symmetric(
        model_cpu.conv1.weight.detach().numpy(),
        scale=w_scale,
    )
    q["conv2_w"], conv2_w_info = quantize_int8_symmetric(
        model_cpu.conv2.weight.detach().numpy(),
        scale=w_scale,
    )
    q["fc1_w"], fc1_w_info = quantize_int8_symmetric(
        model_cpu.fc1.weight.detach().numpy(),
        scale=w_scale,
    )
    q["fc2_w"], fc2_w_info = quantize_int8_symmetric(
        model_cpu.fc2.weight.detach().numpy(),
        scale=w_scale,
    )
    q["fc3_w"], fc3_w_info = quantize_int8_symmetric(
        model_cpu.fc3.weight.detach().numpy(),
        scale=w_scale,
    )

    q["conv1_b"] = quantize_bias_int32(
        model_cpu.conv1.bias.detach().numpy(),
        act_scale=scales["conv1_in"],
        weight_scale=w_scale,
        mode=bias_mode,
    )
    q["conv2_b"] = quantize_bias_int32(
        model_cpu.conv2.bias.detach().numpy(),
        act_scale=scales["conv2_in"],
        weight_scale=w_scale,
        mode=bias_mode,
    )
    q["fc1_b"] = quantize_bias_int32(
        model_cpu.fc1.bias.detach().numpy(),
        act_scale=scales["fc1_in"],
        weight_scale=w_scale,
        mode=bias_mode,
    )
    q["fc2_b"] = quantize_bias_int32(
        model_cpu.fc2.bias.detach().numpy(),
        act_scale=scales["fc2_in"],
        weight_scale=w_scale,
        mode=bias_mode,
    )
    q["fc3_b"] = quantize_bias_int32(
        model_cpu.fc3.bias.detach().numpy(),
        act_scale=scales["fc3_in"],
        weight_scale=w_scale,
        mode=bias_mode,
    )

    q["_scale_flow"] = scales
    q["_quant_info"] = {
        "conv1_w_sat_rate": conv1_w_info["sat_rate"],
        "conv2_w_sat_rate": conv2_w_info["sat_rate"],
        "fc1_w_sat_rate": fc1_w_info["sat_rate"],
        "fc2_w_sat_rate": fc2_w_info["sat_rate"],
        "fc3_w_sat_rate": fc3_w_info["sat_rate"],
    }

    return q