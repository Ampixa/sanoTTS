"""Pure-numpy forward passes for the saanoTTS Root-A student stack.

These mirror the fp32 reference C runtime (mcu/src/snt_front_f32.c and
mcu/src/snt_piperlite.c in the parent research repo), which itself was
ported from the PyTorch training code (tools/train_roota_piper_duration_student.py,
tools/train_roota_piper_latent_student.py, tools/train_roota_piper_decoder_student.py)
and gated against it at correlation 1.0. Operation *order* here does not need to be
bit-identical to either reference (numpy's BLAS-backed matmuls accumulate in a
different order than the scalar C loops or PyTorch's cuDNN/MKL kernels), only
numerically equivalent -- differences are float32-rounding-noise sized, far below
the >0.99 waveform-correlation gate this package is held to.

Only the subgraphs actually observed in the shipped voice packages
(architecture="duration_conv" / "token_context", decoder variant="piperlite",
stage*_branches=[0,1,2], no output adapter) are implemented. Anything else
raises NotImplementedError rather than guessing at an unverified tensor layout.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("sanotts.models")

# Fallback id used when a phoneme id from the frontend's (larger, shared)
# codepoint table exceeds a specific component's trained vocab_size. Schwa
# is a safe, neutral, always-in-vocab choice -- the same fallback used by
# the ESP32/WASM ports for the analogous duration/acoustic vocab mismatch.
_SCHWA_FALLBACK_ID = 59


def clamp_ids_to_vocab(ids: np.ndarray, vocab_size: int, *, label: str) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    out_of_range = (ids < 0) | (ids >= vocab_size)
    if np.any(out_of_range):
        n = int(out_of_range.sum())
        fallback = _SCHWA_FALLBACK_ID if _SCHWA_FALLBACK_ID < vocab_size else 0
        logger.warning(
            "sanotts: %d/%d %s ids fall outside vocab_size=%d, remapping to id=%d",
            n, ids.size, label, vocab_size, fallback,
        )
        ids = np.where(out_of_range, fallback, ids)
    return ids


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def leaky_relu(x: np.ndarray, slope: float) -> np.ndarray:
    return np.where(x > 0.0, x, slope * x)


def linspace01(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float64).astype(np.float32)


def conv1d_same(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    dilation: int = 1,
) -> np.ndarray:
    """PyTorch Conv1d "same"-padding semantics: pad = dilation*(K//2).

    x: [in_ch, T]; weight: [out_ch, in_ch, K]; bias: [out_ch].
    Returns [out_ch, T].
    """
    in_ch, T = x.shape
    out_ch, in_ch_w, K = weight.shape
    if in_ch_w != in_ch:
        raise ValueError(f"conv1d_same: channel mismatch {in_ch_w} != {in_ch}")
    pad = dilation * (K // 2)
    out = np.broadcast_to(bias[:, None].astype(np.float32), (out_ch, T)).copy()
    for k in range(K):
        off = k * dilation - pad
        lo = max(0, -off)
        hi = min(T, T - off)
        if hi <= lo:
            continue
        # out[:, lo:hi] += weight[:, :, k] @ x[:, lo+off:hi+off]
        out[:, lo:hi] += weight[:, :, k] @ x[:, lo + off:hi + off]
    return out


def conv1d_1x1(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Conv1d with kernel_size=1: a plain per-timestep linear projection."""
    w = weight[:, :, 0] if weight.ndim == 3 else weight
    return w @ x + bias[:, None]


def depthwise_conv1d_same(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Depthwise Conv1d, no bias. x: [C, T]; weight: [C, K] (or [C,1,K])."""
    if weight.ndim == 3:
        weight = weight[:, 0, :]
    C, T = x.shape
    _, K = weight.shape
    pad = K // 2
    out = np.zeros_like(x)
    for k in range(K):
        off = k - pad
        lo = max(0, -off)
        hi = min(T, T - off)
        if hi <= lo:
            continue
        out[:, lo:hi] += weight[:, k:k + 1] * x[:, lo + off:hi + off]
    return out


def conv_transpose1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> np.ndarray:
    """PyTorch ConvTranspose1d. x: [in_ch, T]; weight: [in_ch, out_ch, K]; bias: [out_ch].

    out[oc, t*stride + k - padding] += weight[ic, oc, k] * x[ic, t], summed over ic, k;
    output length L = (T - 1) * stride - 2 * padding + K.
    """
    in_ch, T = x.shape
    in_ch_w, out_ch, K = weight.shape
    if in_ch_w != in_ch:
        raise ValueError(f"conv_transpose1d: channel mismatch {in_ch_w} != {in_ch}")
    L = (T - 1) * stride - 2 * padding + K
    out = np.broadcast_to(bias[:, None].astype(np.float32), (out_ch, L)).copy()
    for k in range(K):
        shift = k - padding
        if shift >= L:
            continue
        t_lo = 0
        if shift < 0:
            t_lo = (-shift + stride - 1) // stride
        t_hi = (L - 1 - shift) // stride + 1
        t_hi = min(t_hi, T)
        if t_hi <= t_lo:
            continue
        count = t_hi - t_lo
        j_start = t_lo * stride + shift
        j_end = j_start + (count - 1) * stride + 1
        contrib = weight[:, :, k].T @ x[:, t_lo:t_hi]  # [out_ch, count]
        out[:, j_start:j_end:stride] += contrib
    return out


def residual_conv_block(
    x: np.ndarray,
    tensors: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray:
    """One ResidualConvBlock: x + scale * conv2(silu(conv1(x))). Kernel/pad same as x.

    Kernel size is read from the weight tensor itself (weight.shape[-1]), not
    passed in, so this always matches whatever the checkpoint actually stored.
    """
    scale = float(tensors[f"{prefix}.scale"][0])
    t = conv1d_same(x, tensors[f"{prefix}.net.0.weight"], tensors[f"{prefix}.net.0.bias"])
    t = silu(t)
    u = conv1d_same(t, tensors[f"{prefix}.net.2.weight"], tensors[f"{prefix}.net.2.bias"])
    return x + scale * u


# --------------------------------------------------------------------------
# Duration student ("architecture": "duration_conv" in the manifest)
# --------------------------------------------------------------------------

def duration_forward(
    tensors: dict[str, np.ndarray],
    config: dict[str, Any],
    ids: np.ndarray,
    *,
    length_scale: float,
) -> np.ndarray:
    architecture = str(config.get("architecture") or "")
    if architecture != "duration_conv":
        raise NotImplementedError(f"unsupported duration architecture: {architecture!r}")

    vocab_size = int(config["vocab_size"])
    hidden = int(config["hidden"])
    depth = int(config["depth"])
    max_tokens = int(config["max_tokens"])
    max_duration = int(config["max_duration"])

    ids = clamp_ids_to_vocab(ids, vocab_size, label="duration")
    n = ids.shape[0]
    if n <= 0:
        raise ValueError("duration_forward: empty id sequence")

    embed = tensors["embedding.weight"]  # [vocab, hidden]
    if embed.shape != (vocab_size, hidden):
        raise RuntimeError(f"duration embedding shape {embed.shape} != config ({vocab_size}, {hidden})")
    token_x = embed[ids].T  # [hidden, n]

    positions = linspace01(n)
    length_hint = np.float32(np.log1p(np.float64(n)) / np.log1p(np.float64(max_tokens)))
    valid_hint = np.ones((n,), dtype=np.float32)
    features = np.stack([positions, np.full((n,), length_hint, dtype=np.float32), valid_hint], axis=0)

    x = conv1d_1x1(
        np.concatenate([token_x, features], axis=0),
        tensors["input_proj.weight"],
        tensors["input_proj.bias"],
    )
    for i in range(depth):
        x = residual_conv_block(x, tensors, f"blocks.{i}")

    log_duration = conv1d_1x1(x, tensors["output.weight"], tensors["output.bias"])[0]  # [n]

    duration = np.exp(log_duration)
    duration = np.clip(duration, 1.0, None)
    duration = np.round(duration * float(length_scale))
    duration = np.clip(duration, 1.0, float(max_duration))
    return duration.astype(np.int64)


# --------------------------------------------------------------------------
# Acoustic student ("architecture": "token_context" in the manifest)
# --------------------------------------------------------------------------

def acoustic_forward(
    tensors: dict[str, np.ndarray],
    config: dict[str, Any],
    ids: np.ndarray,
    durations: np.ndarray,
) -> np.ndarray:
    architecture = str(config.get("architecture") or "")
    if architecture != "token_context":
        raise NotImplementedError(f"unsupported acoustic architecture: {architecture!r}")
    unexpected_adapter_keys = [name for name in tensors if "adapter" in name]
    if unexpected_adapter_keys:
        raise NotImplementedError(
            "acoustic checkpoint has an output adapter "
            f"({unexpected_adapter_keys}); this package only implements the "
            "adapter-free token_context path verified against the shipped voices"
        )

    vocab_size = int(config["vocab_size"])
    hidden = int(config["hidden"])
    depth = int(config["depth"])
    token_depth = int(config["token_depth"])
    out_channels = int(config["out_channels"])

    ids = clamp_ids_to_vocab(ids, vocab_size, label="acoustic")
    durations = np.asarray(durations, dtype=np.int64)
    if durations.shape != ids.shape:
        raise ValueError("acoustic_forward: id/duration shape mismatch")
    if np.any(durations < 1):
        raise ValueError("acoustic_forward: non-positive duration")

    n = int(ids.shape[0])
    frames = int(durations.sum())

    # -- token stage --
    embed = tensors["embedding.weight"]
    if embed.shape != (vocab_size, hidden):
        raise RuntimeError(f"acoustic embedding shape {embed.shape} != config ({vocab_size}, {hidden})")
    token_x = embed[ids].T  # [hidden, n]
    token_pos = linspace01(n)
    durations_f = durations.astype(np.float64)
    max_duration = max(float(durations_f.max()), 1.0)
    duration_hint = (np.log1p(durations_f) / np.log1p(max_duration)).astype(np.float32)
    token_features = np.stack([token_pos, duration_hint], axis=0)

    token_x = conv1d_1x1(
        np.concatenate([token_x, token_features], axis=0),
        tensors["token_input_proj.weight"],
        tensors["token_input_proj.bias"],
    )
    for i in range(token_depth):
        token_x = residual_conv_block(token_x, tensors, f"token_blocks.{i}")

    # -- expand token context to frames --
    expanded = np.repeat(token_x, durations, axis=1)  # [hidden, frames]
    if expanded.shape[1] != frames:
        raise RuntimeError("acoustic_forward: context expansion length mismatch")

    frame_pos = linspace01(frames)
    token_count = max(n - 1, 1)
    token_pos_frame = np.empty((frames,), dtype=np.float32)
    duration_pos_frame = np.empty((frames,), dtype=np.float32)
    pos = 0
    for token_index in range(n):
        d = int(durations[token_index])
        if d <= 0:
            continue
        token_pos_frame[pos:pos + d] = np.float32(float(token_index) / float(token_count))
        if d == 1:
            duration_pos_frame[pos] = 0.0
        else:
            duration_pos_frame[pos:pos + d] = (
                np.arange(d, dtype=np.float64) / np.float64(d - 1)
            ).astype(np.float32)
        pos += d

    frame_features = np.stack([frame_pos, token_pos_frame, duration_pos_frame], axis=0)
    x = conv1d_1x1(
        np.concatenate([expanded, frame_features], axis=0),
        tensors["frame_input_proj.weight"],
        tensors["frame_input_proj.bias"],
    )
    for i in range(depth):
        x = residual_conv_block(x, tensors, f"frame_blocks.{i}")

    latent = conv1d_1x1(x, tensors["output.weight"], tensors["output.bias"])
    if latent.shape[0] != out_channels:
        raise RuntimeError("acoustic_forward: unexpected output channel count")
    return latent.astype(np.float32)  # [out_channels, frames]


# --------------------------------------------------------------------------
# Decoder student ("variant": "piperlite" in the manifest)
# --------------------------------------------------------------------------

_BANK_KERNELS = (3, 5, 7)
_BANK_DIL1 = (1, 2, 3)
_BANK_DIL2 = (2, 6, 12)


def _residual_bank(x: np.ndarray, tensors: dict[str, np.ndarray], prefix: str, branches: list[int]) -> np.ndarray:
    """PiperResidualBank: mean over active branches of
    y2 = conv2(lrelu(y1, 0.1)) + y1, y1 = conv1(lrelu(x, 0.1)) + x.
    """
    acc = np.zeros_like(x)
    for branch in branches:
        k = _BANK_KERNELS[branch]
        d1 = _BANK_DIL1[branch]
        d2 = _BANK_DIL2[branch]
        t = leaky_relu(x, 0.1)
        u = conv1d_same(t, tensors[f"{prefix}.blocks.{branch}.conv1.weight"], tensors[f"{prefix}.blocks.{branch}.conv1.bias"], dilation=d1)
        y1 = u + x
        t2 = leaky_relu(y1, 0.1)
        u2 = conv1d_same(t2, tensors[f"{prefix}.blocks.{branch}.conv2.weight"], tensors[f"{prefix}.blocks.{branch}.conv2.bias"], dilation=d2)
        y2 = u2 + y1
        acc = acc + y2
    return acc / float(len(branches))


def _apply_post_filter(audio: np.ndarray, tensors: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    channels = int(config.get("post_filter_channels") or 0)
    layers = int(config.get("post_filter_layers") or 0)
    kernel = int(config.get("post_filter_kernel") or 9)
    scale = float(config.get("post_filter_scale") or 0.0)
    if channels <= 0:
        return audio

    r = conv1d_same(audio[None, :], tensors["post_filter.in_conv.weight"], tensors["post_filter.in_conv.bias"])
    for layer in range(layers):
        unit_scale = float(tensors[f"post_filter.units.{layer}.scale"][0])
        t = leaky_relu(r, 0.1)
        u = conv1d_same(t, tensors[f"post_filter.units.{layer}.conv1.weight"], tensors[f"post_filter.units.{layer}.conv1.bias"], dilation=1 + layer)
        t = leaky_relu(u, 0.1)
        u2 = conv1d_same(t, tensors[f"post_filter.units.{layer}.conv2.weight"], tensors[f"post_filter.units.{layer}.conv2.bias"], dilation=1)
        r = r + unit_scale * u2
    out = conv1d_same(r, tensors["post_filter.out_conv.weight"], tensors["post_filter.out_conv.bias"])[0]
    return np.tanh(audio + scale * out).astype(np.float32)


def decoder_forward(tensors: dict[str, np.ndarray], config: dict[str, Any], latent: np.ndarray) -> np.ndarray:
    variant = str(config.get("variant") or "")
    if variant != "piperlite":
        raise NotImplementedError(f"unsupported decoder variant: {variant!r}")
    if str(config.get("activation") or "leaky_relu") != "leaky_relu":
        raise NotImplementedError("only activation='leaky_relu' decoders are implemented")
    if float(config.get("pre_tanh_repair_channels") or 0) > 0:
        raise NotImplementedError("pre_tanh_repair is not implemented (no shipped voice uses it)")
    res_layers = int(config.get("res_layers") or 1)
    if res_layers != 1:
        raise NotImplementedError(f"only res_layers=1 is implemented, got {res_layers}")

    channels = config["channels"]
    c0, c1, c2, c3 = (int(c) for c in channels[:4])

    x = conv1d_same(latent, tensors["pre.weight"], tensors["pre.bias"])  # [c0, frames]

    stage_specs = [
        (c0, c1, 16, 8, 4, "up0", "res0.0", config.get("stage0_branches")),
        (c1, c2, 16, 8, 4, "up1", "res1.0", config.get("stage1_branches")),
        (c2, c3, 8, 4, 2, "up2", "res2.0", config.get("stage2_branches")),
    ]
    for in_c, out_c, up_k, up_s, up_p, up_name, bank_prefix, branches in stage_specs:
        if branches is None:
            branches = [0, 1, 2]
        x = leaky_relu(x, 0.1)
        x = conv_transpose1d(x, tensors[f"{up_name}.weight"], tensors[f"{up_name}.bias"], stride=up_s, padding=up_p)
        x = _residual_bank(x, tensors, bank_prefix, list(branches))

    x = leaky_relu(x, 0.01)
    audio = conv1d_same(x, tensors["post.weight"], tensors["post.bias"])[0]
    audio = np.tanh(audio).astype(np.float32)
    audio = _apply_post_filter(audio, tensors, config)
    return audio
