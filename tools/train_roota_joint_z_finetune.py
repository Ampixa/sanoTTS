#!/usr/bin/env python3
"""Joint fine-tune the z-acoustic student and Piperlite decoder on waveform loss.

This is the R8 z-path interface repair pass:

    phonemes + teacher w_ceil -> z-acoustic -> z_hat -> Piperlite decoder -> waveform

The anchor target is the teacher generator_input latent itself.  The standalone
acoustic and decoder checkpoints written by this tool keep the original latent
student and DecoderStudent loader contracts unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_roota_joint_c_finetune as joint_common
import train_roota_piper_decoder_student as decoder_trainer
import train_roota_piper_latent_student as latent_trainer


BASE_ARTIFACT_DIR = ROOT / "artifacts" / "sub10m-search" / "root-a-piper-vits"
DEFAULT_PACK_DIR = BASE_ARTIFACT_DIR / "en_US-kristin-medium-heldout12-decoder-piper-native-20260702"
DEFAULT_TEACHER_DECODER = (
    BASE_ARTIFACT_DIR
    / "en_US-kristin-medium-decoder-cut-20260702"
    / "en_US-kristin-medium-decoder-from-generator-input.onnx"
)
DEFAULT_ACOUSTIC_CHECKPOINT = (
    BASE_ARTIFACT_DIR
    / "en_US-kristin-u600-r8-zline-20260704"
    / "acoustic192-14k"
    / "latent-student.pt"
)
DEFAULT_DECODER_CHECKPOINT = (
    BASE_ARTIFACT_DIR
    / "en_US-kristin-u600-r8-pruned-357k-20260704"
    / "decoder-student.pt"
)
DEFAULT_OUT_DIR = BASE_ARTIFACT_DIR / "en_US-kristin-u600-r8-joint-z-smoke20-cpu-20260704"
R8_RANDOM_ACOUSTIC_CONFIG = {
    "architecture": "token_context",
    "vocab_size": 127,
    "hidden": 48,
    "depth": 5,
    "token_depth": 3,
    "kernel_size": 5,
    "out_channels": 192,
}
HOP_LENGTH = decoder_trainer.HOP_LENGTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--teacher-decoder", type=Path, default=DEFAULT_TEACHER_DECODER)
    parser.add_argument("--acoustic-checkpoint", type=Path, default=DEFAULT_ACOUSTIC_CHECKPOINT)
    parser.add_argument(
        "--allow-random-acoustic-init",
        action="store_true",
        help=(
            "When --acoustic-checkpoint is missing, create the documented R8 "
            "205,544-param 192-output acoustic model with random weights."
        ),
    )
    parser.add_argument(
        "--random-acoustic-vocab-size",
        type=int,
        default=127,
        help="Vocab size for --allow-random-acoustic-init; 127 preserves the R8 205,544-param budget.",
    )
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_DECODER_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-frames", type=int, default=64)
    parser.add_argument("--acoustic-lr", type=float, default=2e-5)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--waveform-l1-weight", type=float, default=0.1)
    parser.add_argument("--spectral-weight", type=float, default=0.5)
    parser.add_argument("--z-anchor-weight", type=float, default=0.5)
    parser.add_argument("--adv-weight", type=float, default=0.0)
    parser.add_argument("--adv-feature-weight", type=float, default=0.0)
    parser.add_argument("--adv-delta-weight", type=float, default=0.025)
    parser.add_argument("--adv-delta-feature-weight", type=float, default=0.25)
    parser.add_argument("--adv-start-step", type=int, default=1)
    parser.add_argument("--adv-lr", type=float, default=2e-4)
    parser.add_argument("--adv-periods", type=str, default="2,3,5,7,11")
    parser.add_argument("--adv-channels", type=str, default="8,16,32,64")
    parser.add_argument(
        "--adv-gate-mode",
        choices=("none", "target-energy"),
        default="none",
        help="Optionally gate discriminator audio with a teacher-energy mask before adversarial losses.",
    )
    parser.add_argument("--adv-gate-quantile", type=float, default=0.40)
    parser.add_argument("--adv-gate-sharpness", type=float, default=24.0)
    parser.add_argument("--adv-gate-frame-size", type=int, default=1024)
    parser.add_argument("--adv-gate-frame-hop", type=int, default=256)
    parser.add_argument("--duration-params", type=int, default=36164)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--verify-standalone-reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After saving, prove standalone checkpoints reload through the original loaders.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    joint_common.require_dir(args.pack_dir, "pack directory")
    joint_common.require_file(args.teacher_decoder, "teacher decoder ONNX")
    if args.acoustic_checkpoint.is_file():
        pass
    elif not bool(args.allow_random_acoustic_init):
        joint_common.require_file(args.acoustic_checkpoint, "z-acoustic checkpoint")
    joint_common.require_file(args.decoder_checkpoint, "Piperlite decoder checkpoint")
    if int(args.steps) < 1:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if int(args.batch_size) < 1:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if int(args.crop_frames) < 1:
        raise ValueError(f"--crop-frames must be positive, got {args.crop_frames}")
    for name in ("acoustic_lr", "decoder_lr", "adv_lr"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive, got {value!r}")
    for name in (
        "waveform_l1_weight",
        "spectral_weight",
        "z_anchor_weight",
        "adv_weight",
        "adv_feature_weight",
        "adv_delta_weight",
        "adv_delta_feature_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative, got {value!r}")
    if float(args.waveform_l1_weight) == 0.0 and float(args.spectral_weight) == 0.0:
        raise ValueError("at least one of --waveform-l1-weight or --spectral-weight must be positive")
    if float(args.z_anchor_weight) == 0.0:
        raise ValueError("--z-anchor-weight must be positive for the z interface contract")
    if int(args.adv_start_step) < 1:
        raise ValueError(f"--adv-start-step must be >= 1, got {args.adv_start_step}")
    if not (0.0 < float(args.adv_gate_quantile) < 1.0):
        raise ValueError(f"--adv-gate-quantile must be in (0, 1), got {args.adv_gate_quantile}")
    if float(args.adv_gate_sharpness) <= 0.0:
        raise ValueError(f"--adv-gate-sharpness must be positive, got {args.adv_gate_sharpness}")
    if int(args.adv_gate_frame_size) <= 1:
        raise ValueError(f"--adv-gate-frame-size must be greater than 1, got {args.adv_gate_frame_size}")
    if int(args.adv_gate_frame_hop) <= 0:
        raise ValueError(f"--adv-gate-frame-hop must be positive, got {args.adv_gate_frame_hop}")
    if int(args.duration_params) < 0:
        raise ValueError(f"--duration-params must be non-negative, got {args.duration_params}")
    if int(args.random_acoustic_vocab_size) <= 0:
        raise ValueError(
            f"--random-acoustic-vocab-size must be positive, got {args.random_acoustic_vocab_size}"
        )


def max_phoneme_vocab_size(samples: list[decoder_trainer.ChunkSample]) -> int:
    max_id = 0
    for sample in samples:
        if sample.phoneme_ids.size:
            max_id = max(max_id, int(np.max(sample.phoneme_ids)))
    return max_id + 1


def create_random_acoustic_model(
    *,
    samples: list[decoder_trainer.ChunkSample],
    device: torch.device,
    requested_vocab_size: int,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    required_vocab_size = max_phoneme_vocab_size(samples)
    vocab_size = max(int(requested_vocab_size), int(required_vocab_size))
    config = dict(R8_RANDOM_ACOUSTIC_CONFIG)
    config["vocab_size"] = int(vocab_size)
    model = latent_trainer.create_model_from_config(config).to(device)
    init_summary = {
        "mode": "random_acoustic_init",
        "reason": "acoustic checkpoint missing and --allow-random-acoustic-init was passed",
        "required_vocab_size": int(required_vocab_size),
        "config": config,
        "parameters": int(latent_trainer.count_parameters(model)),
    }
    return model, config, init_summary


def load_or_create_acoustic_model(
    args: argparse.Namespace,
    *,
    samples: list[decoder_trainer.ChunkSample],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    if args.acoustic_checkpoint.is_file():
        model, config = joint_common.load_acoustic_checkpoint(args.acoustic_checkpoint, device)
        return model, config, {
            "mode": "checkpoint",
            "checkpoint": str(args.acoustic_checkpoint),
            "parameters": int(latent_trainer.count_parameters(model)),
        }
    return create_random_acoustic_model(
        samples=samples,
        device=device,
        requested_vocab_size=int(args.random_acoustic_vocab_size),
    )


def load_piperlite_decoder_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[decoder_trainer.DecoderStudent, dict[str, Any]]:
    checkpoint = decoder_trainer.load_torch_checkpoint(checkpoint_path, "Piperlite decoder checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{checkpoint_path}: missing decoder checkpoint config")
    if str(config.get("variant") or "") != "piperlite":
        raise RuntimeError(
            f"{checkpoint_path}: joint z fine-tune requires decoder variant piperlite, got {config.get('variant')!r}"
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"{checkpoint_path}: missing decoder model_state_dict")
    model = joint_common.decoder_from_config(config)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{checkpoint_path}: strict decoder load mismatch "
            f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}"
        )
    model.to(device)
    return model, dict(config)


def joint_z_crop_batch(
    samples: list[decoder_trainer.ChunkSample],
    *,
    acoustic_model: nn.Module,
    out_channels: int,
    batch_size: int,
    crop_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crops = joint_common.select_crops(samples, batch_size=batch_size, crop_frames=crop_frames)
    z_hat_values: list[torch.Tensor] = []
    teacher_latent_values: list[np.ndarray] = []
    audio_values: list[np.ndarray] = []
    for crop in crops:
        sample = crop.sample
        shim = joint_common.make_acoustic_shim(sample, code_dim=int(out_channels))
        features = latent_trainer.expand_features(shim, device)
        z_hat_full = latent_trainer.predict_latent_tensor(acoustic_model, features)
        if z_hat_full.ndim != 2:
            raise RuntimeError(f"z-acoustic prediction must be [T, C], got {z_hat_full.shape}")
        expected_shape = (int(sample.latent.shape[2]), int(out_channels))
        if tuple(z_hat_full.shape) != expected_shape:
            raise RuntimeError(
                f"{sample.row_id} chunk {sample.chunk_index}: z-acoustic prediction shape "
                f"{tuple(z_hat_full.shape)} != {expected_shape}"
            )
        z_hat_values.append(z_hat_full[crop.start : crop.end, :].transpose(0, 1))
        teacher_latent_values.append(sample.latent[:, :, crop.start : crop.end])
        audio_start = crop.start * HOP_LENGTH
        audio_end = crop.end * HOP_LENGTH
        audio_values.append(sample.teacher_audio[audio_start:audio_end].reshape(1, -1))
    z_hat = torch.stack(z_hat_values, dim=0).to(dtype=torch.float32)
    teacher_z = torch.as_tensor(
        np.concatenate(teacher_latent_values, axis=0),
        dtype=torch.float32,
        device=device,
    )
    target = torch.as_tensor(np.stack(audio_values, axis=0), dtype=torch.float32, device=device)
    if z_hat.shape != teacher_z.shape:
        raise RuntimeError(f"z_hat shape {z_hat.shape} != teacher_z shape {teacher_z.shape}")
    return z_hat, teacher_z, target


def parameter_accounting(
    *,
    args: argparse.Namespace,
    acoustic_model: nn.Module,
    decoder_model: nn.Module,
) -> dict[str, Any]:
    acoustic_parameters = latent_trainer.count_parameters(acoustic_model)
    decoder_parameters = decoder_trainer.count_parameters(decoder_model)
    acoustic_trainable = int(sum(param.numel() for param in acoustic_model.parameters() if param.requires_grad))
    decoder_trainable = int(sum(param.numel() for param in decoder_model.parameters() if param.requires_grad))
    duration_parameters = int(args.duration_params)
    return {
        "duration_student_parameters": duration_parameters,
        "acoustic_parameters": int(acoustic_parameters),
        "decoder_parameters": int(decoder_parameters),
        "acoustic_trainable_parameters": int(acoustic_trainable),
        "decoder_trainable_parameters": int(decoder_trainable),
        "inference_total_parameters": int(duration_parameters + acoustic_parameters + decoder_parameters),
        "training_total_parameters": int(duration_parameters + acoustic_parameters + decoder_parameters),
        "trainable_parameters": int(acoustic_trainable + decoder_trainable),
    }


def save_checkpoints(
    *,
    args: argparse.Namespace,
    acoustic_model: nn.Module,
    acoustic_config: dict[str, Any],
    decoder_model: decoder_trainer.DecoderStudent,
    decoder_config: dict[str, Any],
    logs: list[dict[str, Any]],
    accounting: dict[str, Any],
    acoustic_init: dict[str, Any],
) -> dict[str, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    acoustic_state = joint_common.state_dict_cpu(acoustic_model)
    decoder_state = joint_common.state_dict_cpu(decoder_model)
    train_args = joint_common.jsonable_args(args)

    joint_checkpoint = args.out_dir / "joint-z-finetune.pt"
    acoustic_checkpoint = args.out_dir / "latent-student.pt"
    decoder_checkpoint = args.out_dir / "decoder-student.pt"

    torch.save(
        {
            "format": "roota_joint_z_finetune_v1",
            "acoustic_state_dict": acoustic_state,
            "decoder_model_state_dict": decoder_state,
            "acoustic_config": acoustic_config,
            "decoder_config": decoder_config,
            "joint_config": train_args,
            "acoustic_init": acoustic_init,
            "parameter_accounting": accounting,
            "logs": logs,
        },
        joint_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": acoustic_state,
            "config": acoustic_config,
            "train_args": train_args,
            "joint_checkpoint": str(joint_checkpoint),
            "joint_logs": logs,
            "student_parameters": int(accounting["acoustic_parameters"]),
            "acoustic_init": acoustic_init,
        },
        acoustic_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": decoder_state,
            "config": decoder_config,
            "train_args": train_args,
            "joint_checkpoint": str(joint_checkpoint),
            "joint_logs": logs,
            "decoder_parameters": int(accounting["decoder_parameters"]),
            "adversarial_discriminator_parameters": 0,
            "adversarial_delta_discriminator_parameters": 0,
        },
        decoder_checkpoint,
    )
    return {
        "joint_checkpoint": joint_checkpoint,
        "acoustic_checkpoint": acoustic_checkpoint,
        "decoder_checkpoint": decoder_checkpoint,
    }


def verify_standalone_reload(paths: dict[str, Path], device: torch.device) -> dict[str, Any]:
    acoustic_model, acoustic_config = latent_trainer.load_model_from_checkpoint(
        paths["acoustic_checkpoint"],
        device,
    )
    acoustic_checkpoint = latent_trainer.load_torch_checkpoint_windows_safe(
        paths["acoustic_checkpoint"],
        "saved standalone latent checkpoint",
    )
    acoustic_strict_model = latent_trainer.create_model_from_config(acoustic_config).to(device)
    acoustic_incompatible = acoustic_strict_model.load_state_dict(
        acoustic_checkpoint["model_state_dict"],
        strict=True,
    )

    decoder_checkpoint = decoder_trainer.load_torch_checkpoint(
        paths["decoder_checkpoint"],
        "saved standalone decoder checkpoint",
    )
    decoder_config = decoder_checkpoint.get("config")
    if not isinstance(decoder_config, dict):
        raise RuntimeError(f"{paths['decoder_checkpoint']}: saved standalone decoder missing config")
    decoder_state = decoder_checkpoint.get("model_state_dict")
    if not isinstance(decoder_state, dict):
        raise RuntimeError(f"{paths['decoder_checkpoint']}: saved standalone decoder missing model_state_dict")
    decoder_model = joint_common.decoder_from_config(decoder_config).to(device)
    decoder_incompatible = decoder_model.load_state_dict(decoder_state, strict=True)

    return {
        "acoustic_load_model_from_checkpoint": {
            "ok": True,
            "checkpoint": str(paths["acoustic_checkpoint"]),
            "parameters": int(latent_trainer.count_parameters(acoustic_model)),
            "config_out_channels": int(acoustic_config.get("out_channels") or 0),
        },
        "acoustic_create_model_from_config_load_state_dict_strict": {
            "ok": True,
            "parameters": int(latent_trainer.count_parameters(acoustic_strict_model)),
            "missing_keys": list(acoustic_incompatible.missing_keys),
            "unexpected_keys": list(acoustic_incompatible.unexpected_keys),
        },
        "decoder_DecoderStudent_load_state_dict_strict": {
            "ok": True,
            "checkpoint": str(paths["decoder_checkpoint"]),
            "parameters": int(decoder_trainer.count_parameters(decoder_model)),
            "missing_keys": list(decoder_incompatible.missing_keys),
            "unexpected_keys": list(decoder_incompatible.unexpected_keys),
        },
    }


def loss_family_check(logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not logs:
        return {"ok": False, "reason": "no logs"}
    keys = ("spectral", "waveform_l1", "z_anchor")
    result: dict[str, Any] = {"ok": True}
    for key in keys:
        values = [float(log[key]) for log in logs if key in log]
        finite = all(math.isfinite(value) for value in values)
        nonzero = any(abs(value) > 0.0 for value in values)
        result[key] = {
            "finite": bool(finite),
            "nonzero": bool(nonzero),
            "values": values,
        }
        result["ok"] = bool(result["ok"] and finite and nonzero)
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = joint_common.pick_device(str(args.device))
    rows, samples, source_in_channels = decoder_trainer.load_samples(args.pack_dir, args.teacher_decoder)
    if int(source_in_channels) != 192:
        raise RuntimeError(f"{args.pack_dir}: expected 192-channel generator_input, got {source_in_channels}")

    acoustic_model, acoustic_config, acoustic_init = load_or_create_acoustic_model(args, samples=samples, device=device)
    decoder_model, decoder_config = load_piperlite_decoder_checkpoint(args.decoder_checkpoint, device)
    if int(acoustic_config.get("out_channels") or 0) != int(source_in_channels):
        raise RuntimeError(
            f"{args.acoustic_checkpoint}: acoustic out_channels {acoustic_config.get('out_channels')} "
            f"!= z channels {source_in_channels}"
        )
    if int(decoder_config.get("in_channels") or 0) != int(source_in_channels):
        raise RuntimeError(
            f"{args.decoder_checkpoint}: decoder in_channels {decoder_config.get('in_channels')} "
            f"!= pack channels {source_in_channels}"
        )
    if str(decoder_config.get("variant") or "") != "piperlite":
        raise RuntimeError(f"{args.decoder_checkpoint}: expected piperlite decoder")

    acoustic_model.train()
    decoder_model.train()

    accounting = parameter_accounting(args=args, acoustic_model=acoustic_model, decoder_model=decoder_model)
    print(json.dumps({"joint_z_acoustic_init": acoustic_init}, ensure_ascii=False), flush=True)
    print(json.dumps({"joint_z_parameter_accounting": accounting}, ensure_ascii=False), flush=True)

    optimizer_groups: list[dict[str, Any]] = [
        {
            "params": [param for param in acoustic_model.parameters() if param.requires_grad],
            "lr": float(args.acoustic_lr),
            "name": "acoustic",
        },
        {
            "params": [param for param in decoder_model.parameters() if param.requires_grad],
            "lr": float(args.decoder_lr),
            "name": "decoder",
        },
    ]
    optimizer_groups = [group for group in optimizer_groups if group["params"]]
    if not optimizer_groups:
        raise RuntimeError("joint z model has no trainable parameters")
    trainable_parameters = [param for group in optimizer_groups for param in group["params"]]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-5)

    adv_periods = decoder_trainer.parse_positive_int_tuple(str(args.adv_periods), label="--adv-periods", min_value=2)
    adv_channels = decoder_trainer.parse_positive_int_tuple(str(args.adv_channels), label="--adv-channels")
    discriminator: decoder_trainer.MultiPeriodDiscriminator | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    discriminator_parameter_count = 0
    if float(args.adv_weight) > 0.0 or float(args.adv_feature_weight) > 0.0:
        discriminator = decoder_trainer.MultiPeriodDiscriminator(adv_periods, adv_channels).to(device)
        discriminator_parameter_count = decoder_trainer.count_parameters(discriminator)
        discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=float(args.adv_lr), weight_decay=1e-5)
    delta_discriminator: decoder_trainer.MultiPeriodDiscriminator | None = None
    delta_discriminator_optimizer: torch.optim.Optimizer | None = None
    delta_discriminator_parameter_count = 0
    if float(args.adv_delta_weight) > 0.0 or float(args.adv_delta_feature_weight) > 0.0:
        delta_discriminator = decoder_trainer.MultiPeriodDiscriminator(adv_periods, adv_channels).to(device)
        delta_discriminator_parameter_count = decoder_trainer.count_parameters(delta_discriminator)
        delta_discriminator_optimizer = torch.optim.AdamW(
            delta_discriminator.parameters(),
            lr=float(args.adv_lr),
            weight_decay=1e-5,
        )
    print(
        json.dumps(
            {
                "joint_z_adversarial_parameters": {
                    "adversarial_discriminator_parameters": int(discriminator_parameter_count),
                    "adversarial_delta_discriminator_parameters": int(delta_discriminator_parameter_count),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    logs: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        z_hat, teacher_z, target = joint_z_crop_batch(
            samples,
            acoustic_model=acoustic_model,
            out_channels=int(source_in_channels),
            batch_size=int(args.batch_size),
            crop_frames=int(args.crop_frames),
            device=device,
        )
        prediction_value = decoder_model(z_hat)
        if isinstance(prediction_value, tuple):
            raise RuntimeError("joint z decoder returned features unexpectedly")
        prediction = prediction_value
        if prediction.shape != target.shape:
            raise RuntimeError(f"prediction shape {prediction.shape} != target shape {target.shape}")

        waveform_l1 = F.l1_loss(prediction, target)
        spectral = (
            decoder_trainer.multi_resolution_stft_loss(prediction, target)
            if float(args.spectral_weight) > 0.0
            else prediction.new_tensor(0.0)
        )
        z_anchor = F.l1_loss(z_hat, teacher_z)
        joint_common.finite_or_raise(waveform_l1, "waveform_l1", step)
        if float(args.spectral_weight) > 0.0:
            joint_common.finite_or_raise(spectral, "spectral", step)
        joint_common.finite_or_raise(z_anchor, "z_anchor", step)

        (
            adversarial_generator,
            adversarial_feature,
            adversarial_discriminator,
            adversarial_gate_mean,
            discriminator_grad_norm,
        ) = joint_common.run_waveform_adversarial(
            args=args,
            step=step,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            prediction=prediction,
            target=target,
        )
        (
            adversarial_delta_generator,
            adversarial_delta_feature,
            adversarial_delta_discriminator,
            adversarial_delta_gate_mean,
            delta_discriminator_grad_norm,
        ) = joint_common.run_delta_adversarial(
            args=args,
            step=step,
            discriminator=delta_discriminator,
            discriminator_optimizer=delta_discriminator_optimizer,
            prediction=prediction,
            target=target,
        )

        loss = (
            float(args.waveform_l1_weight) * waveform_l1
            + float(args.spectral_weight) * spectral
            + float(args.z_anchor_weight) * z_anchor
            + float(args.adv_weight) * adversarial_generator
            + float(args.adv_feature_weight) * adversarial_feature
            + float(args.adv_delta_weight) * adversarial_delta_generator
            + float(args.adv_delta_feature_weight) * adversarial_delta_feature
        )
        joint_common.finite_or_raise(loss, "loss", step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0).detach().cpu())
        optimizer.step()

        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            log = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "waveform_l1": float(waveform_l1.detach().cpu()),
                "waveform_l1_weight": float(args.waveform_l1_weight),
                "spectral": float(spectral.detach().cpu()),
                "spectral_weight": float(args.spectral_weight),
                "z_anchor": float(z_anchor.detach().cpu()),
                "z_anchor_weight": float(args.z_anchor_weight),
                "adversarial_generator": float(adversarial_generator.detach().cpu()),
                "adversarial_feature": float(adversarial_feature.detach().cpu()),
                "adversarial_discriminator": float(adversarial_discriminator.detach().cpu()),
                "adversarial_gate_mean": float(adversarial_gate_mean),
                "adversarial_delta_generator": float(adversarial_delta_generator.detach().cpu()),
                "adversarial_delta_feature": float(adversarial_delta_feature.detach().cpu()),
                "adversarial_delta_discriminator": float(adversarial_delta_discriminator.detach().cpu()),
                "adversarial_delta_gate_mean": float(adversarial_delta_gate_mean),
                "grad_norm": float(grad_norm),
                "discriminator_grad_norm": float(discriminator_grad_norm),
                "delta_discriminator_grad_norm": float(delta_discriminator_grad_norm),
            }
            logs.append(log)
            print(json.dumps(log, ensure_ascii=False), flush=True)

    accounting = parameter_accounting(args=args, acoustic_model=acoustic_model, decoder_model=decoder_model)
    paths = save_checkpoints(
        args=args,
        acoustic_model=acoustic_model,
        acoustic_config=acoustic_config,
        decoder_model=decoder_model,
        decoder_config=decoder_config,
        logs=logs,
        accounting=accounting,
        acoustic_init=acoustic_init,
    )
    reload_proof = verify_standalone_reload(paths, device) if bool(args.verify_standalone_reload) else None
    if reload_proof is not None:
        print(json.dumps({"standalone_reload_proof": reload_proof}, ensure_ascii=False), flush=True)

    families = loss_family_check(logs)
    print(json.dumps({"loss_family_check": families}, ensure_ascii=False), flush=True)
    report = {
        "passed": True,
        "pack_dir": str(args.pack_dir),
        "teacher_decoder": str(args.teacher_decoder),
        "acoustic_checkpoint": str(args.acoustic_checkpoint),
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "crop_frames": int(args.crop_frames),
        "source_in_channels": int(source_in_channels),
        "rows": int(len(rows)),
        "chunks": int(len(samples)),
        "acoustic_init": acoustic_init,
        "parameter_accounting": accounting,
        "optimizer_groups": [
            {
                "name": str(group["name"]),
                "lr": float(group["lr"]),
                "parameters": int(sum(param.numel() for param in group["params"])),
            }
            for group in optimizer_groups
        ],
        "adversarial_discriminator_parameters": int(discriminator_parameter_count),
        "adversarial_delta_discriminator_parameters": int(delta_discriminator_parameter_count),
        "paths": {key: str(value) for key, value in paths.items()},
        "standalone_reload_proof": reload_proof,
        "loss_family_check": families,
        "logs": logs,
        "train_args": joint_common.jsonable_args(args),
    }
    decoder_trainer.write_json(args.out_dir / "train-report.json", report)
    return report


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
