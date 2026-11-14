"""Single-GPU FP32 trainer for Dense PaiNN and its GAM extension."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from reproduction_painn.data import (
    DeepMDNpyDataset,
    SystemBatchSampler,
    ThreeBPADataset,
    collate_frames,
    fit_atomref,
    load_dataset,
)
from reproduction_painn.models import BackboneAgnosticMLIP, MLIPConfig, build_model

from .metrics import evaluate_model, model_forward, prepare_neighbors, to_device

if TYPE_CHECKING:
    from collections.abc import Iterator


def learning_rate_at_step(
    step: int,
    total_steps: int,
    start: float,
    stop: float | None,
    decay_steps: int,
) -> float:
    """DeePMD-style staircase exponential schedule."""
    if stop is None:
        return start
    if not 0.0 < stop <= start:
        raise ValueError("stop_learning_rate must be in (0, learning_rate]")
    if decay_steps < 1:
        raise ValueError("decay_steps must be positive")
    decay_rate = (stop / start) ** (decay_steps / total_steps)
    return start * decay_rate ** (min(step, total_steps) // decay_steps)


class StaircaseExponentialScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        start: float,
        stop: float | None,
        decay_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.start = start
        self.stop = stop
        self.decay_steps = decay_steps
        self.last_step = 0
        self.group_scales = [
            float(group["lr"]) / start for group in optimizer.param_groups
        ]

    def step(self, step: int) -> float:
        self.last_step = step
        learning_rate = learning_rate_at_step(
            step, self.total_steps, self.start, self.stop, self.decay_steps
        )
        for group, scale in zip(
            self.optimizer.param_groups, self.group_scales, strict=True
        ):
            group["lr"] = learning_rate * scale
        return learning_rate

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step(int(state["last_step"]))


def _infinite(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _dataset(
    data_config: dict[str, Any], split: str, model_config: MLIPConfig
) -> DeepMDNpyDataset | ThreeBPADataset | None:
    return load_dataset(data_config, split, model_config.cutoff)


def _loader(
    dataset: DeepMDNpyDataset | ThreeBPADataset,
    data_config: dict[str, Any],
    device: torch.device,
    seed: int,
    train: bool,
) -> tuple[DataLoader, SystemBatchSampler]:
    sampler = SystemBatchSampler(
        dataset,
        batch_size=data_config.get("batch_size", "auto:512"),
        seed=seed,
        shuffle=train,
        sampling=data_config.get("sampling", "proportional"),
    )
    workers = int(data_config.get("num_workers", 0))
    options: dict[str, Any] = {
        "batch_sampler": sampler,
        "collate_fn": collate_frames,
        "num_workers": workers,
        "pin_memory": bool(data_config.get("pin_memory", device.type == "cuda")),
    }
    if workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
    return DataLoader(dataset, **options), sampler


def save_checkpoint(
    path: Path,
    model: BackboneAgnosticMLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: StaircaseExponentialScheduler,
    sampler: SystemBatchSampler,
    step: int,
    best_metrics: dict[str, float],
    config: dict[str, Any],
    *,
    epoch: int,
    epochs_without_improvement: int = 0,
) -> None:
    """Atomically save all state needed for deterministic resume.

    Step stays inside the checkpoint because optimizers and schedulers update per
    batch. User-facing JSONL logs remain epoch-only.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "step": step,
        "global_step": step,
        "epoch": epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "best_metric": best_metrics["force"],
        "best_metrics": dict(best_metrics),
        "config": config,
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: BackboneAgnosticMLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: StaircaseExponentialScheduler,
    sampler: SystemBatchSampler,
    device: torch.device,
) -> tuple[int, int, int, dict[str, float], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(
        checkpoint.get("scheduler", {"last_step": checkpoint["step"]})
    )
    if "sampler" in checkpoint:
        sampler.load_state_dict(checkpoint["sampler"])
    elif "sampler_generator" in checkpoint:
        sampler.generator.set_state(checkpoint["sampler_generator"].cpu())
    if "python_rng" in checkpoint:
        random.setstate(checkpoint["python_rng"])
    if "numpy_rng" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng"])
    if "torch_rng" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if device.type == "cuda" and "cuda_rng" in checkpoint:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    saved_best = checkpoint.get("best_metrics")
    if saved_best is None:
        best_metrics = {
            "force": float(checkpoint.get("best_metric", math.inf)),
            "energy": math.inf,
            "joint": math.inf,
        }
    else:
        best_metrics = {
            name: float(saved_best.get(name, math.inf))
            for name in ("force", "energy", "joint")
        }
    return (
        int(checkpoint.get("global_step", checkpoint["step"])),
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("epochs_without_improvement", 0)),
        best_metrics,
        checkpoint["config"],
    )


def train_from_config(
    config: dict[str, Any],
    device_name: str | None = None,
    steps_override: int | None = None,
    epochs_override: int | None = None,
    output_dir_override: str | Path | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    training = config["training"]
    model_config = MLIPConfig.from_dict(config["model"])
    configured_epochs = training.get("epochs")
    configured_steps = training.get("steps")
    if configured_epochs is None and configured_steps is None:
        raise ValueError("training config requires epochs or steps")

    output_dir = Path(output_dir_override or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads(json.dumps(config))
    run_config["output_dir"] = str(output_dir)

    seed = int(training.get("seed", 17))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.set_float32_matmul_precision(training.get("matmul_precision", "high"))

    data_config = config["data"]
    train_dataset = _dataset(data_config, "train", model_config)
    validation_dataset = _dataset(data_config, "validation", model_config)
    if train_dataset is None or validation_dataset is None:
        raise ValueError("training requires train and validation datasets")
    minimum_faces = {
        "train": train_dataset.validate_minimum_image(model_config.cutoff),
        "validation": validation_dataset.validate_minimum_image(model_config.cutoff),
    }
    train_loader, train_sampler = _loader(
        train_dataset, data_config, device, seed, train=True
    )
    validation_loader, _ = _loader(
        validation_dataset, data_config, device, seed, train=False
    )
    steps_per_epoch = len(train_loader)
    if epochs_override is not None:
        max_epochs = int(epochs_override)
    elif configured_epochs is not None:
        max_epochs = int(configured_epochs)
    else:
        max_epochs = math.ceil(int(configured_steps) / steps_per_epoch)
    configured_schedule_steps = training.get("schedule_steps")
    schedule_steps = (
        int(configured_schedule_steps)
        if configured_schedule_steps is not None
        else (
            int(configured_steps)
            if configured_steps is not None and configured_epochs is None
            else max_epochs * steps_per_epoch
        )
    )
    target_steps = (
        int(steps_override)
        if steps_override is not None
        else max_epochs * steps_per_epoch
    )
    run_config["resolved_training"] = {
        "max_epochs": max_epochs,
        "steps_per_epoch": steps_per_epoch,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    atomref, energy_scale = (
        fit_atomref(train_dataset, model_config.max_atomic_number)
        if data_config.get("fit_atomref", True)
        else (torch.zeros(model_config.max_atomic_number + 1), 1.0)
    )
    model = build_model(model_config, atomref, energy_scale).to(device)
    backbone_checkpoint = training.get("initial_backbone_checkpoint")
    if backbone_checkpoint is not None and resume is None:
        checkpoint = torch.load(
            Path(backbone_checkpoint), map_location=device, weights_only=False
        )
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in checkpoint["model"].items()
            if key.startswith("backbone.")
        }
        if not backbone_state:
            raise ValueError(
                f"checkpoint contains no backbone parameters: {backbone_checkpoint}"
            )
        model.backbone.load_state_dict(backbone_state, strict=False)

    start_lr = float(training.get("learning_rate", 1.0e-3))
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(model.parameters()),
                "lr": float(training.get("backbone_lr", start_lr)),
                "name": "model",
            }
        ],
        lr=start_lr,
        weight_decay=float(training.get("weight_decay", 0.0)),
        fused=bool(training.get("fused_optimizer", True)) and device.type == "cuda",
    )
    scheduler = StaircaseExponentialScheduler(
        optimizer,
        schedule_steps,
        start_lr,
        training.get("stop_learning_rate"),
        int(training.get("decay_steps", 5_000)),
    )
    start_step = 0
    resumed_epoch = 0
    epochs_without_improvement = 0
    best_metrics = {"force": math.inf, "energy": math.inf, "joint": math.inf}
    primary_metric = str(training.get("primary_checkpoint_metric", "force"))
    if primary_metric not in best_metrics:
        raise ValueError("primary_checkpoint_metric must be force, energy, or joint")
    if resume is not None:
        (
            start_step,
            resumed_epoch,
            epochs_without_improvement,
            best_metrics,
            saved_config,
        ) = load_checkpoint(
            Path(resume), model, optimizer, scheduler, train_sampler, device
        )
        if saved_config["model"] != config["model"]:
            raise ValueError("checkpoint and requested model configurations differ")
    if start_step >= target_steps:
        raise ValueError(
            f"checkpoint at epoch {resumed_epoch} already reached the requested target"
        )

    energy_weight = float(training.get("energy_weight", 1.0))
    force_weight = float(training.get("force_weight", 100.0))
    clip_norm = float(training.get("gradient_clip_norm", 10.0))
    validation_every = int(
        training.get(
            "validation_every",
            int(training.get("validation_every_epochs", 1)) * steps_per_epoch,
        )
    )
    checkpoint_every = int(
        training.get(
            "checkpoint_every",
            int(training.get("checkpoint_every_epochs", 10)) * steps_per_epoch,
        )
    )
    max_validation = int(training.get("max_validation_structures", 64))
    early_stopping = bool(training.get("early_stopping", False))
    patience_epochs = int(training.get("patience_epochs", 256))

    writer = SummaryWriter(output_dir / "tensorboard")
    log_path = output_dir / "train.jsonl"
    iterator = _infinite(train_loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timing = {
        name: 0.0 for name in ("data", "neighbor", "forward", "backward", "optimizer")
    }
    structures_seen = 0
    latest_loss = math.nan
    validation_metrics: dict[str, float] = {}
    epoch_sums = {
        name: torch.zeros((), device=device)
        for name in ("loss", "energy_mse_per_atom", "force_mse", "gradient_norm")
    }
    v2a_memory = model.backbone.irrep_memory_v2a
    v2a_gradient_parameters: dict[str, list[torch.nn.Parameter]] = {}
    if v2a_memory is not None:
        if (
            v2a_memory.l1_amplitude is None
            or v2a_memory.l1_projection is None
            or v2a_memory.alpha1 is None
        ):
            raise RuntimeError("incomplete Irrep-GAM-v2a parameterization")
        v2a_gradient_parameters = {
            "K": list(v2a_memory.geo_key.parameters()),
            "Q": list(v2a_memory.geo_query.parameters()),
            "A1": list(v2a_memory.l1_amplitude.parameters()),
            "P1": list(v2a_memory.l1_projection.parameters()),
            "alpha1": [v2a_memory.alpha1],
        }
        epoch_sums.update(
            {
                f"v2a_gradient_{name}": torch.zeros((), device=device)
                for name in v2a_gradient_parameters
            }
        )
    epoch_batches = 0
    actual_step = start_step
    run_started = time.perf_counter()
    print(
        json.dumps(
            {
                "device": str(device),
                "frames": {
                    "train": len(train_dataset),
                    "validation": len(validation_dataset),
                },
                "minimum_cell_face_angstrom": minimum_faces,
                "parameters": model.parameter_counts(),
                "memory_enabled": model_config.memory_enabled,
                "memory_input_gate_enabled": (
                    model_config.memory_input_gate_enabled
                ),
                "memory_normalizer_enabled": model_config.memory_normalizer_enabled,
                "memory_normalized_addressing_enabled": (
                    model_config.memory_normalized_addressing_enabled
                ),
                "memory_environment_query_enabled": (
                    model_config.memory_environment_query_enabled
                ),
                "preaggregation_memory_enabled": (
                    model_config.preaggregation_memory_enabled
                ),
                "irrep_memory_v2a_enabled": model_config.irrep_memory_v2a_enabled,
                "start_epoch": resumed_epoch,
            }
        ),
        flush=True,
    )

    for step in range(start_step + 1, target_steps + 1):
        actual_step = step
        epoch = (step - 1) // steps_per_epoch + 1
        started = time.perf_counter()
        batch = to_device(next(iterator), device)
        timing["data"] += time.perf_counter() - started
        learning_rate = scheduler.step(step)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        neighbors = prepare_neighbors(model, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["neighbor"] += time.perf_counter() - started
        started = time.perf_counter()
        output = model_forward(model, batch, neighbors)
        energy_error = (output["energy"] - batch["energy"]) / batch["n_atoms"]
        force_error = output["forces"] - batch["forces"]
        energy_loss = energy_error.square().mean()
        force_loss = force_error.square().mean()
        loss = energy_weight * energy_loss + force_weight * force_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in epoch {epoch}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["forward"] += time.perf_counter() - started
        started = time.perf_counter()
        loss.backward()
        for name, parameters in v2a_gradient_parameters.items():
            gradient_terms = [
                parameter.grad.detach().square().sum()
                for parameter in parameters
                if parameter.grad is not None
            ]
            component_norm = (
                torch.stack(gradient_terms).sum().sqrt()
                if gradient_terms
                else loss.new_zeros(())
            )
            epoch_sums[f"v2a_gradient_{name}"] += component_norm
        if clip_norm > 0.0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_norm
            )
        else:
            gradient_terms = [
                parameter.grad.detach().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            gradient_norm = (
                torch.stack(gradient_terms).sum().sqrt()
                if gradient_terms
                else loss.new_zeros(())
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["backward"] += time.perf_counter() - started
        started = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["optimizer"] += time.perf_counter() - started

        structures_seen += int(batch["energy"].shape[0])
        epoch_sums["loss"] += loss.detach()
        epoch_sums["energy_mse_per_atom"] += energy_loss.detach()
        epoch_sums["force_mse"] += force_loss.detach()
        epoch_sums["gradient_norm"] += gradient_norm.detach()
        epoch_batches += 1

        epoch_finished = step % steps_per_epoch == 0 or step == target_steps
        if epoch_finished:
            means = {
                name: float((value / epoch_batches).cpu())
                for name, value in epoch_sums.items()
            }
            latest_loss = means["loss"]
            training_seconds = sum(timing.values())
            completed_epochs = (step - start_step) / steps_per_epoch
            record = {
                "epoch": epoch,
                "loss": latest_loss,
                "energy_mse_per_atom": means["energy_mse_per_atom"],
                "force_mse": means["force_mse"],
                "learning_rate": learning_rate,
                "gradient_clip_norm": clip_norm,
                "total_gradient_norm_before_clip": means["gradient_norm"],
                "training_epochs_per_s": completed_epochs / training_seconds,
                "training_structures_per_s": structures_seen / training_seconds,
                "timing_seconds": dict(timing),
                "seconds_per_epoch": training_seconds / max(completed_epochs, 1.0e-12),
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                ),
            }
            if v2a_gradient_parameters:
                record["v2a_gradient_norms"] = {
                    name: means[f"v2a_gradient_{name}"]
                    for name in v2a_gradient_parameters
                }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            for key in ("loss", "energy_mse_per_atom", "force_mse", "learning_rate"):
                writer.add_scalar(f"train/{key}", record[key], epoch)
            writer.add_scalar(
                "train/gradient_norm_before_clip", means["gradient_norm"], epoch
            )
            writer.add_scalar(
                "performance/sec_per_epoch", record["seconds_per_epoch"], epoch
            )
            writer.add_scalar(
                "performance/gpu_memory", record["peak_gpu_memory_bytes"], epoch
            )
            print(json.dumps(record), flush=True)
            for value in epoch_sums.values():
                value.zero_()
            epoch_batches = 0

        if step % validation_every == 0 or step == target_steps:
            validation_metrics = evaluate_model(
                model, validation_loader, device, max_validation
            )
            candidates = {
                "force": validation_metrics["force_rmse_ev_per_angstrom"],
                "energy": validation_metrics["energy_rmse_per_atom_ev"],
                "joint": math.sqrt(
                    validation_metrics["force_rmse_ev_per_angstrom"]
                    * validation_metrics["energy_rmse_per_atom_ev"]
                ),
            }
            improved = [
                name for name, value in candidates.items() if value < best_metrics[name]
            ]
            for name in improved:
                best_metrics[name] = candidates[name]
            next_stale = 0 if primary_metric in improved else epochs_without_improvement
            for name in improved:
                save_checkpoint(
                    output_dir / f"best_{name}.pt",
                    model,
                    optimizer,
                    scheduler,
                    train_sampler,
                    step,
                    best_metrics,
                    run_config,
                    epoch=epoch,
                    epochs_without_improvement=next_stale,
                )
            if primary_metric in improved:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    train_sampler,
                    step,
                    best_metrics,
                    run_config,
                    epoch=epoch,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += max(
                    1, validation_every // steps_per_epoch
                )
            for key, value in validation_metrics.items():
                if isinstance(value, int | float):
                    writer.add_scalar(f"validation/{key}", value, epoch)
            validation_record = {"epoch": epoch, "validation": validation_metrics}
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(validation_record) + "\n")
            print(json.dumps(validation_record), flush=True)
            if early_stopping and epochs_without_improvement >= patience_epochs:
                print(json.dumps({"early_stopping": True, "epoch": epoch}), flush=True)
                break

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                train_sampler,
                step,
                best_metrics,
                run_config,
                epoch=epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
        milestone_epochs = {
            int(value)
            for value in training.get("milestone_checkpoint_epochs", [])
        }
        if epoch_finished and epoch in milestone_epochs:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                train_sampler,
                step,
                best_metrics,
                run_config,
                epoch=epoch,
                epochs_without_improvement=epochs_without_improvement,
            )

    final_epoch = (actual_step - 1) // steps_per_epoch + 1
    save_checkpoint(
        output_dir / "latest.pt",
        model,
        optimizer,
        scheduler,
        train_sampler,
        actual_step,
        best_metrics,
        run_config,
        epoch=final_epoch,
        epochs_without_improvement=epochs_without_improvement,
    )
    training_seconds = sum(timing.values())
    summary = {
        "epoch": final_epoch,
        "target_epochs": max_epochs,
        "resumed_from_epoch": resumed_epoch,
        "final_loss": latest_loss,
        "validation": validation_metrics,
        "best_force_rmse_ev_per_angstrom": best_metrics["force"],
        "best_energy_rmse_per_atom_ev": best_metrics["energy"],
        "best_joint_rmse": best_metrics["joint"],
        "primary_checkpoint_metric": primary_metric,
        "parameter_counts": model.parameter_counts(),
        "training_epochs_per_s": ((actual_step - start_step) / steps_per_epoch)
        / training_seconds,
        "training_structures_per_s": structures_seen / training_seconds,
        "timing_seconds": timing,
        "wall_seconds": time.perf_counter() - run_started,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "system_id_map": {
            str(index): name
            for index, name in enumerate(validation_dataset.system_names)
        },
        "model_config": model_config.to_dict(),
        "energy_scale": energy_scale,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    writer.close()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


__all__ = [
    "StaircaseExponentialScheduler",
    "learning_rate_at_step",
    "load_checkpoint",
    "save_checkpoint",
    "train_from_config",
]
