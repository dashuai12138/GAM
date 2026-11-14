"""Energy/force evaluation shared by training and the evaluation CLI."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from reproduction_painn.data.neighbor import build_neighbor_graph, rebuild_rij

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from reproduction_painn.models import BackboneAgnosticMLIP


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def prepare_neighbors(
    model: BackboneAgnosticMLIP, batch: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare topology/vectors separately so the trainer can time this stage."""
    positions = batch["positions"]
    if not positions.requires_grad:
        positions.requires_grad_(True)
    if "idx_i" not in batch:
        return build_neighbor_graph(
            positions,
            batch["batch"],
            model.config.cutoff,
            batch["cell"],
            batch["pbc"],
        )
    idx_i, idx_j = batch["idx_i"], batch["idx_j"]
    r_ij = rebuild_rij(
        positions,
        batch["batch"],
        idx_i,
        idx_j,
        batch["cell"],
        batch.get("cell_shifts"),
    )
    return idx_i, idx_j, r_ij


def model_forward(
    model: BackboneAgnosticMLIP,
    batch: dict[str, Any],
    neighbors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if neighbors is None:
        neighbors = prepare_neighbors(model, batch)
    idx_i, idx_j, r_ij = neighbors
    return model(
        {
            "atomic_numbers": batch["atomic_numbers"],
            "positions": batch["positions"],
            "batch": batch["batch"],
            "cell": batch["cell"],
            "pbc": batch["pbc"],
            "idx_i": idx_i,
            "idx_j": idx_j,
            "r_ij": r_ij,
            "n_structures": batch["energy"].shape[0],
        },
        compute_forces=True,
    )


def _per_atom_norm(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values.flatten(start_dim=1), dim=-1)


def _distribution_metrics(values: list[torch.Tensor]) -> dict[str, float | int]:
    combined = torch.cat(values).flatten()
    return {
        "count": combined.numel(),
        "mean": float(combined.mean()),
        "std": float(combined.std(unbiased=False)),
        "p10": float(torch.quantile(combined, 0.10)),
        "p50": float(torch.quantile(combined, 0.50)),
        "p90": float(torch.quantile(combined, 0.90)),
        "p99": float(torch.quantile(combined, 0.99)),
        "min": float(combined.min()),
        "max": float(combined.max()),
    }


def _v2a_distribution_metrics(
    values: list[torch.Tensor],
) -> dict[str, float | int]:
    """Detailed distribution used by the short Irrep-GAM-v2a audit."""
    combined = torch.cat(values).flatten()
    return {
        "count": combined.numel(),
        "mean": float(combined.mean()),
        "std": float(combined.std(unbiased=False)),
        "p01": float(torch.quantile(combined, 0.01)),
        "p10": float(torch.quantile(combined, 0.10)),
        "p50": float(torch.quantile(combined, 0.50)),
        "p90": float(torch.quantile(combined, 0.90)),
        "p99": float(torch.quantile(combined, 0.99)),
        "min": float(combined.min()),
        "max": float(combined.max()),
    }


def _addressing_distributions(
    keys: torch.Tensor,
    queries: torch.Tensor,
    idx_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key_dim = keys.shape[-1]
    scores = (keys * queries[idx_i]).sum(dim=-1) / math.sqrt(key_dim)
    n_atoms = queries.shape[0]
    maximum = scores.new_full((n_atoms,), -torch.inf)
    maximum.scatter_reduce_(0, idx_i, scores, reduce="amax", include_self=True)
    exponentials = torch.exp(scores - maximum[idx_i])
    denominator = scores.new_zeros(n_atoms)
    denominator.index_add_(0, idx_i, exponentials)
    probabilities = exponentials / denominator[idx_i].clamp_min(1.0e-30)
    entropy = scores.new_zeros(n_atoms)
    entropy.index_add_(
        0,
        idx_i,
        -probabilities * torch.log(probabilities.clamp_min(1.0e-30)),
    )
    below_top = torch.where(
        scores < maximum[idx_i], scores, scores.new_full((), -torch.inf)
    )
    second = scores.new_full((n_atoms,), -torch.inf)
    second.scatter_reduce_(0, idx_i, below_top, reduce="amax", include_self=True)
    gap = torch.where(
        torch.isfinite(second), maximum - second, torch.zeros_like(maximum)
    )
    return scores.abs(), entropy, gap


def _memory_adaptation_metrics(
    accumulated: list[dict[str, list[torch.Tensor] | float]],
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for layer_index, values in enumerate(accumulated):
        layer_metrics: dict[str, Any] = {
            "layer": layer_index,
            "alpha": float(values["alpha"]),
            "associative": {},
        }
        for name in (
            "key_norm",
            "query_norm",
            "absolute_key_query_score",
            "score_entropy",
            "top1_top2_gap",
            "memory_write_norm",
            "memory_norm",
            "raw_retrieval_norm",
            "normalized_retrieval_norm",
            "delta_q_norm",
            "delta_q_over_q_post",
            "forget_gate",
        ):
            tensors = values[name]
            if not isinstance(tensors, list):
                raise TypeError(f"expected tensor list for {name}")
            layer_metrics["associative"][name] = _distribution_metrics(tensors)

        input_gates = values["input_gate"]
        if not isinstance(input_gates, list):
            raise TypeError("expected tensor list for input_gate")
        if input_gates:
            layer_metrics["input_gate"] = _distribution_metrics(input_gates)

        denominators = values["denominator"]
        if not isinstance(denominators, list):
            raise TypeError("expected tensor list for denominator")
        if denominators:
            denominator = torch.cat(denominators).flatten()
            normalizer_metrics: dict[str, Any] = {
                "fraction_denominator_eq_1": float((denominator == 1).float().mean()),
                "fraction_denominator_gt_1": float((denominator > 1).float().mean()),
            }
            for name in (
                "normalizer_norm",
                "normalizer_dot_query",
                "normalizer_abs_dot_query",
                "denominator",
            ):
                tensors = values[name]
                if not isinstance(tensors, list):
                    raise TypeError(f"expected tensor list for {name}")
                normalizer_metrics[name] = _distribution_metrics(tensors)
            layer_metrics["normalizer"] = normalizer_metrics
        layers.append(layer_metrics)
    return {"layers": layers}


def _irrep_memory_metrics(
    accumulated: list[dict[str, list[torch.Tensor] | float]],
) -> dict[str, Any]:
    """Summarize per-layer high-order utilization without retaining graphs."""
    layers: list[dict[str, Any]] = []
    for layer_index, values in enumerate(accumulated):
        layer_metrics: dict[str, Any] = {"layer": layer_index}
        for order in (1, 2):
            prefix = f"l{order}_"
            memory_values = values.get(prefix + "memory_norm", [])
            if not isinstance(memory_values, list) or not memory_values:
                continue
            order_metrics: dict[str, Any] = {}
            for name in (
                "memory_norm",
                "retrieval_norm",
                "current_context_norm",
                "matching",
                "delta_q_norm",
                "delta_q_over_q_post",
                "forget",
            ):
                tensors = values[prefix + name]
                if not isinstance(tensors, list):
                    raise TypeError(f"expected tensor list for {prefix + name}")
                order_metrics[name] = _distribution_metrics(tensors)
            order_metrics["alpha"] = float(values[prefix + "alpha"])
            layer_metrics[f"l{order}"] = order_metrics
        layers.append(layer_metrics)
    return {"layers": layers}


def _irrep_memory_v2a_metrics(
    accumulated: list[dict[str, list[torch.Tensor] | float]],
) -> dict[str, Any]:
    """Summarize normalized addressing and invariant self-contraction."""
    layers: list[dict[str, Any]] = []
    for layer_index, values in enumerate(accumulated):
        layer_metrics: dict[str, Any] = {"layer": layer_index}
        for name in (
            "address_score",
            "retrieval_norm",
            "self_contraction",
            "log_feature",
            "delta_q_norm",
            "delta_q_over_q_post",
            "forget",
        ):
            tensors = values[name]
            if not isinstance(tensors, list):
                raise TypeError(f"expected tensor list for v2a {name}")
            layer_metrics[name] = _v2a_distribution_metrics(tensors)
        layer_metrics["alpha"] = float(values["alpha"])
        layers.append(layer_metrics)
    return {"layers": layers}


def evaluate_model(
    model: BackboneAgnosticMLIP,
    loader: DataLoader,
    device: torch.device,
    max_structures: int | None,
    *,
    group_by_system: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    energy_absolute = energy_squared = 0.0
    force_absolute = force_squared = 0.0
    n_structures = n_force_components = 0
    force_norm_errors: list[torch.Tensor] = []
    memory_adaptation_accumulated: list[
        dict[str, list[torch.Tensor] | float]
    ] | None = None
    irrep_memory_accumulated: list[
        dict[str, list[torch.Tensor] | float]
    ] | None = None
    irrep_memory_v2a_accumulated: list[
        dict[str, list[torch.Tensor] | float]
    ] | None = None
    system_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "energy_absolute": 0.0,
            "energy_squared": 0.0,
            "force_absolute": 0.0,
            "force_squared": 0.0,
            "n_structures": 0.0,
            "n_force_components": 0.0,
        }
    )
    started = time.perf_counter()
    batches = tqdm(loader, desc="Evaluating", unit="batch", disable=not show_progress)
    for raw_batch in batches:
        batch = to_device(raw_batch, device)
        with torch.enable_grad():
            output = model_forward(model, batch)
        energy_error = (
            (output["energy"] - batch["energy"]) / batch["n_atoms"]
        ).detach()
        force_error = (output["forces"] - batch["forces"]).detach()
        force_norm_errors.append(torch.linalg.vector_norm(force_error, dim=-1).cpu())
        memory_diagnostics = output.get("memory_adaptation_diagnostics")
        if memory_diagnostics is not None:
            if memory_adaptation_accumulated is None:
                metric_names = (
                    "input_gate",
                    "normalizer_norm",
                    "normalizer_dot_query",
                    "normalizer_abs_dot_query",
                    "denominator",
                    "key_norm",
                    "query_norm",
                    "absolute_key_query_score",
                    "score_entropy",
                    "top1_top2_gap",
                    "memory_write_norm",
                    "memory_norm",
                    "raw_retrieval_norm",
                    "normalized_retrieval_norm",
                    "delta_q_norm",
                    "delta_q_over_q_post",
                    "forget_gate",
                )
                memory_adaptation_accumulated = [
                    {**{name: [] for name in metric_names}, "alpha": 0.0}
                    for _ in memory_diagnostics
                ]
            for layer_values, layer_diagnostics in zip(
                memory_adaptation_accumulated, memory_diagnostics, strict=True
            ):
                layer_values["alpha"] = float(
                    layer_diagnostics["alpha"].detach().cpu()
                )
                tensor_names = {
                    "key_norm": "keys",
                    "query_norm": "queries",
                    "memory_write_norm": "memory_write",
                    "memory_norm": "memory",
                    "raw_retrieval_norm": "raw_retrieval",
                    "normalized_retrieval_norm": "retrieval",
                    "delta_q_norm": "delta",
                    "forget_gate": "forget_gate",
                }
                for metric_name, tensor_name in tensor_names.items():
                    tensors = layer_values[metric_name]
                    if not isinstance(tensors, list):
                        raise TypeError(f"expected tensor list for {metric_name}")
                    tensors.append(
                        _per_atom_norm(layer_diagnostics[tensor_name].detach()).cpu()
                    )
                delta_norm = _per_atom_norm(layer_diagnostics["delta"].detach())
                q_post_norm = _per_atom_norm(
                    layer_diagnostics["q_post"].detach()
                ).clamp_min(1.0e-12)
                ratio_values = layer_values["delta_q_over_q_post"]
                if not isinstance(ratio_values, list):
                    raise TypeError("expected tensor list for delta_q_over_q_post")
                ratio_values.append((delta_norm / q_post_norm).cpu())

                absolute_score, entropy, gap = _addressing_distributions(
                    layer_diagnostics["keys"].detach(),
                    layer_diagnostics["queries"].detach(),
                    layer_diagnostics["idx_i"],
                )
                for metric_name, tensor in (
                    ("absolute_key_query_score", absolute_score),
                    ("score_entropy", entropy),
                    ("top1_top2_gap", gap),
                ):
                    tensors = layer_values[metric_name]
                    if not isinstance(tensors, list):
                        raise TypeError(f"expected tensor list for {metric_name}")
                    tensors.append(tensor.cpu())

                if "input_gate" in layer_diagnostics:
                    gates = layer_values["input_gate"]
                    if not isinstance(gates, list):
                        raise TypeError("expected tensor list for input_gate")
                    gates.append(layer_diagnostics["input_gate"].detach().flatten().cpu())
                if "normalizer" in layer_diagnostics:
                    normalizer = layer_diagnostics["normalizer"].detach()
                    normalizer_dot = layer_diagnostics[
                        "normalizer_dot_query"
                    ].detach().flatten()
                    normalizer_tensors = {
                        "normalizer_norm": _per_atom_norm(normalizer),
                        "normalizer_dot_query": normalizer_dot,
                        "normalizer_abs_dot_query": normalizer_dot.abs(),
                        "denominator": layer_diagnostics["denominator"]
                        .detach()
                        .flatten(),
                    }
                    for metric_name, tensor in normalizer_tensors.items():
                        tensors = layer_values[metric_name]
                        if not isinstance(tensors, list):
                            raise TypeError(f"expected tensor list for {metric_name}")
                        tensors.append(tensor.cpu())
        irrep_diagnostics = output.get("irrep_memory_diagnostics")
        if irrep_diagnostics is not None:
            if irrep_memory_accumulated is None:
                irrep_memory_accumulated = []
                for layer_diagnostics in irrep_diagnostics:
                    layer_values: dict[str, list[torch.Tensor] | float] = {}
                    for order in (1, 2):
                        if f"l{order}_memory" not in layer_diagnostics:
                            continue
                        for name in (
                            "memory_norm",
                            "retrieval_norm",
                            "current_context_norm",
                            "matching",
                            "delta_q_norm",
                            "delta_q_over_q_post",
                            "forget",
                        ):
                            layer_values[f"l{order}_{name}"] = []
                        layer_values[f"l{order}_alpha"] = 0.0
                    irrep_memory_accumulated.append(layer_values)
            for layer_values, layer_diagnostics in zip(
                irrep_memory_accumulated, irrep_diagnostics, strict=True
            ):
                q_post_norm = _per_atom_norm(
                    layer_diagnostics["q_post"].detach()
                ).clamp_min(1.0e-12)
                for order in (1, 2):
                    prefix = f"l{order}_"
                    if prefix + "memory" not in layer_diagnostics:
                        continue
                    tensors_by_name = {
                        "memory_norm": _per_atom_norm(
                            layer_diagnostics[prefix + "memory"].detach()
                        ),
                        "retrieval_norm": _per_atom_norm(
                            layer_diagnostics[prefix + "retrieval"].detach()
                        ),
                        "current_context_norm": _per_atom_norm(
                            layer_diagnostics[prefix + "current"].detach()
                        ),
                        "matching": layer_diagnostics[prefix + "matching"]
                        .detach()
                        .flatten(),
                        "delta_q_norm": _per_atom_norm(
                            layer_diagnostics[prefix + "delta"].detach()
                        ),
                        "forget": layer_diagnostics[prefix + "forget"]
                        .detach()
                        .flatten(),
                    }
                    tensors_by_name["delta_q_over_q_post"] = (
                        tensors_by_name["delta_q_norm"] / q_post_norm
                    )
                    for name, tensor in tensors_by_name.items():
                        stored = layer_values[prefix + name]
                        if not isinstance(stored, list):
                            raise TypeError(f"expected tensor list for {prefix + name}")
                        stored.append(tensor.cpu())
                    layer_values[prefix + "alpha"] = float(
                        layer_diagnostics[f"alpha{order}"].detach().cpu()
                    )
        v2a_diagnostics = output.get("irrep_memory_v2a_diagnostics")
        if v2a_diagnostics is not None:
            if irrep_memory_v2a_accumulated is None:
                names = (
                    "address_score",
                    "retrieval_norm",
                    "self_contraction",
                    "log_feature",
                    "delta_q_norm",
                    "delta_q_over_q_post",
                    "forget",
                )
                irrep_memory_v2a_accumulated = [
                    {**{name: [] for name in names}, "alpha": 0.0}
                    for _ in v2a_diagnostics
                ]
            for layer_values, layer_diagnostics in zip(
                irrep_memory_v2a_accumulated,
                v2a_diagnostics,
                strict=True,
            ):
                q_post_norm = _per_atom_norm(
                    layer_diagnostics["q_post"].detach()
                ).clamp_min(1.0e-12)
                delta_norm = _per_atom_norm(
                    layer_diagnostics["l1_delta"].detach()
                )
                tensors_by_name = {
                    "address_score": layer_diagnostics["address_score"]
                    .detach()
                    .flatten(),
                    "retrieval_norm": _per_atom_norm(
                        layer_diagnostics["l1_retrieval"].detach()
                    ),
                    "self_contraction": layer_diagnostics[
                        "l1_self_contraction"
                    ]
                    .detach()
                    .flatten(),
                    "log_feature": layer_diagnostics["l1_log_feature"]
                    .detach()
                    .flatten(),
                    "delta_q_norm": delta_norm,
                    "delta_q_over_q_post": delta_norm / q_post_norm,
                    "forget": layer_diagnostics["l1_forget"]
                    .detach()
                    .flatten(),
                }
                for name, tensor in tensors_by_name.items():
                    stored = layer_values[name]
                    if not isinstance(stored, list):
                        raise TypeError(f"expected tensor list for v2a {name}")
                    stored.append(tensor.cpu())
                layer_values["alpha"] = float(
                    layer_diagnostics["alpha1"].detach().cpu()
                )
        energy_absolute += float(energy_error.abs().sum().cpu())
        energy_squared += float(energy_error.square().sum().cpu())
        force_absolute += float(force_error.abs().sum().cpu())
        force_squared += float(force_error.square().sum().cpu())
        n_structures += energy_error.numel()
        n_force_components += force_error.numel()
        if group_by_system:
            energy_cpu = energy_error.cpu()
            force_cpu = force_error.cpu()
            atom_batch = batch["batch"].detach().cpu()
            system_names = batch["system_name"]
            for system_name in dict.fromkeys(system_names):
                structure_mask = torch.tensor(
                    [name == system_name for name in system_names], dtype=torch.bool
                )
                system_energy = energy_cpu[structure_mask]
                system_force = force_cpu[structure_mask[atom_batch]]
                totals = system_totals[system_name]
                totals["energy_absolute"] += float(system_energy.abs().sum())
                totals["energy_squared"] += float(system_energy.square().sum())
                totals["force_absolute"] += float(system_force.abs().sum())
                totals["force_squared"] += float(system_force.square().sum())
                totals["n_structures"] += float(system_energy.numel())
                totals["n_force_components"] += float(system_force.numel())
        if max_structures is not None and n_structures >= max_structures:
            break
    elapsed = time.perf_counter() - started
    force_norm_error = torch.cat(force_norm_errors)
    metrics: dict[str, Any] = {
        "energy_mae_per_atom_ev": energy_absolute / n_structures,
        "energy_rmse_per_atom_ev": (energy_squared / n_structures) ** 0.5,
        "force_mae_ev_per_angstrom": force_absolute / n_force_components,
        "force_rmse_ev_per_angstrom": (force_squared / n_force_components) ** 0.5,
        "force_p95_ev_per_angstrom": float(torch.quantile(force_norm_error, 0.95)),
        "force_p99_ev_per_angstrom": float(torch.quantile(force_norm_error, 0.99)),
        "inference_structures_per_s": n_structures / elapsed,
        "evaluated_structures": n_structures,
    }
    if group_by_system:
        metrics["per_system"] = {
            name: {
                "energy_mae_per_atom_ev": values["energy_absolute"]
                / values["n_structures"],
                "energy_rmse_per_atom_ev": (
                    values["energy_squared"] / values["n_structures"]
                )
                ** 0.5,
                "force_mae_ev_per_angstrom": values["force_absolute"]
                / values["n_force_components"],
                "force_rmse_ev_per_angstrom": (
                    values["force_squared"] / values["n_force_components"]
                )
                ** 0.5,
                "evaluated_structures": int(values["n_structures"]),
            }
            for name, values in system_totals.items()
        }
    if memory_adaptation_accumulated is not None:
        metrics["memory_adaptation"] = _memory_adaptation_metrics(
            memory_adaptation_accumulated
        )
    if irrep_memory_accumulated is not None:
        metrics["irrep_memory"] = _irrep_memory_metrics(
            irrep_memory_accumulated
        )
    if irrep_memory_v2a_accumulated is not None:
        metrics["irrep_memory_v2a"] = _irrep_memory_v2a_metrics(
            irrep_memory_v2a_accumulated
        )
    model.train(was_training)
    return metrics


__all__ = ["evaluate_model", "model_forward", "prepare_neighbors", "to_device"]
