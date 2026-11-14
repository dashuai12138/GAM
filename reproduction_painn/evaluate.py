"""Evaluate a Dense or geometric-memory PaiNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from reproduction_painn.data import SystemBatchSampler, collate_frames, load_dataset
from reproduction_painn.models import MLIPConfig, build_model
from reproduction_painn.training.metrics import evaluate_model

WATER_ICE_SYSTEMS = {
    "lw_pimd": "Liquid Water",
    "ice_pimd": "Ice Ih (b)",
    "ice_triple_I": "Ice Ih (c)",
    "ice_triple_II": "Ice Ih (d)",
}


def water_ice_benchmark_metrics(
    per_system: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float | int | str]], dict[str, float]] | None:
    """Convert the four water/ice systems to the DPA3 Table S-3 convention."""
    if not set(WATER_ICE_SYSTEMS).issubset(per_system):
        return None
    rows: dict[str, dict[str, float | int | str]] = {}
    energy_rmse: list[float] = []
    force_rmse: list[float] = []
    for system_name, label in WATER_ICE_SYSTEMS.items():
        values = per_system[system_name]
        row: dict[str, float | int | str] = {
            "label": label,
            **values,
            "energy_rmse_mev_per_h2o": values["energy_rmse_per_atom_ev"] * 3_000.0,
            "force_rmse_mev_per_angstrom": values["force_rmse_ev_per_angstrom"]
            * 1_000.0,
        }
        rows[system_name] = row
        energy_rmse.append(float(row["energy_rmse_mev_per_h2o"]))
        force_rmse.append(float(row["force_rmse_mev_per_angstrom"]))
    lwarmse = {
        "energy_mev_per_h2o": math.exp(
            sum(math.log(value) for value in energy_rmse) / len(energy_rmse)
        ),
        "force_mev_per_angstrom": math.exp(
            sum(math.log(value) for value in force_rmse) / len(force_rmse)
        ),
    }
    return rows, lwarmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--domain", choices=("300K", "600K", "1200K", "dihedral"))
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--device")
    parser.add_argument("--neighbor-mode", choices=("cache", "dynamic"))
    args = parser.parse_args()

    config: dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = MLIPConfig.from_dict(config["model"])
    data_config = config["data"]
    if data_config.get("format") == "3bpa" and args.split == "test" and not args.domain:
        raise ValueError("3BPA test evaluation requires --domain")
    if args.neighbor_mode is not None:
        data_config = {**data_config, "neighbor_mode": args.neighbor_mode}
    dataset = load_dataset(
        data_config, args.split, model_config.cutoff, domain=args.domain
    )
    if dataset is None:
        raise ValueError(f"configuration has no {args.split} split")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    sampler = SystemBatchSampler(
        dataset,
        data_config.get("batch_size", "auto:512"),
        seed=int(config["training"].get("seed", 17)),
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_frames,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", device.type == "cuda")),
    )
    checkpoint_path = args.checkpoint or Path(config["output_dir"]) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate_model(
        model,
        loader,
        device,
        args.max_structures,
        group_by_system=True,
        show_progress=True,
    )
    per_system = metrics.pop("per_system")
    water_ice = water_ice_benchmark_metrics(per_system)
    if water_ice is not None:
        per_system, water_ice_lwarmse = water_ice
    else:
        water_ice_lwarmse = None
    suffix = args.domain or args.split
    result = {
        "model": config.get(
            "name", "geometric_memory" if model_config.memory_enabled else "dense"
        ),
        "seed": int(config["training"].get("seed", 42)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "test_domain": args.domain or args.split,
        "metrics": metrics,
        "per_system": per_system,
        "water_ice_lwarmse": water_ice_lwarmse,
        "system_id_map": {str(i): name for i, name in enumerate(dataset.system_names)},
    }
    output_path = checkpoint_path.parent / f"evaluation_{suffix}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    flat_result = {
        "model": result["model"],
        "seed": result["seed"],
        "epoch": result["epoch"],
        "checkpoint": result["checkpoint"],
        "test_domain": result["test_domain"],
        "energy_mae": metrics["energy_mae_per_atom_ev"],
        "energy_rmse": metrics["energy_rmse_per_atom_ev"],
        "force_mae": metrics["force_mae_ev_per_angstrom"],
        "force_rmse": metrics["force_rmse_ev_per_angstrom"],
        "force_p95": metrics["force_p95_ev_per_angstrom"],
        "force_p99": metrics["force_p99_ev_per_angstrom"],
    }
    csv_path = checkpoint_path.parent / f"evaluation_{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_result))
        writer.writeheader()
        writer.writerow(flat_result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
