"""Neighbor-list construction, compact disk cache, and differentiable vectors."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable


CACHE_VERSION = 1


@dataclass(frozen=True)
class NeighborCache:
    """Memory-mapped ragged neighbors for one DeePMD set."""

    edge_ptr: np.ndarray
    idx_i: np.ndarray
    idx_j: np.ndarray
    cell_shifts: np.ndarray

    def frame(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, stop = int(self.edge_ptr[frame]), int(self.edge_ptr[frame + 1])
        return (
            self.idx_i[start:stop],
            self.idx_j[start:stop],
            self.cell_shifts[start:stop],
        )


def cache_path(cache_root: str | Path, system_name: str, set_name: str) -> Path:
    return Path(cache_root) / system_name / set_name


def load_neighbor_cache(
    cache_root: str | Path,
    system_name: str,
    set_name: str,
    cutoff: float,
    n_frames: int,
    n_atoms: int,
) -> NeighborCache | None:
    directory = cache_path(cache_root, system_name, set_name)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "version": CACHE_VERSION,
        "cutoff": float(cutoff),
        "n_frames": n_frames,
        "n_atoms": n_atoms,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"stale neighbor cache {directory}: {key}={metadata.get(key)!r}, "
                f"expected {value!r}"
            )
    return NeighborCache(
        edge_ptr=np.load(directory / "edge_ptr.npy", mmap_mode="r"),
        idx_i=np.load(directory / "idx_i.npy", mmap_mode="r"),
        idx_j=np.load(directory / "idx_j.npy", mmap_mode="r"),
        cell_shifts=np.load(directory / "cell_shifts.npy", mmap_mode="r"),
    )


def vesin_neighbors(
    positions: np.ndarray, cell: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the directed Vesin list and integer periodic cell shifts."""
    try:
        import vesin
    except ImportError as error:
        raise RuntimeError(
            "Vesin is required for cached neighbor lists; install requirements.txt"
        ) from error
    neighbors = vesin.NeighborList(cutoff=float(cutoff), full_list=True, sorted=True)
    points = np.asarray(positions, dtype=np.float64)
    box = np.asarray(cell, dtype=np.float64)
    idx_i, idx_j, shifts = neighbors.compute(
        points, box, periodic=True, quantities="ijS"
    )
    if positions.shape[0] > np.iinfo(np.uint16).max:
        raise ValueError("uint16 neighbor cache supports at most 65535 atoms/frame")
    return (
        np.asarray(idx_i, dtype=np.uint16),
        np.asarray(idx_j, dtype=np.uint16),
        np.asarray(shifts, dtype=np.int8),
    )


def precompute_set_neighbors(
    coordinates: np.ndarray,
    cells: np.ndarray,
    destination: str | Path,
    cutoff: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one compact ragged cache atomically at set granularity."""
    destination = Path(destination)
    if (destination / "metadata.json").is_file() and not overwrite:
        metadata = json.loads(
            (destination / "metadata.json").read_text(encoding="utf-8")
        )
        expected = {
            "version": CACHE_VERSION,
            "cutoff": float(cutoff),
            "n_frames": int(coordinates.shape[0]),
            "n_atoms": int(coordinates.shape[1]),
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            return metadata
        raise ValueError(
            f"stale neighbor cache at {destination}; rerun with --overwrite"
        )
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    i_parts: list[np.ndarray] = []
    j_parts: list[np.ndarray] = []
    shift_parts: list[np.ndarray] = []
    edge_ptr = np.zeros(coordinates.shape[0] + 1, dtype=np.int64)
    for frame in range(coordinates.shape[0]):
        idx_i, idx_j, shifts = vesin_neighbors(
            np.asarray(coordinates[frame]), np.asarray(cells[frame]), cutoff
        )
        i_parts.append(idx_i)
        j_parts.append(idx_j)
        shift_parts.append(shifts)
        edge_ptr[frame + 1] = edge_ptr[frame] + idx_i.shape[0]
    idx_i_all = np.concatenate(i_parts) if i_parts else np.empty(0, dtype=np.uint16)
    idx_j_all = np.concatenate(j_parts) if j_parts else np.empty(0, dtype=np.uint16)
    shifts_all = (
        np.concatenate(shift_parts) if shift_parts else np.empty((0, 3), dtype=np.int8)
    )
    np.save(temporary / "edge_ptr.npy", edge_ptr)
    np.save(temporary / "idx_i.npy", idx_i_all)
    np.save(temporary / "idx_j.npy", idx_j_all)
    np.save(temporary / "cell_shifts.npy", shifts_all)
    metadata = {
        "version": CACHE_VERSION,
        "backend": "vesin",
        "cutoff": float(cutoff),
        "n_frames": int(coordinates.shape[0]),
        "n_atoms": int(coordinates.shape[1]),
        "n_edges": int(idx_i_all.shape[0]),
        "bytes": int(
            edge_ptr.nbytes + idx_i_all.nbytes + idx_j_all.nbytes + shifts_all.nbytes
        ),
        "offset_encoding": "integer_cell_shift",
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return metadata


def precompute_dataset_neighbors(
    systems: Iterable[Any],
    cache_root: str | Path,
    cutoff: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    total_frames = total_edges = total_bytes = 0
    details: list[dict[str, Any]] = []
    items = [(system, values) for system in systems for values in system.sets]
    for system, values in tqdm(items, desc="Vesin neighbor caches", unit="set"):
        metadata = precompute_set_neighbors(
            values.coordinates,
            values.cells,
            cache_path(cache_root, system.cache_key, values.name),
            cutoff,
            overwrite,
        )
        details.append(
            {
                "system": system.name,
                "cache_key": system.cache_key,
                "set": values.name,
                **metadata,
            }
        )
        total_frames += int(metadata["n_frames"])
        total_edges += int(metadata["n_edges"])
        total_bytes += int(metadata["bytes"])
    return {
        "frames": total_frames,
        "edges": total_edges,
        "bytes": total_bytes,
        "sets": details,
    }


def rebuild_rij(
    positions: torch.Tensor,
    batch: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    cell: torch.Tensor | None,
    cell_shifts: torch.Tensor | None,
) -> torch.Tensor:
    """Rebuild differentiable pair vectors from cached discrete topology."""
    vectors = positions[idx_j] - positions[idx_i]
    if cell_shifts is not None:
        if cell is None:
            raise ValueError("cell is required with cached periodic shifts")
        edge_cells = cell[batch[idx_i]]
        offsets = torch.einsum(
            "ei,eij->ej", cell_shifts.to(dtype=positions.dtype), edge_cells
        )
        vectors = vectors + offsets
    return vectors


def build_neighbor_graph(
    positions: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic fallback for tests/small inference; formal training uses caches."""
    n_structures = int(batch.max().item()) + 1 if batch.numel() else 0
    if n_structures == 0:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, positions.new_empty((0, 3))
    counts = torch.bincount(batch, minlength=n_structures)
    expected_batch = torch.repeat_interleave(
        torch.arange(n_structures, device=batch.device), counts
    )
    if bool(torch.all(counts == counts[0])) and torch.equal(batch, expected_batch):
        n_atoms = int(counts[0].item())
        points = positions.reshape(n_structures, n_atoms, 3)
        vectors = points[:, None, :, :] - points[:, :, None, :]
        if cell is not None:
            fractional = torch.einsum("bijd,bdk->bijk", vectors, torch.linalg.inv(cell))
            periodic = (
                positions.new_ones((n_structures, 3))
                if pbc is None
                else pbc.to(device=positions.device, dtype=positions.dtype)
            )
            fractional = (
                fractional
                - torch.round(fractional).detach() * periodic[:, None, None, :]
            )
            vectors = torch.einsum("bijd,bdk->bijk", fractional, cell)
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        mask = ~torch.eye(n_atoms, dtype=torch.bool, device=positions.device)[None]
        edge_batch, local_i, local_j = torch.nonzero(
            mask & (distances < cutoff) & (distances > 0.0), as_tuple=True
        )
        offsets = edge_batch * n_atoms
        return (
            offsets + local_i,
            offsets + local_j,
            vectors[edge_batch, local_i, local_j],
        )
    return build_neighbor_graph_reference(positions, batch, cutoff, cell, pbc)


def build_neighbor_graph_reference(
    positions: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loop reference path for heterogeneous batches."""
    idx_i_parts: list[torch.Tensor] = []
    idx_j_parts: list[torch.Tensor] = []
    vector_parts: list[torch.Tensor] = []
    n_structures = int(batch.max().item()) + 1 if batch.numel() else 0
    for structure in range(n_structures):
        atoms = torch.nonzero(batch == structure, as_tuple=False).flatten()
        local = torch.arange(atoms.numel(), device=positions.device)
        local_i, local_j = torch.meshgrid(local, local, indexing="ij")
        vectors = positions[atoms[local_j]] - positions[atoms[local_i]]
        mask = local_i != local_j
        if cell is not None:
            fractional = vectors @ torch.linalg.inv(cell[structure])
            periodic = (
                torch.ones(3, device=positions.device, dtype=positions.dtype)
                if pbc is None
                else pbc[structure].to(device=positions.device, dtype=positions.dtype)
            )
            fractional = fractional - torch.round(fractional).detach() * periodic
            vectors = fractional @ cell[structure]
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        mask &= (distances < cutoff) & (distances > 0.0)
        idx_i_parts.append(atoms[local_i[mask]])
        idx_j_parts.append(atoms[local_j[mask]])
        vector_parts.append(vectors[mask])
    if not idx_i_parts:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, positions.new_empty((0, 3))
    return torch.cat(idx_i_parts), torch.cat(idx_j_parts), torch.cat(vector_parts)


__all__ = [
    "NeighborCache",
    "build_neighbor_graph",
    "build_neighbor_graph_reference",
    "load_neighbor_cache",
    "precompute_dataset_neighbors",
    "precompute_set_neighbors",
    "rebuild_rij",
    "vesin_neighbors",
]
