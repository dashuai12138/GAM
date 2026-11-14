"""Streaming DeePMD-NPY dataset with system-homogeneous batching."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .neighbor import NeighborCache, load_neighbor_cache

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@dataclass(frozen=True)
class DataSetBlock:
    name: str
    coordinates: np.ndarray
    energies: np.ndarray
    forces: np.ndarray
    cells: np.ndarray
    neighbors: NeighborCache | None


@dataclass(frozen=True)
class DataSystem:
    name: str
    cache_key: str
    atomic_numbers: np.ndarray
    sets: tuple[DataSetBlock, ...]


def _atomic_number_map(type_map: Sequence[str]) -> np.ndarray:
    periodic = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "P": 15,
        "S": 16,
        "Cl": 17,
    }
    missing = [symbol for symbol in type_map if symbol not in periodic]
    if missing:
        raise ValueError(f"unsupported elements in type_map.raw: {missing}")
    return np.asarray([periodic[symbol] for symbol in type_map], dtype=np.int64)


class DeepMDNpyDataset(Dataset[dict[str, Any]]):
    """Read frames lazily and optionally attach mmap neighbor topology."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        set_range: Sequence[int | None] | None = None,
        neighbor_mode: str = "dynamic",
        neighbor_cache_dir: str | Path = "reproduction_painn/cache/neighbors",
        cutoff: float = 5.0,
        include_frames: dict[str, Sequence[int]] | None = None,
        exclude_frames: dict[str, Sequence[int]] | None = None,
    ) -> None:
        if neighbor_mode not in {"dynamic", "auto", "cache"}:
            raise ValueError("neighbor_mode must be dynamic, auto, or cache")
        self.neighbor_mode = neighbor_mode
        self.neighbor_cache_dir = Path(neighbor_cache_dir)
        self.cutoff = float(cutoff)
        self.systems: list[DataSystem] = []
        self.index: list[tuple[int, int, int]] = []
        self.indices_by_system: list[list[int]] = []
        for root_value in roots:
            root = Path(root_value).resolve()
            root_digest = hashlib.sha1(
                str(root).lower().encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:10]
            cache_key = f"{root.name}-{root_digest}"
            type_map = (root / "type_map.raw").read_text(encoding="utf-8").split()
            type_index = np.loadtxt(root / "type.raw", dtype=np.int64, ndmin=1)
            atomic_numbers = _atomic_number_map(type_map)[type_index]
            set_dirs = sorted(path for path in root.glob("set.*") if path.is_dir())
            if set_range is not None:
                if len(set_range) != 2:
                    raise ValueError("set_range must be [start, stop]")
                set_dirs = set_dirs[set_range[0] : set_range[1]]
            if not set_dirs:
                raise FileNotFoundError(f"no set.* directory under {root}")
            n_atoms = atomic_numbers.shape[0]
            set_data: list[DataSetBlock] = []
            for set_dir in set_dirs:
                coordinates = np.load(set_dir / "coord.npy", mmap_mode="r")
                energies = np.load(set_dir / "energy.npy", mmap_mode="r").reshape(-1)
                forces = np.load(set_dir / "force.npy", mmap_mode="r")
                cells = np.load(set_dir / "box.npy", mmap_mode="r")
                n_frames = coordinates.shape[0]
                if not (
                    energies.shape[0] == forces.shape[0] == cells.shape[0] == n_frames
                ):
                    raise ValueError(f"inconsistent frame counts under {set_dir}")
                neighbors = None
                if neighbor_mode != "dynamic":
                    neighbors = load_neighbor_cache(
                        self.neighbor_cache_dir,
                        cache_key,
                        set_dir.name,
                        self.cutoff,
                        n_frames,
                        n_atoms,
                    )
                    if neighbors is None and neighbor_mode == "cache":
                        raise FileNotFoundError(
                            f"missing neighbor cache for {root.name}/{set_dir.name}; "
                            "run python -m reproduction_painn.precompute_neighbors "
                            "CONFIG"
                        )
                set_data.append(
                    DataSetBlock(
                        name=set_dir.name,
                        coordinates=coordinates.reshape(n_frames, n_atoms, 3),
                        energies=energies,
                        forces=forces.reshape(n_frames, n_atoms, 3),
                        cells=cells.reshape(n_frames, 3, 3),
                        neighbors=neighbors,
                    )
                )
            system_id = len(self.systems)
            self.systems.append(
                DataSystem(root.name, cache_key, atomic_numbers, tuple(set_data))
            )
            system_indices: list[int] = []
            included = (
                None
                if include_frames is None
                else set(int(item) for item in include_frames.get(root.name, ()))
            )
            excluded = (
                set()
                if exclude_frames is None
                else set(int(item) for item in exclude_frames.get(root.name, ()))
            )
            total_frames = sum(values.energies.shape[0] for values in set_data)
            candidate_indices = list(excluded)
            if included is not None:
                candidate_indices.extend(included)
            invalid = [
                item for item in candidate_indices if item < 0 or item >= total_frames
            ]
            if invalid:
                raise IndexError(
                    f"frame indices outside [0, {total_frames}) for {root.name}: "
                    f"{invalid[:5]}"
                )
            global_frame = 0
            for set_id, values in enumerate(set_data):
                for frame in range(values.energies.shape[0]):
                    selected = (included is None or global_frame in included) and (
                        global_frame not in excluded
                    )
                    global_frame += 1
                    if not selected:
                        continue
                    system_indices.append(len(self.index))
                    self.index.append((system_id, set_id, frame))
            self.indices_by_system.append(system_indices)

    @property
    def system_names(self) -> list[str]:
        return [system.name for system in self.systems]

    def __len__(self) -> int:
        return len(self.index)

    def validate_minimum_image(self, cutoff: float) -> float:
        minimum_face = math.inf
        for system in self.systems:
            for values in system.sets:
                inverse_cells = np.linalg.inv(np.asarray(values.cells))
                face_distances = 1.0 / np.linalg.norm(inverse_cells, axis=-2)
                minimum_face = min(minimum_face, float(np.min(face_distances)))
        if 2.0 * cutoff >= minimum_face:
            raise ValueError(
                "minimum-image neighbor search is unsafe: "
                f"2 * cutoff={2.0 * cutoff:g} >= minimum cell face "
                f"distance={minimum_face:g}"
            )
        return minimum_face

    def __getitem__(self, item: int) -> dict[str, Any]:
        system_id, set_id, frame = self.index[item]
        system = self.systems[system_id]
        values = system.sets[set_id]
        result: dict[str, Any] = {
            "atomic_numbers": torch.as_tensor(system.atomic_numbers, dtype=torch.long),
            "positions": torch.from_numpy(
                np.array(values.coordinates[frame], dtype=np.float32, copy=True)
            ),
            "energy": torch.tensor(float(values.energies[frame]), dtype=torch.float32),
            "forces": torch.from_numpy(
                np.array(values.forces[frame], dtype=np.float32, copy=True)
            ),
            "cell": torch.from_numpy(
                np.array(values.cells[frame], dtype=np.float32, copy=True)
            ),
            "pbc": torch.ones(3, dtype=torch.bool),
            "system_id": system_id,
            "system_name": system.name,
            "phase": system.name,
            "frame_index": sum(
                block.energies.shape[0] for block in system.sets[:set_id]
            )
            + frame,
        }
        if values.neighbors is not None:
            idx_i, idx_j, shifts = values.neighbors.frame(frame)
            result.update(
                {
                    "idx_i": torch.from_numpy(
                        np.array(idx_i, dtype=np.int64, copy=True)
                    ),
                    "idx_j": torch.from_numpy(
                        np.array(idx_j, dtype=np.int64, copy=True)
                    ),
                    "cell_shifts": torch.from_numpy(
                        np.array(shifts, dtype=np.int8, copy=True)
                    ),
                }
            )
        return result


class ThreeBPADataset(Dataset[dict[str, Any]]):
    """Memory-mapped, non-periodic view of the converted public 3BPA data."""

    TEST_DOMAINS = {
        "300K": "test_300K",
        "600K": "test_600K",
        "1200K": "test_1200K",
        "dihedral": "test_dihedral",
    }

    def __init__(
        self,
        root: str | Path,
        split: str,
        split_file: str | Path,
        domain: str | None = None,
    ) -> None:
        self.root = Path(root)
        manifest = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if split in {"train", "validation"}:
            domain_name = "train_300K"
            key = "train_indices" if split == "train" else "validation_indices"
            indices = [int(value) for value in manifest[key]]
        elif split == "test":
            if domain not in self.TEST_DOMAINS:
                raise ValueError(
                    f"3BPA test domain must be one of {sorted(self.TEST_DOMAINS)}"
                )
            domain_name = self.TEST_DOMAINS[domain]
            indices = list(range(int(manifest["domains"][domain_name]["structures"])))
        else:
            raise ValueError(f"unsupported 3BPA split: {split}")
        domain_root = self.root / domain_name
        self.atomic_numbers = np.load(domain_root / "atomic_numbers.npy", mmap_mode="r")
        self.positions = np.load(domain_root / "positions.npy", mmap_mode="r")
        self.energies = np.load(domain_root / "energies.npy", mmap_mode="r")
        self.forces = np.load(domain_root / "forces.npy", mmap_mode="r")
        self.frame_indices = indices
        self.domain_name = domain_name
        self.systems = [
            DataSystem(
                name=domain_name,
                cache_key=domain_name,
                atomic_numbers=self.atomic_numbers,
                sets=(),
            )
        ]
        self.indices_by_system = [list(range(len(indices)))]

    @property
    def system_names(self) -> list[str]:
        return [self.domain_name]

    def __len__(self) -> int:
        return len(self.frame_indices)

    def validate_minimum_image(self, cutoff: float) -> float:
        del cutoff
        return math.inf

    def __getitem__(self, item: int) -> dict[str, Any]:
        frame = self.frame_indices[item]
        return {
            "atomic_numbers": torch.as_tensor(
                np.array(self.atomic_numbers, dtype=np.int64, copy=True)
            ),
            "positions": torch.from_numpy(
                np.array(self.positions[frame], dtype=np.float32, copy=True)
            ),
            "energy": torch.tensor(float(self.energies[frame]), dtype=torch.float32),
            "forces": torch.from_numpy(
                np.array(self.forces[frame], dtype=np.float32, copy=True)
            ),
            "cell": torch.eye(3, dtype=torch.float32) * 50.0,
            "pbc": torch.zeros(3, dtype=torch.bool),
            "system_id": 0,
            "system_name": self.domain_name,
            "phase": self.domain_name,
            "frame_index": frame,
        }

    def fit_atomref(self, max_atomic_number: int) -> tuple[torch.Tensor, float]:
        references = np.array(np.load(self.root / "atomref.npy"), copy=True)
        if references.shape[0] < max_atomic_number + 1:
            raise ValueError("3BPA atomref does not cover max_atomic_number")
        references = references[: max_atomic_number + 1]
        baseline = float(np.sum(references[np.asarray(self.atomic_numbers)]))
        selected = np.asarray(self.energies[self.frame_indices], dtype=np.float64)
        residual_per_atom = (selected - baseline) / self.atomic_numbers.shape[0]
        # The public isolated-atom references leave a large, composition-wide
        # molecular offset. Center it using the 450 training structures only;
        # adding the same offset to every present element contributes exactly
        # one offset per atom and keeps initialization on the residual scale.
        offset = float(np.mean(residual_per_atom))
        references[np.unique(self.atomic_numbers)] += offset
        residual_per_atom = residual_per_atom - offset
        scale = float(np.std(residual_per_atom))
        if not np.isfinite(scale) or scale < 1.0e-6:
            scale = 1.0
        return torch.as_tensor(references, dtype=torch.float32), scale


def split_frame_selection(
    split_file: str | Path | None, split: str
) -> tuple[dict[str, list[int]] | None, dict[str, list[int]] | None]:
    """Load a compact split manifest; test is the complement of train/validation."""
    if split_file is None:
        return None, None
    manifest = json.loads(Path(split_file).read_text(encoding="utf-8"))
    systems = manifest["systems"]
    if split in {"train", "validation"}:
        return {
            name: [int(item) for item in values.get(split, ())]
            for name, values in systems.items()
        }, None
    if split == "test":
        return None, {
            name: [
                int(item)
                for subset in ("train", "validation")
                for item in values.get(subset, ())
            ]
            for name, values in systems.items()
        }
    raise ValueError(f"unsupported split: {split}")


def load_dataset(
    data_config: dict[str, Any],
    split: str,
    cutoff: float,
    *,
    domain: str | None = None,
) -> DeepMDNpyDataset | ThreeBPADataset | None:
    """Build a dataset without leaking format-specific logic into trainers."""
    if data_config.get("format") == "3bpa":
        return ThreeBPADataset(
            data_config["root"],
            split,
            data_config["split_file"],
            domain=domain,
        )
    roots = data_config.get(split)
    if not roots:
        return None
    include_frames, exclude_frames = split_frame_selection(
        data_config.get("split_file"), split
    )
    return DeepMDNpyDataset(
        roots,
        set_range=data_config.get(f"{split}_set_range"),
        neighbor_mode=data_config.get("neighbor_mode", "dynamic"),
        neighbor_cache_dir=data_config.get(
            "neighbor_cache_dir", "reproduction_painn/cache/neighbors"
        ),
        cutoff=cutoff,
        include_frames=include_frames,
        exclude_frames=exclude_frames,
    )


class SystemBatchSampler(Sampler[list[int]]):
    """Keep batches within one system; optionally sample phases uniformly."""

    def __init__(
        self,
        dataset: DeepMDNpyDataset,
        batch_size: int | str,
        seed: int = 17,
        shuffle: bool = True,
        sampling: str = "proportional",
    ) -> None:
        if sampling not in {"proportional", "phase_balanced"}:
            raise ValueError("sampling must be proportional or phase_balanced")
        self.dataset = dataset
        self.shuffle = shuffle
        self.sampling = sampling
        self.generator = torch.Generator().manual_seed(seed)
        if isinstance(batch_size, str):
            if not batch_size.startswith("auto:"):
                raise ValueError("string batch_size must use the form 'auto:N'")
            target_atoms = int(batch_size.split(":", maxsplit=1)[1])
            if target_atoms < 1:
                raise ValueError("auto batch atom target must be positive")
            self.batch_sizes = [
                max(1, math.ceil(target_atoms / system.atomic_numbers.shape[0]))
                for system in dataset.systems
            ]
        else:
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
            self.batch_sizes = [batch_size] * len(dataset.systems)
        self._batches: list[list[int]] | None = None
        self._position = 0

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / batch_size)
            for indices, batch_size in zip(
                self.dataset.indices_by_system, self.batch_sizes, strict=True
            )
        )

    def _ordered(self, indices: list[int]) -> list[int]:
        if not self.shuffle:
            return list(indices)
        order = torch.randperm(len(indices), generator=self.generator).tolist()
        return [indices[item] for item in order]

    def _make_batches(self) -> list[list[int]]:
        if self.sampling == "phase_balanced" and self.shuffle:
            pools = [
                self._ordered(indices) for indices in self.dataset.indices_by_system
            ]
            cursors = [0] * len(pools)
            batches: list[list[int]] = []
            for _ in range(len(self)):
                system_id = int(
                    torch.randint(len(pools), (1,), generator=self.generator).item()
                )
                batch_size = self.batch_sizes[system_id]
                if cursors[system_id] + batch_size > len(pools[system_id]):
                    pools[system_id] = self._ordered(
                        self.dataset.indices_by_system[system_id]
                    )
                    cursors[system_id] = 0
                start = cursors[system_id]
                cursors[system_id] += batch_size
                batches.append(pools[system_id][start : start + batch_size])
            return batches

        batches = []
        for indices, batch_size in zip(
            self.dataset.indices_by_system, self.batch_sizes, strict=True
        ):
            ordered = self._ordered(indices)
            batches.extend(
                ordered[start : start + batch_size]
                for start in range(0, len(ordered), batch_size)
            )
        if self.shuffle:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[item] for item in order]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self._batches is None or self._position >= len(self._batches):
            self._batches = self._make_batches()
            self._position = 0
        while self._position < len(self._batches):
            batch = self._batches[self._position]
            self._position += 1
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator.get_state(),
            "batches": self._batches,
            "position": self._position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generator.set_state(state["generator"].cpu())
        self._batches = state.get("batches")
        self._position = int(state.get("position", 0))


def collate_frames(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    atom_counts = [frame["atomic_numbers"].shape[0] for frame in frames]
    atom_offsets = np.cumsum([0, *atom_counts[:-1]]).tolist()
    result: dict[str, Any] = {
        "atomic_numbers": torch.cat([frame["atomic_numbers"] for frame in frames]),
        "positions": torch.cat([frame["positions"] for frame in frames]),
        "energy": torch.stack([frame["energy"] for frame in frames]),
        "forces": torch.cat([frame["forces"] for frame in frames]),
        "cell": torch.stack([frame["cell"] for frame in frames]),
        "pbc": torch.stack([frame["pbc"] for frame in frames]),
        "batch": torch.cat(
            [
                torch.full((count,), index, dtype=torch.long)
                for index, count in enumerate(atom_counts)
            ]
        ),
        "atom_index": torch.cat([torch.arange(count) for count in atom_counts]),
        "n_atoms": torch.tensor(atom_counts, dtype=torch.long),
        "system_id": torch.tensor([frame["system_id"] for frame in frames]),
        "system_name": [frame["system_name"] for frame in frames],
        "phase": [frame["phase"] for frame in frames],
        "frame_index": torch.tensor([frame["frame_index"] for frame in frames]),
    }
    cached = ["idx_i" in frame for frame in frames]
    if any(cached) and not all(cached):
        raise ValueError("a batch cannot mix cached and dynamic neighbor frames")
    if all(cached):
        result["idx_i"] = torch.cat(
            [
                frame["idx_i"] + offset
                for frame, offset in zip(frames, atom_offsets, strict=True)
            ]
        )
        result["idx_j"] = torch.cat(
            [
                frame["idx_j"] + offset
                for frame, offset in zip(frames, atom_offsets, strict=True)
            ]
        )
        result["cell_shifts"] = torch.cat([frame["cell_shifts"] for frame in frames])
    return result


def fit_atomref(
    dataset: DeepMDNpyDataset | ThreeBPADataset, max_atomic_number: int
) -> tuple[torch.Tensor, float]:
    """Fit an additive elemental baseline and streaming residual scale."""
    if isinstance(dataset, ThreeBPADataset):
        return dataset.fit_atomref(max_atomic_number)
    counts: list[np.ndarray] = []
    energies: list[float] = []
    weights: list[int] = []
    selected_by_system: list[dict[int, list[int]]] = [{} for _ in dataset.systems]
    for system_id, set_id, frame in dataset.index:
        selected_by_system[system_id].setdefault(set_id, []).append(frame)
    for system_id, system in enumerate(dataset.systems):
        count = np.bincount(
            system.atomic_numbers, minlength=max_atomic_number + 1
        ).astype(np.float64)
        frame_count = sum(
            len(items) for items in selected_by_system[system_id].values()
        )
        if frame_count == 0:
            raise ValueError(f"split contains no frames for system {system.name}")
        counts.append(count)
        energies.append(
            sum(
                float(np.sum(values.energies[frame_ids], dtype=np.float64))
                for set_id, frame_ids in selected_by_system[system_id].items()
                for values in (system.sets[set_id],)
            )
            / frame_count
        )
        weights.append(frame_count)
    sqrt_weights = np.sqrt(np.asarray(weights, dtype=np.float64))
    matrix = np.stack(counts) * sqrt_weights[:, None]
    target = np.asarray(energies) * sqrt_weights
    atomref, *_ = np.linalg.lstsq(matrix, target, rcond=1.0e-12)
    residual_sum = residual_square_sum = 0.0
    residual_count = 0
    for system_id, (system, count) in enumerate(
        zip(dataset.systems, counts, strict=True)
    ):
        baseline = float(count @ atomref)
        atom_count = system.atomic_numbers.shape[0]
        for set_id, frame_ids in selected_by_system[system_id].items():
            values = system.sets[set_id]
            residual = (np.asarray(values.energies[frame_ids]) - baseline) / atom_count
            residual_sum += float(np.sum(residual, dtype=np.float64))
            residual_square_sum += float(np.sum(residual**2, dtype=np.float64))
            residual_count += residual.shape[0]
    mean = residual_sum / residual_count
    scale = float(np.sqrt(max(residual_square_sum / residual_count - mean**2, 0.0)))
    if not np.isfinite(scale) or scale < 1.0e-6:
        scale = 1.0
    return torch.as_tensor(atomref, dtype=torch.float32), scale


__all__ = [
    "DataSetBlock",
    "DataSystem",
    "DeepMDNpyDataset",
    "SystemBatchSampler",
    "ThreeBPADataset",
    "collate_frames",
    "fit_atomref",
    "load_dataset",
    "split_frame_selection",
]
