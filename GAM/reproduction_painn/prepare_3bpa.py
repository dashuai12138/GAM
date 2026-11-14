"""Download, verify, split, and convert the public BOTNet 3BPA benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from ase.io import read

SOURCE_COMMIT = "29e6d467317e4b5967b7ea5cbee54de953fa0d45"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/"
    f"{SOURCE_COMMIT}/dataset_3BPA.tar.gz"
)
ARCHIVE_SHA256 = "21fb6ff7a2386405f74fa283433f69650915b1af69303ff3aa7c3c684146b9c3"
DOMAINS = {
    "train_300K": "train_300K.xyz",
    "test_300K": "test_300K.xyz",
    "test_600K": "test_600K.xyz",
    "test_1200K": "test_1200K.xyz",
    "test_dihedral": "test_dih.xyz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)


def _extract_safely(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"unsafe archive member: {member.name}")
        stream.extractall(destination, filter="data")
    return destination / "dataset_3BPA"


def _convert_xyz(source: Path, destination: Path) -> dict[str, object]:
    frames = read(source, index=":")
    if not frames:
        raise ValueError(f"no structures in {source}")
    atomic_numbers = np.asarray(frames[0].numbers, dtype=np.int64)
    if any(not np.array_equal(frame.numbers, atomic_numbers) for frame in frames):
        raise ValueError(f"atomic ordering changes across {source}")
    if any(frame.pbc.any() for frame in frames):
        raise ValueError(f"3BPA must be non-periodic: {source}")
    positions = np.stack([frame.positions for frame in frames]).astype(np.float32)
    energies = np.asarray(
        [frame.get_potential_energy() for frame in frames], dtype=np.float64
    )
    forces = np.stack([frame.get_forces() for frame in frames]).astype(np.float32)
    dihedrals = np.asarray(
        [frame.info.get("dihedrals", []) for frame in frames], dtype=np.float32
    )
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "atomic_numbers.npy", atomic_numbers)
    np.save(destination / "positions.npy", positions)
    np.save(destination / "energies.npy", energies)
    np.save(destination / "forces.npy", forces)
    np.save(destination / "dihedrals.npy", dihedrals)
    return {
        "structures": len(frames),
        "atoms_per_structure": int(atomic_numbers.size),
        "atomic_numbers": sorted(int(value) for value in np.unique(atomic_numbers)),
        "source_file": source.name,
    }


def _isolated_atom_energies(source: Path, max_atomic_number: int = 100) -> np.ndarray:
    references = np.zeros(max_atomic_number + 1, dtype=np.float32)
    frames = read(source, index=":")
    for frame in frames:
        if len(frame) != 1:
            raise ValueError("iso_atoms.xyz must contain single-atom structures")
        references[int(frame.numbers[0])] = float(frame.get_potential_energy())
    return references


def prepare(root: Path, seed: int = 42, force_download: bool = False) -> dict:
    raw_root = root / "raw"
    archive = raw_root / "dataset_3BPA.tar.gz"
    if force_download or not archive.is_file():
        _download(ARCHIVE_URL, archive)
    actual_hash = _sha256(archive)
    if actual_hash != ARCHIVE_SHA256:
        raise ValueError(
            f"3BPA archive SHA256 mismatch: expected {ARCHIVE_SHA256}, "
            f"got {actual_hash}"
        )
    source_root = raw_root / "dataset_3BPA"
    if not source_root.is_dir():
        source_root = _extract_safely(archive, raw_root)

    processed_root = root / "processed"
    domains = {
        name: _convert_xyz(source_root / filename, processed_root / name)
        for name, filename in DOMAINS.items()
    }
    np.save(
        processed_root / "atomref.npy",
        _isolated_atom_energies(source_root / "iso_atoms.xyz"),
    )

    pool_size = int(domains["train_300K"]["structures"])
    if pool_size != 500:
        raise ValueError(f"expected the public 500-structure pool, got {pool_size}")
    permutation = np.random.default_rng(seed).permutation(pool_size)
    train_indices = sorted(int(value) for value in permutation[:450])
    validation_indices = sorted(int(value) for value in permutation[450:])
    manifest = {
        "dataset": "3BPA",
        "source_repository": "https://github.com/davkovacs/BOTNet-datasets",
        "source_commit": SOURCE_COMMIT,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": actual_hash,
        "public_split_available": False,
        "split_method": "NumPy default_rng(seed).permutation; first 450 train",
        "seed": seed,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "domains": domains,
        "periodic": False,
        "units": {
            "distance": "angstrom",
            "energy": "eV",
            "force": "eV/angstrom",
        },
    }
    (root / "split_seed42.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("reproduction_painn/datasets/3bpa")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        manifest = json.loads(
            (args.root / "split_seed42.json").read_text(encoding="utf-8")
        )
    else:
        manifest = prepare(args.root, args.seed, args.force_download)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
