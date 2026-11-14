from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from reproduction_painn.data import ThreeBPADataset

THREE_BPA_ROOT = Path(__file__).parents[1] / "datasets" / "3bpa"


def test_3bpa_manifest_is_a_disjoint_reproducible_450_50_split():
    manifest = json.loads(
        (THREE_BPA_ROOT / "split_seed42.json").read_text(encoding="utf-8")
    )
    train = set(manifest["train_indices"])
    validation = set(manifest["validation_indices"])
    assert manifest["seed"] == 42
    assert manifest["public_split_available"] is False
    assert len(train) == 450
    assert len(validation) == 50
    assert train.isdisjoint(validation)
    assert train | validation == set(range(500))
    assert manifest["periodic"] is False
    assert {
        name: values["structures"] for name, values in manifest["domains"].items()
    } == {
        "train_300K": 500,
        "test_300K": 1669,
        "test_600K": 2138,
        "test_1200K": 2139,
        "test_dihedral": 7047,
    }


def test_3bpa_dataset_uses_only_train_pool_for_fit_and_is_nonperiodic():
    processed = THREE_BPA_ROOT / "processed"
    if not processed.is_dir():
        pytest.skip("run prepare_3bpa.py before the local integration test")
    split_file = THREE_BPA_ROOT / "split_seed42.json"
    train = ThreeBPADataset(processed, "train", split_file)
    validation = ThreeBPADataset(processed, "validation", split_file)
    test = ThreeBPADataset(processed, "test", split_file, domain="1200K")
    assert (len(train), len(validation), len(test)) == (450, 50, 2139)
    sample = train[0]
    assert sample["positions"].shape == (27, 3)
    assert not sample["pbc"].any()
    assert train.validate_minimum_image(5.0) == float("inf")
    atomref, scale = train.fit_atomref(max_atomic_number=100)
    assert atomref.shape == (101,)
    assert scale > 0.0
    baseline = float(atomref[train[0]["atomic_numbers"]].sum())
    train_energies = torch.tensor(
        [float(train[index]["energy"]) for index in range(len(train))]
    )
    centered = (train_energies - baseline) / 27
    assert abs(float(centered.mean())) < 1.0e-4
    assert float(centered.std(unbiased=False)) == pytest.approx(scale, rel=1.0e-4)
