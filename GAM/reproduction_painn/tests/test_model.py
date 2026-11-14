# ruff: noqa: TID253

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch

from reproduction_painn.data.neighbor import (
    build_neighbor_graph,
    build_neighbor_graph_reference,
)
from reproduction_painn.models import MLIPConfig, build_model
from reproduction_painn.models.painn import UPSTREAM_PAINN, OfficialPaiNN, properties


def _geometry(dtype: torch.dtype = torch.float64):
    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1])
    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.96, 0.00, 0.00],
            [-0.24, 0.93, 0.00],
            [4.00, 0.10, 0.20],
            [4.88, 0.30, 0.10],
            [3.75, 0.91, 0.31],
        ],
        dtype=dtype,
    )
    return atomic_numbers, positions, torch.tensor([0, 0, 0, 1, 1, 1])


def _config(**overrides) -> MLIPConfig:
    values = {
        "type": "dense",
        "n_atom_basis": 12,
        "n_interactions": 2,
        "n_rbf": 8,
        "cutoff": 2.5,
    }
    values.update(overrides)
    return MLIPConfig(**values)


def _batch(positions: torch.Tensor | None = None):
    atomic_numbers, original_positions, atom_batch = _geometry()
    return {
        "atomic_numbers": atomic_numbers,
        "positions": original_positions if positions is None else positions,
        "batch": atom_batch,
        "n_structures": 2,
    }


def test_painn_adapter_calls_pinned_official_representation():
    model = build_model(_config()).double()
    representation = model.backbone.representation_model
    assert type(representation) is OfficialPaiNN
    assert (
        Path(inspect.getfile(type(representation))).resolve()
        == UPSTREAM_PAINN.resolve()
    )

    batch = _batch()
    idx_i, idx_j, r_ij = build_neighbor_graph(
        batch["positions"], batch["batch"], model.config.cutoff
    )
    official = representation(
        {
            properties.Z: batch["atomic_numbers"],
            properties.Rij: r_ij,
            properties.idx_i: idx_i,
            properties.idx_j: idx_j,
        }
    )
    adapted = model.backbone({**batch, "idx_i": idx_i, "idx_j": idx_j, "r_ij": r_ij})
    torch.testing.assert_close(
        adapted["node_features"], official["scalar_representation"]
    )


def test_vectorized_neighbor_graph_matches_reference_loop():
    _, positions, atom_batch = _geometry()
    actual = build_neighbor_graph(positions, atom_batch, cutoff=2.5)
    expected = build_neighbor_graph_reference(
        positions, atom_batch, cutoff=2.5, cell=None, pbc=None
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_vectorized_neighbor_graph_applies_periodic_minimum_image():
    positions = torch.tensor([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    atom_batch = torch.zeros(2, dtype=torch.long)
    cell = torch.eye(3).unsqueeze(0) * 10.0
    pbc = torch.ones(1, 3, dtype=torch.bool)
    idx_i, idx_j, vectors = build_neighbor_graph(
        positions, atom_batch, cutoff=1.0, cell=cell, pbc=pbc
    )
    assert idx_i.tolist() == [0, 1]
    assert idx_j.tolist() == [1, 0]
    torch.testing.assert_close(
        vectors,
        torch.tensor([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_dense_energy_force_forward_backward_is_finite():
    model = build_model(_config()).double().train()
    output = model(_batch())
    loss = output["energy"].square().mean() + output["forces"].square().mean()
    loss.backward()
    assert output["energy"].shape == (2,)
    assert output["forces"].shape == (6, 3)
    assert output["node_features"].shape == (6, 12)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_rotation_invariant_energy_and_equivariant_force():
    torch.manual_seed(9)
    model = build_model(_config()).double().eval()
    angle = 0.731
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    original = model(_batch())
    rotated = model(_batch(_geometry()[1] @ rotation.T))
    torch.testing.assert_close(
        original["energy"], rotated["energy"], atol=1.0e-10, rtol=1.0e-10
    )
    torch.testing.assert_close(
        original["forces"] @ rotation.T,
        rotated["forces"],
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_non_dense_model_type_is_rejected():
    with pytest.raises(ValueError, match="model type must be 'dense'"):
        MLIPConfig(type="unsupported")
