from __future__ import annotations

import torch

from reproduction_painn.models import GAMCore, PaiNNGAMAdapter
from reproduction_painn.models.geometric_memory import (
    NormalizedAddressGeometricAssociativeMemory,
)


def _inputs():
    torch.manual_seed(901)
    nodes = torch.randn(5, 12, dtype=torch.float64)
    edges = torch.randn(9, 8, dtype=torch.float64)
    idx_i = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 4])
    idx_j = torch.tensor([1, 2, 0, 2, 3, 4, 1, 4, 3])
    memory = torch.randn(5, 7, 5, dtype=torch.float64)
    return nodes, edges, idx_i, idx_j, memory


def test_gam_core_matches_explicit_normalized_write_then_read():
    torch.manual_seed(902)
    core = GAMCore(12, 8, 5, 7, epsilon=1.0e-8).double()
    nodes, edges, idx_i, idx_j, old_memory = _inputs()

    context = torch.cat([nodes[idx_i], nodes[idx_j], edges], dim=-1)
    keys = core.key(context)
    keys = keys / torch.sqrt(
        keys.square().sum(dim=-1, keepdim=True) + core.epsilon
    )
    values = core.value(context)
    writes = values.unsqueeze(-1) * keys.unsqueeze(-2)
    expected_write = torch.zeros_like(old_memory)
    expected_write.index_add_(0, idx_i, writes)
    degree = nodes.new_zeros(nodes.shape[0])
    degree.index_add_(0, idx_i, degree.new_ones(idx_i.shape[0]))
    expected_write = expected_write / degree.clamp_min(1).sqrt()[:, None, None]
    expected_memory = (
        torch.sigmoid(core.forget(nodes)).unsqueeze(-1) * old_memory
        + expected_write
    )
    queries = core.query(nodes)
    queries = queries / torch.sqrt(
        queries.square().sum(dim=-1, keepdim=True) + core.epsilon
    )
    retrieval = torch.bmm(expected_memory, queries.unsqueeze(-1)).squeeze(-1)
    expected_delta = core.alpha * core.read_projection(retrieval)

    memory, delta = core(nodes, edges, idx_i, idx_j, old_memory)
    torch.testing.assert_close(memory, expected_memory, atol=0.0, rtol=0.0)
    torch.testing.assert_close(delta, expected_delta, atol=0.0, rtol=0.0)


def test_legacy_norm_name_is_state_and_output_compatible_with_gam_core():
    torch.manual_seed(903)
    legacy = NormalizedAddressGeometricAssociativeMemory(12, 8, 5, 7).double()
    torch.manual_seed(903)
    core = GAMCore(12, 8, 5, 7).double()
    assert legacy.state_dict().keys() == core.state_dict().keys()
    assert all(
        torch.equal(legacy.state_dict()[name], core.state_dict()[name])
        for name in core.state_dict()
    )
    inputs = _inputs()
    legacy_memory, legacy_delta = legacy(*inputs)
    core_memory, core_delta = core(*inputs)
    torch.testing.assert_close(core_memory, legacy_memory, atol=0.0, rtol=0.0)
    torch.testing.assert_close(core_delta, legacy_delta, atol=0.0, rtol=0.0)


def test_painn_adapter_exposes_only_q_and_never_changes_mu():
    torch.manual_seed(904)
    q = torch.randn(5, 1, 12, dtype=torch.float64)
    mu = torch.randn(5, 3, 12, dtype=torch.float64)
    residual = torch.randn(5, 12, dtype=torch.float64)

    invariant = PaiNNGAMAdapter.invariant_node_features((q, mu))
    q_out, mu_out = PaiNNGAMAdapter.inject_scalar_residual((q, mu), residual)

    torch.testing.assert_close(invariant, q.squeeze(1), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        q_out, q + residual.unsqueeze(1), atol=0.0, rtol=0.0
    )
    assert mu_out is mu


def test_gam_memory_initialization_is_per_node_and_zero():
    core = GAMCore(12, 8, 5, 7).double()
    nodes = torch.randn(6, 12, dtype=torch.float64)
    memory = core.initialize_memory(nodes)
    assert memory.shape == (6, 7, 5)
    assert memory.dtype == nodes.dtype
    assert memory.device == nodes.device
    assert torch.count_nonzero(memory) == 0
