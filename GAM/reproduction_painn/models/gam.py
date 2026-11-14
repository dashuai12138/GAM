"""Backbone-agnostic scalar geometric associative memory (GAM).

The core in this file only consumes invariant node features, invariant radial
edge features, graph indices, and a per-node memory state.  It deliberately
contains no PaiNN, vector-feature, energy-head, or force-specific logic.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import torch
from torch import nn


BackboneState = TypeVar("BackboneState")


class GAMBackboneAdapter(Protocol[BackboneState]):
    """Minimal interface a backbone must provide to host :class:`GAMCore`.

    Concrete adapters decide how to expose invariant scalar features and how
    to put a scalar residual back into their own representation.  They do not
    own or alter any associative-memory equation.
    """

    @staticmethod
    def invariant_node_features(state: BackboneState) -> torch.Tensor:
        """Return invariant node features with shape ``[n_atoms, node_dim]``."""
        ...

    @staticmethod
    def inject_scalar_residual(
        state: BackboneState, residual: torch.Tensor
    ) -> BackboneState:
        """Add ``[n_atoms, node_dim]`` to the backbone's scalar channels."""
        ...


class GAMCore(nn.Module):
    """Normalized-address GAM-S core shared by arbitrary atomistic backbones.

    For invariant node features ``h`` and invariant radial features ``phi``,
    the module constructs ``x_ij = [h_i, h_j, phi_ij]`` internally.  It then
    performs normalized key/query addressing, a degree-scaled write into a
    persistent per-node matrix, retrieval from the updated matrix, and a
    scalar residual projection.

    Persistence is across interaction depth within one structure forward.
    The host adapter is responsible for creating a fresh zero state for every
    new structure/batch and for choosing the post-block injection point.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        key_dim: int,
        value_dim: int,
        alpha_init: float = 0.1,
        *,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        edge_context_dim = 2 * node_dim + edge_dim
        # Keep this registration and initialization order identical to the
        # trained GAM implementation for strict checkpoint continuity.
        self.key = nn.Linear(edge_context_dim, key_dim)
        self.value = nn.Linear(edge_context_dim, value_dim)
        self.query = nn.Linear(node_dim, key_dim)
        self.forget = nn.Linear(node_dim, 1)
        self.read_projection = nn.Linear(value_dim, node_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.epsilon = float(epsilon)

    def initialize_memory(self, node_features: torch.Tensor) -> torch.Tensor:
        """Create the zero state for one independent structure forward."""
        return node_features.new_zeros(
            node_features.shape[0], self.value_dim, self.key_dim
        )

    def relation_context(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """Construct invariant relation contexts ``[h_i, h_j, phi(r_ij)]``."""
        return torch.cat(
            [node_features[idx_i], node_features[idx_j], edge_features], dim=-1
        )

    def _normalize(self, values: torch.Tensor) -> torch.Tensor:
        return values / torch.sqrt(
            values.square().sum(dim=-1, keepdim=True) + self.epsilon
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write current relations, read updated memory, and return ``delta_h``."""
        edge_context = self.relation_context(
            node_features, edge_features, idx_i, idx_j
        )
        keys = self._normalize(self.key(edge_context))
        values = self.value(edge_context)

        edge_writes = values.unsqueeze(-1) * keys.unsqueeze(-2)
        memory_write = memory.new_zeros(memory.shape)
        memory_write.index_add_(0, idx_i, edge_writes)
        degree = node_features.new_zeros(node_features.shape[0])
        degree.index_add_(0, idx_i, degree.new_ones(idx_i.shape[0]))
        memory_write = memory_write / degree.clamp_min(1).sqrt()[:, None, None]

        forget_gate = torch.sigmoid(self.forget(node_features))
        memory = forget_gate.unsqueeze(-1) * memory + memory_write

        queries = self._normalize(self.query(node_features))
        retrieved = torch.bmm(memory, queries.unsqueeze(-1)).squeeze(-1)
        delta_h = self.alpha * self.read_projection(retrieved)
        return memory, delta_h

    def parameter_counts(self) -> dict[str, int]:
        """Return the same component audit used by existing training reports."""
        return {
            "key": sum(parameter.numel() for parameter in self.key.parameters()),
            "value": sum(parameter.numel() for parameter in self.value.parameters()),
            "query": sum(parameter.numel() for parameter in self.query.parameters()),
            "forget": sum(parameter.numel() for parameter in self.forget.parameters()),
            "read_projection": sum(
                parameter.numel() for parameter in self.read_projection.parameters()
            ),
            "alpha": self.alpha.numel(),
            "input_gate": 0,
        }


__all__ = ["GAMBackboneAdapter", "GAMCore"]
