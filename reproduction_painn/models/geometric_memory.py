"""Scalar geometric associative memory for message-passing networks."""

from __future__ import annotations

import math

import torch
from torch import nn

from .gam import GAMCore


class GeometricAssociativeMemory(nn.Module):
    """Store invariant edge key-value bindings in a per-node matrix."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        key_dim: int,
        value_dim: int,
        alpha_init: float = 0.1,
        *,
        input_gate_enabled: bool = False,
        normalizer_enabled: bool = False,
    ) -> None:
        super().__init__()
        edge_context_dim = 2 * node_dim + edge_dim
        self.key = nn.Linear(edge_context_dim, key_dim)
        self.value = nn.Linear(edge_context_dim, value_dim)
        self.query = nn.Linear(node_dim, key_dim)
        self.forget = nn.Linear(node_dim, 1)
        self.read_projection = nn.Linear(value_dim, node_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        # This is a molecular relation-level adaptation of the gLSTM input
        # gate: each invariant edge context decides how much of that relation
        # is written. It is not the original node/time-step gLSTM gate.
        # Preserve the random stream used by every pre-existing GAM-S/PaiNN
        # parameter. This keeps the 2x2 initialization matched: enabling the
        # new gate adds only its own weights and does not perturb later layers.
        with torch.random.fork_rng(devices=[]):
            self.input_gate = (
                nn.Linear(edge_context_dim, 1) if input_gate_enabled else None
            )
        if self.input_gate is not None:
            nn.init.zeros_(self.input_gate.bias)
        self.input_gate_enabled = input_gate_enabled
        self.normalizer_enabled = normalizer_enabled
        self.key_dim = key_dim
        self.value_dim = value_dim

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update ``memory`` and return its projected query result."""
        edge_context = torch.cat(
            [node_features[idx_i], node_features[idx_j], edge_features], dim=-1
        )
        keys = self.key(edge_context)
        values = self.value(edge_context)

        edge_writes = values.unsqueeze(-1) * keys.unsqueeze(-2)
        memory_write = memory.new_zeros(memory.shape)
        memory_write.index_add_(0, idx_i, edge_writes)

        degree = node_features.new_zeros(node_features.shape[0])
        degree.index_add_(0, idx_i, degree.new_ones(idx_i.shape[0]))
        memory_write = memory_write / degree.clamp_min(1).sqrt()[:, None, None]

        forget_gate = torch.sigmoid(self.forget(node_features))
        memory = forget_gate.unsqueeze(-1) * memory + memory_write
        queries = self.query(node_features)
        retrieved = torch.bmm(memory, queries.unsqueeze(-1)).squeeze(-1)
        retrieved = retrieved / math.sqrt(self.key_dim)
        return memory, self.alpha * self.read_projection(retrieved)

    def forward_adapted(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        memory: torch.Tensor,
        normalizer: torch.Tensor | None,
        *,
        input_gate_override: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Run the relation-gate/normalizer ablation without changing GAM-S.

        The existing :meth:`forward` remains the complete legacy GAM-S path.
        This separate path implements only the clean 2x2 ablation: optional
        relation-level selective writes and optional mLSTM-style normalized
        retrieval. Exponential gates, stabilizer state, and output gates are
        deliberately absent.
        """
        if not (self.input_gate_enabled or self.normalizer_enabled):
            raise RuntimeError("adapted forward requires input gate or normalizer")
        if self.normalizer_enabled and normalizer is None:
            raise ValueError("normalizer state is required when enabled")
        if not self.normalizer_enabled and normalizer is not None:
            raise ValueError("normalizer state must be None when disabled")

        edge_context = torch.cat(
            [node_features[idx_i], node_features[idx_j], edge_features], dim=-1
        )
        keys = self.key(edge_context)
        values = self.value(edge_context)

        input_gate: torch.Tensor | None = None
        if self.input_gate_enabled:
            if self.input_gate is None:
                raise RuntimeError("input gate module is unavailable")
            input_gate = (
                torch.sigmoid(self.input_gate(edge_context))
                if input_gate_override is None
                else input_gate_override
            )
            if input_gate.shape != (idx_i.shape[0], 1):
                raise ValueError("input gate must have shape [n_edges, 1]")
            gated_values = input_gate * values
            gated_keys = input_gate * keys
        else:
            if input_gate_override is not None:
                raise ValueError("cannot override a disabled input gate")
            gated_values = values
            gated_keys = keys

        edge_writes = gated_values.unsqueeze(-1) * keys.unsqueeze(-2)
        memory_write = memory.new_zeros(memory.shape)
        memory_write.index_add_(0, idx_i, edge_writes)

        degree = node_features.new_zeros(node_features.shape[0])
        degree.index_add_(0, idx_i, degree.new_ones(idx_i.shape[0]))
        degree_scale = degree.clamp_min(1).sqrt()
        memory_write = memory_write / degree_scale[:, None, None]

        forget_gate = torch.sigmoid(self.forget(node_features))
        memory = forget_gate.unsqueeze(-1) * memory + memory_write

        normalizer_write: torch.Tensor | None = None
        if self.normalizer_enabled:
            if normalizer is None:
                raise RuntimeError("normalizer state is unavailable")
            normalizer_write = normalizer.new_zeros(normalizer.shape)
            normalizer_write.index_add_(0, idx_i, gated_keys)
            normalizer_write = normalizer_write / degree_scale[:, None]
            normalizer = forget_gate * normalizer + normalizer_write

        queries = self.query(node_features)
        raw_retrieval = torch.bmm(memory, queries.unsqueeze(-1)).squeeze(-1)
        if self.normalizer_enabled:
            if normalizer is None:
                raise RuntimeError("normalizer state is unavailable")
            normalizer_dot_query = (normalizer * queries).sum(dim=-1, keepdim=True)
            denominator = normalizer_dot_query.abs().clamp_min(1.0)
            retrieved = raw_retrieval / denominator
        else:
            normalizer_dot_query = raw_retrieval.new_zeros(
                (raw_retrieval.shape[0], 1)
            )
            denominator = raw_retrieval.new_full(
                (raw_retrieval.shape[0], 1), math.sqrt(self.key_dim)
            )
            retrieved = raw_retrieval / math.sqrt(self.key_dim)

        projected_retrieval = self.read_projection(retrieved)
        delta = self.alpha * projected_retrieval
        diagnostics = {
            "edge_context": edge_context,
            "keys": keys,
            "values": values,
            "queries": queries,
            "memory_write": memory_write,
            "memory": memory,
            "raw_retrieval": raw_retrieval,
            "retrieval": retrieved,
            "projected_retrieval": projected_retrieval,
            "delta": delta,
            "forget_gate": forget_gate,
            "normalizer_dot_query": normalizer_dot_query,
            "denominator": denominator,
            "alpha": self.alpha,
            "idx_i": idx_i,
        }
        if input_gate is not None:
            diagnostics["input_gate"] = input_gate
        if normalizer is not None and normalizer_write is not None:
            diagnostics["normalizer"] = normalizer
            diagnostics["normalizer_write"] = normalizer_write
        return memory, normalizer, delta, diagnostics

    def parameter_counts(self) -> dict[str, int]:
        """Return a direct audit of each mathematical component."""
        return {
            "key": sum(parameter.numel() for parameter in self.key.parameters()),
            "value": sum(parameter.numel() for parameter in self.value.parameters()),
            "query": sum(parameter.numel() for parameter in self.query.parameters()),
            "forget": sum(parameter.numel() for parameter in self.forget.parameters()),
            "read_projection": sum(
                parameter.numel() for parameter in self.read_projection.parameters()
            ),
            "alpha": self.alpha.numel(),
            "input_gate": (
                sum(parameter.numel() for parameter in self.input_gate.parameters())
                if self.input_gate is not None
                else 0
            ),
        }


class NormalizedAddressGeometricAssociativeMemory(GAMCore):
    """Compatibility name for the backbone-agnostic GAM core.

    Existing imports and checkpoints keep working while new backbones can use
    :class:`reproduction_painn.models.gam.GAMCore` directly.
    """

    def __init__(
        self,
        *args,
        input_gate_enabled: bool = False,
        normalizer_enabled: bool = False,
        **kwargs,
    ) -> None:
        if input_gate_enabled or normalizer_enabled:
            raise ValueError("GAM core cannot use i/n ablations")
        super().__init__(*args, **kwargs)


class EnvironmentQueryGeometricAssociativeMemory(
    GeometricAssociativeMemory
):
    """GAM-S-EnvQ with an independently initialized neighbor query block.

    ``query(q_i) + neighbor_query(neighbor_q_i)`` is exactly one affine
    projection of ``[q_i; neighbor_q_i]``.  Keeping the center block under the
    legacy ``query`` name preserves every GAM-S parameter bitwise.
    """

    def __init__(self, node_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, node_dim=node_dim, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.neighbor_query = nn.Linear(node_dim, self.key_dim, bias=False)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_context = torch.cat(
            [node_features[idx_i], node_features[idx_j], edge_features], dim=-1
        )
        keys = self.key(edge_context)
        values = self.value(edge_context)
        edge_writes = values.unsqueeze(-1) * keys.unsqueeze(-2)
        memory_write = memory.new_zeros(memory.shape)
        memory_write.index_add_(0, idx_i, edge_writes)
        degree = node_features.new_zeros(node_features.shape[0])
        degree.index_add_(0, idx_i, degree.new_ones(idx_i.shape[0]))
        degree_scale = degree.clamp_min(1).sqrt()
        memory_write = memory_write / degree_scale[:, None, None]
        forget_gate = torch.sigmoid(self.forget(node_features))
        memory = forget_gate.unsqueeze(-1) * memory + memory_write

        neighbor_features = node_features.new_zeros(node_features.shape)
        neighbor_features.index_add_(0, idx_i, node_features[idx_j])
        neighbor_features = neighbor_features / degree_scale[:, None]
        queries = self.query(node_features) + self.neighbor_query(neighbor_features)
        retrieved = torch.bmm(memory, queries.unsqueeze(-1)).squeeze(-1)
        retrieved = retrieved / math.sqrt(self.key_dim)
        return memory, self.alpha * self.read_projection(retrieved)

    def parameter_counts(self) -> dict[str, int]:
        counts = super().parameter_counts()
        counts["query"] += sum(
            parameter.numel() for parameter in self.neighbor_query.parameters()
        )
        return counts


class IrrepGeometricAssociativeMemory(nn.Module):
    """Persistent, relation-addressed l=1/l=2 geometric memory.

    The invariant address and learned value-channel basis are shared across
    every PaiNN depth.  The main recurrence first incorporates the current
    relation write into the persistent state and then retrieves from that
    updated state, exactly matching the temporal semantics of GAM-S (l=0).
    Read-before-write remains available only as an explicit future ablation.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        key_dim: int,
        value_dim: int,
        orders: tuple[int, ...],
        alpha_init: float = 0.01,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if not orders or not set(orders).issubset({1, 2}):
            raise ValueError("irrep orders must be a non-empty subset of {1, 2}")
        edge_context_dim = 2 * node_dim + edge_dim
        self.geo_key = nn.Linear(edge_context_dim, key_dim)
        self.geo_query = nn.Linear(node_dim, key_dim)
        self.orders = tuple(sorted(set(orders)))
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.epsilon = float(epsilon)

        self.l1_amplitude = (
            nn.Linear(edge_context_dim, value_dim) if 1 in self.orders else None
        )
        self.l1_forget = nn.Linear(node_dim, 1) if 1 in self.orders else None
        self.l1_projection = (
            nn.Linear(value_dim, node_dim, bias=False) if 1 in self.orders else None
        )
        self.alpha1 = (
            nn.Parameter(torch.tensor(float(alpha_init))) if 1 in self.orders else None
        )

        self.l2_amplitude = (
            nn.Linear(edge_context_dim, value_dim) if 2 in self.orders else None
        )
        self.l2_forget = nn.Linear(node_dim, 1) if 2 in self.orders else None
        self.l2_projection = (
            nn.Linear(value_dim, node_dim, bias=False) if 2 in self.orders else None
        )
        self.alpha2 = (
            nn.Parameter(torch.tensor(float(alpha_init))) if 2 in self.orders else None
        )

    def initialize_memories(
        self, node_features: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Allocate per-structure memory; state never crosses structures/frames."""
        n_atoms = node_features.shape[0]
        memories: dict[int, torch.Tensor] = {}
        if 1 in self.orders:
            memories[1] = node_features.new_zeros(
                n_atoms, self.value_dim, 3, self.key_dim
            )
        if 2 in self.orders:
            memories[2] = node_features.new_zeros(
                n_atoms, self.value_dim, 5, self.key_dim
            )
        return memories

    @staticmethod
    def real_spherical_harmonics(
        directions: torch.Tensor, order: int
    ) -> torch.Tensor:
        """Return ViSNet's norm-normalized real Y_l component ordering.

        For unit ``directions`` both the three-dimensional l=1 block and the
        compact five-dimensional l=2 block have Euclidean norm one.  Keeping
        the formulas local avoids adding an e3nn dependency and makes the
        otherwise convention-dependent component order explicit.
        """
        x, y, z = directions.unbind(dim=-1)
        if order == 1:
            return torch.stack((x, y, z), dim=-1)
        if order == 2:
            sqrt_three = math.sqrt(3.0)
            return torch.stack(
                (
                    sqrt_three * x * z,
                    sqrt_three * x * y,
                    y.square() - 0.5 * (x.square() + z.square()),
                    sqrt_three * y * z,
                    0.5 * sqrt_three * (z.square() - x.square()),
                ),
                dim=-1,
            )
        raise ValueError("real spherical harmonics order must be 1 or 2")

    @staticmethod
    def _degree_scale(
        node_features: torch.Tensor, idx_i: torch.Tensor
    ) -> torch.Tensor:
        degree = node_features.new_zeros(node_features.shape[0])
        degree.index_add_(0, idx_i, degree.new_ones(idx_i.shape[0]))
        return degree.clamp_min(1).sqrt()

    def _normalized_match(
        self, retrieved: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        spatial_dims = tuple(range(2, retrieved.ndim))
        retrieved_norm = torch.sqrt(
            retrieved.square().sum(dim=spatial_dims, keepdim=True) + self.epsilon
        )
        current_norm = torch.sqrt(
            current.square().sum(dim=spatial_dims, keepdim=True) + self.epsilon
        )
        return ((retrieved / retrieved_norm) * (current / current_norm)).sum(
            dim=spatial_dims
        )

    def forward_depth(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        directions: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        memories: dict[int, torch.Tensor],
        *,
        persistent: bool = True,
        read_before_write: bool = False,
    ) -> tuple[
        dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, torch.Tensor]
    ]:
        """Write/read/match one depth and return invariant scalar corrections.

        ``persistent=False`` is the fair current-only control: it writes into a
        zero memory, immediately reads/matches that current relation, and the
        caller discards the state at the end of the depth.
        """
        edge_context = torch.cat(
            [node_features[idx_i], node_features[idx_j], edge_features], dim=-1
        )
        keys = self.geo_key(edge_context)
        queries = self.geo_query(node_features)
        degree_scale = self._degree_scale(node_features, idx_i)
        corrections: dict[int, torch.Tensor] = {}
        new_memories: dict[int, torch.Tensor] = {}
        diagnostics: dict[str, torch.Tensor] = {
            "geo_keys": keys,
            "geo_queries": queries,
            "idx_i": idx_i,
        }

        if 1 in self.orders:
            if self.l1_amplitude is None or self.l1_forget is None:
                raise RuntimeError("l=1 modules are unavailable")
            if self.l1_projection is None or self.alpha1 is None:
                raise RuntimeError("l=1 output modules are unavailable")
            amplitudes1 = self.l1_amplitude(edge_context)
            harmonics1 = self.real_spherical_harmonics(directions, 1)
            values1 = amplitudes1.unsqueeze(-1) * harmonics1.unsqueeze(1)
            current1 = values1.new_zeros(
                node_features.shape[0], self.value_dim, 3
            )
            current1.index_add_(0, idx_i, values1)
            current1 = current1 / degree_scale[:, None, None]
            writes1 = values1.unsqueeze(-1) * keys[:, None, None, :]
            write1 = writes1.new_zeros(
                node_features.shape[0], self.value_dim, 3, self.key_dim
            )
            write1.index_add_(0, idx_i, writes1)
            write1 = write1 / degree_scale[:, None, None, None]
            old1 = memories[1]
            forget1 = torch.sigmoid(self.l1_forget(node_features))
            updated1 = forget1[:, :, None, None] * old1 + write1
            read_memory1 = (
                old1 if persistent and read_before_write else updated1
            )
            retrieved1 = torch.einsum("nvxk,nk->nvx", read_memory1, queries)
            retrieved1 = retrieved1 / math.sqrt(self.key_dim)
            matching1 = self._normalized_match(retrieved1, current1)
            delta1 = self.alpha1 * self.l1_projection(matching1)
            corrections[1] = delta1
            new_memories[1] = updated1 if persistent else old1.new_zeros(old1.shape)
            diagnostics.update(
                {
                    "l1_values": values1,
                    "l1_harmonics": harmonics1,
                    "l1_memory_write": write1,
                    "l1_memory": updated1,
                    "l1_retrieval": retrieved1,
                    "l1_current": current1,
                    "l1_matching": matching1,
                    "l1_delta": delta1,
                    "l1_forget": forget1,
                    "alpha1": self.alpha1,
                }
            )

        if 2 in self.orders:
            if self.l2_amplitude is None or self.l2_forget is None:
                raise RuntimeError("l=2 modules are unavailable")
            if self.l2_projection is None or self.alpha2 is None:
                raise RuntimeError("l=2 output modules are unavailable")
            amplitudes2 = self.l2_amplitude(edge_context)
            harmonics2 = self.real_spherical_harmonics(directions, 2)
            values2 = amplitudes2.unsqueeze(-1) * harmonics2.unsqueeze(1)
            current2 = values2.new_zeros(
                node_features.shape[0], self.value_dim, 5
            )
            current2.index_add_(0, idx_i, values2)
            current2 = current2 / degree_scale[:, None, None]
            writes2 = values2.unsqueeze(-1) * keys[:, None, None, :]
            write2 = writes2.new_zeros(
                node_features.shape[0], self.value_dim, 5, self.key_dim
            )
            write2.index_add_(0, idx_i, writes2)
            write2 = write2 / degree_scale[:, None, None, None]
            old2 = memories[2]
            forget2 = torch.sigmoid(self.l2_forget(node_features))
            updated2 = forget2[:, :, None, None] * old2 + write2
            read_memory2 = (
                old2 if persistent and read_before_write else updated2
            )
            retrieved2 = torch.einsum("nvsk,nk->nvs", read_memory2, queries)
            retrieved2 = retrieved2 / math.sqrt(self.key_dim)
            matching2 = self._normalized_match(retrieved2, current2)
            delta2 = self.alpha2 * self.l2_projection(matching2)
            corrections[2] = delta2
            new_memories[2] = updated2 if persistent else old2.new_zeros(old2.shape)
            diagnostics.update(
                {
                    "l2_values": values2,
                    "l2_harmonics": harmonics2,
                    "l2_memory_write": write2,
                    "l2_memory": updated2,
                    "l2_retrieval": retrieved2,
                    "l2_current": current2,
                    "l2_matching": matching2,
                    "l2_delta": delta2,
                    "l2_forget": forget2,
                    "alpha2": self.alpha2,
                }
            )
        return new_memories, corrections, diagnostics

    def parameter_counts(self) -> dict[str, int]:
        """Return the requested component-level high-order parameter audit."""
        modules = {
            "geo_key": self.geo_key,
            "geo_query": self.geo_query,
            "l1_amplitude": self.l1_amplitude,
            "l1_forget": self.l1_forget,
            "l1_projection": self.l1_projection,
            "l2_amplitude": self.l2_amplitude,
            "l2_forget": self.l2_forget,
            "l2_projection": self.l2_projection,
        }
        counts = {
            name: (
                sum(parameter.numel() for parameter in module.parameters())
                if module is not None
                else 0
            )
            for name, module in modules.items()
        }
        counts["l1_alpha"] = 0 if self.alpha1 is None else self.alpha1.numel()
        counts["l2_alpha"] = 0 if self.alpha2 is None else self.alpha2.numel()
        return counts


class IrrepGeometricAssociativeMemoryV2A(
    IrrepGeometricAssociativeMemory
):
    """Isolated l=1 experiment with normalized signed addressing.

    V2a deliberately keeps the established l=1 value construction and
    persistent matrix recurrence.  Its only mathematical changes are
    unit-normalized keys/queries and an invariant retrieval self-contraction
    in place of the legacy retrieval/current-context cosine.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        key_dim: int,
        value_dim: int,
        alpha_init: float = 0.01,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            orders=(1,),
            alpha_init=alpha_init,
            epsilon=epsilon,
        )

    def _unit_normalize(self, values: torch.Tensor) -> torch.Tensor:
        denominator = torch.sqrt(
            values.square().sum(dim=-1, keepdim=True) + self.epsilon
        )
        return values / denominator

    def forward_depth(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        directions: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Write normalized-key bindings, read, and self-contract l=1."""
        if self.l1_amplitude is None or self.l1_forget is None:
            raise RuntimeError("v2a l=1 modules are unavailable")
        if self.l1_projection is None or self.alpha1 is None:
            raise RuntimeError("v2a l=1 output modules are unavailable")

        edge_context = torch.cat(
            [node_features[idx_i], node_features[idx_j], edge_features], dim=-1
        )
        raw_keys = self.geo_key(edge_context)
        raw_queries = self.geo_query(node_features)
        keys = self._unit_normalize(raw_keys)
        queries = self._unit_normalize(raw_queries)
        address = (keys * queries[idx_i]).sum(dim=-1)

        amplitudes = self.l1_amplitude(edge_context)
        harmonics = self.real_spherical_harmonics(directions, 1)
        values = amplitudes.unsqueeze(-1) * harmonics.unsqueeze(1)
        degree_scale = self._degree_scale(node_features, idx_i)
        edge_writes = values.unsqueeze(-1) * keys[:, None, None, :]
        memory_write = edge_writes.new_zeros(
            node_features.shape[0], self.value_dim, 3, self.key_dim
        )
        memory_write.index_add_(0, idx_i, edge_writes)
        memory_write = memory_write / degree_scale[:, None, None, None]

        forget = torch.sigmoid(self.l1_forget(node_features))
        updated_memory = forget[:, :, None, None] * memory + memory_write
        retrieval = torch.einsum("nvxk,nk->nvx", updated_memory, queries)
        self_contraction = retrieval.square().sum(dim=-1)
        log_feature = torch.log1p(self_contraction)
        delta = self.alpha1 * self.l1_projection(log_feature)
        diagnostics = {
            "raw_geo_keys": raw_keys,
            "raw_geo_queries": raw_queries,
            "geo_keys": keys,
            "geo_queries": queries,
            "idx_i": idx_i,
            "address_score": address,
            "l1_values": values,
            "l1_harmonics": harmonics,
            "l1_memory_write": memory_write,
            "l1_memory": updated_memory,
            "l1_retrieval": retrieval,
            "l1_self_contraction": self_contraction,
            "l1_log_feature": log_feature,
            "l1_delta": delta,
            "l1_forget": forget,
            "alpha1": self.alpha1,
        }
        return updated_memory, delta, diagnostics


class PreAggregationInteractionMemory(nn.Module):
    """GAM-Mij binds native pre-aggregation PaiNN messages to invariant keys."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        key_dim: int,
        value_dim: int,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        edge_context_dim = 2 * node_dim + edge_dim
        self.key = nn.Linear(edge_context_dim, key_dim)
        self.query = nn.Linear(node_dim, key_dim)
        self.scalar_value_projection = nn.Linear(node_dim, value_dim)
        self.scalar_forget = nn.Linear(node_dim, 1)
        self.scalar_read_projection = nn.Linear(value_dim, node_dim)
        self.scalar_alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.key_dim = key_dim
        self.value_dim = value_dim

    def edge_keys(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """Return invariant chemical/geometric addresses for each edge."""
        edge_context = torch.cat(
            [node_features[idx_i], node_features[idx_j], edge_features], dim=-1
        )
        return self.key(edge_context)

    @staticmethod
    def _degree(node_features: torch.Tensor, idx_i: torch.Tensor) -> torch.Tensor:
        degree = node_features.new_zeros(node_features.shape[0])
        degree.index_add_(0, idx_i, degree.new_ones(idx_i.shape[0]))
        return degree.clamp_min(1).sqrt()

    def write(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        scalar_edge_messages: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        scalar_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Write native edge interactions before their PaiNN aggregation."""
        keys = self.edge_keys(node_features, edge_features, idx_i, idx_j)
        degree = self._degree(node_features, idx_i)

        scalar_values = self.scalar_value_projection(scalar_edge_messages)
        scalar_writes = scalar_values.unsqueeze(-1) * keys.unsqueeze(-2)
        scalar_update = scalar_memory.new_zeros(scalar_memory.shape)
        scalar_update.index_add_(0, idx_i, scalar_writes)
        scalar_update = scalar_update / degree[:, None, None]
        scalar_forget = torch.sigmoid(self.scalar_forget(node_features))
        scalar_memory = scalar_forget.unsqueeze(-1) * scalar_memory + scalar_update

        diagnostics = {
            "keys": keys,
            "scalar_values": scalar_values,
            "scalar_forget": scalar_forget,
        }
        return scalar_memory, diagnostics

    def read(
        self,
        node_features: torch.Tensor,
        scalar_memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Query after mixing and return invariant scalar-path corrections."""
        queries = self.query(node_features)
        scalar_retrieved = torch.bmm(scalar_memory, queries.unsqueeze(-1)).squeeze(-1)
        scalar_retrieved = scalar_retrieved / math.sqrt(self.key_dim)
        scalar_delta = self.scalar_alpha * self.scalar_read_projection(scalar_retrieved)
        return {
            "queries": queries,
            "scalar_retrieved": scalar_retrieved,
            "scalar_delta": scalar_delta,
        }

    def parameter_counts(self) -> dict[str, int]:
        """Return a direct audit of every pre-aggregation memory component."""
        modules = {
            "key": self.key,
            "query": self.query,
            "scalar_value_projection": self.scalar_value_projection,
            "scalar_forget": self.scalar_forget,
            "scalar_read_projection": self.scalar_read_projection,
        }
        counts = {
            name: (
                sum(parameter.numel() for parameter in module.parameters())
                if module is not None
                else 0
            )
            for name, module in modules.items()
        }
        counts["scalar_alpha"] = self.scalar_alpha.numel()
        return counts


__all__ = [
    "EnvironmentQueryGeometricAssociativeMemory",
    "GeometricAssociativeMemory",
    "IrrepGeometricAssociativeMemory",
    "IrrepGeometricAssociativeMemoryV2A",
    "NormalizedAddressGeometricAssociativeMemory",
    "PreAggregationInteractionMemory",
]
