"""PaiNN adapter with an optional geometric associative memory.

This module owns all PaiNN-specific details and returns a stable energy and
rotation-invariant node-feature contract to the training pipeline.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from reproduction_painn.data.neighbor import build_neighbor_graph, rebuild_rij
from reproduction_painn.models.gam import GAMBackboneAdapter, GAMCore
from reproduction_painn.models.geometric_memory import (
    EnvironmentQueryGeometricAssociativeMemory,
    GeometricAssociativeMemory,
    IrrepGeometricAssociativeMemory,
    IrrepGeometricAssociativeMemoryV2A,
    PreAggregationInteractionMemory,
)
from reproduction_painn.models.painn import OfficialPaiNN, properties, snn


class PaiNNGAMAdapter(
    GAMBackboneAdapter[tuple[torch.Tensor, torch.Tensor]]
):
    """PaiNN-only bridge between scalar ``q`` and the shared GAM core.

    GAM never reads or changes ``mu``.  The adapter exposes pre-interaction
    scalar ``q`` as ``h`` and adds ``delta_h`` after PaiNN interaction/mixing.
    """

    @staticmethod
    def invariant_node_features(
        state: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        q, _ = state
        return q.squeeze(1)

    @staticmethod
    def inject_scalar_residual(
        state: tuple[torch.Tensor, torch.Tensor], residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, mu = state
        return q + residual.unsqueeze(1), mu


class PaiNNBackbone(nn.Module):
    """Adapt official PaiNN to the common ``energy + node_features`` contract."""

    def __init__(
        self,
        *,
        n_atom_basis: int,
        n_interactions: int,
        n_rbf: int,
        cutoff: float,
        max_atomic_number: int,
        epsilon: float = 1.0e-8,
        atomref: torch.Tensor | None = None,
        energy_scale: float = 1.0,
        memory_enabled: bool = False,
        memory_key_dim: int = 16,
        memory_value_dim: int = 64,
        memory_alpha_init: float = 0.1,
        memory_input_gate_enabled: bool = False,
        memory_normalizer_enabled: bool = False,
        memory_normalized_addressing_enabled: bool = False,
        memory_environment_query_enabled: bool = False,
        preaggregation_memory_enabled: bool = False,
        preaggregation_memory_key_dim: int = 16,
        preaggregation_memory_value_dim: int = 64,
        preaggregation_memory_alpha_init: float = 0.1,
        irrep_memory_enabled: bool = False,
        irrep_memory_orders: tuple[int, ...] = (1, 2),
        irrep_memory_key_dim: int = 16,
        irrep_memory_value_dim: int = 64,
        irrep_memory_alpha_init: float = 0.01,
        irrep_memory_read_before_write: bool = False,
        irrep_memory_persistent: bool = True,
        irrep_memory_v2a_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.feature_dim = int(n_atom_basis)
        self.representation_model = OfficialPaiNN(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=snn.GaussianRBF(n_rbf, cutoff),
            cutoff_fn=snn.CosineCutoff(cutoff),
            activation=F.silu,
            shared_interactions=False,
            shared_filters=False,
            epsilon=epsilon,
            nuclear_embedding=nn.Embedding(max_atomic_number + 1, n_atom_basis),
        )
        self.memory_enabled = memory_enabled
        self.memory_input_gate_enabled = memory_input_gate_enabled
        self.memory_normalizer_enabled = memory_normalizer_enabled
        self.memory_adaptation_enabled = (
            memory_input_gate_enabled or memory_normalizer_enabled
        )
        self.memory_normalized_addressing_enabled = (
            memory_normalized_addressing_enabled
        )
        self.memory_environment_query_enabled = memory_environment_query_enabled
        self.memory_layers = nn.ModuleList()
        if memory_enabled:
            memory_class: type[nn.Module]
            if memory_normalized_addressing_enabled:
                memory_class = GAMCore
            elif memory_environment_query_enabled:
                memory_class = EnvironmentQueryGeometricAssociativeMemory
            else:
                memory_class = GeometricAssociativeMemory
            memory_kwargs: dict[str, Any] = {
                "node_dim": n_atom_basis,
                "edge_dim": n_rbf,
                "key_dim": memory_key_dim,
                "value_dim": memory_value_dim,
                "alpha_init": memory_alpha_init,
            }
            if memory_normalized_addressing_enabled:
                memory_kwargs["epsilon"] = epsilon
            else:
                memory_kwargs["input_gate_enabled"] = memory_input_gate_enabled
                memory_kwargs["normalizer_enabled"] = memory_normalizer_enabled
            self.memory_layers.extend(
                memory_class(**memory_kwargs)
                for _ in range(n_interactions)
            )
        self.preaggregation_memory_enabled = preaggregation_memory_enabled
        self.preaggregation_memory_layers = nn.ModuleList()
        if preaggregation_memory_enabled:
            self.preaggregation_memory_layers.extend(
                PreAggregationInteractionMemory(
                    node_dim=n_atom_basis,
                    edge_dim=n_rbf,
                    key_dim=preaggregation_memory_key_dim,
                    value_dim=preaggregation_memory_value_dim,
                    alpha_init=preaggregation_memory_alpha_init,
                )
                for _ in range(n_interactions)
            )
        self.irrep_memory_enabled = irrep_memory_enabled
        self.irrep_memory_read_before_write = irrep_memory_read_before_write
        self.irrep_memory_persistent = irrep_memory_persistent
        if irrep_memory_enabled and not memory_enabled:
            raise ValueError("Irrep-GAM requires the GAM-S l=0 memory path")
        self.irrep_memory = (
            IrrepGeometricAssociativeMemory(
                node_dim=n_atom_basis,
                edge_dim=n_rbf,
                key_dim=irrep_memory_key_dim,
                value_dim=irrep_memory_value_dim,
                orders=irrep_memory_orders,
                alpha_init=irrep_memory_alpha_init,
                epsilon=epsilon,
            )
            if irrep_memory_enabled
            else None
        )
        self.irrep_memory_v2a_enabled = irrep_memory_v2a_enabled
        if irrep_memory_v2a_enabled and not memory_enabled:
            raise ValueError("Irrep-GAM-v2a requires the GAM-S l=0 memory path")
        if irrep_memory_v2a_enabled and irrep_memory_enabled:
            raise ValueError("Irrep-GAM-v2a must be isolated from legacy Irrep-GAM")
        # The fork restores the ambient RNG state after initializing v2a, so
        # every pre-existing parameter created below (notably energy_head)
        # remains bitwise identical to GAM-S under the same seed.
        with torch.random.fork_rng(devices=[]):
            self.irrep_memory_v2a = (
                IrrepGeometricAssociativeMemoryV2A(
                    node_dim=n_atom_basis,
                    edge_dim=n_rbf,
                    key_dim=irrep_memory_key_dim,
                    value_dim=irrep_memory_value_dim,
                    alpha_init=irrep_memory_alpha_init,
                    epsilon=epsilon,
                )
                if irrep_memory_v2a_enabled
                else None
            )
        self.energy_head = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis // 2),
            nn.SiLU(),
            nn.Linear(n_atom_basis // 2, 1),
        )
        if atomref is None:
            atomref = torch.zeros(max_atomic_number + 1)
        if atomref.shape != (max_atomic_number + 1,):
            raise ValueError("atomref must have shape [max_atomic_number + 1]")
        self.register_buffer("atomref", atomref.to(torch.get_default_dtype()))
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))

    def _representation_with_memory(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run the official PaiNN blocks with scalar memory between depths."""
        representation = self.representation_model
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = representation.radial_basis(d_ij)
        fcut = representation.cutoff_fn(d_ij)
        filters = representation.filter_net(phi_ij) * fcut[..., None]
        if representation.share_filters:
            filter_list = [filters] * representation.n_interactions
        else:
            filter_list = torch.split(filters, 3 * representation.n_atom_basis, dim=-1)
        edge_features = (phi_ij * fcut[..., None]).squeeze(1)

        q = representation.embedding(atomic_numbers)
        for embedding in representation.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)
        mu = q.new_zeros((n_atoms, 3, representation.n_atom_basis))
        memory = q.new_zeros(
            n_atoms,
            self.memory_layers[0].value_dim,
            self.memory_layers[0].key_dim,
        )
        for layer, (interaction, mixing) in enumerate(
            zip(representation.interactions, representation.mixing, strict=True)
        ):
            node_features = PaiNNGAMAdapter.invariant_node_features((q, mu))
            memory, memory_read = self.memory_layers[layer](
                node_features, edge_features, idx_i, idx_j, memory
            )
            q, mu = interaction(
                q, mu, filter_list[layer], dir_ij, idx_i, idx_j, n_atoms
            )
            q, mu = mixing(q, mu)
            q, mu = PaiNNGAMAdapter.inject_scalar_residual(
                (q, mu), memory_read
            )

        return {
            "scalar_representation": q.squeeze(1),
            "vector_representation": mu,
        }

    def _representation_with_adapted_memory(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Run the clean GAM-S relation-gate/normalizer 2x2 ablation."""
        representation = self.representation_model
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = representation.radial_basis(d_ij)
        fcut = representation.cutoff_fn(d_ij)
        filters = representation.filter_net(phi_ij) * fcut[..., None]
        if representation.share_filters:
            filter_list = [filters] * representation.n_interactions
        else:
            filter_list = torch.split(filters, 3 * representation.n_atom_basis, dim=-1)
        edge_features = (phi_ij * fcut[..., None]).squeeze(1)

        q = representation.embedding(atomic_numbers)
        for embedding in representation.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)
        mu = q.new_zeros((n_atoms, 3, representation.n_atom_basis))
        memory_layer = self.memory_layers[0]
        memory = q.new_zeros(
            n_atoms, memory_layer.value_dim, memory_layer.key_dim
        )
        normalizer = (
            q.new_zeros(n_atoms, memory_layer.key_dim)
            if self.memory_normalizer_enabled
            else None
        )
        layer_diagnostics: list[dict[str, torch.Tensor]] = []

        for layer, (interaction, mixing) in enumerate(
            zip(representation.interactions, representation.mixing, strict=True)
        ):
            node_features = q.squeeze(1)
            memory, normalizer, memory_delta, diagnostics = self.memory_layers[
                layer
            ].forward_adapted(
                node_features,
                edge_features,
                idx_i,
                idx_j,
                memory,
                normalizer,
            )
            q, mu = interaction(
                q, mu, filter_list[layer], dir_ij, idx_i, idx_j, n_atoms
            )
            q, mu = mixing(q, mu)
            diagnostics["q_post"] = q.squeeze(1)
            layer_diagnostics.append(diagnostics)
            q = q + memory_delta.unsqueeze(1)

        return {
            "scalar_representation": q.squeeze(1),
            "vector_representation": mu,
            "memory_adaptation_diagnostics": layer_diagnostics,
        }

    def _representation_with_irrep_memory(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Run unchanged GAM-S plus cross-depth l=1/l=2 recall/matching."""
        if self.irrep_memory is None:
            raise RuntimeError("Irrep-GAM module is unavailable")
        representation = self.representation_model
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = representation.radial_basis(d_ij)
        fcut = representation.cutoff_fn(d_ij)
        filters = representation.filter_net(phi_ij) * fcut[..., None]
        if representation.share_filters:
            filter_list = [filters] * representation.n_interactions
        else:
            filter_list = torch.split(filters, 3 * representation.n_atom_basis, dim=-1)
        edge_features = (phi_ij * fcut[..., None]).squeeze(1)

        q = representation.embedding(atomic_numbers)
        for embedding in representation.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)
        mu = q.new_zeros((n_atoms, 3, representation.n_atom_basis))

        # The l=0 state and update below are deliberately identical to the
        # established GAM-S path.  Only the separate l>=1 states are new.
        scalar_layer = self.memory_layers[0]
        scalar_memory = q.new_zeros(
            n_atoms, scalar_layer.value_dim, scalar_layer.key_dim
        )
        irrep_memories = self.irrep_memory.initialize_memories(q.squeeze(1))
        layer_diagnostics: list[dict[str, torch.Tensor]] = []

        for layer, (interaction, mixing) in enumerate(
            zip(representation.interactions, representation.mixing, strict=True)
        ):
            q_pre = q.squeeze(1)
            scalar_memory, scalar_delta = self.memory_layers[layer](
                q_pre, edge_features, idx_i, idx_j, scalar_memory
            )
            irrep_memories, corrections, diagnostics = (
                self.irrep_memory.forward_depth(
                    q_pre,
                    edge_features,
                    dir_ij,
                    idx_i,
                    idx_j,
                    irrep_memories,
                    persistent=self.irrep_memory_persistent,
                    read_before_write=self.irrep_memory_read_before_write,
                )
            )
            q, mu = interaction(
                q, mu, filter_list[layer], dir_ij, idx_i, idx_j, n_atoms
            )
            q, mu = mixing(q, mu)
            q_post = q.squeeze(1)
            diagnostics["q_post"] = q_post
            diagnostics["l0_delta"] = scalar_delta
            layer_diagnostics.append(diagnostics)
            total_delta = scalar_delta
            for order in self.irrep_memory.orders:
                total_delta = total_delta + corrections[order]
            q = q + total_delta.unsqueeze(1)

        return {
            "scalar_representation": q.squeeze(1),
            "vector_representation": mu,
            "irrep_memory_diagnostics": layer_diagnostics,
        }

    def _representation_with_irrep_memory_v2a(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Run unchanged GAM-S plus the isolated normalized-address l=1 path."""
        if self.irrep_memory_v2a is None:
            raise RuntimeError("Irrep-GAM-v2a module is unavailable")
        representation = self.representation_model
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = representation.radial_basis(d_ij)
        fcut = representation.cutoff_fn(d_ij)
        filters = representation.filter_net(phi_ij) * fcut[..., None]
        if representation.share_filters:
            filter_list = [filters] * representation.n_interactions
        else:
            filter_list = torch.split(filters, 3 * representation.n_atom_basis, dim=-1)
        edge_features = (phi_ij * fcut[..., None]).squeeze(1)

        q = representation.embedding(atomic_numbers)
        for embedding in representation.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)
        mu = q.new_zeros((n_atoms, 3, representation.n_atom_basis))

        scalar_layer = self.memory_layers[0]
        scalar_memory = q.new_zeros(
            n_atoms, scalar_layer.value_dim, scalar_layer.key_dim
        )
        l1_memory = q.new_zeros(
            n_atoms,
            self.irrep_memory_v2a.value_dim,
            3,
            self.irrep_memory_v2a.key_dim,
        )
        layer_diagnostics: list[dict[str, torch.Tensor]] = []

        for layer, (interaction, mixing) in enumerate(
            zip(representation.interactions, representation.mixing, strict=True)
        ):
            q_pre = q.squeeze(1)
            scalar_memory, scalar_delta = self.memory_layers[layer](
                q_pre, edge_features, idx_i, idx_j, scalar_memory
            )
            l1_memory, l1_delta, diagnostics = (
                self.irrep_memory_v2a.forward_depth(
                    q_pre,
                    edge_features,
                    dir_ij,
                    idx_i,
                    idx_j,
                    l1_memory,
                )
            )
            q, mu = interaction(
                q, mu, filter_list[layer], dir_ij, idx_i, idx_j, n_atoms
            )
            q, mu = mixing(q, mu)
            diagnostics["q_post"] = q.squeeze(1)
            diagnostics["l0_delta"] = scalar_delta
            layer_diagnostics.append(diagnostics)
            q = q + (scalar_delta + l1_delta).unsqueeze(1)

        return {
            "scalar_representation": q.squeeze(1),
            "vector_representation": mu,
            "irrep_memory_v2a_diagnostics": layer_diagnostics,
        }

    @staticmethod
    def _native_edge_interactions(
        interaction: nn.Module,
        q: torch.Tensor,
        mu: torch.Tensor,
        filters: torch.Tensor,
        directions: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the official PaiNN edge tensors immediately before scatter."""
        context = interaction.interatomic_context_net(q)
        neighbor_context = context[idx_j]
        neighbor_vectors = mu[idx_j]
        filtered_context = filters * neighbor_context
        scalar_edges, radial_vectors, transported_vectors = torch.split(
            filtered_context, interaction.n_atom_basis, dim=-1
        )
        directional_edges = (
            radial_vectors * directions[..., None]
            + transported_vectors * neighbor_vectors
        )
        return scalar_edges, directional_edges

    @staticmethod
    def _aggregate_native_edge_interactions(
        q: torch.Tensor,
        mu: torch.Tensor,
        scalar_edges: torch.Tensor,
        directional_edges: torch.Tensor,
        idx_i: torch.Tensor,
        n_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the unchanged official PaiNN scatter/add operations."""
        scalar_update = snn.scatter_add(scalar_edges, idx_i, dim_size=n_atoms)
        directional_update = snn.scatter_add(directional_edges, idx_i, dim_size=n_atoms)
        return q + scalar_update, mu + directional_update

    def _representation_with_preaggregation_memory(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run GAM-Mij and add its memory residual after PaiNN mixing."""
        representation = self.representation_model
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = representation.radial_basis(d_ij)
        fcut = representation.cutoff_fn(d_ij)
        filters = representation.filter_net(phi_ij) * fcut[..., None]
        if representation.share_filters:
            filter_list = [filters] * representation.n_interactions
        else:
            filter_list = torch.split(filters, 3 * representation.n_atom_basis, dim=-1)
        edge_features = (phi_ij * fcut[..., None]).squeeze(1)

        q = representation.embedding(atomic_numbers)
        for embedding in representation.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)
        mu = q.new_zeros((n_atoms, 3, representation.n_atom_basis))
        memory_layer = self.preaggregation_memory_layers[0]
        scalar_memory = q.new_zeros(
            n_atoms, memory_layer.value_dim, memory_layer.key_dim
        )
        for layer, (interaction, mixing) in enumerate(
            zip(representation.interactions, representation.mixing, strict=True)
        ):
            node_features = q.squeeze(1)
            scalar_edges, directional_edges = self._native_edge_interactions(
                interaction,
                q,
                mu,
                filter_list[layer],
                dir_ij,
                idx_j,
            )
            scalar_memory, _ = self.preaggregation_memory_layers[layer].write(
                node_features=node_features,
                edge_features=edge_features,
                scalar_edge_messages=scalar_edges.squeeze(1),
                idx_i=idx_i,
                idx_j=idx_j,
                scalar_memory=scalar_memory,
            )
            memory_read = self.preaggregation_memory_layers[layer].read(
                node_features=node_features,
                scalar_memory=scalar_memory,
            )
            q, mu = self._aggregate_native_edge_interactions(
                q,
                mu,
                scalar_edges,
                directional_edges,
                idx_i,
                n_atoms,
            )
            q, mu = mixing(q, mu)
            q = q + memory_read["scalar_delta"].unsqueeze(1)

        return {
            "scalar_representation": q.squeeze(1),
            "vector_representation": mu,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Return base energy and invariant per-atom features.

        ``batch`` is intentionally framework-level.  PaiNN property names and
        neighbor-vector construction remain encapsulated in this adapter.
        """
        atomic_numbers = batch["atomic_numbers"]
        positions = batch["positions"]
        atom_batch = batch["batch"]
        idx_i = batch.get("idx_i")
        idx_j = batch.get("idx_j")
        r_ij = batch.get("r_ij")
        if r_ij is not None and (idx_i is None or idx_j is None):
            raise ValueError("precomputed r_ij requires idx_i and idx_j")
        if r_ij is None and (idx_i is None or idx_j is None):
            idx_i, idx_j, r_ij = build_neighbor_graph(
                positions,
                atom_batch,
                self.cutoff,
                batch.get("cell"),
                batch.get("pbc"),
            )
        elif r_ij is None:
            r_ij = rebuild_rij(
                positions,
                atom_batch,
                idx_i,
                idx_j,
                batch.get("cell"),
                batch.get("cell_shifts"),
            )
        representation_inputs = {
            properties.Z: atomic_numbers,
            properties.Rij: r_ij,
            properties.idx_i: idx_i,
            properties.idx_j: idx_j,
        }
        if self.irrep_memory_v2a_enabled:
            outputs = self._representation_with_irrep_memory_v2a(
                representation_inputs
            )
        elif self.irrep_memory_enabled:
            outputs = self._representation_with_irrep_memory(representation_inputs)
        elif self.memory_adaptation_enabled:
            outputs = self._representation_with_adapted_memory(
                representation_inputs
            )
        elif self.preaggregation_memory_enabled:
            outputs = self._representation_with_preaggregation_memory(
                representation_inputs
            )
        elif self.memory_enabled:
            outputs = self._representation_with_memory(representation_inputs)
        else:
            outputs = self.representation_model(representation_inputs)
        node_features = outputs["scalar_representation"]
        atomic_energy = self.energy_head(node_features).squeeze(-1)
        atomic_energy = atomic_energy * self.energy_scale + self.atomref[atomic_numbers]
        n_structures = (
            int(batch["n_structures"])
            if "n_structures" in batch
            else (int(atom_batch.max().item()) + 1 if atom_batch.numel() else 0)
        )
        energy = atomic_energy.new_zeros(n_structures)
        energy.index_add_(0, atom_batch, atomic_energy)
        result = {
            "energy": energy,
            "atomic_energy": atomic_energy,
            "node_features": node_features,
            "vector_node_features": outputs["vector_representation"],
        }
        if "memory_adaptation_diagnostics" in outputs:
            result["memory_adaptation_diagnostics"] = outputs[
                "memory_adaptation_diagnostics"
            ]
        if "irrep_memory_diagnostics" in outputs:
            result["irrep_memory_diagnostics"] = outputs[
                "irrep_memory_diagnostics"
            ]
        if "irrep_memory_v2a_diagnostics" in outputs:
            result["irrep_memory_v2a_diagnostics"] = outputs[
                "irrep_memory_v2a_diagnostics"
            ]
        return result

    def memory_parameter_counts(self) -> dict[str, int]:
        counts = {
            "key": 0,
            "value": 0,
            "query": 0,
            "forget": 0,
            "read_projection": 0,
            "alpha": 0,
            "input_gate": 0,
        }
        for layer in self.memory_layers:
            for name, count in layer.parameter_counts().items():
                counts[name] += count
        return counts

    def preaggregation_memory_parameter_counts(self) -> dict[str, int]:
        counts = {
            "key": 0,
            "query": 0,
            "scalar_value_projection": 0,
            "scalar_forget": 0,
            "scalar_read_projection": 0,
            "scalar_alpha": 0,
        }
        for layer in self.preaggregation_memory_layers:
            for name, count in layer.parameter_counts().items():
                counts[name] += count
        return counts

    def irrep_memory_parameter_counts(self) -> dict[str, int]:
        names = (
            "geo_key",
            "geo_query",
            "l1_amplitude",
            "l1_forget",
            "l1_projection",
            "l1_alpha",
            "l2_amplitude",
            "l2_forget",
            "l2_projection",
            "l2_alpha",
        )
        if self.irrep_memory is not None:
            return self.irrep_memory.parameter_counts()
        if self.irrep_memory_v2a is not None:
            return self.irrep_memory_v2a.parameter_counts()
        return {name: 0 for name in names}


__all__ = ["PaiNNBackbone", "PaiNNGAMAdapter"]
