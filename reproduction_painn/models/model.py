"""Dense PaiNN and its optional geometric associative memory extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .backbones import PaiNNBackbone


@dataclass(frozen=True)
class MLIPConfig:
    """Configuration shared by the Dense and GAM PaiNN variants."""

    type: str = "dense"
    backbone: str = "painn"
    n_atom_basis: int = 128
    n_interactions: int = 3
    n_rbf: int = 20
    cutoff: float = 5.0
    max_atomic_number: int = 100
    epsilon: float = 1.0e-8
    memory_enabled: bool = False
    memory_key_dim: int = 16
    memory_value_dim: int = 64
    memory_alpha_init: float = 0.1
    memory_input_gate_enabled: bool = False
    memory_normalizer_enabled: bool = False
    memory_normalized_addressing_enabled: bool = False
    memory_environment_query_enabled: bool = False
    preaggregation_memory_enabled: bool = False
    preaggregation_memory_key_dim: int = 16
    preaggregation_memory_value_dim: int = 64
    preaggregation_memory_alpha_init: float = 0.1
    irrep_memory_enabled: bool = False
    irrep_memory_orders: tuple[int, ...] = (1, 2)
    irrep_memory_key_dim: int = 16
    irrep_memory_value_dim: int = 64
    irrep_memory_alpha_init: float = 0.01
    irrep_memory_read_before_write: bool = False
    irrep_memory_persistent: bool = True
    irrep_memory_v2a_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "irrep_memory_orders", tuple(self.irrep_memory_orders)
        )
        if self.type != "dense":
            raise ValueError(
                "model type must be 'dense'; enable GAM with memory_enabled"
            )
        if self.backbone != "painn":
            raise ValueError(f"unsupported backbone {self.backbone!r}")
        if self.n_atom_basis < 1 or self.n_interactions < 1 or self.n_rbf < 1:
            raise ValueError(
                "backbone dimensions and interaction count must be positive"
            )
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if self.memory_key_dim < 1 or self.memory_value_dim < 1:
            raise ValueError("memory dimensions must be positive")
        if (
            self.preaggregation_memory_key_dim < 1
            or self.preaggregation_memory_value_dim < 1
        ):
            raise ValueError("pre-aggregation memory dimensions must be positive")
        if self.irrep_memory_key_dim < 1 or self.irrep_memory_value_dim < 1:
            raise ValueError("irrep memory dimensions must be positive")
        if (
            self.memory_input_gate_enabled or self.memory_normalizer_enabled
        ) and not self.memory_enabled:
            raise ValueError("memory input gate and normalizer require GAM-S")
        scalar_experiments = (
            self.memory_normalized_addressing_enabled,
            self.memory_environment_query_enabled,
        )
        if any(scalar_experiments) and not self.memory_enabled:
            raise ValueError("GAM and GAM-S-EnvQ require GAM-S")
        if sum(scalar_experiments) > 1:
            raise ValueError("GAM and GAM-S-EnvQ are independent experiments")
        if any(scalar_experiments) and (
            self.memory_input_gate_enabled or self.memory_normalizer_enabled
        ):
            raise ValueError("GAM/EnvQ cannot use i/n ablations")
        if self.preaggregation_memory_enabled and self.memory_enabled:
            raise ValueError(
                "pre-aggregation memory and legacy geometric memory are separate "
                "experimental variants"
            )
        if self.irrep_memory_enabled:
            if not self.memory_enabled:
                raise ValueError("Irrep-GAM requires the unchanged GAM-S l=0 anchor")
            if not self.irrep_memory_orders or not set(
                self.irrep_memory_orders
            ).issubset({1, 2}):
                raise ValueError(
                    "irrep_memory_orders must be a non-empty subset of {1, 2}"
                )
            if self.memory_input_gate_enabled or self.memory_normalizer_enabled:
                raise ValueError("Irrep-GAM cannot use input-gate/normalizer ablations")
            if self.preaggregation_memory_enabled:
                raise ValueError("Irrep-GAM cannot use GAM-Mij native messages")
        if self.irrep_memory_v2a_enabled:
            if not self.memory_enabled:
                raise ValueError("Irrep-GAM-v2a requires the unchanged GAM-S anchor")
            if self.irrep_memory_enabled:
                raise ValueError("Irrep-GAM-v2a is isolated from legacy Irrep-GAM")
            if self.memory_input_gate_enabled or self.memory_normalizer_enabled:
                raise ValueError("Irrep-GAM-v2a cannot use i/n ablations")
            if self.preaggregation_memory_enabled:
                raise ValueError("Irrep-GAM-v2a cannot use GAM-Mij native messages")
        if any(scalar_experiments) and (
            self.preaggregation_memory_enabled
            or self.irrep_memory_enabled
            or self.irrep_memory_v2a_enabled
        ):
            raise ValueError("GAM/EnvQ cannot be mixed with other memories")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MLIPConfig:
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BackboneAgnosticMLIP(nn.Module):
    """Expose a small energy/force contract around the PaiNN backbone."""

    def __init__(self, backbone: nn.Module, *, config: MLIPConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        compute_forces: bool = True,
    ) -> dict[str, Any]:
        positions = batch["positions"]
        if compute_forces and not positions.requires_grad:
            positions = positions.requires_grad_(True)
            batch = dict(batch)
            batch["positions"] = positions
        result: dict[str, Any] = {**self.backbone(batch), "positions": positions}
        if "energy" not in result or "node_features" not in result:
            raise KeyError("backbone must return 'energy' and 'node_features'")
        if compute_forces:
            result["forces"] = -torch.autograd.grad(
                result["energy"].sum(),
                positions,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
        return result

    def parameter_counts(self) -> dict[str, int]:
        memory = self.backbone.memory_parameter_counts()
        preaggregation = self.backbone.preaggregation_memory_parameter_counts()
        irrep = self.backbone.irrep_memory_parameter_counts()
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "memory_total": sum(memory.values()),
            "memory_key": memory["key"],
            "memory_value": memory["value"],
            "memory_query": memory["query"],
            "memory_forget": memory["forget"],
            "memory_read_projection": memory["read_projection"],
            "memory_alpha": memory["alpha"],
            "memory_input_gate": memory["input_gate"],
            "preaggregation_memory_total": sum(preaggregation.values()),
            **{
                f"preaggregation_memory_{name}": count
                for name, count in preaggregation.items()
            },
            "irrep_memory_total": sum(irrep.values()),
            **{f"irrep_memory_{name}": count for name, count in irrep.items()},
        }


def build_model(
    config: MLIPConfig,
    atomref: torch.Tensor | None = None,
    energy_scale: float = 1.0,
) -> BackboneAgnosticMLIP:
    """Build Dense PaiNN, optionally enabling GAM in every interaction block."""
    backbone = PaiNNBackbone(
        n_atom_basis=config.n_atom_basis,
        n_interactions=config.n_interactions,
        n_rbf=config.n_rbf,
        cutoff=config.cutoff,
        max_atomic_number=config.max_atomic_number,
        epsilon=config.epsilon,
        atomref=atomref,
        energy_scale=energy_scale,
        memory_enabled=config.memory_enabled,
        memory_key_dim=config.memory_key_dim,
        memory_value_dim=config.memory_value_dim,
        memory_alpha_init=config.memory_alpha_init,
        memory_input_gate_enabled=config.memory_input_gate_enabled,
        memory_normalizer_enabled=config.memory_normalizer_enabled,
        memory_normalized_addressing_enabled=(
            config.memory_normalized_addressing_enabled
        ),
        memory_environment_query_enabled=config.memory_environment_query_enabled,
        preaggregation_memory_enabled=config.preaggregation_memory_enabled,
        preaggregation_memory_key_dim=config.preaggregation_memory_key_dim,
        preaggregation_memory_value_dim=config.preaggregation_memory_value_dim,
        preaggregation_memory_alpha_init=config.preaggregation_memory_alpha_init,
        irrep_memory_enabled=config.irrep_memory_enabled,
        irrep_memory_orders=config.irrep_memory_orders,
        irrep_memory_key_dim=config.irrep_memory_key_dim,
        irrep_memory_value_dim=config.irrep_memory_value_dim,
        irrep_memory_alpha_init=config.irrep_memory_alpha_init,
        irrep_memory_read_before_write=config.irrep_memory_read_before_write,
        irrep_memory_persistent=config.irrep_memory_persistent,
        irrep_memory_v2a_enabled=config.irrep_memory_v2a_enabled,
    )
    return BackboneAgnosticMLIP(backbone, config=config)


__all__ = ["BackboneAgnosticMLIP", "MLIPConfig", "build_model"]
