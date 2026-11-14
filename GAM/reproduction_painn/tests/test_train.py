from __future__ import annotations

import pytest
import torch

from reproduction_painn.models import MLIPConfig, build_model
from reproduction_painn.training.trainer import (
    StaircaseExponentialScheduler,
    learning_rate_at_step,
    load_checkpoint,
    save_checkpoint,
)


class _SamplerState:
    def __init__(self):
        self.value = {"position": 7}

    def state_dict(self):
        return dict(self.value)

    def load_state_dict(self, state):
        self.value = dict(state)


def test_deepmd_style_learning_rate_schedule_reaches_and_holds_stop():
    assert learning_rate_at_step(0, 1_000_000, 1.0e-3, 1.0e-6, 5_000) == 1.0e-3
    assert learning_rate_at_step(1_000_000, 1_000_000, 1.0e-3, 1.0e-6, 5_000) == (
        pytest.approx(1.0e-6)
    )
    assert learning_rate_at_step(2_000_000, 1_000_000, 1.0e-3, 1.0e-6, 5_000) == (
        pytest.approx(1.0e-6)
    )


def test_constant_learning_rate_without_stop_value():
    assert learning_rate_at_step(123, 1_000, 2.0e-4, None, 100) == 2.0e-4


def test_checkpoint_persists_epoch_global_step_and_early_stop_state(tmp_path):
    config = MLIPConfig(n_atom_basis=8, n_interactions=1, n_rbf=4, cutoff=2.5)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = StaircaseExponentialScheduler(optimizer, 100, 1.0e-3, 1.0e-5, 10)
    scheduler.step(37)
    sampler = _SamplerState()
    path = tmp_path / "checkpoint.pt"
    run_config = {"model": config.to_dict()}
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        sampler,
        37,
        {"force": 0.4, "energy": 0.03, "joint": 0.1},
        run_config,
        epoch=4,
        epochs_without_improvement=3,
    )
    raw = torch.load(path, weights_only=False)
    assert raw["step"] == raw["global_step"] == 37
    assert raw["epoch"] == 4
    assert raw["epochs_without_improvement"] == 3

    restored_model = build_model(config)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1.0e-3)
    restored_scheduler = StaircaseExponentialScheduler(
        restored_optimizer, 100, 1.0e-3, 1.0e-5, 10
    )
    restored_sampler = _SamplerState()
    restored_sampler.value = {"position": 0}
    step, epoch, stale_epochs, metrics, restored_config = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_sampler,
        torch.device("cpu"),
    )
    assert (step, epoch, stale_epochs) == (37, 4, 3)
    assert metrics["force"] == 0.4
    assert restored_config == run_config
    assert restored_sampler.value == {"position": 7}
