# PaiNN + GAM on 3BPA

Example implementation of **Geometric Associative Memory (GAM)** for atomistic
energy and force prediction. This repository provides the PaiNN baseline and
PaiNN + GAM using normalized scalar addressing.

GAM forms keys and values from invariant relation features
`[h_i, h_j, phi(r_ij)]`, writes them into a per-atom matrix memory, and retrieves
a scalar residual with a normalized query. Memory persists across interaction
layers within a forward pass and starts from zero for each new batch. The
PaiNN adapter adds the residual after interaction/mixing. Forces are computed
as the negative energy gradient.

## Files

```text
configs/
  painn.json
  painn_gam.json
reproduction_painn/
  models/
    gam.py                  # Standalone normalized-address GAM core
    geometric_memory.py     # Existing shared GAM helpers
    model.py                # Energy/force model and configuration
    painn.py                # Hash-verified upstream PaiNN loader
    backbones/painn.py      # PaiNN and GAM integration
  _vendor/schnetpack/        # Minimal PaiNN sources and upstream MIT license
  data/                     # Dataset, batching and neighbor construction
  training/                 # Trainer and evaluation metrics
  datasets/3bpa/split_seed42.json
  tests/
  prepare_3bpa.py
  train.py
  evaluate.py
requirements.txt
```

The implementation and experiment settings are extracted from the original
PaiNN and PaiNN + GAM 3BPA runs. Shared modules include the existing PaiNN/GAM
helper classes and imports. Experiment names and output paths use GAM naming.

## Environment

Use Python 3.11.15. The validated environment uses
PyTorch 2.10.0+cu126 and the package versions pinned in `requirements.txt`.
Run all commands from the repository root, the directory containing this file.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. For the validated CUDA
12.6 wheel, install dependencies with the following commands.

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Choose the appropriate PyTorch wheel for your driver and platform using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/).
The requirements also support an installed CPU wheel. Training and evaluation
accept `--device cpu`. The required SchNetPack PaiNN sources are vendored with
their original license and provenance. See
[`UPSTREAM.md`](reproduction_painn/_vendor/schnetpack/UPSTREAM.md).

## Prepare 3BPA

```bash
python -m reproduction_painn.prepare_3bpa
```

This downloads the BOTNet 3BPA archive at commit
`29e6d467317e4b5967b7ea5cbee54de953fa0d45`, checks SHA-256
`21fb6ff7a2386405f74fa283433f69650915b1af69303ff3aa7c3c684146b9c3`,
and creates `reproduction_painn/datasets/3bpa/raw/` and `processed/`.
An offline copy of `dataset_3BPA.tar.gz` can be placed in that `raw/` directory
before running the same command. Dataset source and usage information are
available in the [BOTNet dataset repository](https://github.com/davkovacs/BOTNet-datasets).

The versioned split uses `numpy.random.default_rng(42)` to select 450 training
and 50 validation structures from the 500-structure 300 K pool. Preparation
regenerates the same split. Each structure contains 27 atoms with non-periodic
boundary conditions. Positions, energies and forces use Å, eV and eV/Å.
Test domains contain 1,669 structures at 300 K, 2,138 at 600 K, 2,139 at
1200 K, and 7,047 in the dihedral set. Energy references and scaling are fitted
using the 450 training structures.

## Train

```bash
python -m reproduction_painn.train configs/painn.json --device cuda
python -m reproduction_painn.train configs/painn_gam.json --device cuda
```

Both runs use seed 42, 3,000 epochs, batch size 5, 128 features, 3 interactions,
20 radial basis functions, a 5 Å cutoff, dynamic neighbors, Adam, and the
existing staircase exponential learning-rate schedule from 1e-3 to 1e-6
with `decay_steps=4500`. The training objective is
`mean(((E_pred - E_true) / N_atoms)^2) + 100 * mean((F_pred - F_true)^2)`.
Validation uses the 50 validation structures every epoch. Early stopping is
disabled. The GAM configuration uses key dimension 16, value dimension 64,
and residual scale initialized to 0.1.

Outputs are written to `reproduction_painn/runs/3bpa/dense_3000/` and
`reproduction_painn/runs/3bpa/gam/`. `best_force.pt` selects the
lowest validation force RMSE. Resume a run with the same training command
plus `--resume`, which loads its `latest.pt`.

## Test

Evaluate the selected checkpoint on each complete test domain.

```bash
python -m reproduction_painn.evaluate configs/painn.json --checkpoint reproduction_painn/runs/3bpa/dense_3000/best_force.pt --domain 300K --device cuda
python -m reproduction_painn.evaluate configs/painn_gam.json --checkpoint reproduction_painn/runs/3bpa/gam/best_force.pt --domain 300K --device cuda
```

Repeat both commands with `--domain 600K`, `--domain 1200K` and
`--domain dihedral` to evaluate the other three domains.

Each command writes `evaluation_<domain>.json` and `.csv` beside the checkpoint.
Energy MAE/RMSE are reported in eV/atom and force MAE/RMSE in eV/Å. Multiply
the energy metrics by 27 to obtain eV/molecule for the paper's 3BPA convention.

## Code checks

```bash
python -m pytest -q reproduction_painn/tests
```

The data integration test runs after dataset preparation. Unit tests can also
run before preparation.

## Distribution and validation

The repository includes source code, two experiment configurations and the
fixed split. Dataset arrays are generated by the preparation command, and
model checkpoints are generated by training. `.gitignore` covers generated
data, weights, logs, caches and local environments.

The vendored SchNetPack code retains its MIT license. The authors still need
to select a project-wide license for their original GAM code before release.
