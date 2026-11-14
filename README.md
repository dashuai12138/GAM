# GAM: Geometric Associative Memory

Official implementation of **Geometric Associative Memory (GAM)**.

This repository provides the PaiNN baseline and PaiNN + GAM on the 3BPA benchmark.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Prepare the 3BPA dataset:

```bash
python -m reproduction_painn.prepare_3bpa
```

## Training

```bash
python -m reproduction_painn.train configs/painn.json --device cuda
python -m reproduction_painn.train configs/painn_gam.json --device cuda
```

## Evaluation

```bash
python -m reproduction_painn.evaluate configs/painn.json --checkpoint reproduction_painn/runs/3bpa/dense_3000/best_force.pt --domain 300K --device cuda

python -m reproduction_painn.evaluate configs/painn_gam.json --checkpoint reproduction_painn/runs/3bpa/gam/best_force.pt --domain 300K --device cuda
```

Other test sets: `600K`, `1200K`, `dihedral`.
