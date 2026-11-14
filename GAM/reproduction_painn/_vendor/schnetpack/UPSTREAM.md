# SchNetPack upstream provenance

- Repository: <https://github.com/atomistic-machine-learning/schnetpack.git>
- Release tag: `v2.2.0`
- Release commit: `4c967da`
- Downloaded archive: <https://github.com/atomistic-machine-learning/schnetpack/archive/refs/tags/v2.2.0.zip>
- Archive SHA-256: `9137c30bef0f8e8232aff4025c38373205c226a410297acf0b2b0bcb0e6b8ecd`
- Download date: 2026-09-03
- License: MIT; the upstream `LICENSE` is preserved beside this file.

The official files required by PaiNN were copied byte-for-byte from the release:

- `properties.py`
- `representation/painn.py`
- `nn/base.py`
- `nn/cutoff.py`
- `nn/radial.py`
- `nn/scatter.py`
- `nn/utils.py`

The small package `__init__.py` files are local glue which export only the symbols
needed by PaiNN. The official `representation/painn.py` is not modified and has
SHA-256:

`227414bcf7505ca008f9fcc92681a5731950efa686fa8f4352fb7c8b4211960b`
