"""Hash-verified official SchNetPack PaiNN core.

Only the minimal upstream sources required by PaiNN are stored inside this
package. The official representation file is executed byte-for-byte unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

PINNED_TAG = "v2.2.0"
PINNED_COMMIT = "4c967da"
PINNED_PAINN_SHA256 = "227414bcf7505ca008f9fcc92681a5731950efa686fa8f4352fb7c8b4211960b"

UPSTREAM_PACKAGE = Path(__file__).resolve().parents[1] / "_vendor" / "schnetpack"
UPSTREAM_PAINN = UPSTREAM_PACKAGE / "representation" / "painn.py"


def _verify_upstream() -> None:
    if not UPSTREAM_PAINN.is_file():
        raise RuntimeError(f"Pinned SchNetPack source is missing: {UPSTREAM_PAINN}")
    digest = hashlib.sha256(UPSTREAM_PAINN.read_bytes()).hexdigest()
    if digest != PINNED_PAINN_SHA256:
        raise RuntimeError(
            "Vendored SchNetPack painn.py differs from the pinned official file: "
            f"expected {PINNED_PAINN_SHA256}, got {digest}"
        )


def _namespace_package(name: str, path: Path) -> types.ModuleType:
    package = types.ModuleType(name)
    package.__file__ = str(path / "__init__.py")
    package.__package__ = name
    package.__path__ = [str(path)]
    spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    if spec is not None:
        spec.submodule_search_locations = [str(path)]
    package.__spec__ = spec
    sys.modules[name] = package
    return package


def _load_painn_module():
    _verify_upstream()
    loaded_root = sys.modules.get("schnetpack")
    if loaded_root is None:
        loaded_root = _namespace_package("schnetpack", UPSTREAM_PACKAGE)
        loaded_root.__version__ = PINNED_TAG.removeprefix("v")
    else:
        loaded_paths = [
            Path(path).resolve() for path in getattr(loaded_root, "__path__", [])
        ]
        if UPSTREAM_PACKAGE.resolve() not in loaded_paths:
            raise RuntimeError(
                "A different SchNetPack installation was imported before the pinned "
                "vendored source. Import reproduction_painn first."
            )
    properties_module = importlib.import_module("schnetpack.properties")
    nn_module = importlib.import_module("schnetpack.nn")
    module_name = "schnetpack.representation.painn"
    painn_module = sys.modules.get(module_name)
    if painn_module is None:
        representation_name = "schnetpack.representation"
        if representation_name not in sys.modules:
            _namespace_package(representation_name, UPSTREAM_PACKAGE / "representation")
        spec = importlib.util.spec_from_file_location(module_name, UPSTREAM_PAINN)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load official PaiNN: {UPSTREAM_PAINN}")
        painn_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = painn_module
        spec.loader.exec_module(painn_module)
    return painn_module, nn_module, properties_module


_painn, snn, properties = _load_painn_module()
OfficialPaiNN = _painn.PaiNN


__all__ = [
    "OfficialPaiNN",
    "PINNED_COMMIT",
    "PINNED_PAINN_SHA256",
    "PINNED_TAG",
    "UPSTREAM_PAINN",
    "properties",
    "snn",
]
