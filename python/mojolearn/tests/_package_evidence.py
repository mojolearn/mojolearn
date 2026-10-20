"""Test-only package provenance; load by filename without selecting mojolearn."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


_DISTRIBUTIONS = ("mojolearn", "mojolearn-core", "mojolearn-nvidia", "mojolearn-amd", "mojolearn-nvidia-ptx")


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_evidence(package, *, require_installed=False):
    """Record the imported package and native modules; optionally require wheel ownership."""
    path = Path(package.__file__).resolve()
    record = {
        "selection": "installed" if require_installed else "current",
        "package": str(path),
        "version": getattr(package, "__version__", None),
        "python": sys.executable,
        "prefix": sys.prefix,
        "native_bindings": [],
    }
    distributions = {}
    if require_installed:
        for name in _DISTRIBUTIONS:
            try:
                dist = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            direct = json.loads(dist.read_text("direct_url.json") or "{}")
            if direct.get("dir_info", {}).get("editable"):
                raise RuntimeError(f"Installed-package check refuses editable distribution {name}")
            distributions[name] = dist
        owners = [dist for dist in distributions.values()
                  if any(str(entry) == "mojolearn/__init__.py" for entry in dist.files or ())]
        if len(owners) != 1:
            raise RuntimeError("Installed-package check requires the mojolearn distribution")
        owner = owners[0]
        expected = Path(owner.locate_file("mojolearn/__init__.py")).resolve()
        if (path != expected or not path.is_relative_to(Path(sys.prefix).resolve())
                or not {"site-packages", "dist-packages"}.intersection(path.parts)):
            raise RuntimeError(f"Installed-package check detected checkout shadowing: {path}; expected {expected}")
        if record["version"] != owner.version:
            raise RuntimeError("Imported mojolearn version differs from installed distribution metadata")
        record["distributions"] = {name: dist.version for name, dist in distributions.items()}
        # RECORD binds files to the base wheel or a split GPU wheel. Merely
        # finding a .so under site-packages would admit a copied local binary.
        owned = {}
        for name, dist in distributions.items():
            for entry in dist.files or ():
                if str(entry).endswith((".so", ".pyd", ".dylib")):
                    owned[Path(dist.locate_file(entry)).resolve()] = (name, entry)

    native = {}
    for name, module in tuple(sys.modules.items()):
        if not (name.startswith("mojolearn.") or name.startswith("_mojolearn")):
            continue
        filename = vars(module).get("__file__") if module is not None else None
        if filename and str(filename).endswith((".so", ".pyd", ".dylib")):
            native.setdefault(Path(filename).resolve(), []).append(name)
    for binding, names in sorted(native.items()):
        digest = _sha256(binding)
        row = {"path": str(binding), "sha256": digest, "modules": sorted(names)}
        if require_installed:
            if (binding not in owned or not binding.is_relative_to(Path(sys.prefix).resolve())
                    or not {"site-packages", "dist-packages"}.intersection(binding.parts)):
                raise RuntimeError(f"Native binding is not owned by an installed mojolearn wheel: {binding}")
            owner, entry = owned[binding]
            if entry.hash is None or entry.hash.mode != "sha256":
                raise RuntimeError(f"Native binding has no SHA-256 wheel RECORD entry: {binding}")
            import base64
            recorded = base64.urlsafe_b64decode(entry.hash.value + "==").hex()
            if digest != recorded:
                raise RuntimeError(f"Native binding differs from its installed wheel RECORD: {binding}")
            row["distribution"] = owner
        record["native_bindings"].append(row)
    return record
