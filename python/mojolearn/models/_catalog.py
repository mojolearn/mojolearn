# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The verified pretrained-model catalog: :func:`catalog` and :func:`verify_catalog_entry`."""
# Engineering notes. catalog.json beside this file is written by tools/model_catalog.py from
# tools/real_model_import.py box runs; it holds revisions, hashes and numbers only, never weights, model
# files or evaluation images. verify_catalog_entry rebuilds a model the way real_model_import.py does
# (same source, same export call, same import shapes, same seeded check input) so its output digest is
# comparable with the digests the catalog records per version, numeric mode and vendor. The check input
# and the digest are the tool's: numpy.random.default_rng(seed) and sha256 of the output bytes, 16 hex.
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

__all__ = ["catalog", "verify_catalog_entry"]

_HERE = Path(__file__).resolve().parent
_CATALOG = _HERE / "catalog.json"
_SCHEMA = "mojolearn-model-catalog/1"
_CACHE = os.path.join("~", ".cache", "mojolearn", "models")


def _load():
    try:
        doc = json.loads(_CATALOG.read_text())  # cpu-route: the shipped catalog file is read on the host
    except FileNotFoundError:
        return []
    if doc.get("schema") != _SCHEMA:
        raise RuntimeError(f"the model catalog has schema {doc.get('schema')!r}, expected {_SCHEMA!r}")
    return doc.get("models", [])


def catalog():
    """List the pretrained models in mojolearn's model catalog.

    Each entry describes one published pretrained model that mojolearn
    imports with :func:`mojolearn.import_model`: where it comes from, pinned
    exactly (torchvision weights and versions, a Hugging Face repository and
    commit, or an ONNX file's URL and SHA-256), the input shapes, the import
    call, its accuracy where a public evaluation exists, and for each
    mojolearn version and numeric mode the output digest on each GPU vendor
    with the latency and device memory measured there. In `identical` mode,
    mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs; an entry
    whose ``status`` is ``'verified'`` has the same output digest on every
    vendor it was run on, NVIDIA and AMD included.

    The catalog holds no weights or data: :func:`verify_catalog_entry`
    downloads the model from its source.

    Returns
    -------
    list of dict
        One dict per model, with the keys ``name``, ``kind``, ``status``,
        ``source``, ``inputs``, ``outputs``, ``import``, ``check``,
        ``accuracy`` and ``versions``. ``versions`` maps a mojolearn version
        to a numeric mode to ``{'digests': {vendor: digest}, 'vendors':
        {vendor: {'latency_ms': ..., 'device_memory_mb': ..., ...}}}``.

    Examples
    --------
    >>> import mojolearn
    >>> [m["name"] for m in mojolearn.models.catalog() if m["status"] == "verified"]
    """
    return copy.deepcopy(_load())


def _entry(name):
    for e in _load():  # glue: catalog entries, one per model
        if e.get("name") == name:
            return e
    names = ", ".join(e.get("name", "?") for e in _load())  # glue: catalog entry names for the message
    raise KeyError(f"no model {name!r} in the catalog (models: {names or 'none'})")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:  # cpu-route: hashing a downloaded model file on the host
        for chunk in iter(lambda: fh.read(1 << 20), b""):  # cpu-route: file bytes in 1 MB reads
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest):
    import urllib.request
    tmp = str(dest) + ".part"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:  # cpu-route: model file download
        while True:  # cpu-route: model file download in 1 MB reads
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)


def _no_tf32(torch):
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except AttributeError:
        pass


def _check_input(entry):
    """The catalog's seeded check input (the box tool's recipe), as one array."""
    from .._optional_numpy import require_numpy
    np = require_numpy("mojolearn.models.verify_catalog_entry")
    check = entry["check"]
    shape = tuple(entry["inputs"][0]["shape"])
    if check.get("input_kind") == "ids":
        raise ValueError(f"{entry['name']}: a token-id check input needs the model's vocabulary size; "
                         "this entry cannot be checked here")
    # cpu-route: the fixed check input is generated on the host, as the box runs generated it
    return np.random.default_rng(int(check["seed"])).standard_normal(shape, dtype=np.float32)


def _source_torchvision(entry, x, report):
    try:
        import torch
        import torchvision
    except ImportError:
        raise ImportError(f"checking {entry['name']} needs torch and torchvision") from None
    _no_tf32(torch)
    src = entry["source"]
    cls, _, member = src["weights"].partition(".")
    weights = getattr(getattr(torchvision.models, cls), member)
    ctor = getattr(torchvision.models, entry["name"])
    model = ctor(weights=weights).eval()
    report["source"] = {"weights": str(weights), "torch_version": torch.__version__,
                        "torchvision_version": torchvision.__version__}
    with torch.no_grad():
        return torch.export.export(model, (torch.from_numpy(x),))


def _source_hf(entry, x, report):
    try:
        import torch
        import transformers
    except ImportError:
        raise ImportError(f"checking {entry['name']} needs torch and transformers") from None
    _no_tf32(torch)
    src = entry["source"]
    lm = transformers.AutoModelForCausalLM.from_pretrained(src["repo"], revision=src.get("revision"),
                                                           torch_dtype=torch.float32)
    lm.config.use_cache = False

    class Logits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            return self.m(input_ids=input_ids, use_cache=False, return_dict=True).logits

    report["source"] = {"repo": src["repo"], "revision": src.get("revision"), "torch_version": torch.__version__,
                        "transformers_version": transformers.__version__}
    with torch.no_grad():
        try:
            return torch.export.export(Logits(lm).eval(), (torch.from_numpy(x),))
        except Exception:  # noqa: BLE001  the non-strict tracer, as the box runs fell back to
            return torch.export.export(Logits(lm).eval(), (torch.from_numpy(x),), strict=False)


def _source_onnx(entry, cache, report):
    src = entry["source"]
    cache = Path(os.path.expanduser(cache))
    cache.mkdir(parents=True, exist_ok=True)
    name = src["url"].rsplit("/", 1)[1]
    path = cache / name
    if not path.is_file():
        _download(src["url"], path)
    sha = _sha256_file(path)
    if sha != src["sha256"]:
        raise RuntimeError(f"{path} has SHA-256 {sha}; the catalog's file has {src['sha256']}. "
                           "Delete it to download it again.")
    report["source"] = {"url": src["url"], "sha256": sha}
    if not src.get("converted_sha256"):
        return str(path)
    try:
        import onnx
        from onnx import version_converter
    except ImportError:
        raise ImportError(f"checking {entry['name']} needs the onnx package (its file is converted to "
                          "opset 13 first)") from None
    conv = cache / (path.stem + "-opset13.onnx")
    if not conv.is_file():
        onnx.save(version_converter.convert_version(onnx.load(str(path)), 13), str(conv))
    report["source"]["converted_sha256"] = _sha256_file(conv)
    report["source"]["onnx_version"] = onnx.__version__
    return str(conv)


def verify_catalog_entry(name, *, cache=None):
    """Import a catalog model on this install and compare its output digest.

    Builds the model from its pinned source (downloading the weights or the
    model file, which needs network access and torch with torchvision,
    transformers or onnx as the source requires), imports it with
    :func:`mojolearn.import_model`, runs it once on the catalog's seeded
    check input and compares the SHA-256 digest of the output with the
    digest the catalog records for this mojolearn version and numeric mode.
    In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA
    and AMD GPUs, so the digest matches on any of them.

    Parameters
    ----------
    name : str
        A model name from :func:`catalog`, such as ``'resnet18'``.
    cache : str or path-like, optional
        Where ONNX model files are downloaded (default
        ``~/.cache/mojolearn/models``). torchvision and Hugging Face
        weights use those libraries' own caches.

    Returns
    -------
    dict
        ``name``, ``version``, ``numeric_mode``, ``vendor``, ``digest``
        (this install's output digest), ``expected`` (the catalog's digest,
        or None when the catalog has none for this version and mode),
        ``match`` (True or False, or None when there is nothing to compare
        with), ``input_digest``, ``run_ms`` (the wall time of this first
        run, warm-up included) and ``source``.

    Raises
    ------
    KeyError
        ``name`` is not in the catalog.
    ValueError
        The entry's status is not ``'verified'``, or the check input on this
        install differs from the catalog's.

    Examples
    --------
    >>> import mojolearn
    >>> report = mojolearn.models.verify_catalog_entry("resnet18")
    >>> report["match"]
    True
    """
    import time

    import mojolearn
    entry = _entry(name)
    if entry.get("status") != "verified":
        raise ValueError(f"{name}: catalog status is {entry.get('status')!r}; only verified models are checked")
    version = mojolearn.__version__
    mode = mojolearn.numeric_mode()
    report = {"name": name, "version": version, "numeric_mode": mode, "vendor": mojolearn.vendor()}
    x = _check_input(entry)
    want_in = entry["check"]["input_digest"]
    report["input_digest"] = hashlib.sha256(x.tobytes()).hexdigest()[:16]
    if report["input_digest"] != want_in:
        raise ValueError(f"{name}: the check input on this install has digest {report['input_digest']}, the "
                         f"catalog's has {want_in} (a different numpy random generator)")
    kind = entry["kind"]
    if kind == "torchvision":
        source = _source_torchvision(entry, x, report)
    elif kind == "hf-causal-lm":
        source = _source_hf(entry, x, report)
    else:
        source = _source_onnx(entry, cache or _CACHE, report)
    shapes = entry["check"].get("import_shapes")
    shapes = {k: tuple(v) for k, v in shapes.items()} if shapes else None  # glue: one entry per model input
    prog = mojolearn.import_model(source, shapes=shapes)
    t0 = time.perf_counter()
    y = prog.run(x)
    report["run_ms"] = (time.perf_counter() - t0) * 1e3
    y = y if isinstance(y, tuple) else (y,)
    h = hashlib.sha256()
    for a in y:  # glue: one digest update per model output
        h.update(a.tobytes())
    report["digest"] = h.hexdigest()[:16]
    cell = (entry.get("versions", {}).get(version) or {}).get(mode) or {}
    digests = cell.get("digests") or {}
    report["expected"] = next(iter(digests.values()), None)  # glue: the vendors' digests are equal
    report["match"] = None if report["expected"] is None else report["digest"] == report["expected"]
    return report
