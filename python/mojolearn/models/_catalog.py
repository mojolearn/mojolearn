# SPDX-License-Identifier: Apache-2.0
"""Native catalog metadata and verification of already exported model artifacts.

Export a model and prepare its catalog check input in application/test code.
The native verifier admits the artifact and input, runs the existing imported
program provider, and completes both digests and the comparison report.
"""
from pathlib import Path
from .. import _backend
from .._version import __version__
from .._array import Array
from ..tokenizer import _binding
from ..importers._program import _IMPORT_CONTEXT

__all__ = ["catalog", "verify_catalog_entry"]
_NATIVE = _binding()._module
_CATALOG = Path(__file__).resolve().parent / "catalog.json"


def catalog():
    """Return independent native-decoded records from the shipped model catalog."""
    return _NATIVE.catalog_public_call("catalog", (), _NATIVE_CONTEXT)


def verify_catalog_entry(name, *, source=None, inputs=None, numeric_mode=None, cache=None):
    """Verify an exported catalog model using explicit native check inputs.

    ``source`` is an existing ONNX artifact or native ``GraphIR``. ``inputs``
    is one native Array/buffer, or a tuple/list of native input arrays in model
    order. Its byte digest must match the catalog's recorded check input.
    Export and seeded check-input generation belong to application/test setup;
    no Python model forward, exporter, downloader or random generator runs here.
    ``cache`` remains accepted for call compatibility; the artifact is supplied
    directly. ``numeric_mode`` uses the normal import provider selection.

    Returns the name, version, numeric mode, vendor, source, input/output digests,
    elapsed run milliseconds, expected digest and match result. A missing
    expected digest yields ``match=None``. Missing source/inputs are refused
    with the explicit native alternative.
    """
    return _NATIVE.catalog_public_call("verify", (name, source, inputs, numeric_mode, cache), _NATIVE_CONTEXT)


# Import-time connections only; product calls never resolve package bindings.
_NATIVE_CONTEXT = {"path": str(_CATALOG), "version": __version__, "vendor": _backend.vendor(),
                   "default_mode": _backend.default_mode(),
                   "backend_fields": vars(_backend), "import_context": _IMPORT_CONTEXT,
                   "array_class": Array}
_NATIVE.catalog_public_call("connect", (), _NATIVE_CONTEXT)
