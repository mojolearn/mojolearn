# SPDX-License-Identifier: Apache-2.0
"""Compatibility refusal for the retired implicit graph-capture backend."""
# Owner authorization 2026-10-27: exported artifacts replace Python graph
# capture and implicit weight refresh. Keep a native refusal for direct imports
# of the old backend, but do not register it with TorchDynamo.
from ..importers._program import _IMPORT_NATIVE, _IMPORT_CONTEXT

BACKEND_NAME = "mojolearn"


def backend(gm, example_inputs, *, numeric_mode=None, **options):
    return _IMPORT_NATIVE("torch_capture", (gm, example_inputs, numeric_mode, options), _IMPORT_CONTEXT)
