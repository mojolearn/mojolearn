# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Verification: the Torch and MAX adapters (adapters/, python/mojolearn/adapters/, the wheels'
# _adapters/) are the SAME API in another format: front-ends over the kernels the Python API calls,
# computing nothing of their own. They get NO identity lanes, NO verifier rows and NO sabotage arms
# (Andrew, 2026-10-24; AGENTS.md "Framework adapters are the same API in another format"). Their
# qualification is the installed-wheel smoke on each vendor (tools/adapter_wheel_smoke.py, run by
# the release gate).
"""Optional PyTorch and MAX adapters.

``mojolearn.adapters.torch`` registers mojolearn's operators with the PyTorch
dispatcher (``torch.ops.mojolearn.operation`` and ``torch.ops.mojolearn.train``).
``mojolearn.adapters.max`` gives a MAX graph mojolearn's custom operations.
Both call the compiled mojolearn library of your GPU install; the files ship
in the mojolearn GPU wheels.

Importing ``mojolearn`` never imports PyTorch or MAX: each adapter loads only
when its module is imported, and refuses with an install hint when its
framework is missing (``pip install "mojolearn[torch]"``,
``pip install "mojolearn[max]"``).
"""

__all__ = []
