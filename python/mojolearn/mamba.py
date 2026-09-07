# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public FP32 Mamba-1, Mamba-2 and Mamba-3 block APIs for the alpha surface.

All six Block/State names are also exported from ``mojolearn``. Each block
accepts caller-supplied weights and exposes ``forward(x, state=None)``,
``step(x, state)``, ``allocate_state(...)`` and ``backward(x, grad_output)``.
Forward supports fast/deterministic/identical modes; backward currently
supports IDENTICAL zero-state prefill only and recomputes forward internally.
Backward returns ``x`` plus every constructor weight gradient (11 leaves for
Mamba1, 10 each for Mamba2 and Mamba3). No PyTorch dependency is required by
these native-backed public methods.

Shapes and fixed architectural constants are documented on each class.
FAST/DETERMINISTIC backward, carried-state or decode backward, final-state
cotangents, automatic differentiation integration, and a complete Mamba model
trainer are not exposed. The package requires a matching native extension
for the selected vendor and numeric mode; Python symbols alone are not a
working GPU wheel.

Evidence is fixture-scoped. Historical three-vendor native certificates and
new Python API checks have separate scopes. The retained September 6 NVIDIA
run3 reports 102 forward/API checks passing in each of FAST and IDENTICAL,
and five Mamba2/3 backward surface tests passing. Those source-run results do
not certify an unbuilt alpha wheel or every vendor/shape/mode. See
``mamba/PUBLIC_ALPHA_SURFACE.md`` for the exact retained paths and outstanding
wheel qualification.

Example with correctly shaped FP32 ``weights``, ``x`` and ``dy``::

    from mojolearn.mamba import Mamba2Block
    block = Mamba2Block(weights, numeric_mode="identical")
    y = block.forward(x)
    gradients = block.backward(x, dy)
    state = block.allocate_state(batch_size=x.shape[0])
    y0 = block.step(x[:, :1], state)
"""

from ._mamba_impl import (
    Mamba1Block,
    Mamba1State,
    Mamba2Block,
    Mamba2State,
    Mamba3Block,
    Mamba3State,
)

__all__ = [
    "Mamba1Block",
    "Mamba1State",
    "Mamba2Block",
    "Mamba2State",
    "Mamba3Block",
    "Mamba3State",
]
