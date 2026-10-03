# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-py2mojo-linear: the build-time switch for the data-path
work this lane moved out of Python (svm, kernel_methods, estimators,
x_linear). Every binding that carries one of the moved entries exports
`py2mojo_linear_flags()`; Python reads it and takes the Mojo entry when the
bit is set. `-D MOJOLEARN_PY2MOJO_linear_OFF` answers 0 and restores the
old Python path (the A/B arm A)."""

from std.sys.compile import is_defined

#: gamma='scale' exact variance (SVC, SVR, RBFSampler): `scale_gamma_limbs`
comptime PY2MOJO_SCALE_GAMMA = 1
#: per-row glue of the linear estimators (weights, thresholds, row normalise)
comptime PY2MOJO_ROWS = 2


def py2mojo_linear_flags() -> Int:
    comptime if is_defined["MOJOLEARN_PY2MOJO_linear_OFF"]():
        return 0
    return PY2MOJO_SCALE_GAMMA | PY2MOJO_ROWS
