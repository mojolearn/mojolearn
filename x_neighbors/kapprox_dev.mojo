# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-kapprox: the KernelPCA resident switch (GPU binding).

FAST + Apple only, behind `-D MOJOLEARN_KPCA_RESIDENT`; any other build
answers 0, and the Python layer asks the binding's constant before taking
that path, so IDENTICAL runs main's code. The chi2 samplers' device ops
(`-D MOJOLEARN_KAPPROX_DEVICE`) and SparseRandomProjection's one-launch draw
(`-D MOJOLEARN_SPARSE_RP_DEVICE`) were DROPPED-quality; their code is on
lane/apple-fast-kapprox @ 10d5a7970.
"""
from std.python import PythonObject
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST


#: KernelPCA.fit resident on x_decomp's kit (one upload of X; the kernel
#: matrix, its centering and the top-k solve never leave the device):
#: FAST + Apple + `-D MOJOLEARN_KPCA_RESIDENT`. A Python-only route; this
#: binary only answers whether it is on.
comptime XN_KPCA_RESIDENT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_KPCA_RESIDENT"]()
)


def kpca_resident_binding() raises -> PythonObject:
    """1 when KernelPCA.fit takes the resident kit route."""
    comptime if XN_KPCA_RESIDENT:
        return PythonObject(1)
    return PythonObject(0)
