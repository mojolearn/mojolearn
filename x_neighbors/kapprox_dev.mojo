# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-kapprox: the KernelPCA resident switch (GPU binding).

FAST + Apple default for the validated RBF top-k route; rollback with
`-D MOJOLEARN_KPCA_RESIDENT_OFF`. Any other vendor or numeric mode
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
#: KEEP, 2026-10-04, w2-w4d-kpca-{taxi,istella}, measured 34b4f6c72:
#: taxi 850.7 -> 127.5 ms; istella 919.5 -> 201.0 ms. w2-w4d-kpca-q
#: PASS: eigenvalue relative differences < 9.88e-7, subspace angles < 3.79e-6.
#: First read 0.1 ms in both arms; only internal kernel storage is resident.
#: Python limits the default to the measured RBF / auto top-k route.
#: FAST + Apple only; MOJOLEARN_KPCA_RESIDENT_OFF restores the old route.
#: See docs/apple-fast/EXPERIMENTS.md (KPCA_RESIDENT).
comptime XN_KPCA_RESIDENT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_KPCA_RESIDENT_OFF"]()
)


def kpca_resident_binding() raises -> PythonObject:
    """1 when KernelPCA.fit takes the resident kit route."""
    comptime if XN_KPCA_RESIDENT:
        return PythonObject(1)
    return PythonObject(0)
