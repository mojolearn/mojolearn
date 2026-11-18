# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 636 over a borrowed host buffer, for the CPU host binding
(`bindings/_mojolearn_svm_host.mojo`) only. Moved out of
`svm/impl/svm_parameter.mojo` (cpu-gpu-cleanup c-svm, 2026-10-02): the GPU
binding checks X on the device after its one upload
(`svm/checks/device_select.mojo::check_finite_device`), so no GPU module
imports the host pool for it any more.

DEVIATION 2665 (2026-09-11): one threaded pass. The verdict is the one
parallel SIMD scan of `core/host_finite_scan.mojo` (`isfinite`'s predicate by
bits: the exponent all ones is a NaN or an infinity; lane fx-host-scans,
2026-11-18: this file's own span pool and 16-cell OR walk are gone). Only on
a refusal the first failing flat index is found by one serial walk, and the
message is `check_finite_list`'s.
"""

from core.host_finite_scan import host_all_finite_f32, host_first_nonfinite_f32


def check_finite_ptr(
    values: MutPointer[Float32, MutUntrackedOrigin], n: Int, what: String
) raises:
    """`check_finite_list` over `n` borrowed cells (DEVIATION 2665)."""
    if n <= 0:
        return
    # parallel: the one host finite scan; the index walk runs only on a refusal
    if host_all_finite_f32(values, n):
        return
    var first = host_first_nonfinite_f32(values, n)
    if first >= 0:
        raise Error(
            "svm: " + what + " contains a non-finite value at flat index "
            + String(first) + " (DEVIATION 636: a NaN or inf input cannot be"
            " fitted; a computed NaN has a vendor-specific payload)"
        )
