# SPDX-License-Identifier: Apache-2.0
"""Typed input marshaling moved word for word from bindings/classical_arrays.mojo."""
from core.family_cast_impl import family_cast_rows_finite
from core.host_finite_scan import host_all_finite_f32_strided


def classical_cast_rows(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int, name: String,
    borrowed: Bool, dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """Refuse NaN/infinity in the rows x cols input at `addr` (byte strides
    s0, s1) and, unless `borrowed` (float32 in C order, used as is), write
    its classical_f32_bits words to `dst` in C order.
    A borrowed input is not rebuilt: its words are only tested, by the
    shared SIMD scan split across the host's cores
    (`core/host_finite_scan.mojo`). Any other input goes through
    `family_cast_rows_finite`, the same per-item classical_f32_bits word in
    row tasks across the host's cores. The refusal and its wording are the
    serial walk's; `dst` is the caller's fresh result, discarded on a
    refusal, so writing every cell before refusing changes no output."""
    # parallel: finiteness scan / cast across host cores (0.8.66 regression C1, 13 ns/element serial)
    var finite: Bool
    if borrowed:
        finite = host_all_finite_f32_strided(addr, s0, s1, rows, cols)
    else:
        finite = family_cast_rows_finite(addr, s0, s1, rows, cols, item, kind, dst)
    if not finite:
        raise Error("Input " + name + " contains NaN or infinity")


def classical_check_weights(
    input: MutPointer[Float32, MutUntrackedOrigin], output: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, scalar: Bool,
) raises -> Bool:
    var any_positive = False
    for i in range(rows):  # native input marshaling: weight validation and scalar expansion
        var v = input[0 if scalar else i]
        if v < 0:
            raise Error("Negative values in data passed to `sample_weight`")
        any_positive = any_positive or v > 0
        if scalar:
            output[i] = v
    return any_positive
