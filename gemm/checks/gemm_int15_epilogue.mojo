# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The epilogue of `mojolearn.identical.gemm.int15i64.v1` as ONE device
function a kernel calls at its store: the three exact Int32 sums of a cell
(`HH`, `HL + LH`, `LL`) to the cell's float32.

Lane lane/lowbit-int15, 2026-09-29, the epilogue fold (maintainer's
approval of the five-point interface, same day). Contract clauses W-5 (the
recombination in Int64), W-6 (the pinned Int64 to float32 seam) and W-7
(the scale `2^(ea[i] + eb[j])`); the seams are `checks/numerics_int15.mojo`'s,
called with the arguments every other plan calls them with.

TWO CALLERS, ONE FUNCTION.
  FUSED      lane/lowbit-mma-speed's sums kernel with `FUSED = True` calls it
             where it would store the three sums, so the product is one
             launch and no sums reach device memory.
  TWO LAUNCH `gemm_int15_tuned.mojo::int15_sums_epilogue_kernel` calls it on
             the sums the kernel with `FUSED = False` stored. Kept as a gate
             arm: fused and two-launch must print one digest.

THE SABOTAGE ARMS, each a define read here, so they reach both callers.
  `-D MOJOLEARN_LOWBIT_SABOTAGE=1`         every stored value flipped
  `-D MOJOLEARN_INT15_EPILOGUE_SABOTAGE=1` the high sum read from the middle slot
  `-D MOJOLEARN_INT15_EXPONENT_SABOTAGE=1` the column exponent read at the row
                                           index (`eb[i mod n]` for `eb[j]`)
Lane D's own arm (`MOJOLEARN_INT8_PIECES_SABOTAGE`, its middle sum takes
`HL` twice) lives in its kernel and reaches both callers too.

This file imports only the seams and the oracle's value flip, so the sums
kernel's file can import it without a cycle.
"""

from std.sys import is_defined

from checks.numerics_int15 import dequant_int15_pinned, int15_recombine
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: DEVIATION 2973, the value arm, the define every fifteen-bit plan reads.
comptime INT15_STORE_VALUE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: The epilogue's defect arm: the high sum is read from the middle slot.
comptime INT15_EPILOGUE_SABOTAGE = is_defined["MOJOLEARN_INT15_EPILOGUE_SABOTAGE"]()

#: The scale's defect arm: the column exponent is read at the row index.
comptime INT15_EXPONENT_SABOTAGE = is_defined["MOJOLEARN_INT15_EXPONENT_SABOTAGE"]()


def int15_epilogue_sabotage_name() -> String:
    comptime if INT15_EPILOGUE_SABOTAGE and INT15_EXPONENT_SABOTAGE:
        return String("INT15_EPILOGUE_READS_THE_WRONG_SUM+INT15_EXPONENT_OF_I_READ_AT_J")
    elif INT15_EPILOGUE_SABOTAGE:
        return String("INT15_EPILOGUE_READS_THE_WRONG_SUM")
    elif INT15_EXPONENT_SABOTAGE:
        return String("INT15_EXPONENT_OF_I_READ_AT_J")
    else:
        return String("none")


@always_inline
def int15_store_cell(
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    hh_in: Int32,
    mid: Int32,
    ll: Int32,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """Cell `(i, j)` of the `m x n` output `c` (row-major) from its three
    sums: `ftz(f32_pinned(HH 2^14 + mid 2^7 + LL) * 2^(ea[i] + eb[j]))`.
    Masked to the output: a thread outside it stores nothing."""
    if i >= m or j >= n:
        return
    var hh = hh_in
    comptime if INT15_EPILOGUE_SABOTAGE:
        # THE DEFECT ARM: a slot read one place off.
        hh = mid
    var jb = j
    comptime if INT15_EXPONENT_SABOTAGE:
        # THE DEFECT ARM: the exponent of the row index read for the column.
        jb = i % n
    var out = dequant_int15_pinned(
        int15_recombine(hh, mid, ll), Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(jb))
    )
    comptime if INT15_STORE_VALUE_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(i * n + j, out)
