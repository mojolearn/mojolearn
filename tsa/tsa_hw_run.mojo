# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/tsa_hw_native.mojo` (wave 8 run split): the Holt-Winters
component gather of `hw_component`, typed Mojo only. Imports no Python
(tools/python_free_dirs.sh). The loop is the former door loop unchanged (lane
rs-sequence-tokenizer, 2026-11-12)."""
from core.xseq_marshal import xseq_column_copy_f32

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def hw_component_gather(comps_addr: Int, base: Int, which: Int, rows: Int, b: Int):
    """Component `which` of the time-major packed components at `comps_addr`
    into the series-major float32 output at `base` (`b` series of `rows`)."""
    for series in range(b):  # driver-loop: gather each native component column into its series-major output
        xseq_column_copy_f32(_F32P(unsafe_from_address=comps_addr + which * rows * b * 4),
            _F32P(unsafe_from_address=base + series * rows * 4), rows, b, series)
