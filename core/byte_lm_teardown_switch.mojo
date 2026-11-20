# SPDX-License-Identifier: Apache-2.0
# The byte LM teardown switch, a build constant. No GPU or Python import:
# the GPU extension (_mojolearn_byte_lm) and the host tokenizer extension
# (through bindings/byte_lm_trainer_native.mojo) both read it.
from std.sys.compile import is_defined


comptime BYTE_LM_TEARDOWN_WITH_GIL = is_defined["MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL"]()
"""DEVIATION 2518 variant (b): release the stateless step's trainer and
context with the GIL held. Default off (unchanged); `-D
MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL` is the diagnostic arm
(tools/byte_lm_lifetime_diag.py runs each variant in its own build). It was
`MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL=1` in the environment, read once per
process into a SharedSlot (plan row N101), until lane envgate-neural-lm
(2026-11-15); the slot and its struct existed only for that read and are
gone."""


def byte_lm_teardown_with_gil() raises -> Bool:
    """The build's BYTE_LM_TEARDOWN_WITH_GIL switch."""
    return BYTE_LM_TEARDOWN_WITH_GIL
