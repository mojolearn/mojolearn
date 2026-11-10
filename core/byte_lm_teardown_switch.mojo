# SPDX-License-Identifier: Apache-2.0
# The byte LM teardown switch, read once per process. No GPU or Python import:
# the GPU extension (_mojolearn_byte_lm) and the host tokenizer extension
# (through bindings/byte_lm_trainer_native.mojo) both read it.
from std.os import getenv
from core.shared_slot import SharedSlot


struct _TeardownSwitch(Defaultable, Movable):
    """DEVIATION 2518 variant (b), MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL=1, read
    ONCE per process (plan row N101: it was a getenv per stateless step).
    tools/byte_lm_lifetime_diag.py runs each variant in its own process."""
    var locked: Bool

    def __init__(out self):
        self.locked = String(getenv("MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL")) == "1"


comptime BYTE_LM_TEARDOWN_SWITCH = SharedSlot[_TeardownSwitch, "MojolearnByteLMTeardownWithGilV2"]


def byte_lm_teardown_with_gil() raises -> Bool:
    """The process's MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL switch."""
    return BYTE_LM_TEARDOWN_SWITCH.get_or_create_ptr()[].locked
