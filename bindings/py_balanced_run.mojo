# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/py_balanced.mojo: typed Mojo only (no Python imports).
from core.shared_slot import SharedSlot


struct _ExactDictType(Defaultable, Movable):
    # Address of CPython's dict type (a static, immortal type), 0 until read.
    var address: Int

    def __init__(out self):
        self.address = 0


comptime _EXACT_DICT_TYPE = SharedSlot[_ExactDictType, "MojolearnPyBalancedDictTypeV2"]
