# SPDX-License-Identifier: Apache-2.0
"""THE FOLD X WORD's numbering (core/fold_x_device.mojo), with no device import (lane vdf-scope-cluster3).

A member that hands its kernel entry a fold slot as a negative X address word needs only this numbering; the
device copy (`fold_x_fill`) and the store stay in core/fold_x_device.mojo. Kept apart so a member file shared by a
device image and its CPU-only host image (core/xneighbors_member.mojo, core/xcluster_member.mojo, through
core/member_fold_x.mojo) imports no max.gpu module for it.
"""

#: `fold_x_slot` of a host address word.
comptime FOLD_X_NONE = -1


def fold_x_word(slot: Int) -> Int:
    """The X address word naming store slot `slot` (negative)."""
    return -(slot + 1)


def fold_x_slot(x_word: Int) -> Int:
    """The store slot a negative X address word names, else FOLD_X_NONE."""
    if x_word < 0:
        return -x_word - 1
    return FOLD_X_NONE
