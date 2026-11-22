# SPDX-License-Identifier: Apache-2.0
"""RUN of the base binding's host helpers (bindings/host_helpers.mojo): the
typed int64 buffer pointer. No Python import (tools/python_free_dirs.sh). Lane
rs-shared-infra-b (2026-11-12): moved word for word."""


def _i64_ptr(addr: Int) raises -> MutPointer[Int64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int64 buffer address")
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)
