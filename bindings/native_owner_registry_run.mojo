# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/native_owner_registry.mojo: typed Mojo only (no Python imports).


def native_owner_no_pins(value: Int, index: Int) abi("C") -> Int:
    # For audited owners consisting entirely of native words/handles.
    return 0 if index == -1 else -1
