# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/sequence_dtype_native.mojo` (wave 8 run split): the
SyntaxError offset of an empty compact-shape component, typed Mojo only.
Imports no Python (tools/python_free_dirs.sh). The statements are the former
scan in `sequence_dtype_syntax`, unchanged (lane rs-sequence-tokenizer,
2026-11-12)."""


def sequence_dtype_syntax_offset(text: String) -> Int:
    """The 1-based offset NumPy's literal parser reports for the first empty
    component of a compact dtype shape (1 when none is found)."""
    var offset = 1
    var empty = True
    for i in range(text.byte_length()):  # driver-loop: locate an empty compact-shape component in native parser text
        var character = UInt8(text.unsafe_ptr()[i])
        if character == 32 or character == 40:
            continue
        if character == 44:
            if empty:
                offset = i + 1
                break
            empty = True
        else:
            empty = False
    return offset
