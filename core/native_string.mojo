# SPDX-License-Identifier: Apache-2.0
"""Owned UTF-8 text copied from a borrowed, NUL-terminated byte pointer."""
from std.ffi import CStringSpan, c_char


@always_inline
def unsafe_string_from_utf8_cstr[dtype: DType, src_origin: Origin, //](
    source: Pointer[Scalar[dtype], src_origin],
) -> String:
    """Copy bytes preceding the first NUL into one owned String.

    The source must be non-null, NUL-terminated, valid UTF-8, and alive for
    this call. Its origin is retained by the C-string/byte views; neither
    view escapes. This is the Mojo 1.2 spelling of the former unsafe
    String pointer constructor, with no decoding or replacement pass.
    """
    comptime assert dtype == DType.int8 or dtype == DType.uint8
    var text = CStringSpan(unsafe_from_ptr=source.bitcast[c_char]().as_imm())
    return String(unsafe_from_utf8=text.as_bytes())
