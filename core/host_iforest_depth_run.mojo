# SPDX-License-Identifier: Apache-2.0
"""RUN of the IsolationForest depth step (bindings/host_iforest_depth_native.mojo,
`host_iforest_depth_step`): the integer and float `+1`/`-1` on a decoded
scalar's bits, written into the caller's 32-byte result buffer. Typed Mojo
only, no Python import (tools/python_free_dirs.sh). The door decodes the
scalar and resolves the promoted code, kind and width (UNPACK), then
reports the returned status and boxes the bytes (PACK). Lane
rs-boosting-trees (2026-11-12): moved word for word; the cast-invalid
report that sat between the cast check and the arithmetic now runs in the
door right after this call, before the status report, so a warning that
raises still raises before any result is published. One scalar per call:
nothing to split across cores."""
from core.native_scalar_decode import NativeRealBits
from core.native_depth_state import NativeDepthFormat, native_depth_float_cast, native_depth_add_one
from core.sequence_array_cast import SEQ_CAST_OVERFLOW, SEQ_CAST_INVALID


def host_iforest_depth_step_run(
    source_real: NativeRealBits,
    source_imag: NativeRealBits,
    code: String,
    kind: String,
    width: Int,
    subtract: Bool,
    ndarray: Bool,
    data: MutPointer[UInt8, MutUntrackedOrigin],
    mut cast_invalid: Bool,
) raises -> Int:
    """Write the stepped value into `data`; return the status word.
    `cast_invalid` is set when a signaling NaN fails the promotion cast
    (the door reports it with "cast" text before the status)."""
    var status = 0
    if kind == "i" or kind == "u":
        var bits = UInt64(0)-source_real.low if source_real.negative else source_real.low
        var mask = ~UInt64(0) if width == 8 else (UInt64(1) << UInt64(width*8))-1
        bits &= mask
        var signed = kind == "i"
        var overflow = bits == ((mask >> 1)+1 if signed else UInt64(0)) if subtract else bits == (mask >> 1 if signed else mask)
        bits = (bits-UInt64(1) if subtract else bits+UInt64(1)) & mask
        for i in range(width):  # driver-loop: integer ufunc arithmetic wraps within the promoted primitive width
            data[i] = UInt8((bits >> UInt64(i*8)) & 255)
        if overflow and not ndarray:
            status = SEQ_CAST_OVERFLOW
    else:
        var complex = kind == "c"
        var real_code = String("f") if code == "F" else String("d") if code == "D" else String("g") if code == "G" else code
        var format = NativeDepthFormat(real_code,width//2 if complex else width)
        cast_invalid = source_real.precision != format.precision and source_real.special == 2 and source_real.bit(source_real.precision-2) == 0
        if complex:
            cast_invalid = cast_invalid or (source_imag.precision != format.precision and source_imag.special == 2 and source_imag.bit(source_imag.precision-2) == 0)
        var real = native_depth_float_cast(source_real,format.precision,format.minimum_power)
        if real.special == 2 and real.bit(format.precision-2) == 0:
            status = SEQ_CAST_INVALID
        real = native_depth_add_one(real,subtract,format.precision,format.minimum_power)
        format.write(data,0,real)
        if complex:
            var imag = native_depth_float_cast(source_imag,format.precision,format.minimum_power)
            if imag.special == 2:
                if imag.bit(format.precision-2) == 0:
                    status = SEQ_CAST_INVALID
                # Complex imaginary +/-0 quiets a NaN like the real part.
                imag = native_depth_add_one(imag,subtract,format.precision,format.minimum_power)
            elif not imag.truth() and not subtract:
                imag.negative = False
            format.write(data,format.width,imag)
    return status
