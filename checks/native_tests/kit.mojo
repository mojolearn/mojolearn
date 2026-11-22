# SPDX-License-Identifier: Apache-2.0
"""The one transfer kit of the native checks: host words to a device buffer
and back, and the comparisons of what came back.

A host-to-device copy is queued: the driver reads the host memory after
`enqueue_copy` returns, and Mojo frees a local at its last named use. A check
that writes the sequence itself (pinned buffer, fill, `enqueue_copy`, drain,
keep-alive) frees the pinned buffer early whenever the keep-alive is missing
or sits before the drain. The functions of the first set below hold the
pinned buffer themselves and DRAIN BEFORE THEY RETURN, so nothing a caller
can do frees host memory a queued copy still reads.

    var d = upload[DType.float32](ctx, values)        # a new device buffer
    upload_into[DType.uint8](ctx, flags, host_flags)   # into an existing one
    var got = download[DType.uint32](ctx, d, n)         # an ordinary List

A check that fills a pinned buffer itself hands the buffer over instead
(`upload_host`, `upload_host_into`, `download_host_into`, the buffer a
`var` of the caller): the call borrows it and drains, so no keep-alive line
follows.

What the kit does not cover: a DEVICE buffer that a queued launch reads is
still the caller's to keep alive until the caller's next drain (`download`
drains, so a device input named after the download that follows its launch
is safe).

The second set (`..._undrained_keep`) leaves the queue undrained for a check
whose subject is the queue, an ordering or the drain itself. Each returns
the pinned host buffer the queued copy reads; the caller must hold it and
name it (`_ = keep^`) AFTER its own drain.

Free functions only: no struct here holds a check's buffers (a struct of
per-dtype buffer lists is the nested native buffer list shape behind the
IPCA compiler crash). No kernels, no `main()`, no default tolerance: a
caller always writes its tolerances. One drain per transfer instead of one
per batch is the price; a check is not a hot path and no product file
imports this module. Tested by `checks/native_tests/kit_roundtrip.mojo`.
"""
from std.math import isfinite
from std.memory import unsafe_memcpy
from std.sys.info import size_of

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.native_tests.testsuite_cases import require


# --- host words to the device, drained before the return -------------------


def upload[dt: DType](ctx: DeviceContext, values: List[Scalar[dt]]) raises -> DeviceBuffer[dt]:
    """A new device buffer holding `values`, drained before it returns. An
    empty list gets a one-element zero buffer, since a zero-size allocation
    is refused on some backends."""
    var n = len(values)
    var cells = n if n > 0 else 1
    var buf = ctx.enqueue_create_buffer[dt](cells)
    var host = ctx.enqueue_create_host_buffer[dt](cells)
    ctx.synchronize()
    if n > 0:
        unsafe_memcpy(dest=host.unsafe_ptr(), src=values.unsafe_ptr(), count=n)
    else:
        host.unsafe_ptr().unsafe_store(0, Scalar[dt](0))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def upload_into[dt: DType](ctx: DeviceContext, mut dst: DeviceBuffer[dt], values: List[Scalar[dt]]) raises:
    """`values` into the first `len(values)` elements of `dst`, drained
    before it returns. The rest of `dst` is not written."""
    var n = len(values)
    require(n <= len(dst), "kit.upload_into: " + String(n) + " words into a buffer of " + String(len(dst)))
    if n == 0:
        return
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.synchronize()
    unsafe_memcpy(dest=host.unsafe_ptr(), src=values.unsafe_ptr(), count=n)
    # A whole buffer is copied as itself; a prefix through a view of it.
    if n == len(dst):
        ctx.enqueue_copy(dst_buf=dst, src_ptr=host.unsafe_ptr())
        ctx.synchronize()
    else:
        var view = dst.create_sub_buffer[dt](0, n)
        ctx.enqueue_copy(dst_buf=view, src_ptr=host.unsafe_ptr())
        ctx.synchronize()
        _ = view^
    _ = host^


def upload_whole_into[dt: DType](ctx: DeviceContext, mut dst: DeviceBuffer[dt], values: List[Scalar[dt]]) raises:
    """`values` against the whole of `dst`, as a raw copy from the list's
    pointer was written: the first `len(dst)` words when the list is at
    least that long. A shorter list writes only its own words (the raw copy
    read past the list's end there). Drained before it returns."""
    var n = len(dst)
    if len(values) < n:
        n = len(values)
    if n == 0:
        return
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.synchronize()
    unsafe_memcpy(dest=host.unsafe_ptr(), src=values.unsafe_ptr(), count=n)
    if n == len(dst):
        ctx.enqueue_copy(dst_buf=dst, src_ptr=host.unsafe_ptr())
        ctx.synchronize()
    else:
        var view = dst.create_sub_buffer[dt](0, n)
        ctx.enqueue_copy(dst_buf=view, src_ptr=host.unsafe_ptr())
        ctx.synchronize()
        _ = view^
    _ = host^


def upload_host[dt: DType](ctx: DeviceContext, mut host: HostBuffer[dt], count: Int) raises -> DeviceBuffer[dt]:
    """A new device buffer of `count` elements holding the first `count`
    elements of the pinned buffer `host` (filled by the caller), drained
    before it returns: the caller may free `host` at once."""
    require(count > 0, "kit.upload_host: count must be positive")
    var buf = ctx.enqueue_create_buffer[dt](count)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    return buf^


def upload_host_into[dt: DType](ctx: DeviceContext, mut dst: DeviceBuffer[dt], mut host: HostBuffer[dt]) raises:
    """The pinned buffer `host` (at least `len(dst)` elements, filled by the
    caller) into the whole of `dst`, drained before it returns."""
    ctx.enqueue_copy(dst_buf=dst, src_ptr=host.unsafe_ptr())
    ctx.synchronize()


def upload_address[dt: DType](ctx: DeviceContext, address: Int, count: Int) raises -> DeviceBuffer[dt]:
    """A new device buffer holding the `count` elements at the host address
    `address`, staged through a pinned buffer and drained before it returns.
    An address carries no origin: the caller names the owner of that memory
    after this call (`_ = owner`), as for any address handed to a callee."""
    require(count > 0, "kit.upload_address: count must be positive")
    var buf = ctx.enqueue_create_buffer[dt](count)
    var host = ctx.enqueue_create_host_buffer[dt](count)
    ctx.synchronize()
    var src = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=address)
    unsafe_memcpy(dest=host.unsafe_ptr(), src=src, count=count)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


# --- the device to host words, drained before the return -------------------


def download[dt: DType](ctx: DeviceContext, mut src: DeviceBuffer[dt], n: Int) raises -> List[Scalar[dt]]:
    """The first `n` elements of `src` as an ordinary list, drained (so every
    launch queued before the call has finished too)."""
    var words = List[Scalar[dt]](length=(n if n > 0 else 0), fill=Scalar[dt](0))
    if n <= 0:
        return words^
    require(n <= len(src), "kit.download: " + String(n) + " words from a buffer of " + String(len(src)))
    var host = ctx.enqueue_create_host_buffer[dt](n)
    if n == len(src):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=src)
        ctx.synchronize()
    else:
        var view = src.create_sub_buffer[dt](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    unsafe_memcpy(dest=words.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
    _ = host^
    return words^


def download_into[dt: DType](ctx: DeviceContext, mut src: DeviceBuffer[dt], mut dst: List[Scalar[dt]]) raises:
    """The first `len(dst)` elements of `src` into `dst`, drained."""
    var n = len(dst)
    if n == 0:
        return
    require(n <= len(src), "kit.download_into: " + String(n) + " words from a buffer of " + String(len(src)))
    var host = ctx.enqueue_create_host_buffer[dt](n)
    if n == len(src):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=src)
        ctx.synchronize()
    else:
        var view = src.create_sub_buffer[dt](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    unsafe_memcpy(dest=dst.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
    _ = host^


def download_whole_into[dt: DType](ctx: DeviceContext, src: DeviceBuffer[dt], mut dst: List[Scalar[dt]]) raises:
    """The whole of `src` into `dst`, as a raw copy to the list's pointer
    was written, drained. A list shorter than `src` receives only its own
    length (the raw copy wrote past the list's end there)."""
    var n = len(src)
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=src)
    ctx.synchronize()
    if len(dst) < n:
        n = len(dst)
    if n > 0:
        unsafe_memcpy(dest=dst.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
    _ = host^


def download_host_into[dt: DType](ctx: DeviceContext, src: DeviceBuffer[dt], mut host: HostBuffer[dt]) raises:
    """The whole of `src` into the caller's pinned buffer `host` (at least
    `len(src)` elements), drained before it returns: `host` is readable at
    once."""
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=src)
    ctx.synchronize()


# --- buffer-to-buffer forms (lane fx-checks-orphans, 2026-11-18) -------------
# The copies a check wrote with two buffers and no pointer. Each makes the
# same `enqueue_copy(dst_buf=, src_buf=)` call the check made, then drains.


def copy_device_into[dt: DType](ctx: DeviceContext, mut dst: DeviceBuffer[dt], src: DeviceBuffer[dt]) raises:
    """The device buffer `src` into the device buffer `dst` (a device to
    device copy, as `enqueue_copy(dst_buf=dst, src_buf=src)` sizes it),
    drained before it returns: `src` may be freed or overwritten at once."""
    ctx.enqueue_copy(dst_buf=dst, src_buf=src)
    ctx.synchronize()


def download_buffer_into[dt: DType](ctx: DeviceContext, src: DeviceBuffer[dt], mut host: HostBuffer[dt]) raises:
    """The device buffer `src` into the pinned buffer `host` of the caller,
    written as a buffer to buffer copy (`enqueue_copy(dst_buf=host,
    src_buf=src)`), drained before it returns: `host` is readable at once."""
    ctx.enqueue_copy(dst_buf=host, src_buf=src)
    ctx.synchronize()


def upload_buffer_into[dt: DType](ctx: DeviceContext, mut dst: DeviceBuffer[dt], host: HostBuffer[dt]) raises:
    """The pinned buffer `host` of the caller into the device buffer `dst`,
    written as a buffer to buffer copy (`enqueue_copy(dst_buf=dst,
    src_buf=host)`), drained before it returns: `host` may be freed at once."""
    ctx.enqueue_copy(dst_buf=dst, src_buf=host)
    ctx.synchronize()


# --- the queue left undrained: the caller holds what is returned ------------


def upload_into_undrained_keep[dt: DType](
    ctx: DeviceContext, mut dst: DeviceBuffer[dt], values: List[Scalar[dt]]
) raises -> HostBuffer[dt]:
    """DANGER, for a check of the queue only. Queues the copy of `values`
    (exactly `len(dst)` words) into `dst` and returns WITHOUT draining. The
    returned pinned buffer is the memory the queued copy reads: hold it and
    name it (`_ = keep^`) after your own `ctx.synchronize()`. Freed earlier,
    the copy reads freed storage."""
    var n = len(values)
    require(n > 0 and n == len(dst), "kit.upload_into_undrained_keep: " + String(n) + " words, buffer of " + String(len(dst)))
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.synchronize()
    unsafe_memcpy(dest=host.unsafe_ptr(), src=values.unsafe_ptr(), count=n)
    ctx.enqueue_copy(dst_buf=dst, src_ptr=host.unsafe_ptr())
    return host^


def download_undrained_keep[dt: DType](ctx: DeviceContext, mut src: DeviceBuffer[dt]) raises -> HostBuffer[dt]:
    """DANGER, for a check of the queue only. Queues the copy of the whole of
    `src` into a new pinned buffer and returns it WITHOUT draining. Its
    contents are defined only after your own `ctx.synchronize()`; hold it and
    name it (`_ = keep^`) after that drain and after your last read of it."""
    var host = ctx.enqueue_create_host_buffer[dt](len(src))
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=src)
    return host^


# --- comparisons ------------------------------------------------------------


def word_hex[dt: DType](words: List[Scalar[dt]], index: Int) -> String:
    """The bits of `words[index]` in hex, most significant byte first."""
    comptime DIGITS = "0123456789abcdef"
    var width = size_of[Scalar[dt]]()
    var raw = words.unsafe_ptr().unsafe_bitcast[UInt8]()
    var text = String("0x")
    for b in range(width):
        # the host is little-endian: the last byte of the element is the top one
        var value = Int(raw[unsafe_offset=index * width + width - 1 - b])
        text += String(DIGITS[byte=value >> 4])
        text += String(DIGITS[byte=value & 15])
    return text


def same_words[dt: DType](got: List[Scalar[dt]], want: List[Scalar[dt]], what: String) raises:
    """Bit-exact equality (a NaN equals only the same NaN bits, +0.0 differs
    from -0.0). The error names the first differing index with both words
    in hex."""
    require(len(got) == len(want), what + ": extent " + String(len(got)) + " vs " + String(len(want)))
    var width = size_of[Scalar[dt]]()
    var gp = got.unsafe_ptr().unsafe_bitcast[UInt8]()
    var wp = want.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(len(got)):
        var differs = False
        for b in range(width):
            if gp[unsafe_offset=i * width + b] != wp[unsafe_offset=i * width + b]:
                differs = True
        if differs:
            raise Error(what + ": at " + String(i) + " got " + word_hex[dt](got, i) + ", want " + word_hex[dt](want, i))


def allclose[dt: DType](
    actual: List[Scalar[dt]], expected: List[Float64], rtol: Float64, atol: Float64, what: String
) raises:
    """|a - e| <= atol + rtol |e| for every element, in float64 (a NaN
    fails). Both tolerances are always the caller's."""
    require(len(actual) == len(expected), what + ": extent " + String(len(actual)) + " vs " + String(len(expected)))
    for i in range(len(actual)):
        var a = actual[i].cast[DType.float64]()
        if not (abs(a - expected[i]) <= atol + rtol * abs(expected[i])):
            raise Error(what + ": at " + String(i) + " got " + String(a) + ", expected " + String(expected[i]))


def all_finite[dt: DType](values: List[Scalar[dt]], what: String) raises:
    """Every element finite (no NaN, no infinity); floating dtypes only."""
    for i in range(len(values)):
        require(isfinite(values[i]), what + ": non-finite at " + String(i))
