# SPDX-License-Identifier: Apache-2.0
"""SHA-256 (FIPS 180-4) over host bytes, one-shot and streamed.

The compression function and constants moved here word for word from
autograd/model_verify.mojo (which imports `sha256_hex` back), so the model
report, the tokenizer fingerprint and the checkpoint/transport digests of the
binding shells share one implementation. A streamed `Sha256` gives the same
digest as `sha256_hex` over the concatenation of every `update`, which is the
contract of Python's hashlib it replaces in the shells (bindings/sha256_door.mojo).

cpu-route: integrity digests over host bytes (checkpoint files, transport
payloads, saved generation state), outside any GPU fit, transform or predict.
"""

comptime _BP = UnsafePointer[UInt8, MutUntrackedOrigin]


def _sha256_k() -> List[UInt32]:
    var k: List[UInt32] = [
        0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
        0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
        0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
        0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
        0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
        0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
    ]
    return k^


def _sha256_iv() -> List[UInt32]:
    var h: List[UInt32] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ]
    return h^


@always_inline
def _rotr(x: UInt32, n: UInt32) -> UInt32:
    return (x >> n) | (x << (UInt32(32) - n))


def _sha256_block(mut h: List[UInt32], k: List[UInt32], mut w: List[UInt32], src: _BP, off: Int):
    """One 64-byte block of `src` at `off` into the state `h`. UInt32
    arithmetic wraps, as SHA-256 requires."""
    for t in range(16):  # the block's message words
        var o = off + 4 * t
        w[t] = (
            (src[o].cast[DType.uint32]() << 24)
            | (src[o + 1].cast[DType.uint32]() << 16)
            | (src[o + 2].cast[DType.uint32]() << 8)
            | src[o + 3].cast[DType.uint32]()
        )
    for t in range(16, 64):  # the message schedule
        var s0 = _rotr(w[t - 15], 7) ^ _rotr(w[t - 15], 18) ^ (w[t - 15] >> 3)
        var s1 = _rotr(w[t - 2], 17) ^ _rotr(w[t - 2], 19) ^ (w[t - 2] >> 10)
        w[t] = w[t - 16] + s0 + w[t - 7] + s1
    var a = h[0]
    var b = h[1]
    var c = h[2]
    var d = h[3]
    var e = h[4]
    var f = h[5]
    var g = h[6]
    var hh = h[7]
    for t in range(64):  # the compression rounds
        var big1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        var ch = (e & f) ^ ((~e) & g)
        var t1 = hh + big1 + ch + k[t] + w[t]
        var big0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        var maj = (a & b) ^ (a & c) ^ (b & c)
        var t2 = big0 + maj
        hh = g
        g = f
        f = e
        e = d + t1
        d = c
        c = b
        b = a
        a = t1 + t2
    h[0] = h[0] + a
    h[1] = h[1] + b
    h[2] = h[2] + c
    h[3] = h[3] + d
    h[4] = h[4] + e
    h[5] = h[5] + f
    h[6] = h[6] + g
    h[7] = h[7] + hh


def _hex32(v: UInt32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(8):  # the word's hex digits
        var nib = Int((v >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


struct Sha256(Movable):
    """A streamed SHA-256: `update` with host bytes any number of times, then
    `hexdigest` (64 lowercase hex digits). `hexdigest` does not end the
    stream, as with hashlib."""
    var h: List[UInt32]
    var pending: List[UInt8]   # the bytes of the current partial block, fewer than 64
    var length: UInt64         # bytes hashed so far

    def __init__(out self):
        self.h = _sha256_iv()
        self.pending = List[UInt8]()
        self.length = UInt64(0)

    def update(mut self, addr: Int, n: Int):
        """Hash the `n` bytes at host address `addr` (the caller keeps them alive)."""
        if n <= 0:
            return
        var src = _BP(unsafe_from_address=addr)
        var h = self.h.copy()
        var k = _sha256_k()
        var w = List[UInt32](length=64, fill=UInt32(0))
        self.length += UInt64(n)
        var i = 0
        if len(self.pending) > 0:
            while i < n and len(self.pending) < 64:  # cpu-route: fill the held partial block
                self.pending.append(src[i])
                i += 1
            if len(self.pending) == 64:
                var held = self.pending.copy()
                _sha256_block(h, k, w, _BP(unsafe_from_address=Int(held.unsafe_ptr())), 0)
                _ = held^
                self.pending.clear()
        while i + 64 <= n:  # cpu-route: the host bytes hashed block by block
            _sha256_block(h, k, w, src, i)
            i += 64
        while i < n:  # cpu-route: the new partial block, fewer than 64 bytes
            self.pending.append(src[i])
            i += 1
        self.h = h^

    def update_list(mut self, data: List[UInt8]):
        self.update(Int(data.unsafe_ptr()), len(data))

    def update_text(mut self, text: String):
        """Hash the UTF-8 bytes of `text`."""
        var raw = List[UInt8]()
        var view = text.as_bytes()
        for i in range(len(view)):  # cpu-route: the text's bytes (short strings: hex digests, scalars)
            raw.append(view[i])
        self.update_list(raw)

    def digest_words(self) -> List[UInt32]:
        """The eight state words of the digest of everything hashed so far."""
        var h = self.h.copy()
        var k = _sha256_k()
        var w = List[UInt32](length=64, fill=UInt32(0))
        # the tail, the 0x80 marker, zeros and the message length in bits (big-endian)
        var tail = self.pending.copy()
        tail.append(UInt8(0x80))
        while len(tail) % 64 != 56:  # padding bytes
            tail.append(UInt8(0))
        var bits = self.length * UInt64(8)
        for i in range(8):  # the length bytes
            tail.append(((bits >> UInt64(56 - 8 * i)) & UInt64(0xFF)).cast[DType.uint8]())
        var src = _BP(unsafe_from_address=Int(tail.unsafe_ptr()))
        for blk in range(len(tail) // 64):  # the padded tail blocks
            _sha256_block(h, k, w, src, 64 * blk)
        _ = tail^
        return h^

    def hexdigest(self) -> String:
        var h = self.digest_words()
        var out = String("")
        for i in range(8):  # the state words
            out += _hex32(h[i])
        return out


def sha256_hex_addr(addr: Int, n: Int) -> String:
    """SHA-256 of the `n` host bytes at `addr`, 64 lowercase hex digits."""
    var s = Sha256()
    s.update(addr, n)
    return s.hexdigest()


def sha256_hex(data: List[UInt8]) -> String:
    """SHA-256 of `data`, 64 lowercase hex digits."""
    return sha256_hex_addr(Int(data.unsafe_ptr()), len(data))


def sha256_hex_text(text: String) -> String:
    """SHA-256 of the UTF-8 bytes of `text`, 64 lowercase hex digits."""
    var s = Sha256()
    s.update_text(text)
    return s.hexdigest()


def sha256_sliced8_hex(addr: Int, n: Int) -> String:
    """The cross-vendor v2 slice digest of the `n` host bytes at `addr`: the
    SHA-256 of the ASCII concatenation of the hex SHA-256 of eight slices,
    slice i spanning bytes [n*i//8, n*(i+1)//8). Eight slices are the fixed v2
    format, not a size rule."""
    var joined = String("")
    for i in range(8):  # the eight v2 slices
        var start = n * i // 8
        var end = n * (i + 1) // 8
        joined += sha256_hex_addr(addr + start, end - start)
    return sha256_hex_text(joined)
