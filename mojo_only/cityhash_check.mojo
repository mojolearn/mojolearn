"""Gate the category hash stack against CatBoost's own compiled source.

    pixi run check-cityhash

Three functions under test, the whole key chain a category ever passes
through in their system:

  `city_hash_64`           gbdt/digest/city.mojo      (util/digest/city.cpp)
  `calc_cat_feature_hash`  gbdt/cat_feature/          (libs/cat_feature)
  `calc_hash` + the (ui64)(int) widening
                           gbdt/models/hash.mojo      (libs/model/hash.h,
                                                       ctr_provider.h:107)

There is no live oracle: their binary never prints a hash, and PUBLIC
CityHash vectors are the trap rather than the oracle -- their city.h says
its results DIFFER from mainline CityHash. So `tools/cityhash_oracle/`
compiles their own `city.cpp` (byte for byte, includes shimmed) plus a
transcription of the two one-liners above it, and dumps
`bench/cityhash_oracle.txt`. This check recomputes every row and compares
cell for cell. Per the sabotage rule, a per-cell match against the
competitor's own compiled output IS the reach proof; no planted-defect arm
is needed.

## Two sections

1. **STRINGS**: 43 byte strings -- every length branch of CityHash64 (0,
   1..3, 4..8, 9..16, 17..32, 33..64, and >64 at 1, 2, 4 and 15 loop
   iterations), category spellings from the real datasets ("Private",
   ">50K", "?", "nan", integers-as-strings), UTF-8, an embedded NUL and a
   0xff byte. Both the 64-bit hash and its low-32 truncation are compared,
   because the truncation is the function the rest of CatBoost calls.

2. **CHAINS**: the apply-time combination fold at lengths 1..8, mixing
   sign-extended category hashes with bare 0/1 binary-split arms. The
   elements are read from the fixture itself, so this section isolates
   `calc_hash` and the widening rule from any hashing already gated above.
   The 8-element row contains hashes at and above 2^31, where a zero
   extension in `cat_hash_chain_element` would diverge.
"""

from gbdt.cat_feature.cat_feature import calc_cat_feature_hash
from gbdt.digest.city import city_hash_64
from gbdt.models.hash import calc_hash, cat_hash_chain_element


def _hex_digit(b: UInt8) raises -> Int:
    var v = Int(b)
    if v >= 48 and v <= 57:  # '0'..'9'
        return v - 48
    if v >= 97 and v <= 102:  # 'a'..'f'
        return v - 97 + 10
    raise Error("bad hex digit: " + String(v))


def _parse_hex_u64(tok: StringSlice) raises -> UInt64:
    var v = UInt64(0)
    for b in tok.as_bytes():
        v = (v << 4) | UInt64(_hex_digit(b))
    return v


def _decode_hex_bytes(tok: StringSlice) raises -> List[UInt8]:
    var out = List[UInt8]()
    if tok == "-":  # the empty string's marker
        return out^
    var bs = tok.as_bytes()
    if len(bs) % 2 != 0:
        raise Error("odd hex dump length")
    for i in range(0, len(bs), 2):
        out.append(UInt8(_hex_digit(bs[i]) * 16 + _hex_digit(bs[i + 1])))
    return out^


def main() raises:
    var f = open("bench/cityhash_oracle.txt", "r")
    var txt = f.read()
    f.close()
    var lines = txt.split("\n")

    var head = lines[0].split(" ")
    if String(head[0]) != "strings":
        raise Error("fixture header missing")
    var n_strings = Int(String(head[1]))

    var wrong64 = 0
    var wrong32 = 0
    for i in range(n_strings):
        var fields = lines[1 + i].split(" ")
        if String(fields[0]) != "str":
            raise Error("bad str row " + String(i))
        var length = Int(String(fields[1]))
        var data = _decode_hex_bytes(fields[2])
        if len(data) != length:
            raise Error("hex dump length mismatch at row " + String(i))
        var want64 = _parse_hex_u64(fields[3])
        var want32 = _parse_hex_u64(fields[4]).cast[DType.uint32]()
        var got64 = city_hash_64(Span(data))
        var got32 = calc_cat_feature_hash(Span(data))
        if got64 != want64:
            wrong64 += 1
            print(
                "  MISMATCH len", length, ": city_hash_64", hex(got64),
                "want", hex(want64),
            )
        if got32 != want32:
            wrong32 += 1
            print(
                "  MISMATCH len", length, ": calc_cat_feature_hash",
                hex(got32), "want", hex(want32),
            )
    print(
        "cityhash strings:", n_strings, "rows,", wrong64,
        "wrong city_hash_64,", wrong32, "wrong calc_cat_feature_hash",
    )

    var chain_head = lines[1 + n_strings].split(" ")
    if String(chain_head[0]) != "chains":
        raise Error("chains header missing")
    var n_chains = Int(String(chain_head[1]))
    var wrong_chain = 0
    for i in range(n_chains):
        var fields = lines[2 + n_strings + i].split(" ")
        if String(fields[0]) != "chain":
            raise Error("bad chain row " + String(i))
        var k = Int(String(fields[1]))
        var acc = UInt64(0)
        for e in range(k):
            var tok = fields[2 + e]
            var kind = tok.as_bytes()[0]
            var payload = _parse_hex_u64(
                StringSlice(unsafe_from_utf8=tok.as_bytes()[1:])
            )
            if kind == UInt8(104):  # 'h': sign-extended category hash
                acc = calc_hash(
                    acc, cat_hash_chain_element(payload.cast[DType.uint32]())
                )
            elif kind == UInt8(98):  # 'b': bare binary-split arm
                acc = calc_hash(acc, payload)
            else:
                raise Error("bad chain token")
        var want = _parse_hex_u64(fields[2 + k])
        if acc != want:
            wrong_chain += 1
            print(
                "  MISMATCH chain", k, ":", hex(acc), "want", hex(want)
            )
    print("cityhash chains:", n_chains, "rows,", wrong_chain, "wrong")

    if wrong64 + wrong32 + wrong_chain != 0:
        raise Error("cityhash check FAILED")
    print("cityhash check OK")
