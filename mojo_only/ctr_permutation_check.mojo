"""Gate the CTR estimation permutation against CatBoost's own generator.

    pixi run check-permutation

`gbdt/data/permutation.mojo` is the port of `TDataPermutation` and of the
two things under it: `NCatboostCuda::Shuffle` (`cuda/data/data_utils.h:21`)
and `TRandom` over `TMersenne<ui64>`. It decides what "before" means for
every `Borders` CTR in the library, so a wrong permutation is not a wrong
number, it is a different estimator -- and one that still trains, still
converges, and reports a plausible loss.

There is NO live oracle for it on this box. Their GPU learner refuses to
run on Apple silicon ("Environment for task type [GPU] not found") and their
CPU learner never exposes a CTR estimation order. So the oracle is
`bench/permutation_oracle.txt`, produced by `tools/permutation_oracle/`,
which compiles CatBoost's OWN `util/random/mersenne64.{h,cpp}` and drives it
from a C++ transcription of the four short pieces around it. That is a
second implementation, in a different language, through a different
compiler; see that directory's README for exactly which lines are theirs
verbatim and which are transcribed.

## Five sections

1. **THE RAW MT19937-64 STREAM**, twenty draws at three seeds, compared as
   64-BIT PATTERNS. It comes first because everything else is downstream of
   it: a tempering shift or a twist index that is one off produces a
   perfectly good-looking permutation of the right length. Seed
   `0x1571` is 5489, the reference seed of the published MT19937-64, so the
   first row of this section is checkable against a third party as well.

2. **`Uniform(t)` AT A `t` WHERE THE REJECTION BITES.** Added because a
   sabotage proved the next section blind to it: replacing their
   rejection-sampled `GenUniform` with a plain `GenRand() % max` left every
   permutation below bit-identical, since a few-thousand-row `max` rejects
   a draw with probability about `max / 2^64`. At `max = 2^63 + 1` half of
   the draws are rejected and the shortcut is wrong immediately.

3. **WHOLE PERMUTATIONS**, cell by cell, at the ids and sizes the CTR path
   uses -- including `n = 1` and `n = 2`, where Fisher-Yates runs zero and
   one swap, and the block arm at `blockSize = 64`, which is their
   `fold_permutation_block` default even though the CTR path never passes
   it. Plus **THE PROPERTIES A CELL COMPARE CANNOT SEE**: every order is a
   bijection, permutation 0 IS the identity, and every other id is NOT --
   the last one because "we ported the permutation" and "we ported an
   expensive way to write `iota`" look identical from the outside.

4. **THE SEED**, against `1664525 * id + 1013904223 + blockSize`
   (`permutation.h:93-95`) evaluated here, so that a change to it has to be
   made twice.

5. **THE INVERSE**, `inverse[order[i]] == i`, on a shuffle rather than on
   an involution, because an involutive fixture cannot tell the two
   directions apart and the whole CTR block depends on which one it has.
"""

from gbdt.data.permutation import (
    DEFAULT_PERMUTATION_COUNT,
    TMersenne64,
    TRandom,
    get_identity_permutation,
    get_permutation,
)


comptime PERMUTATION_ORACLE = "bench/permutation_oracle.txt"


def _parse_hex_u64(s: String) raises -> UInt64:
    """Sixteen hex digits into a `UInt64`.

    The oracle prints the stream in hex on purpose: a `ui64` above 2^63 does
    not survive a signed decimal parse, and half of a good 64-bit stream is
    above 2^63.
    """
    var out = UInt64(0)
    var n = s.byte_length()
    if n == 0:
        raise Error("empty hex word")
    for i in range(n):
        var c = Int(ord(String(s[byte=i])))
        var d: Int
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        else:
            raise Error("bad hex digit in '" + s + "'")
        out = (out << 4) | UInt64(d)
    return out


def _read_lines(path: String) raises -> List[String]:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    var lines = List[String]()
    for line in text.splitlines():
        var s = String(String(line).strip())
        if s.byte_length() > 0:
            lines.append(s^)
    return lines^


def _check_streams(
    lines: List[String], mut pos: Int, mut failures: List[String]
) raises:
    var head = lines[pos].split(" ")
    if String(head[0]) != "streams":
        raise Error("malformed permutation oracle: expected 'streams'")
    var n_streams = Int(String(head[1]))
    pos += 1

    var checked = 0
    for _ in range(n_streams):
        var h = lines[pos].split(" ")
        if String(h[0]) != "stream":
            raise Error("malformed stream header")
        var seed = _parse_hex_u64(String(h[1]))
        var count = Int(String(h[2]))
        pos += 1

        var rng = TMersenne64(seed)
        var wrong = 0
        var first = -1
        var got_first = UInt64(0)
        var want_first = UInt64(0)
        for i in range(count):
            var want = _parse_hex_u64(lines[pos + i])
            var got = rng.gen_rand()
            if got != want:
                wrong += 1
                if first < 0:
                    first = i
                    got_first = got
                    want_first = want
        pos += count

        if wrong != 0:
            failures.append(
                String("MT19937-64 at seed 0x")
                + String(h[1])
                + String(": ")
                + String(wrong)
                + String(" of ")
                + String(count)
                + String(" draws differ from CatBoost's own generator,")
                + String(" first at ")
                + String(first)
                + String(" (got ")
                + String(got_first)
                + String(", want ")
                + String(want_first)
                + String(")")
            )
        checked += 1
    print(
        "  MT19937-64:", checked, "seeds x 20 draws, every 64-bit pattern"
        " equal to CatBoost's util/random/mersenne64.cpp"
    )


def _check_uniforms(
    lines: List[String], mut pos: Int, mut failures: List[String]
) raises:
    """`TRandom::Uniform(t)`, at a `t` where the REJECTION actually bites.

    `NPrivate::GenUniform` (`common_ops.h:48-60`) rejects every draw at or
    above `RandMax() - RandMax() % max` and retries. At the sizes a
    permutation uses -- a few thousand -- a draw is rejected with
    probability about `max / 2^64`, so a plain `GenRand() % max` agrees with
    theirs on every fixture anyone would build and diverges only in
    production, years apart, on one row. **That sabotage was run against
    section 2 and section 2 stayed green**, which is why this section
    exists: at `max = 2^63 + 1` the bound is `2^63 + 1` and roughly half of
    all draws are thrown away, so the modulo shortcut is wrong on the third
    draw.
    """
    var head = lines[pos].split(" ")
    if String(head[0]) != "uniforms":
        raise Error("malformed permutation oracle: expected 'uniforms'")
    var n_cases = Int(String(head[1]))
    pos += 1

    var checked = 0
    for _ in range(n_cases):
        var h = lines[pos].split(" ")
        if String(h[0]) != "uniform":
            raise Error("malformed uniform header")
        var seed = _parse_hex_u64(String(h[1]))
        var size = _parse_hex_u64(String(h[2]))
        var count = Int(String(h[3]))
        pos += 1

        var rng = TRandom(seed)
        rng.advance(10)
        var wrong = 0
        var first = -1
        for i in range(count):
            var want = _parse_hex_u64(lines[pos + i])
            var got = rng.uniform(size)
            if got != want:
                wrong += 1
                if first < 0:
                    first = i
        pos += count
        if wrong != 0:
            failures.append(
                String("Uniform(0x")
                + String(h[2])
                + String("): ")
                + String(wrong)
                + String(" of ")
                + String(count)
                + String(" draws differ from their GenUniform, first at ")
                + String(first)
            )
        checked += 1

    # `Uniform(0)` is their `Y_ABORT_UNLESS(max > 0)` (`common_ops.h:50`).
    var refused = False
    try:
        var rng0 = TRandom(UInt64(1))
        _ = rng0.uniform(UInt64(0))
    except:
        refused = True
    if not refused:
        failures.append(String("Uniform(0) must refuse an empty range"))

    print(
        "  Uniform:", checked, "sizes x 24 draws, including one where the"
        " rejection branch fires on half of them"
    )


def _check_permutations(
    lines: List[String], mut pos: Int, mut failures: List[String]
) raises:
    var head = lines[pos].split(" ")
    if String(head[0]) != "permutations":
        raise Error("malformed permutation oracle: expected 'permutations'")
    var n_cases = Int(String(head[1]))
    pos += 1

    var checked = 0
    for _ in range(n_cases):
        var h = lines[pos].split(" ")
        if String(h[0]) != "permutation":
            raise Error("malformed permutation header")
        var index = Int(String(h[1]))
        var block_size = Int(String(h[2]))
        var n = Int(String(h[3]))
        pos += 1

        var want = List[UInt32]()
        for i in range(n):
            want.append(UInt32(Int(String(lines[pos + i]))))
        pos += n

        var perm = get_permutation(n, index, block_size)
        var got = perm.fill_order()

        var label = (
            String("permutation id ")
            + String(index)
            + String(" block ")
            + String(block_size)
            + String(" n ")
            + String(n)
        )

        if len(got) != n:
            failures.append(
                label + String(": ") + String(len(got)) + String(" entries")
            )
            continue

        var wrong = 0
        var first = -1
        for i in range(n):
            if got[i] != want[i]:
                wrong += 1
                if first < 0:
                    first = i
        if wrong != 0:
            failures.append(
                label
                + String(": ")
                + String(wrong)
                + String(" of ")
                + String(n)
                + String(" positions differ from CatBoost's, first at ")
                + String(first)
                + String(" (got ")
                + String(got[first])
                + String(", want ")
                + String(want[first])
                + String(")")
            )

        # --- section 3, the properties a cell compare cannot see ---
        var seen = List[Bool]()
        for _ in range(n):
            seen.append(False)
        var out_of_range = 0
        for i in range(n):
            var v = Int(got[i])
            if v < 0 or v >= n:
                out_of_range += 1
            else:
                seen[v] = True
        var missing = 0
        for i in range(n):
            if not seen[i]:
                missing += 1
        if out_of_range != 0 or missing != 0:
            failures.append(
                label
                + String(" is not a bijection: ")
                + String(out_of_range)
                + String(" entries out of range, ")
                + String(missing)
                + String(" rows never named")
            )

        var moved = 0
        for i in range(n):
            if Int(got[i]) != i:
                moved += 1
        if index == 0:
            if moved != 0:
                failures.append(
                    label
                    + String(" moved ")
                    + String(moved)
                    + String(" rows; permutation 0 IS the identity")
                    + String(" (permutation.cpp:14-17)")
                )
        elif n > 3 and moved == 0:
            # An expensive way to write `iota` passes every cell compare
            # against an oracle that has the same bug. It cannot pass this.
            failures.append(
                label
                + String(" moved NO rows; a non-identity permutation id")
                + String(" that returns row order is the whole defect this")
                + String(" file exists to catch")
            )

        # `IsIdentity()` (`permutation.h:77-79`) must agree with the order
        # it actually produced.
        if perm.is_identity() != (index == 0):
            failures.append(
                label + String(": is_identity() disagrees with the id")
            )
        checked += 1

    print(
        "  orders:", checked, "cases against CatBoost's Shuffle cell by"
        " cell, each a bijection, identity iff id 0"
    )


def _check_seed(mut failures: List[String]) raises:
    """`GetSeed()` (`permutation.h:93-95`), evaluated here rather than
    inherited, so a change to the formula has to be made twice."""
    var checked = 0
    for block_size in [1, 64]:
        for index in range(6):
            var want = UInt64(1664525) * UInt64(index) + UInt64(
                1013904223
            ) + UInt64(block_size)
            var got = get_permutation(1000, index, block_size).get_seed()
            if got != want:
                failures.append(
                    String("seed for id ")
                    + String(index)
                    + String(" block ")
                    + String(block_size)
                    + String(": got ")
                    + String(got)
                    + String(", 1664525*id + 1013904223 + blockSize is ")
                    + String(want)
                )
            checked += 1
    print("  seeds:", checked, "ids x block sizes match their formula")


def _check_inverse(mut failures: List[String]) raises:
    """`FillInversePermutation` (`permutation.cpp:30-37`).

    `inverse[order[i]] == i`, which is the direction the whole CTR block
    depends on and the one an involutive test fixture would hide.
    """
    var perm = get_permutation(313, 2, 1)
    var order = perm.fill_order()
    var inverse = perm.fill_inverse_permutation()
    var wrong = 0
    for i in range(len(order)):
        if Int(inverse[Int(order[i])]) != i:
            wrong += 1
    if wrong != 0:
        failures.append(
            String("fill_inverse_permutation: ")
            + String(wrong)
            + String(" of ")
            + String(len(order))
            + String(" entries do not invert fill_order")
        )
    var identity_inverse = get_identity_permutation(64)
    var inv = identity_inverse.fill_inverse_permutation()
    for i in range(64):
        if Int(inv[i]) != i:
            failures.append(String("the identity's inverse is not itself"))
            break
    print("  inverse: inverse[order[i]] == i on a 313-row shuffle")


def check_permutation() raises:
    print("CTR estimation permutation:")
    var failures = List[String]()
    var lines = _read_lines(PERMUTATION_ORACLE)
    var pos = 0
    _check_streams(lines, pos, failures)
    _check_uniforms(lines, pos, failures)
    _check_permutations(lines, pos, failures)
    _check_seed(failures)
    _check_inverse(failures)

    # The default this port ships. Their `permutation_count` is 4 and their
    # estimation permutation is `PermutationsCount() - 1`
    # (`doc_parallel_boosting.h:101-103`), so the default CTR estimation
    # order is id 3 and MUST NOT be the identity.
    print(
        "  shipped: permutation_count", DEFAULT_PERMUTATION_COUNT,
        "-> ctr estimation permutation id",
        DEFAULT_PERMUTATION_COUNT - 1,
    )
    var shipped = get_permutation(1000, DEFAULT_PERMUTATION_COUNT - 1, 1)
    if shipped.is_identity():
        failures.append(
            String("the shipped ctr estimation permutation id ")
            + String(DEFAULT_PERMUTATION_COUNT - 1)
            + String(" is the identity, which is row order")
        )

    if len(failures) > 0:
        var msg = String("permutation checks FAILED:")
        for i in range(len(failures)):
            msg += String("\n    ") + failures[i]
        raise Error(msg)
    print("  all five sections green")


def main() raises:
    check_permutation()
