# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuRAND's XORWOW generator, the one cuML's Isolation Forest instantiates.

MIRRORS the XORWOW section of `nvidia/curand/include/curand_kernel.h`
(cuRAND 10.x headers at `upstream/curand-headers`, the `curandState`
typedef at `:301-302`), `curand_uniform.h:69-71` (`_curand_uniform`) and
`curand_globals.h:56` (`CURAND_2POW32_INV`). Constant for constant, shift
for shift.

cuML's `build_isolation_trees_global_kernel`
(`cpp/src/isolation_forest/isolation_tree_builder.cuh:263-264`) does

    curandState rng_state;
    curand_init(seed, tree_id, 0, &rng_state);

and thread 0 of the tree's block then draws from that ONE state in
program order: `sample_bounded` (row subsample, then the feature
subsample), then per node `sample_bounded(n_cols)` for the feature start
and `curand_uniform` for the split fraction. `curandState` IS
`curandStateXORWOW_t`, so the stream every tree sees is XORWOW seeded by
the `_curand_init_inplace` scramble (`curand_kernel.h:818-840`) and
skipped ahead `tree_id` subsequences of 2^67 draws
(`_skipahead_sequence_inplace`, `:719-736`). NOT RAFT's PCG/Philox: the
brief named RAFT's `rng.cuh`, the file cuML actually includes is
`<curand_kernel.h>` (`isolation_tree_builder.cuh:28`).

THE STREAM IS A PURE FUNCTION OF (seed, tree_id). `subsequence = tree_id
= blockIdx.x` is a semantic key, not a scheduling accident: one block per
tree, thread 0 the only consumer, and the consumption order is the serial
stack walk. Nothing here depends on block size, grid shape or launch
order, so the brief's DEVIATION 680 re-keying is NOT NEEDED; the port
keeps their key and the launch-invariance gate measures it (block 32 /
128 / 256 give the same bytes, `isolation_forest/original/if_check.mojo`).

What is ported and how it was verified (all integer arithmetic):
  * `curand_init` (seed scramble `:823-834`, then `skipahead_sequence`,
    then `skipahead`), `curand` (`:863-874`, the Marsaglia xorwow step
    with the Weyl `d += 362437`), `curand_uniform` (`x * 2^-32f + 2^-33f`:
    `CURAND_2POW32_INV = 2.3283064e-10f`, which IS 2^-32 exactly in
    float32; the product is exact so fma-or-not cannot move it).
  * The two precalculated tables `precalc_xorwow_matrix[32][800]` and
    `precalc_xorwow_offset_matrix[32][800]` (`curand_precalc.h:67`,
    `:1803`) are POWERS of the xorwow step's GF(2) transition matrix:
    `A^(2^67 * 4^i)` and `A^(4^i)`, `i = 0..31`, in cuRAND's own row
    layout (`matrix[n*(i*32+j)+k]` = word k of the image of bit j of word
    i). They are REBUILT here from the step (`build_xorwow_tables`) by the
    repeated squaring `__curand_matpow` uses (`:399-413`), and the Python
    reference `isolation_forest/original/xorwow_reference.py` (IN TREE,
    re-run 2026-08-23) parses BOTH of the header's 25,600-constant tables
    and finds the 32 rebuilt sequence matrices AND the 32 rebuilt offset
    matrices equal to the header's, word for word. See DEVIATION 683
    below for why rebuild rather than embed, and DEVIATION 751 for why
    the offset table needed its own check.

================= DEVIATION BLOCK =================
DEVIATION 683. THE PRECALC TABLES ARE REBUILT ON THE HOST, NOT EMBEDDED,
AND LIVE IN GLOBAL MEMORY, NOT `__constant__`. Theirs: 51,200 literal
`unsigned int`s in `__constant__` (device) and static (host) arrays.
Ours: `build_xorwow_tables()` computes the same two tables from the step
matrix (integer GF(2) arithmetic, ~130 matrix squarings, host, once per
process) and the kernel receives them as two `UInt32` pointers. WHAT is
said is unchanged: the Python reference shows the rebuilt tables equal
the header's, and the RNG gate below holds the first 1024 draws of six
(seed, tree) pairs against a reference derived from the header's own
constants. HOW changed because a 1.4 MB header of literals is not a
source file anyone can review, and `__constant__` has no portable
spelling (Metal has no equivalent address space in MAX).

DEVIATION 751. THE OFFSET TABLE IS BUILT, UPLOADED AND NEVER STEPPED BY
cuML, SO IT GETS ITS OWN REFERENCE. cuML calls `curand_init(seed, tree_id,
0, &state)` and nothing else, so `_skipahead_inplace` always runs its
`while p != 0` loop ZERO times and `precalc_xorwow_offset_matrix` is dead
weight in every isolation-forest kernel launch. That is REACHED BUT INERT
in the exact sense of the standing rule: `build_xorwow_tables()` computes
the offset table, `XorwowDeviceTables` uploads its 25,600 words on every
fit, `curand_init` takes it as an argument -- and no gate in the lane
could tell a correct offset table from garbage, because nothing reads it.
Half of DEVIATION 683's "the rebuilt tables equal the header's" was
therefore an unverified claim (the Python reference checked
`precalc_xorwow_matrix` only).

Closed rather than deleted, because the table is part of the ported
`curand_init` contract and a future caller (any port that passes a nonzero
`offset`) would inherit an untested one:
  * `xorwow_reference.py` now parses `precalc_xorwow_offset_matrix` too and
    asserts all 32 rebuilt `A^(4^i)` matrices equal the header's, word for
    word (they do), and asserts that skipping k equals stepping k times for
    k in {1,2,3,4,5,17,64};
  * it emits `original/xorwow_offset_reference.tsv`, 6 x 256 draws at
    offsets {1, 3, 4, 64, 4095, 1};
  * `check_if_xorwow_on_device` runs those triples THROUGH THE KERNEL and
    compares bits, and asserts offset 1 is exactly a one-draw advance of
    offset 0 -- the single-draw perturbation this whole lane is built to
    detect, planted deliberately and observed.
Nothing about cuML's behavior changed: `offset` is still 0 everywhere the
forest is built.
===================================================
"""

comptime XORWOW_N = 5
"""Words of xorwow state (`curandStateXORWOW::v[5]`)."""
comptime XORWOW_MATRIX_WORDS = XORWOW_N * XORWOW_N * 32
"""800: one `n x n` GF(2) matrix in cuRAND's 32-bit-unit layout."""
comptime PRECALC_NUM_MATRICES = 32
comptime PRECALC_BLOCK_SIZE = 2
comptime PRECALC_BLOCK_MASK = (1 << PRECALC_BLOCK_SIZE) - 1
comptime XORWOW_SEQUENCE_SPACING = 67
comptime XORWOW_TABLE_WORDS = PRECALC_NUM_MATRICES * XORWOW_MATRIX_WORDS
"""25,600 words per table."""

comptime CURAND_2POW32_INV = Float32(2.3283064e-10)
"""`curand_globals.h:56`. Rounds to exactly 2^-32 in float32 (the
nearest float to 2.3283064e-10 is 2^-32; spacing there is 2^-55)."""


@fieldwise_init
struct curandStateXORWOW(Copyable, Movable):
    """`struct curandStateXORWOW { unsigned int d, v[5]; ... }`
    (`curand_kernel.h:150-156`). The Box-Muller cache fields are not
    ported: nothing in the isolation forest draws a normal."""

    var d: UInt32
    var v0: UInt32
    var v1: UInt32
    var v2: UInt32
    var v3: UInt32
    var v4: UInt32

    @staticmethod
    def zero() -> Self:
        return Self(0, 0, 0, 0, 0, 0)

    def word(self, i: Int) -> UInt32:
        if i == 0:
            return self.v0
        if i == 1:
            return self.v1
        if i == 2:
            return self.v2
        if i == 3:
            return self.v3
        return self.v4

    def set_word(mut self, i: Int, x: UInt32):
        if i == 0:
            self.v0 = x
        elif i == 1:
            self.v1 = x
        elif i == 2:
            self.v2 = x
        elif i == 3:
            self.v3 = x
        else:
            self.v4 = x


# ---------------------------------------------------------------------------
# `curand_kernel.h:312-334`: __curand_matvec_inplace<N>, on the state words.
# ---------------------------------------------------------------------------


def _curand_matvec_inplace(
    mut state: curandStateXORWOW,
    matrix: MutPointer[UInt32, MutAnyOrigin],
    matrix_offset: Int,
):
    """`__curand_matvec_inplace<5>(state->v, matrix)`: result[k] ^=
    matrix[N*(i*32+j)+k] for every set bit j of word i; then v = result."""
    var r0: UInt32 = 0
    var r1: UInt32 = 0
    var r2: UInt32 = 0
    var r3: UInt32 = 0
    var r4: UInt32 = 0
    for i in range(XORWOW_N):
        var w = state.word(i)
        for j in range(32):
            if (w & (UInt32(1) << UInt32(j))) != 0:
                var base = matrix_offset + XORWOW_N * (i * 32 + j)
                r0 ^= matrix.unsafe_load(base + 0)
                r1 ^= matrix.unsafe_load(base + 1)
                r2 ^= matrix.unsafe_load(base + 2)
                r3 ^= matrix.unsafe_load(base + 3)
                r4 ^= matrix.unsafe_load(base + 4)
    state.v0 = r0
    state.v1 = r1
    state.v2 = r2
    state.v3 = r3
    state.v4 = r4


# ---------------------------------------------------------------------------
# `curand_kernel.h:698-736`: the in-place skipaheads curand_init uses.
# ---------------------------------------------------------------------------


def _skipahead_inplace(
    x: UInt64,
    mut state: curandStateXORWOW,
    offset_table: MutPointer[UInt32, MutAnyOrigin],
):
    """`_skipahead_inplace<curandStateXORWOW_t, 5>(x, state)` (`:698-717`):
    two bits of `x` per precalc matrix, then `d += 362437 * (unsigned)x`."""
    var p = x
    var matrix_num = 0
    while p != 0:
        var reps = Int(p & UInt64(PRECALC_BLOCK_MASK))
        for _ in range(reps):
            _curand_matvec_inplace(
                state, offset_table, matrix_num * XORWOW_MATRIX_WORDS
            )
        p >>= UInt64(PRECALC_BLOCK_SIZE)
        matrix_num += 1
    state.d = state.d + UInt32(362437) * UInt32(x & 0xFFFFFFFF)


def _skipahead_sequence_inplace(
    x_in: UInt64,
    mut state: curandStateXORWOW,
    sequence_table: MutPointer[UInt32, MutAnyOrigin],
):
    """`_skipahead_sequence_inplace<curandStateXORWOW_t, 5>(x, state)`
    (`:719-736`). "No update of state->d needed, guaranteed to be a
    multiple of 2^32" (their comment at `:735`)."""
    var x = x_in
    var matrix_num = 0
    while x != 0:
        var reps = Int(x & UInt64(PRECALC_BLOCK_MASK))
        for _ in range(reps):
            _curand_matvec_inplace(
                state, sequence_table, matrix_num * XORWOW_MATRIX_WORDS
            )
        x >>= UInt64(PRECALC_BLOCK_SIZE)
        matrix_num += 1


# ---------------------------------------------------------------------------
# `curand_kernel.h:818-859`: curand_init for curandStateXORWOW_t.
# ---------------------------------------------------------------------------


def curand_init(
    seed: UInt64,
    subsequence: UInt64,
    offset: UInt64,
    mut state: curandStateXORWOW,
    sequence_table: MutPointer[UInt32, MutAnyOrigin],
    offset_table: MutPointer[UInt32, MutAnyOrigin],
):
    """`curand_init(seed, subsequence, offset, curandStateXORWOW_t*)` ->
    `_curand_init_inplace` (`:818-840`): the seed scramble (constants
    theirs, "arbitrary nonzero values" / "arbitrary odd values"), then
    `skipahead_sequence(subsequence)`, then `skipahead(offset)`. The two
    tables are DEVIATION 683's (see the module docstring)."""
    var s0: UInt32 = UInt32(seed & 0xFFFFFFFF) ^ UInt32(0xAAD26B49)
    var s1: UInt32 = UInt32((seed >> 32) & 0xFFFFFFFF) ^ UInt32(0xF7DCEFDD)
    var t0: UInt32 = UInt32(1099087573) * s0
    var t1: UInt32 = UInt32(2591861531) * s1
    state.d = UInt32(6615241) + t1 + t0
    state.v0 = UInt32(123456789) + t0
    state.v1 = UInt32(362436069) ^ t0
    state.v2 = UInt32(521288629) + t1
    state.v3 = UInt32(88675123) ^ t1
    state.v4 = UInt32(5783321) + t0
    _skipahead_sequence_inplace(subsequence, state, sequence_table)
    _skipahead_inplace(offset, state, offset_table)


def curand(mut state: curandStateXORWOW) -> UInt32:
    """`curand(curandStateXORWOW_t*)` (`:863-874`): Marsaglia's xorwow
    step plus the Weyl sequence, returns `v[4] + d`."""
    var t: UInt32 = state.v0 ^ (state.v0 >> 2)
    state.v0 = state.v1
    state.v1 = state.v2
    state.v2 = state.v3
    state.v3 = state.v4
    state.v4 = (state.v4 ^ (state.v4 << 4)) ^ (t ^ (t << 1))
    state.d = state.d + UInt32(362437)
    return state.v4 + state.d


def _curand_uniform(x: UInt32) -> Float32:
    """`curand_uniform.h:69-71`: `x * CURAND_2POW32_INV +
    (CURAND_2POW32_INV/2.0f)`. The uint -> float conversion is
    round-to-nearest (`cvt.rn.f32.u32`; Mojo's `Float32(UInt32)` is the
    same rounding, measured in the RNG gate); the product by 2^-32 is
    exact, so the sum is ONE rounding whether or not a backend contracts
    it, and no `identical_mul_add` seam is needed here. Range (0, 1]."""
    return Float32(x) * CURAND_2POW32_INV + (CURAND_2POW32_INV / Float32(2.0))


def curand_uniform(mut state: curandStateXORWOW) -> Float32:
    """`curand_uniform(curandStateXORWOW_t*)` (`curand_uniform.h:108-111`)."""
    return _curand_uniform(curand(state))


# ---------------------------------------------------------------------------
# DEVIATION 683: rebuilding `precalc_xorwow_matrix` and
# `precalc_xorwow_offset_matrix` from the step. HOST ONLY.
# ---------------------------------------------------------------------------


def _xorwow_step_words(
    v0: UInt32, v1: UInt32, v2: UInt32, v3: UInt32, v4: UInt32
) -> SIMD[DType.uint32, 8]:
    """The linear part of `curand()` on explicit words (no `d`)."""
    var t: UInt32 = v0 ^ (v0 >> 2)
    var n4: UInt32 = (v4 ^ (v4 << 4)) ^ (t ^ (t << 1))
    var out = SIMD[DType.uint32, 8](0)
    out[0] = v1
    out[1] = v2
    out[2] = v3
    out[3] = v4
    out[4] = n4
    return out


def _matvec_words(
    vec: SIMD[DType.uint32, 8],
    mat: List[UInt32],
    mat_off: Int,
) -> SIMD[DType.uint32, 8]:
    """`__curand_matvec(vector, matrix, result, 5)` (`:336-348`) on a
    List-backed matrix starting at `mat_off`."""
    var res = SIMD[DType.uint32, 8](0)
    for i in range(XORWOW_N):
        var w = vec[i]
        for j in range(32):
            if (w & (UInt32(1) << UInt32(j))) != 0:
                var base = mat_off + XORWOW_N * (i * 32 + j)
                for k in range(XORWOW_N):
                    res[k] = res[k] ^ mat[base + k]
    return res


def _matmat(a: List[UInt32], b: List[UInt32]) -> List[UInt32]:
    """`__curand_matmat(matrixA, matrixB, 5)` (`:371-379`): row r of A
    becomes `matvec(row r of A, B)`. Returns the product (theirs stores
    back into A)."""
    var out = List[UInt32]()
    for _ in range(XORWOW_MATRIX_WORDS):
        out.append(UInt32(0))
    for r in range(XORWOW_N * 32):
        var row = SIMD[DType.uint32, 8](0)
        for k in range(XORWOW_N):
            row[k] = a[r * XORWOW_N + k]
        var res = _matvec_words(row, b, 0)
        for k in range(XORWOW_N):
            out[r * XORWOW_N + k] = res[k]
    return out^


def xorwow_step_matrix() -> List[UInt32]:
    """The 160x160 GF(2) matrix of one xorwow step, in cuRAND's layout:
    row `i*32+j` is the image of the unit vector with bit j of word i."""
    var mat = List[UInt32]()
    for _ in range(XORWOW_MATRIX_WORDS):
        mat.append(UInt32(0))
    for i in range(XORWOW_N):
        for j in range(32):
            var e = SIMD[DType.uint32, 8](0)
            e[i] = UInt32(1) << UInt32(j)
            var img = _xorwow_step_words(e[0], e[1], e[2], e[3], e[4])
            for k in range(XORWOW_N):
                mat[XORWOW_N * (i * 32 + j) + k] = img[k]
    return mat^


@fieldwise_init
struct XorwowTables(Movable):
    """DEVIATION 683: the two rebuilt tables. `sequence` is
    `precalc_xorwow_matrix` (A^(2^67 * 4^i)), `offset` is
    `precalc_xorwow_offset_matrix` (A^(4^i)), i = 0..31, each 32 x 800
    words, flattened."""

    var sequence: List[UInt32]
    var offset: List[UInt32]


def build_xorwow_tables() -> XorwowTables:
    """Rebuild both tables from `xorwow_step_matrix()` by repeated squaring
    (`__curand_matpow`'s arithmetic, `:399-413`). Host, integer only."""
    var a = xorwow_step_matrix()
    var offset = List[UInt32]()
    var sequence = List[UInt32]()
    # offset table: A^(4^i)
    var p = a.copy()
    for _ in range(PRECALC_NUM_MATRICES):
        for w in range(XORWOW_MATRIX_WORDS):
            offset.append(p[w])
        var p2 = _matmat(p, p)
        p = _matmat(p2, p2)
    # sequence table: A^(2^67) then ^4 each step
    var s = a.copy()
    for _ in range(XORWOW_SEQUENCE_SPACING):
        s = _matmat(s, s)
    for _ in range(PRECALC_NUM_MATRICES):
        for w in range(XORWOW_MATRIX_WORDS):
            sequence.append(s[w])
        var s2 = _matmat(s, s)
        s = _matmat(s2, s2)
    return XorwowTables(sequence^, offset^)
