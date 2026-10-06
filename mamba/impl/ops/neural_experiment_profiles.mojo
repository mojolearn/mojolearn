# SPDX-License-Identifier: Apache-2.0
"""NEURAL Mamba experiment controls and portable arithmetic building blocks.

Source-development only: no compilation, tests, measurements or identity
claims accompany this file. The fixed leaf/profile constants are independent
of launch geometry, device vendor and requested sequence length.
"""

from std.sys.compile import is_defined
from std.memory import stack_allocation
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul, identical_mul_add

comptime _NEURAL_IDN_EXPERIMENT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

# M07: 64 and 256 are launch resource alternatives around the existing 128
# independent output owners. They never define a reduction leaf or sequence
# chunk. All shapes use the selected block size; no benchmark dimensions.
comptime IDN_MAMBA_TPB_64 = (
    _NEURAL_IDN_EXPERIMENT
    and is_defined["MOJOLEARN_NI_M07_TPB_64"]()  # NOT TESTED — NOT COMPILED — NOT MEASURED; default OFF.
)
comptime IDN_MAMBA_TPB_256 = (
    _NEURAL_IDN_EXPERIMENT
    and is_defined["MOJOLEARN_NI_M07_TPB_256"]()  # NOT TESTED — NOT COMPILED — NOT MEASURED; default OFF.
    and not IDN_MAMBA_TPB_64  # Explicit precedence if both are requested.
)
comptime NEURAL_MAMBA_TPB = 64 if IDN_MAMBA_TPB_64 else (256 if IDN_MAMBA_TPB_256 else 128)

comptime IDN_M3_RETAIN_STATE_DECAY = (
    _NEURAL_IDN_EXPERIMENT
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and is_defined["MOJOLEARN_NI_M03_RETAIN_STATE_DECAY"]()  # NOT TESTED — NOT COMPILED — NOT MEASURED; default OFF.
)

comptime IDN_M3_RESOURCE_YINTRA = (
    _NEURAL_IDN_EXPERIMENT
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and is_defined["MOJOLEARN_NI_M04_RESOURCE_YINTRA"]()  # NOT TESTED — NOT COMPILED — NOT MEASURED; default OFF.
)

comptime IDN_M3_ANGLE_SUFFIX_SEEDS = (
    _NEURAL_IDN_EXPERIMENT
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and is_defined["MOJOLEARN_NI_M09_SUFFIX_SEEDS"]()  # NOT TESTED — NOT COMPILED — NOT MEASURED; default OFF.
)

# M10 is an explicit numerical revision: fixed 256-row leaves, adjacent
# pairwise merges and unpaired-node promotion. No extra +0 seed at any merge.
# The device and generated host import this same portable leaf-merge helper.
comptime IDN_M2_BALANCED_GRADS = (
    _NEURAL_IDN_EXPERIMENT
    and is_defined["MOJOLEARN_NI_M10_BALANCED_GRADS"]()  # NOT TESTED — NOT COMPILED — NOT MEASURED; default OFF.
)


def m2_balanced_gradient_fold(
    part: MutPointer[Float32, MutAnyOrigin], tiles: Int, cols: Int, column: Int,
) -> Float32:
    """Consume one scratch column using adjacent pairs, promoting odd tails.

    Only this output owner reads/writes this column. Scratch may be compacted
    in place because each pair's inputs are read before its lower slot is
    written; another column never aliases those slots. A single leaf returns
    verbatim, retaining signed zero. Padded leaves are never invented.
    """
    var active = tiles
    while active > 1:
        var pairs = active // 2
        for i in range(pairs):
            var left = ftz(part.unsafe_load((2 * i) * cols + column))
            var right = ftz(part.unsafe_load((2 * i + 1) * cols + column))
            part.unsafe_store(i * cols + column, ftz(left + right))
        if active % 2 != 0:
            part.unsafe_store(pairs * cols + column,
                              part.unsafe_load((active - 1) * cols + column))
        active = pairs + active % 2
    return part.unsafe_load(column)


# M08 building block only: NOT TESTED — NOT COMPILED — NOT MEASURED.
# Deliberately no live A/B selector: selective-scan backward, absolute token
# positions and checkpoint/resume serialization must be wired together first.
struct Mamba1AffineV2(Copyable, Movable):
    """One proposed FP32 affine-map node h -> a*h+b, with explicit rounding.

    left.then(right) means right(left(h)); the product coefficient is a
    separately rounded FTZ multiply and the offset is one FTZ FMA. This is
    an arithmetic revision, not an assertion that serial recurrence agrees.
    """

    var a: Float32
    var b: Float32

    def __init__(out self, a: Float32, b: Float32):
        self.a = ftz(a)
        self.b = ftz(b)

    def then(self, right: Self) -> Self:
        return Self(ftz(identical_mul(right.a, self.a)),
                    ftz(identical_mul_add(right.a, self.b, right.b)))

    def apply(self, state: Float32) -> Float32:
        return ftz(identical_mul_add(self.a, ftz(state), self.b))


def mamba1_absolute_chunk_v2(absolute_position: Int) -> Int:
    """64-token profile chunks anchored to absolute position, never a call."""
    return absolute_position // 64


def mamba1_chunk_prefix_v2(
    out_a: MutPointer[Float32, MutAnyOrigin],
    out_b: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    count: Int,
):
    """Proposed chunk-local inclusive affine prefixes, count in [1, 64].

    Source primitive only, not wired to a model. Six fixed Hillis-Steele
    rounds use distances 1,2,4,8,16,32 and two separate pages. Each round
    composes the older prefix on the left with the current prefix on the
    right. Unpaired positions copy verbatim, avoiding a synthetic identity
    multiply/add that could change signed zero. GPU schedule work may map
    these same six rounds to cooperative lanes after caller integration.

    A streaming caller MUST retain the pending absolute chunk's original
    leaves, recompute its prefixes on extension, and seal only full chunks.
    Calling this separately on fragments would define a different profile.
    """
    var pa = stack_allocation[128, Scalar[DType.float32]]()
    var pb = stack_allocation[128, Scalar[DType.float32]]()
    for i in range(count):
        pa[i] = ftz(a.unsafe_load(i))
        pb[i] = ftz(b.unsafe_load(i))
    var read_page = 0
    var distance = 1
    for round in range(6):
        var write_page = 64 - read_page
        for i in range(count):
            var right = Mamba1AffineV2(pa[read_page + i], pb[read_page + i])
            if i >= distance:
                var left = Mamba1AffineV2(pa[read_page + i - distance],
                                          pb[read_page + i - distance])
                var combined = left.then(right)
                pa[write_page + i] = combined.a
                pb[write_page + i] = combined.b
            else:
                pa[write_page + i] = right.a
                pb[write_page + i] = right.b
        read_page = write_page
        distance *= 2
    for i in range(count):
        out_a.unsafe_store(i, pa[read_page + i])
        out_b.unsafe_store(i, pb[read_page + i])
