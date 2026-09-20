# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One table: every kernel, every tunable it takes, per GPU vendor."""

from std.sys.compile import is_defined
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_amd_rdna_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

from checks.numerics import (
    NumericMode,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    GLOBAL_NUMERIC_MODE,
)


comptime COLUMN_BIT_IDENTICAL = 0
comptime COLUMN_APPLE = 1
comptime COLUMN_NVIDIA = 2
comptime COLUMN_AMD = 3

comptime COLUMN_AMD_RDNA = 4
comptime COLUMN_QUALCOMM = 5
comptime COLUMN_INTEL = 6

comptime COLUMN_SPEC_BASELINE = 7

#: THE CPU COLUMN (the CPU training lane, 2026-09-13; brief
#: docs/lanes/BRIEF_cpu_training_2026-09-13.md section 2). A build with NO
#: accelerator target compiles this column, never `COLUMN_APPLE`, which the
#: fallthrough of `TARGET_COLUMN` handed every host build until this column
#: existed. It is not a vendor: no kernel is launched on it, and every
#: NUMERIC row it answers is the pinned `COLUMN_BIT_IDENTICAL` reading, so a
#: host routine that MIRRORS a device fold reads the same widths and budgets
#: the identical contract pins (32 lanes, the 32 KB identity floor, 1024
#: threads). Every SCHEDULING and ROUTING row answers the non-vendor default,
#: never NVIDIA's or AMD's measured schedule and never Apple's repairs (the
#: kNN zero-FMA repair and preflight, the two-level quantize search, the
#: subnormal-flushing compare), because a CPU FMA rounds once and flushes
#: nothing. FAST rows never reach it: every `build_*_host.sh` is IDENTICAL
#: only. `bindings/build_byte_lm_host.sh` and `bindings/build_forest_host.sh`
#: pass `-D MOJOLEARN_COLUMN_CPU`, and each host binding carries
#: `comptime assert TARGET_COLUMN == COLUMN_CPU`.
comptime COLUMN_CPU = 8

#: THE TPU AND TRAINIUM COLUMNS (2026-09-14, lane/declared-graph-columns;
#: docs/lanes/DECLARED_TPU_TRAINIUM_COLUMNS_2026-09-14.md). DECLARED, NOT
#: BUILDABLE, and NOT SIMULATABLE: neither has a `-D MOJOLEARN_COLUMN_*`
#: define in `TARGET_COLUMN`, so no build compiles against them and no row
#: they answer can reach a kernel. Mojo emits code for neither. Each takes
#: user kernels only through its vendor's own kernel language: Google's
#: Pallas on the TPU (`jax.experimental.pallas.tpu`) and AWS's Neuron Kernel
#: Interface on Trainium (`nki.isa`). Both machines are kernel-shaped the way
#: our GPU columns are (128 lanes, an on-chip scratchpad, explicit placement),
#: so the kernel-shaped rows answer the identical reading, as `COLUMN_CPU`
#: does. What refuses them is ARITHMETIC. The primitive rows below
#: (`column_fma_instruction`, `column_float32_division`, ...) record what
#: each vendor's documentation lets a kernel name, and IDENTICAL builds every
#: float it produces from those primitives.
#:
#: TPU pinned to v6 (Trillium), the generation the Pallas TPU details page
#: describes ("8x128 for 32-bit values (as of TPU v6)"). Trainium pinned to
#: NeuronCore-v2/v3 (Trn2, Trn3), the generations the NKI ISA reference
#: pages are marked relevant for.
comptime COLUMN_TPU = 9
comptime COLUMN_TRAINIUM = 10

comptime COLUMN_COUNT = 11

comptime COLUMN_METAL = COLUMN_APPLE
comptime COLUMN_CUDA = COLUMN_NVIDIA
comptime COLUMN_HIP = COLUMN_AMD
comptime COLUMN_CDNA = COLUMN_AMD
comptime COLUMN_RDNA = COLUMN_AMD_RDNA
comptime COLUMN_ADRENO = COLUMN_QUALCOMM
comptime COLUMN_XE = COLUMN_INTEL
comptime COLUMN_HOST = COLUMN_CPU


def column_name(column: Int) -> String:
    if column == COLUMN_BIT_IDENTICAL:
        return String("bit-identical")
    if column == COLUMN_APPLE:
        return String("apple")
    if column == COLUMN_NVIDIA:
        return String("nvidia")
    if column == COLUMN_AMD:
        return String("amd")
    if column == COLUMN_QUALCOMM:
        return String("qualcomm")
    if column == COLUMN_INTEL:
        return String("intel")
    if column == COLUMN_AMD_RDNA:
        return String("amd-rdna")
    if column == COLUMN_SPEC_BASELINE:
        return String("spec-baseline")
    if column == COLUMN_CPU:
        return String("cpu")
    if column == COLUMN_TPU:
        return String("tpu")
    if column == COLUMN_TRAINIUM:
        return String("trainium")
    return String("unknown")


def column_is_buildable(column: Int) -> Bool:
    """Whether Mojo can emit a kernel for this column TODAY. The CPU column
    emits no kernel and is buildable in the only sense that matters here: a
    host build with no accelerator target compiles it, on seven CI runners
    (the byte LM CPU gate)."""
    return (
        column == COLUMN_BIT_IDENTICAL
        or column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
        or column == COLUMN_CPU
    )


#: CONTRACT PRIMITIVES (2026-09-14, lane/declared-graph-columns). IDENTICAL
#: builds every float it produces from a short list, namely binary32 add,
#: subtract and multiply; ONE fused multiply-add instruction (`numerics.identical_mul_add`,
#: IDENTITY_PATHS row 9, and the residual inside every portable transcendental
#: and `portable_sqrtf`); ONE hardware division (`portable_divf`, row 49);
#: exact 32-bit integer arithmetic (the fixed-point accumulators, Philox);
#: and reading a float's bits (`ftz` flushes by bits, row 10). Square root,
#: rsqrt, exp, log and the trig functions are NOT primitives, because each is
#: built from the list. Each row below answers whether a kernel on the column can
#: NAME the primitive. Whether its rounding is right is a measurement, not a
#: row: `check-ieee-arith` (with its built-to-separate FMA arm) and
#: `check-division` are the first gates on every new column.
comptime CAP_ABSENT = 0
comptime CAP_PRESENT = 1
comptime CAP_UNAUDITED = 2


def cap_name(cap: Int) -> String:
    if cap == CAP_PRESENT:
        return String("yes")
    if cap == CAP_ABSENT:
        return String("NO")
    return String("?")


def column_kernel_language(column: Int) -> String:
    """What a user kernel on this column is written in. `mojo` for every column Mojo emits code for; `none` for the declared GPU columns, whose kernel language is whatever a future Mojo target emits; the vendor's own language for the TPU (Pallas, `jax.experimental.pallas.tpu`) and Trainium (the Neuron Kernel Interface, `nki.isa`), because those are the only doors either vendor documents and Mojo emits for neither."""
    if column == COLUMN_TPU:
        return String("pallas")
    if column == COLUMN_TRAINIUM:
        return String("nki")
    if column_is_buildable(column):
        return String("mojo")
    return String("none")


def column_fma_instruction(column: Int) -> Int:
    """CONTRACT PRIMITIVE: a fused multiply-add a kernel can name.

    - bit-identical, apple, nvidia, amd, amd-rdna, cpu: PRESENT. `std.math.fma`
      is the shipped spelling on all of them (row 9); Metal through MAX
      measured FUSED on 1,629 of 1,629 separating patterns.
    - tpu: ABSENT. The Pallas TPU op list (docs.jax.dev/en/latest/pallas/tpu/
      details.html, "Elementwise operations") has add, sub, mul, divide, max,
      min, select, abs, bitwise ops, shifts, compares, casts, exp, tanh, pow,
      sin and cos, and warns the list "might not be comprehensive" because JAX
      functions compose primitives. No JAX primitive and no StableHLO op
      (openxla/stablehlo docs/spec.md at 5a1e6d92, the full op index) is a
      fused multiply-add, so there is nothing to compose one from.
    - trainium: ABSENT. The NKI ISA operator table ("Supported Math Operators
      for NKI ISA", nki/api/nki.api.shared.html) lists add, subtract,
      multiply, max, min, compares, logical ops, power, abs_max, abs_min, abs,
      square, relu, rsqrt and reciprocal. `scalar_tensor_tensor` applies two
      operators "in sequence", documented as equivalent to two instructions
      back to back, which is the unfused spelling.
    - qualcomm, intel: PRESENT (audited 2026-09-15 against the Khronos
      specifications, docs/lanes/DECLARED_TPU_TRAINIUM_COLUMNS_2026-09-14.md).
      Both parts ship OpenCL (Adreno through Qualcomm's OpenCL driver, Xe
      through Intel's compute runtime), and the OpenCL C `fma` builtin
      "Returns the correctly rounded floating-point representation of the
      sum of c with the infinitely precise product of a and b. Rounding of
      intermediate products shall not occur" (OpenCL C 3.0, math functions;
      Table 65 lists fma as "Correctly rounded"). SPIR-V's OpenCL.std `fma`
      (instruction 25) and SYCL 2020's `sycl::fma` say the same. Whether the
      device does it in hardware is the `CL_FP_FMA` device flag; a software
      fma is slower and still one rounding. The hazard this row pins is in
      the same spec: `#pragma OPENCL FP_CONTRACT` defaults to ON, so a plain
      `a*b+c` may or may not be fused.
    - spec-baseline: ABSENT. The baseline is what BOTH portable
      specifications guarantee, and Vulkan's does not guarantee a fused one.
      GLSL.std.450 `Fma` "Computes a * b + c", and the GLSL 4.60 precision
      table allows `a * b + c` as a "Correctly rounded single operation or
      sequence of two correctly rounded operations", with `fma()` "Inherited
      from a * b + c".
    """
    if column == COLUMN_TPU or column == COLUMN_TRAINIUM:
        return CAP_ABSENT
    if column == COLUMN_SPEC_BASELINE:
        return CAP_ABSENT
    if column == COLUMN_QUALCOMM or column == COLUMN_INTEL:
        return CAP_PRESENT
    if column_is_buildable(column):
        return CAP_PRESENT
    return CAP_UNAUDITED


def column_float32_division(column: Int) -> Int:
    """CONTRACT PRIMITIVE: one binary32 division a kernel can name (`portable_divf` is one hardware division between two bit flushes, row 49).

    - bit-identical, apple, nvidia, amd, amd-rdna, cpu: PRESENT. Apple
      measured correctly rounded on the normal class (`check-division`); the
      NVIDIA/AMD re-print is still owed per row 49.
    - tpu: PRESENT as a name (`/` in the Pallas TPU op list, cost class
      medium); its rounding is unmeasured.
    - trainium: ABSENT. The NKI ISA operator table has no divide. The nearest
      instruction is `reciprocal`, and the same page says the Vector Engine
      computes it "at a higher precision compared to Scalar Engine" (whose
      activations are "approximated with piece-wise polynomials"), which
      documents neither one as a correctly rounded binary32 division.
    - qualcomm, intel, spec-baseline: PRESENT as a name (OpenCL C `/`,
      SPIR-V `OpFDiv`), and THIS IS THE ROW'S MEASUREMENT RISK. Neither
      specification requires a correctly rounded single-precision division.
      OpenCL C 3.0 Table 65 allows `x / y` "<= 2.5 ulp" (Table 66, the
      embedded profile, "<= 3 ulp"), and correct rounding is the optional
      `CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT` device flag ("divide and sqrt are
      correctly rounded as defined by the IEEE754 specification"). The GLSL
      4.60 precision table allows `a / b` "2.5 ULP". A device that does not
      report the flag is `check-division`'s to catch, and a column whose
      division fails it keeps a correctly rounded fma, which is enough to
      correct a quotient the way `portable_sqrtf` corrects a root (unbuilt).
    """
    if column == COLUMN_TRAINIUM:
        return CAP_ABSENT
    if column == COLUMN_TPU:
        return CAP_PRESENT
    if (
        column == COLUMN_QUALCOMM
        or column == COLUMN_INTEL
        or column == COLUMN_SPEC_BASELINE
    ):
        return CAP_PRESENT
    if column_is_buildable(column):
        return CAP_PRESENT
    return CAP_UNAUDITED


def column_int32_exact(column: Int) -> Int:
    """CONTRACT PRIMITIVE: exact 32-bit integer elementwise arithmetic (the fixed-point accumulators, the Philox draws).

    - bit-identical, apple, nvidia, amd, amd-rdna, cpu: PRESENT.
    - tpu: PRESENT. Pallas TPU supports `jnp.int*` and `jnp.uint*` and says
      the hardware "generally only supports elementwise computation using
      32-bit types". Integer REDUCTIONS are not supported there, so an
      integer fold is spelled elementwise. Wraparound on overflow is not
      documented (StableHLO leaves integer overflow implementation-defined).
    - trainium: PRESENT. `nki.isa.tensor_tensor`: all-int32/uint32 operands
      default to the GpSimd Engine, "which uses native integer arithmetic.
      This ensures exact results for all 32-bit integer values."
    - qualcomm, intel, spec-baseline: PRESENT. OpenCL C `uint` is "An
      unsigned 32-bit integer" and `int` a two's complement 32-bit integer;
      SPIR-V and GLSL carry the same 32-bit integer types. The accumulators
      and Philox use unsigned words, whose wraparound C99 defines.
    """
    if column == COLUMN_TPU or column == COLUMN_TRAINIUM:
        return CAP_PRESENT
    if (
        column == COLUMN_QUALCOMM
        or column == COLUMN_INTEL
        or column == COLUMN_SPEC_BASELINE
    ):
        return CAP_PRESENT
    if column_is_buildable(column):
        return CAP_PRESENT
    return CAP_UNAUDITED


def column_float_bits_readable(column: Int) -> Int:
    """CONTRACT PRIMITIVE: reinterpreting a binary32 as its 32 bits and back (`ftz` flushes by bits, because Metal's compares flush their operands, row 49's finding (i)).

    - bit-identical, apple, nvidia, amd, amd-rdna, cpu: PRESENT (`bitcast`).
    - tpu: UNAUDITED. StableHLO has `bitcast_convert`; the Pallas TPU op list
      names type casts (`.astype`, a value conversion) and does not say
      whether a bit reinterpretation lowers.
    - trainium: UNAUDITED. NKI bitvec operators treat INTEGER tiles as bit
      patterns; the reference read for this lane does not say how a float32
      tile's bits reach an integer tile.
    - qualcomm, intel, spec-baseline: PRESENT. SPIR-V's core `OpBitcast`
      reinterprets a value's bits in both the OpenCL and Vulkan flavors,
      SYCL 2020 pre-adopts `std::bit_cast` as `sycl::bit_cast`, and GLSL has
      `floatBitsToUint`. GLSL warns that `intBitsToFloat` may flush a
      subnormal to zero, which is the direction `ftz` flushes anyway.
    """
    if column == COLUMN_TPU or column == COLUMN_TRAINIUM:
        return CAP_UNAUDITED
    if (
        column == COLUMN_QUALCOMM
        or column == COLUMN_INTEL
        or column == COLUMN_SPEC_BASELINE
    ):
        return CAP_PRESENT
    if column_is_buildable(column):
        return CAP_PRESENT
    return CAP_UNAUDITED


def column_arithmetic_refusal_reason(column: Int) -> String:
    """Why the column cannot run IDENTICAL's arithmetic, naming EVERY absent primitive, or empty. Only a documented ABSENT refuses: UNAUDITED is a measurement owed, never a verdict in either direction."""
    var missing = String("")
    if column_fma_instruction(column) == CAP_ABSENT:
        missing = missing + (
            " no fused multiply-add instruction (IDENTICAL spells every a*b+c,"
            " every portable transcendental and portable_sqrtf's residual with"
            " one fma, IDENTITY_PATHS row 9);"
        )
    if column_float32_division(column) == CAP_ABSENT:
        missing = missing + (
            " no binary32 division instruction (portable_divf is one hardware"
            " division, row 49);"
        )
    if column_int32_exact(column) == CAP_ABSENT:
        missing = missing + (
            " no exact 32-bit integer arithmetic (the fixed-point accumulators"
            " and Philox);"
        )
    if column_float_bits_readable(column) == CAP_ABSENT:
        missing = missing + (
            " no way to read a float's bits (ftz flushes by bits, row 10);"
        )
    if missing.byte_length() == 0:
        return missing
    return (
        column_name(column)
        + " (kernel language "
        + column_kernel_language(column)
        + ") documents"
        + missing
        + " the column stays refused until the vendor documents the"
        " instruction or an exact-integer construction of it is gated"
    )

comptime K_HIST_BINARY = 0
comptime K_HIST_HALF_BYTE = 1
comptime K_HIST_ONE_BYTE = 2
comptime K_SCAN = 3
comptime K_SUBTRACT = 4
comptime K_SCORES = 5
comptime K_SPLIT_POINTS = 6
comptime K_HIST_2_ONE_BYTE = 7

comptime K_POINTWISE_HIST_2 = 8

comptime K_POINTWISE_HIST_2_HALF_BYTE = 9

comptime PINNED_REPLICATION_LANES = 32

comptime PINNED_REDUCE_WIDTH = 512


@fieldwise_init
struct KernelSpec(Copyable, Movable):
    """Every knob one kernel takes. Resolved once, never re-derived."""

    var block_size: Int
    """SCHEDULING. Threads per threadgroup."""

    var hist_floats_per_thread: Int
    """NUMERIC, and it reads as a memory-budget row."""

    var features_per_int: Int
    """NUMERIC in effect: it is the packing, so it decides which features share a load and therefore which sums are formed."""

    var replication_lanes: Int
    """NUMERIC. See PINNED_REPLICATION_LANES."""

    var reduce_width: Int
    """NUMERIC. See PINNED_REDUCE_WIDTH."""

    var deterministic_flush: Bool
    """NUMERIC."""

    var flush_forced_by_vendor: Bool
    """Whether `deterministic_flush` is the mode's choice or the vendor's constraint."""

    def shared_bytes(self) -> Int:
        """What this spec asks of threadgroup memory."""
        return self.block_size * self.hist_floats_per_thread * 4



comptime IDENTITY_PROFILE = 1

comptime IDENTITY_FLOOR_SHARED_BYTES = 32 * 1024

comptime IDENTITY_FLOOR_LANES = 32

comptime IDENTITY_FLOOR_BLOCK = 512


def column_meets_identity_floor(column: Int) -> Bool:
    """Whether this vendor can join `IDENTICAL` without the floor moving. The arithmetic clause (2026-09-14) refuses only on a documented ABSENT primitive, so every column declared before it resolves as it did."""
    return (
        column_shared_limit(column) >= IDENTITY_FLOOR_SHARED_BYTES
        and column_has_threadgroup_int_atomics(column)
        and column_max_block_size(column) >= IDENTITY_FLOOR_BLOCK
        and column_arithmetic_refusal_reason(column).byte_length() == 0
    )


def identity_refusal_reason(column: Int) -> String:
    """Why `IDENTICAL` refuses this column, or empty if it does not. The arithmetic reason comes first and the kernel-shaped reason follows it, so a column refused on both (the spec baseline, since 2026-09-15) says both."""
    var arithmetic = column_arithmetic_refusal_reason(column)
    var kernel = _kernel_floor_refusal_reason(column)
    if arithmetic.byte_length() > 0 and kernel.byte_length() > 0:
        return arithmetic + "; also " + kernel
    if arithmetic.byte_length() > 0:
        return arithmetic
    return kernel


def _kernel_floor_refusal_reason(column: Int) -> String:
    """The kernel-shaped half of `identity_refusal_reason`: memory, atomics and block size."""
    if column_shared_limit(column) < IDENTITY_FLOOR_SHARED_BYTES:
        return (
            column_name(column)
            + " allows "
            + String(column_shared_limit(column) // 1024)
            + " KB of threadgroup memory per block; the identity floor"
            " (profile "
            + String(IDENTITY_PROFILE)
            + ") needs "
            + String(IDENTITY_FLOOR_SHARED_BYTES // 1024)
            + " KB, because the block size it buys decides the replication"
            " factor and the replication factor decides which partial sums"
            " combine"
        )
    if not column_has_threadgroup_int_atomics(column):
        return (
            column_name(column)
            + " has no threadgroup integer atomic add; the identity column"
            " accumulates the histogram in shared Int32 and there is no"
            " substitute that keeps addition associative"
        )
    if column_max_block_size(column) < IDENTITY_FLOOR_BLOCK:
        return (
            column_name(column)
            + " dispatches at most "
            + String(column_max_block_size(column))
            + " threads per block; the identity column's hist_2 arm runs "
            + String(IDENTITY_FLOOR_BLOCK)
        )
    return String("")


def column_shared_limit(column: Int) -> Int:
    """Threadgroup / shared / LDS / SLM bytes a single block may claim."""
    if column == COLUMN_APPLE:
        return 32 * 1024
    if column == COLUMN_NVIDIA:
        return 48 * 1024
    if column == COLUMN_AMD:
        return 64 * 1024
    if column == COLUMN_QUALCOMM:
        return 32 * 1024
    if column == COLUMN_INTEL:
        return 64 * 1024
    if column == COLUMN_AMD_RDNA:
        return 64 * 1024
    if column == COLUMN_SPEC_BASELINE:
        return 16 * 1024
    if column == COLUMN_CPU:
        # No threadgroup memory exists on the host. The row sizes device
        # pages only, and the identical reading is the one every host twin
        # of a device fold must see, so this is the frozen floor and not a
        # vendor's budget: the byte LM host binding compiles
        # `lib_smem_pages_for` / `lib_smem_page_fits_for` through
        # gemm/checks/gemm_identical.mojo, and 32 KB is the value it compiled
        # under the Apple fallthrough on seven runners.
        return IDENTITY_FLOOR_SHARED_BYTES
    if column == COLUMN_TPU:
        # The identical reading, not a vendor budget. A Pallas TPU kernel
        # computes in VMEM, which the details page calls "fairly large for
        # such a low-level memory hierarchy (16MB+)", shared by the kernel
        # rather than claimed per block; no build compiles this row.
        return IDENTITY_FLOOR_SHARED_BYTES
    if column == COLUMN_TRAINIUM:
        # The identical reading, not a vendor budget. An NKI kernel computes
        # in SBUF ("On-chip scratchpad SRAM that serves as a software-managed
        # cache"); the per-partition size is not recorded here, and no
        # build compiles this row.
        return IDENTITY_FLOOR_SHARED_BYTES
    return IDENTITY_FLOOR_SHARED_BYTES  # BIT_IDENTICAL: frozen, not derived


def column_has_float_atomics(column: Int) -> Bool:
    """Whether this vendor can do `atomicAdd` on a `float` at all."""
    return (
        column == COLUMN_BIT_IDENTICAL
        or column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
        or column == COLUMN_INTEL
        # the identical reading; a host build is IDENTICAL only, where
        # `deterministic_flush_for` is true whatever this row says
        or column == COLUMN_CPU
    )


def column_compares_flush_subnormals(column: Int) -> Bool:
    """CAPABILITY. The CPU column answers False: `ftz` is explicit under
    IDENTICAL (checks/numerics.mojo flushes by bits, not by MXCSR), and a
    host compare sees the subnormal it is handed."""
    if column == COLUMN_CPU:
        return False
    return column == COLUMN_APPLE


comptime VENDOR_TF32_PRODUCT_REL_BOUND = Float64(1.0e-3)


def column_vendor_fp32_matmul_is_tf32(column: Int) -> Bool:
    """CAPABILITY. The CPU column answers False: a host fp32 product is fp32."""
    if column == COLUMN_CPU:
        return False
    return column == COLUMN_NVIDIA


def vendor_fp32_matmul_is_lossy(column: Int, compute_capability: Int) -> Bool:
    """The runtime form of `column_vendor_fp32_matmul_is_tf32`: the column predicate OR'd with the one generation fact the column cannot carry -- an Apple part reporting `compute_capability == 5` (M5) runs MAX 26.5.0's fp19 simdgroup path by default."""
    if column_vendor_fp32_matmul_is_tf32(column):
        return True
    return column == COLUMN_APPLE and compute_capability == 5


def vendor_fp32_matmul_precision_name(
    column: Int, compute_capability: Int
) -> String:
    """What a check prints beside its tolerance: the precision class of the vendor fp32 product on this build and device."""
    if column_vendor_fp32_matmul_is_tf32(column):
        return String("TF32 (10-bit mantissa tensor-core product)")
    if column == COLUMN_APPLE and compute_capability == 5:
        return String("fp19 (Apple M5 simdgroup MMA, 10-bit mantissa)")
    return String("fp32")


def column_has_threadgroup_int_atomics(column: Int) -> Bool:
    """Whether a block can `atomicAdd` an `Int32` in THREADGROUP memory.

    TPU and Trainium answer False, because neither vendor's kernel pages
    document an atomic add. The TPU column is the interesting one. A Pallas TPU grid runs
    "sequentially, in lexicographic order", and consecutive invocations may
    write the same output slice "without any risk of race conditions", so a
    plain integer store has a fixed order there and needs no atomic. The
    floor's clause is written for concurrent blocks; restating it for a
    sequential grid is a profile question owed at bring-up, not decided here.
    """
    return column != COLUMN_TPU and column != COLUMN_TRAINIUM


def column_has_dedicated_shared_memory(column: Int) -> Bool:
    """Whether "shared memory" is an on-chip scratchpad or just cached RAM."""
    return True


def column_spec_guarantees_onchip_shared(column: Int) -> Bool:
    """Whether anything PROMISES the shared memory is on chip. The TPU column answers False, since the Pallas TPU pages call VMEM "small but fast" and a lower level of the memory hierarchy, and nothing read for this lane says where it lives. Trainium answers True, since the NKI overview calls SBUF "On-chip scratchpad SRAM"."""
    return column != COLUMN_SPEC_BASELINE and column != COLUMN_TPU


def column_max_block_size(column: Int) -> Int:
    """Largest threadgroup the vendor will dispatch, before our budget bites."""
    if column == COLUMN_SPEC_BASELINE:
        return 128
    if column == COLUMN_CPU:
        return 1024  # the identical reading; no block is ever dispatched
    if column == COLUMN_TPU:
        return 1024  # the identical reading; also one 8x128 vector register tile (TPU v6)
    if column == COLUMN_TRAINIUM:
        return 1024  # the identical reading; no NKI instruction maps to a thread block
    return 1024


def column_lane_width(column: Int) -> Int:
    """Hardware lanes that move in lockstep: warp on NVIDIA, SIMD group on Apple, WAVEFRONT on AMD, wave on Adreno, sub-group on Intel."""
    if column == COLUMN_AMD:
        return 64
    if column == COLUMN_QUALCOMM:
        return 8
    if column == COLUMN_INTEL:
        return 8
    if column == COLUMN_AMD_RDNA:
        return 32
    if column == COLUMN_SPEC_BASELINE:
        return 1
    if column == COLUMN_CPU:
        # PINNED_REPLICATION_LANES, the identical reading (`lane_width_for`
        # answers it under IDENTICAL on every column). Not 1: a host has no
        # lanes, but a host twin of a device fold restates a 32-lane group,
        # and `GEMM_HEAD_LANES = lib_lane_width_for[TARGET_COLUMN]()` in
        # gemm/checks/gemm_identical.mojo:2882 is compiled into the byte LM
        # host binding, which matched the GPU bits on seven runners at 32.
        return PINNED_REPLICATION_LANES
    if column == COLUMN_TPU:
        # "TPUs perform the bulk of the computation on 2D vector registers,
        # which are typically of size 8x128 for 32-bit values (as of TPU
        # v6)", sublanes and lanes respectively (Pallas TPU details page).
        return 128
    if column == COLUMN_TRAINIUM:
        # NKI instructions run over a partition axis that "must not exceed
        # 128", and the Scalar Engine keeps "one 32-bit register per compute
        # lane, 128 registers in total" (nki.isa.activation). The pinned 32
        # lanes divide 128.
        return 128
    return 32


def column_lane_width_is_fixed(column: Int) -> Bool:
    """Whether `column_lane_width` is a property of the DEVICE or a decision the vendor's compiler makes per kernel. The CPU column answers True: its width is the pinned constant, not a compiler's choice."""
    if column == COLUMN_CPU:
        return True
    return (
        column != COLUMN_QUALCOMM
        and column != COLUMN_INTEL
        and column != COLUMN_SPEC_BASELINE
    )


def spec_for(kernel: Int, device: Int, mode: NumericMode) raises -> KernelSpec:
    """The resolved knobs for one kernel, substituting column by column."""
    var identical = mode.mode == NUMERIC_IDENTICAL
    var numeric_column = COLUMN_BIT_IDENTICAL if identical else device

    var floats_per_thread = 16
    var per_int = 8
    if kernel == K_HIST_BINARY:
        per_int = 32
    elif kernel == K_HIST_ONE_BYTE or kernel == K_HIST_2_ONE_BYTE:
        floats_per_thread = 32
        per_int = 4
    elif kernel == K_HIST_HALF_BYTE:
        per_int = 8
    else:
        floats_per_thread = 0
        per_int = 0

    var catboost_block = 384 if (
        kernel == K_HIST_ONE_BYTE or kernel == K_HIST_2_ONE_BYTE
    ) else 768
    var block = catboost_block
    if floats_per_thread > 0:
        var limit = column_shared_limit(numeric_column) // (
            floats_per_thread * 4
        )
        if limit < block:
            block = limit
    var hard_cap = column_max_block_size(device)
    if hard_cap < block:
        block = hard_cap
    elif kernel == K_SCORES:
        block = 128  # compute_scores.cu:167
    elif kernel == K_SPLIT_POINTS:
        block = 256  # compute_scores.cu:493
    else:
        block = 512

    if block < 32:
        raise Error(
            "kernel "
            + String(kernel)
            + " cannot fit a block in column "
            + column_name(numeric_column)
            + ": "
            + String(floats_per_thread)
            + " floats per thread leaves room for "
            + String(block)
            + " threads, and the replication geometry needs at least one"
            " full lane group"
        )

    var vendor_forces_flush = not column_has_float_atomics(device)
    var flush = mode.deterministic_flush() or vendor_forces_flush

    return KernelSpec(
        block,
        floats_per_thread,
        per_int,
        PINNED_REPLICATION_LANES,
        PINNED_REDUCE_WIDTH if identical else block,
        flush,
        vendor_forces_flush,
    )



#: The build's column. The `-D MOJOLEARN_COLUMN_*` define wins, the CPU
#: define first (a host build script passes it and nothing else may); then
#: the accelerator target the compiler was handed; and a build with no
#: accelerator target at all is the CPU column. Until 2026-09-13 the last
#: line read `COLUMN_APPLE` unconditionally, so every host build compiled as
#: the Apple column (bit-inert for the rows the byte LM reaches, wrong for the
#: kNN rows a host fit would reach). Apple is now behind its own predicate,
#: the one `checks/vendor.mojo` already folds into `COMPILED_VENDOR`, so a
#: Metal build still compiles Apple and a build with no accelerator does not.
comptime TARGET_COLUMN = (
    COLUMN_CPU if is_defined["MOJOLEARN_COLUMN_CPU"]() else
    COLUMN_APPLE if is_defined["MOJOLEARN_COLUMN_APPLE"]() else
    COLUMN_NVIDIA if is_defined["MOJOLEARN_COLUMN_NVIDIA"]() else
    COLUMN_AMD if is_defined["MOJOLEARN_COLUMN_AMD"]() else
    COLUMN_AMD_RDNA if is_defined["MOJOLEARN_COLUMN_AMD_RDNA"]() else
    # Explicit declaration simulation only; these do not enable a backend.
    COLUMN_QUALCOMM if is_defined["MOJOLEARN_COLUMN_QUALCOMM"]() else
    COLUMN_INTEL if is_defined["MOJOLEARN_COLUMN_INTEL"]() else
    COLUMN_SPEC_BASELINE if is_defined["MOJOLEARN_COLUMN_SPEC_BASELINE"]() else
    COLUMN_AMD_RDNA if has_amd_rdna_gpu_accelerator() else
    COLUMN_AMD if has_amd_gpu_accelerator() else
    COLUMN_NVIDIA if has_nvidia_gpu_accelerator() else
    COLUMN_APPLE if has_apple_gpu_accelerator() else
    COLUMN_CPU
)


comptime DETECTED_COLUMN = (
    COLUMN_AMD_RDNA if has_amd_rdna_gpu_accelerator() else
    COLUMN_AMD if has_amd_gpu_accelerator() else
    COLUMN_NVIDIA if has_nvidia_gpu_accelerator() else
    COLUMN_APPLE if has_apple_gpu_accelerator() else
    COLUMN_CPU
)


def column_is_simulated() -> Bool:
    """True when `-D MOJOLEARN_COLUMN_*` names a vendor this device is not."""
    return TARGET_COLUMN != DETECTED_COLUMN


def column_is_host(column: Int) -> Bool:
    """Whether `column` is the CPU column, the one a build with no accelerator target compiles."""
    return column == COLUMN_CPU


def hist_floats_per_thread_for[kernel: Int]() -> Int:
    """Shared floats per thread. `GetHistSize()` is this times the block."""
    if (
        kernel == K_HIST_ONE_BYTE
        or kernel == K_HIST_2_ONE_BYTE
        or kernel == K_POINTWISE_HIST_2
    ):
        return 32
    return 16


def catboost_block_for[kernel: Int]() -> Int:
    """What CatBoost uses, before our shared-memory budget bites."""
    if (
        kernel == K_HIST_ONE_BYTE
        or kernel == K_HIST_2_ONE_BYTE
        or kernel == K_POINTWISE_HIST_2
    ):
        return 384
    return 768


def block_size_for[kernel: Int, column: Int]() -> Int:
    """SCHEDULING row, bounded by a NUMERIC one."""
    comptime floats = hist_floats_per_thread_for[kernel]()
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime cb_cap = catboost_block_for[kernel]()
    comptime cap = (
        IDENTITY_FLOOR_BLOCK if identical
        and IDENTITY_FLOOR_BLOCK < cb_cap else cb_cap
    )
    comptime budget = (
        IDENTITY_FLOOR_SHARED_BYTES if identical
        else column_shared_limit(column)
    )
    comptime limit = budget // (floats * 4)
    comptime by_smem = limit if limit < cap else cap
    comptime hard = column_max_block_size(column)
    return by_smem if by_smem < hard else hard


def lane_width_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC."""
    if identical:
        return PINNED_REPLICATION_LANES
    return column_lane_width(column)


def replication_lanes_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC, and it is the row the whole CatBoost histogram family's layout actually rests on: the LOGICAL width of one private-replica group."""
    comptime assert PINNED_REPLICATION_LANES == IDENTITY_FLOOR_LANES, (
        "the logical replication width and the identity floor's lane count"
        " are one guarantee with two names; keep them equal"
    )
    return PINNED_REPLICATION_LANES


def reduce_width_for[kernel: Int, column: Int, identical: Bool]() -> Int:
    """NUMERIC."""
    comptime block = block_size_for[kernel, column]()
    comptime pinned = PINNED_REDUCE_WIDTH if identical else block
    return block if block < pinned else pinned



comptime SYNC_BLOCK = 0

comptime SYNC_LANE = 1


def sync_granularity_for[column: Int]() -> Int:
    """The finest sync a kernel may rely on."""
    return SYNC_BLOCK


def requires_uniform_iteration_for[column: Int]() -> Bool:
    """Whether every thread of a block must run the SAME iteration count."""
    return sync_granularity_for[column]() == SYNC_BLOCK


def sub_byte_lane_sync_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 1947): which barrier the CatBoost histogram accumulators use for their TURN-TAKING sync, the one that stands where theirs writes `tiled_partition<8>::sync()` or `tiled_partition<32>::sync()` between two writes to the same private slice."""
    if not column_lane_width_is_fixed(column):
        return SYNC_BLOCK
    if column_lane_width(column) != PINNED_REPLICATION_LANES:
        return SYNC_BLOCK
    return SYNC_LANE


def deterministic_flush_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row, comptime, so a kernel can branch on it."""
    return identical or not column_has_float_atomics(column)



comptime HIST_SMEM_WARP_PRIVATE_F32 = 0

comptime HIST_SMEM_SHARED2_I32 = 1


def pointwise_one_byte_fixed_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row: whether the POINTWISE one-byte family routes EVERY width through the 8-bit fixed-point accumulator."""
    comptime if identical:
        return True
    return column == COLUMN_APPLE


def pointwise_doc_split_for[column: Int, ordered: Bool]() -> Bool:
    """NUMERIC row (DEVIATION 2624): whether the POINTWISE histogram launchers split the DOCUMENT axis `EstimateBlockPerFeatureMultiplier` ways. Above one, every document block of a feature float-`atomicAdd`s its partial into the same `binSums` cell in whatever order the device finishes them, and the multiplier itself follows `sm_count`, so the ordered tiers (deterministic and identical) keep one block per feature group per part; see `pw_block_multiplier` in `gbdt/methods/pointwise_kernels.mojo`."""
    comptime if ordered:
        return False
    return True


#: DEVIATION 2670 (OPT-IN, NOT FLIPPED): the multiprocessor count the ordered
#: tiers' pointwise multiplier estimate reads on EVERY vendor, in place of
#: the device's own. A power of two near an H100's 132; NUMERIC, never tuned
#: per vendor, because the multiplier decides which document lands in which
#: block and therefore the float fold.
comptime PW_2670_PINNED_SM = 128


def pointwise_private_doc_slots_sm_for[column: Int, ordered: Bool]() -> Int:
    """NUMERIC row (DEVIATION 2670, opt-in behind `-D MOJOLEARN_2670_PW_PRIVATE_DOC_SLOTS=1`): 0 keeps DEVIATION 2624 (one document block per feature group per part). Non-zero splits the document axis in the ordered tiers at the multiplier `EstimateBlockPerFeatureMultiplier` gives for THIS pinned SM count, every document block writes its partial into its OWN scratch slot with a plain store, and one launch folds the slots into `binSums` in block order 0..M-1. New bits against 2624 (float addition is not associative, DEVIATION 2669), the same bits at every launch geometry and on every vendor; see `pw_block_multiplier` in `gbdt/methods/pointwise_kernels.mojo`."""
    comptime if ordered and is_defined["MOJOLEARN_2670_PW_PRIVATE_DOC_SLOTS"]():
        return PW_2670_PINNED_SM
    return 0


def greedy_one_byte_fixed_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1906, NARROWED by DEVIATION 1947): whether the GREEDY one-byte family routes EVERY width through the fused 8-bit fixed-point kernel (`hist_2_one_byte_8bit.mojo`) instead of CatBoost's maxBins ladder."""
    comptime if identical:
        return False
    comptime if is_defined["MOJOLEARN_2043_FAST_FUSED_ONE_BYTE"]():
        return True
    return True


def greedy_sub_byte_excluded_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row, RETRACTED 2026-09-01 (DEVIATION 1947 supersedes DEVIATION 1910): whether the GREEDY sub-byte histogram families -- BINARY (32 features per word) and HALF-BYTE (8 per word) -- are comptime-EXCLUDED from the build, their launch sites refusing at runtime BY NAME."""
    return False


def greedy_quantized_hist_for[column: Int, identical: Bool]() -> Bool:
    """NUMERIC row (DEVIATIONS 1911/1912): whether the NON-SYMMETRIC drivers' one-byte histogram build routes through the QUANTIZED SHARED-HISTOGRAM family (`kernel/hist_quantized_shared.mojo`) -- per-round fixed-point gradient pairs packed one 64-bit word per row, ONE shared-memory Int32 histogram per thread block accumulated with threadgroup integer..."""
    comptime if identical:
        return False
    comptime if is_defined["MOJOLEARN_2045_FAST_NO_QUANT_HIST"]():
        return False
    return (
        column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
    )


def quantized_hist_group_features_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 1913): how many one-byte features one thread block's shared histogram covers in the quantized family."""
    comptime limit = column_shared_limit(column)
    var g = limit // (256 * 2 * 4)
    g = (g // 4) * 4
    if g < 4:
        g = 4
    if g > 32:
        g = 32
    return g


def reorder_single_pass_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1907): stable partition above 500,000 rows.

    IDENTICAL's NVIDIA candidate requires an explicit build define until
    a large-input identity and timing run exercises the routed kernel.
    Small identity fixtures cannot reach this branch. The kill switch wins
    over the opt-in, and other vendors retain the established partition.
    """
    comptime if is_defined["MOJOLEARN_2042_FAST_NO_LOOKBACK"]():
        return False
    comptime if identical:
        return (
            column == COLUMN_NVIDIA
            and is_defined["MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION"]()
        )
    return column == COLUMN_NVIDIA


def ridx_only_splits_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1902): whether the NON-SYMMETRIC driver's split moves only the row index, leaving the stat planes stationary for the life of the fit, with every stat reader gathering `stats[row_index[pos]]` instead of reading a permuted plane."""
    comptime if identical:
        return False
    comptime if is_defined["MOJOLEARN_2044_FAST_NO_RIDX_ONLY"]():
        return False
    return (
        column == COLUMN_APPLE
        or column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
    )


def hist_smem_mode_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC row: HOW the hist_2 family accumulates in shared memory."""

    comptime CATBOOST_PRIVATE_BYTES = 384 * 32 * 4
    comptime limit = column_shared_limit(column)

    comptime if is_defined["MOJOLEARN_2046_FAST_SHARED_I32"]():
        return HIST_SMEM_SHARED2_I32
    comptime if identical:
        return HIST_SMEM_SHARED2_I32  # the BIT_IDENTICAL column's value
    elif limit < CATBOOST_PRIVATE_BYTES:
        return HIST_SMEM_SHARED2_I32
    else:
        return HIST_SMEM_WARP_PRIVATE_F32



comptime PINNED_PARTITION_CHUNKS_SM = 32


def partition_chunks_sm_for[identical: Bool](device_sm: Int) -> Int:
    """The `sm_count` the partition-stats chunk formula is fed."""
    comptime if identical:
        return PINNED_PARTITION_CHUNKS_SM
    else:
        return device_sm


def hist2_block_size_for[column: Int, smem_mode: Int]() -> Int:
    """SCHEDULING row bounded by the NUMERIC budget, per accumulation mode."""

    comptime hard = column_max_block_size(column)
    comptime if smem_mode == HIST_SMEM_SHARED2_I32:
        comptime limit = column_shared_limit(column) // 64
        comptime by_smem = 512 if limit >= 512 else limit
        return by_smem if by_smem < hard else hard
    else:
        return block_size_for[K_HIST_2_ONE_BYTE, column]()


def pw_hist2_block_size_for[column: Int, fixed: Bool]() -> Int:
    """SCHEDULING row for the POINTWISE one-byte family's block, per route."""
    return block_size_for[K_POINTWISE_HIST_2, column]()


def pw_hist2_smem_floats_for[column: Int, fixed: Bool]() -> Int:
    """Companion to `pw_hist2_block_size_for`: the shared scratch, in 4-byte slots."""
    return 32 * pw_hist2_block_size_for[column, fixed]()


def replicas_for(hist_cells: Int) -> Int:
    """DELETED IN SPIRIT."""
    return -16
    return 1



comptime K_LIB_ROW_NORM = 100
comptime K_LIB_COLUMN_STATS = 101
comptime K_LIB_TRANSPOSE = 102
comptime K_LIB_GEMM_CONTRACTION = 103
comptime K_LIB_FUSED_DISTANCE_NN = 104
comptime K_LIB_REDUCE_BY_KEY = 105
comptime K_LIB_PLUS_PLUS = 106
comptime K_LIB_EPS_NEIGHBORHOOD = 107
comptime K_LIB_ADJ_SCAN = 108
comptime K_LIB_WEAK_CC = 109
comptime K_LIB_SELECT_RADIX = 110
comptime K_LIB_SELECT_WARPSORT = 111
comptime K_LIB_BALL_COVER_EPS = 112
comptime K_LIB_JACOBI_EIGH = 113
comptime K_LIB_GRAM_SPLITK = 114
comptime K_LIB_WEIGHTED_VERTEX_DEG = 115


comptime PINNED_LIB_REDUCE_LANES = 32

comptime PINNED_ACC_ROWS_PER_TH = 4
comptime PINNED_ACC_COLS_PER_TH = 4
comptime PINNED_KBLK = 32
comptime PINNED_VECLEN = 4


@fieldwise_init
struct LibKernelSpec(Copyable, Movable):
    """The knobs one library kernel takes, resolved per column."""

    var block_size: Int
    """SCHEDULING. Threads per threadgroup."""

    var lane_width: Int
    """SCHEDULING **only for indexing**, NUMERIC when it bounds a reduction."""

    var reduce_lanes: Int
    """NUMERIC. See `PINNED_LIB_REDUCE_LANES`."""

    var acc_rows_per_th: Int
    """NUMERIC. Policy4x4 accumulation geometry."""

    var acc_cols_per_th: Int
    """NUMERIC. Policy4x4 accumulation geometry."""

    var kblk: Int
    """NUMERIC. Policy4x4 K-block."""

    var veclen: Int
    """NUMERIC. Policy4x4 vector length."""

    var shared_limit: Int
    """SCHEDULING. Threadgroup bytes this column allows a block to claim."""


def lib_block_size(kernel: Int, column: Int) -> Int:
    """SCHEDULING."""
    if kernel == K_LIB_SELECT_RADIX:
        return 256
    if kernel == K_LIB_SELECT_WARPSORT:
        return 8 * lib_lane_width(column)
    if kernel == K_LIB_TRANSPOSE:
        return 32 * 32 // 4
    if kernel == K_LIB_JACOBI_EIGH:
        return 32
    if kernel == K_LIB_GEMM_CONTRACTION or kernel == K_LIB_FUSED_DISTANCE_NN:
        return 16 * 16
    if kernel == K_LIB_GRAM_SPLITK:
        return 256
    return 128


def lib_lane_width(column: Int) -> Int:
    """SCHEDULING."""
    return column_lane_width(column)


def lib_spec_for(
    kernel: Int, device: Int, mode: NumericMode
) raises -> LibKernelSpec:
    """The resolved knobs for one library kernel."""
    var identical = mode.mode == NUMERIC_IDENTICAL
    var numeric_column = COLUMN_BIT_IDENTICAL if identical else device

    var reduce_lanes = PINNED_LIB_REDUCE_LANES
    if not identical:
        reduce_lanes = PINNED_LIB_REDUCE_LANES

    var spec = LibKernelSpec(
        lib_block_size(kernel, device),
        lib_lane_width(device),
        reduce_lanes,
        PINNED_ACC_ROWS_PER_TH,
        PINNED_ACC_COLS_PER_TH,
        PINNED_KBLK,
        PINNED_VECLEN,
        column_shared_limit(numeric_column),
    )

    if spec.block_size < spec.reduce_lanes:
        raise Error(
            "library kernel "
            + String(kernel)
            + " resolves to a block of "
            + String(spec.block_size)
            + " threads in column "
            + column_name(device)
            + ", which is narrower than the "
            + String(spec.reduce_lanes)
            + "-lane fold it has to perform"
        )
    return spec^



def lib_smem_pages(kernel: Int, column: Int, page_bytes: Int) -> Int:
    """SCHEDULING."""
    if 2 * page_bytes <= column_shared_limit(column):
        return 2
    return 1



def lib_block_bounds_a_float_fold[kernel: Int]() -> Bool:
    """NUMERIC CLASSIFIER, and the correction of a label this file got wrong."""
    return (
        kernel == K_LIB_ROW_NORM
        or kernel == K_LIB_REDUCE_BY_KEY
        or kernel == K_LIB_PLUS_PLUS
        or kernel == K_LIB_COLUMN_STATS
        or kernel == K_LIB_JACOBI_EIGH
        or kernel == K_LIB_WEIGHTED_VERTEX_DEG
    )


def lib_block_size_for[kernel: Int, column: Int]() -> Int:
    """SCHEDULING for most rows, NUMERIC for three of them."""
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime numeric_row = lib_block_bounds_a_float_fold[kernel]()
    comptime resolved = (
        COLUMN_BIT_IDENTICAL if identical and numeric_row else column
    )
    comptime lanes = column_lane_width(resolved)
    if kernel == K_LIB_SELECT_RADIX:
        return 256
    if kernel == K_LIB_SELECT_WARPSORT:
        return 8 * lanes
    if kernel == K_LIB_TRANSPOSE:
        return 256
    if kernel == K_LIB_JACOBI_EIGH:
        return 32 if lanes <= 32 else lanes
    if kernel == K_LIB_GEMM_CONTRACTION or kernel == K_LIB_FUSED_DISTANCE_NN:
        return 256
    if kernel == K_LIB_GRAM_SPLITK:
        return 256
    return 128


def lib_lane_width_for[column: Int]() -> Int:
    """SCHEDULING."""
    return column_lane_width(column)


def lib_reduce_lanes_for[column: Int, identical: Bool]() -> Int:
    """NUMERIC."""
    return PINNED_LIB_REDUCE_LANES


def lib_smem_pages_for[column: Int, page_bytes: Int]() -> Int:
    """SCHEDULING."""
    comptime limit = column_shared_limit(column)
    return 2 if 2 * page_bytes <= limit else 1


def lib_smem_page_fits_for[column: Int, page_bytes: Int]() -> Bool:
    """SCHEDULING row (2026-09-09, orchestrator, Apple RUN OWED of the split-K lane): whether ONE shared page of `page_bytes` fits under the column's shared limit at all. `lib_smem_pages_for` answers "one page or two"; it cannot say "not even one". The 128x128 tuned pair at K step 32 is 36,864 bytes a page, which no NVIDIA leg noticed (48 KB) and which Metal refuses at pipeline creation (32 KB: "Threadgroup memory size (36864) exceeds the maximum threadgroup memory allowed (32768)", gemm_device_check and gemm_backward_check on the M4). A plan whose page does not fit resolves its K step down through this row instead of naming a vendor. The fused attention kernels ask the same question per head dim: the head-dim-128 backward path claims 35,600 bytes and takes the eager path on a 32 KB column; register-blocked forward needs 18,624 bytes and fits."""
    return page_bytes <= column_shared_limit(column)


def lib_hardware_ftz_fma_for[column: Int]() -> Bool:
    """Capability row for NVIDIA's explicit round-to-nearest FMA intrinsics.

    This is not permission to replace round-then-flush with a bare .ftz
    instruction. At a smallest-normal rounding boundary, fma.rn.ftz can
    return zero where round-then-flush returns 0x00800000. Callers must
    preserve explicit round-then-flush semantics or correct that case;
    see the adversarial seam evidence from 2026-09-09.
    Other columns retain their existing software spelling. The CPU column
    answers False, the software spelling: the host has no `.ftz` instruction
    and the fold flush is `ftz` in checks/numerics.mojo.
    """
    if column == COLUMN_CPU:
        return False
    return column == COLUMN_NVIDIA


def lib_gemm_stage_ftz_for[column: Int]() -> Bool:
    """Flush input operands once before shared staging, preserving their bits.

    NVIDIA already uses this transport. AMD CDNA qualified 2026-09-18:
    22.4% lower fixed-shape GEMM sum; 700-step enwik8/Pile comparisons
    improve 13.9%/15.3% with identical loss and state witnesses. The FMA
    rounding seam and fold topology do not change. Other columns remain
    on their existing path. See LANE_STATUS_gemm-kernel-speed.md.
    """
    return column == COLUMN_NVIDIA or column == COLUMN_AMD


def lib_postround_class_flush_for[column: Int]() -> Bool:
    """AMD post-round class flush, measured 2026-09-18.

    Same 262,144-triple hash as shipped round-then-flush; 700-step loss and
    final-state witnesses match NVIDIA on both corpora. The product seam
    change reduces AMD's late step by 8.43% across those corpora. See
    docs/lanes/LANE_STATUS_amd-gemm-class.md. Fold topology is unchanged.
    """
    return column == COLUMN_AMD


def lib_zero_fma_repair_for[column: Int]() -> Bool:
    """NUMERIC row (lane `lane/apple-seam-repair`, 2026-09-18): the column's
    native FMA flushes BEFORE rounding, so an rtf-spelled seam
    (`ftz(fma(a, b, acc))`) must repair a signed-zero result whose exact value
    rounds up to the smallest normal. Apple M4 hashes `fbr` over the
    262,144-triple seam probe (GEMM brief 14.3); NVIDIA and AMD compute `rtf`
    already. Implementation: `checks/rtf_seam.mojo`. `-D
    MOJOLEARN_NO_ZERO_FMA_REPAIR` is the never-shipped price arm."""
    comptime if is_defined["MOJOLEARN_NO_ZERO_FMA_REPAIR"]():
        return False
    return column == COLUMN_APPLE


def attn_masked_tail_replay_for[column: Int]() -> Bool:
    """Exact omitted-tail replay, measured H100 and MI300X 2026-09-18.

    NVIDIA: paired 700-step enwik8/Pile GitHub runs: 3697/1768 backward
    corner refusals become zero; every loss and state witness matches. Late
    medians 0.455439 -> 0.197157 and 0.377134 -> 0.196831 seconds (geomean
    2.1038x). See docs/lanes/LANE_STATUS_lm-attention-fallback.md.

    AMD (Hot Aisle MI300X, same probe, seed and shape): all 700 losses, the
    six state hashes at steps 0/699 AND every per-step, per-layer status and
    replay-site vector equal NVIDIA's, in both arms. Refusals 3697/1768 -> 0
    (dQ replay 3449/1612 sites; zdot 0). Late medians 0.835384 -> 0.632183
    and 0.773939 -> 0.630843 seconds (1.3214x / 1.2268x, geomean 1.2733x).
    AMD's default arm has no estash zdot kernel, so its zdot keeps refusing
    on a corner (zero in training on either corpus).
    Merged with AMD GEMM operand staging, an MI325X enwik8 pair again equals
    NVIDIA everywhere: 0.762693 -> 0.550012 seconds (1.3867x).

    Apple: the Metal corner fixtures (zdot, 64 dQ cells, dk/dv) equal eager
    bit for bit, the replay-disabled arms differ (0x80000000 vs 0x00000000),
    and all 130 sites equal NVIDIA's. Apple's DEFAULT arm (`stash_tiled`)
    reaches none of the replay kernels, so on Apple this row is INERT in
    default training; a reduced HD64 training witness under the NVIDIA
    schedule define equals NVIDIA's. See
    docs/lanes/LANE_STATUS_attention-replay-vendors.md.
    """
    return column == COLUMN_NVIDIA or column == COLUMN_AMD or column == COLUMN_APPLE


def byte_lm_release_eager_for[column: Int]() -> Bool:
    """Bound byte-LM eager buffers at their last consumers.

    NVIDIA: the active 700-step eager lifetime arm matches all losses and
    checkpoint hashes and returns retained eager capacity to 432 bytes; aexp
    is kept separately. Automatic fused/eager switching matches too, with
    sampled peak 31537 -> 18497 MiB. AMD (MI300X, enwik8, legacy arithmetic
    with release): every loss, hash and routing vector equals legacy, eager
    capacity 432 bytes after every step (release active on 499 of 700 steps).
    Apple: the reduced HD64 witness with this row on equals NVIDIA's. This is
    a storage bound, not a speed claim. See
    docs/lanes/LANE_STATUS_attention-replay-vendors.md.
    """
    return column == COLUMN_NVIDIA or column == COLUMN_AMD or column == COLUMN_APPLE


def attn_zdot_rows_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2528, 2026-09-11, trial arm only; brief docs/lanes/BRIEF_attention_step_2026-09-11.md section 12): query rows per 256-thread block of the fused attention's register-blocked y/dy kernel (`fused_bwd_ydy_tiled_kernel`), 64 or 32. The kernel's shared page is `(2 * rows + 128) * 20` floats (20,480 B at 64, 15,360 B at 32), and on a column whose shared memory is partitioned per compute unit the page bounds the resident blocks. The rows are a schedule, never a numeric term: every chain keeps its terms and order at either value. UNMEASURED on every column. AMD reads 32 as the variant section 11.3 named to price; the page-only count (3 blocks x 64 rows vs 4 x 32 rows per CU) does not favor it, so the AMD leg prices both through the `_r32` / `_r64` arm names and this row follows that measurement. The shipped build reads it nowhere."""
    if column == COLUMN_AMD:
        return 32
    if column == COLUMN_CPU:
        return 64  # the non-vendor default; no attention kernel runs here
    return 64


def lib_gemm_block_parallelism_for[column: Int]() -> Int:
    """SCHEDULING row, SHIPPED since DEVIATION 2595 (2026-09-11; brief docs/lanes/BRIEF_gemm_long_k_2026-09-11.md sections 3, 4 and 10; first added by DEVIATION 2591 as a trial-arm row): how many 256-thread GEMM blocks the column runs side by side. A value above 0 TURNS ON the `ksplit` default in `gemm/checks/gemm_identical.mojo::identical_gemm_shipped_into`: every call the long-k group rule takes (section 4, rules 1, 2 and 4, at `S` = this value) runs the 128x128 group kernel over power-of-two leaf groups plus one fold launch, and every other call runs the plan `choose_gemm_plan` picks, as before. 0 turns it off: the dispatch compiles to the old line and the TUNED 128x128 plan runs exactly as it did. NVIDIA 132, MEASURED: the H100's SM count (docs/lanes/BRIEF_attention_step_2026-09-11.md section 3.1), and the value the `ksplit` arm ran at on the H100 leg that flipped it (bench/results/e1g/2026-09-11_152822-nvidia-h100-80gb-hbm3-gemm-longk, lean step geomean 0.895 on enwik8 and Pile GitHub, every step witness equal). AMD 110, MEASURED 2026-09-11 on the Hot Aisle MI300X (bench/results/e1g/2026-09-11_164818-amd-mi300x-hotaisle-gemm-longk): lean step 1.953 -> 1.198 s on both corpora, `ksplit` verdict FLIP at geomean 0.6136 and `ksplit_leaf` 0.6090, every step witness equal to shipped. The classical callers were then checked to HOLD under this row (68be1c79: SVC, kmeans, PCA and KDE on the same MI300X). This sentence previously said "AMD 0, OFF UNTIL THE MI300X LEG DECIDES" and contradicted the body below it for two days. The trial arm still runs on AMD at the column's reading through `lib_gemm_block_parallelism_trial_for`. Every other column 0 (Apple included, so the Apple identity card compiles the old line). A wrong value costs time and can never move a bit, because the group size reaches no leaf boundary and no tree level (brief section 5.5)."""
    if column == COLUMN_NVIDIA:
        return 132
    if column == COLUMN_AMD:
        # Measured 2026-09-11 on the Hot Aisle MI300X with the trial arm at S=110
        # (e1g/2026-09-11_164818-amd-mi300x-hotaisle-gemm-longk): lean step
        # 1.953 -> 1.198 s on both corpora, geomean 0.614, witnesses equal.
        return 110
    if column == COLUMN_CPU:
        # 0: the host oracle (`gemm_oracle`) folds the contract tree without
        # groups, and the dispatch compiles to the old line as it did under
        # the Apple fallthrough.
        return 0
    return 0


def lib_gemm_kernel_body_for[column: Int]() -> Int:
    """SCHEDULING row, SHIPPED since DEVIATION 2707 (2026-09-13; brief docs/lanes/BRIEF_gemm_kernel_2026-09-11.md sections 16 to 18): which KERNEL BODY the IDENTICAL GEMM dispatch runs on every call the TUNED 128x128 plan serves. 0 is the 2595 dispatch as it stood (`identical_gemm_tuned_kernel` where the ksplit rule declines the call, `identical_gemm_ksplit_kernel` groups where it takes it). 1 is the `kpack_hg` body (DEVIATION 2706): `identical_gemm_kpack_kernel` at the shipped 128x128 geometry with the padded, 16-byte-aligned packed page (2700, 2703), ONE 8-wide conflict-free shared store per thread per window instead of sixteen scalar stores at a four-way bank conflict (gather staging), and the fold's flush spelled as the hardware `mul.rn.ftz` by one that the step seam already uses (one instruction for six); the same group rule at the same row, the same fold tree, the same words at the same addresses in the same order, so no bit moves and the M4 arms check and the H100 step check say so. NVIDIA 1, MEASURED 2026-09-13 on a RunPod H100 (bench/results/e1g/2026-09-13_175602-nvidia-h100-gemm-hfgs): lean step 0.232 -> 0.211 s on enwik8 and Pile GitHub (geomean 0.9085), GEMM sum 143 -> 122 ms (0.852), every step witness equal to shipped; the decomposition that named the two costs is bench/results/e1g/2026-09-13_174125-nvidia-h100-gemm-diag2 (staging phase a third of the window, fold a fifth). AMD 1 as well, MEASURED 2026-09-13 on a Hot Aisle MI300X (the body's comment names the leg): the gather staging is placement and applies, the fold flush is NVIDIA's instruction and compiles out there. Apple and every other column 0: they compile exactly the line they compiled before. A wrong value costs time and can never move a bit. This row is the switch: 0 here is the revert."""
    if column == COLUMN_NVIDIA:
        return 1
    if column == COLUMN_AMD:
        # AMD 1, MEASURED 2026-09-13 on a Hot Aisle MI300X (gfx942), 8core VM,
        # one heat window, commit bfde0442
        # (bench/results/e1g/2026-09-13_193949-amd-mi300x-hotaisle-gemm-amd-row):
        #   verdict kpack_gs FLIP geomean=0.9570 enwik8=0.9559 pilegithub=0.9580
        #   verdict kpack_hg FLIP geomean=0.9585 enwik8=0.9573 pilegithub=0.9597
        #   every step witness equal to shipped on both corpora; step check PASS
        # GEMM sum 592 -> 561 ms (kpack_gs 0.947) and 559 ms (kpack_hg 0.944).
        # On this column the body's hardware fold flush compiles out
        # (`comptime if HW and TUNED_HW_FTZ_FMA`, NVIDIA only), so 1 here IS
        # the gather staging alone and the two arms price the same to 0.15
        # percent; one row value keeps one code path. Shipped-build gate on
        # the MI300X: brief section 18.4.
        return 1
    if column == COLUMN_CPU:
        return 0  # the revert line; no GEMM kernel body runs on the host
    return 0


def lib_gemm_block_parallelism_trial_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2595, 2026-09-11, trial arm only; brief docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section 10): the `S` the `ksplit` TRIAL arm reads, so a leg can still force the arm on a column whose shipped row is 0. The shipped row wherever it is above 0 (NVIDIA 132, so the arm and the default split identically there). AMD 110, from a READING, not a measurement (DEVIATION 2591): the attention brief section 11.1 records 110 CUs (pinned to the MI250X; the MI325X and MI300X counts are not in the repository) and resident blocks per CU as `min(2048 // 256, 65536 // page bytes)`; the shipped 128x128 GEMM block holds two 20,480 B pages (40,960 B), so one block per CU and 110 side by side. The MI300X leg's CONTROL pair `ctl_nt_1536x1408x768` / `ctl_nt_1664x1408x768` (`bench/gemm_step_price_main.mojo`) reads the real value. Every other column 0, meaning no reading: the arm then takes the finest split the workspace cap allows. The shipped build reads it nowhere."""
    comptime shipped = lib_gemm_block_parallelism_for[column]()
    if shipped > 0:
        return shipped
    if column == COLUMN_AMD:
        return 110
    if column == COLUMN_CPU:
        return 0  # no reading: the host has no blocks to run side by side
    return 0


def attn_fwd_rows_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2531, 2026-09-11; brief docs/lanes/BRIEF_attention_step_2026-09-11.md sections 14 and 15): query rows per 256-thread block of the fused attention's second-round forward kernel (`fused_attn_forward_r2_kernel`), 64 (the shipped hd-64 sstash geometry) or 32, read by the bare `_fgrid` arm token (`_fgrid_r32` / `_fgrid_r64` force it). The kernel's shared page is `(32 * 64 + rows * 35) * 4` bytes (17,152 B at 64 rows, 12,672 B at 32), and on a column whose shared memory is partitioned per compute unit the page bounds the resident blocks. The rows are a schedule, never a numeric term: the score, denominator and context chains keep their terms and order at either value, and the row maximum is an `identical_fmax` fold whose grouping is free. NVIDIA 32, MEASURED (DEVIATION 2534, H100 leg bench/results/e1g/2026-09-11_154257-nvidia-h100-80gb-hbm3-attention-round3, commit 5bcfa71d): the lean LM step under `stash_tiled_fgrid_r32` was 0.3718 / 0.3716 s against `stash_tiled` 0.3845 / 0.3819 s (enwik8 / Pile GitHub), every step witness equal, and `fgrid_r64` priced at 1.00x of stash_tiled on real activations while `fgrid_r32` priced 1.07x. AMD 32 is still the variant brief section 11.4 named to price (the page-only count, 3 blocks x 64 rows against 5 x 32 per CU, does not settle it); THE MI300X LEG DECIDES IT through the `_fgrid_r32` / `_fgrid_r64` arm names. Every other column 64, unmeasured. The shipped default arm (`attn_default_arm_for`) forces its rows with `_fgrid_r32`, so this row never moves a shipped path."""
    if column == COLUMN_NVIDIA:
        return 32
    if column == COLUMN_AMD:
        return 32
    if column == COLUMN_CPU:
        return 64  # the non-vendor default; no attention kernel runs here
    return 64


def attn_dkdv_keys_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2597, 2026-09-11; brief docs/lanes/BRIEF_attention_step_2026-09-11.md sections 16 and 18): keys per 256-thread block of the fused attention's trial dk/dv folds over the stash (`fused_bwd_dkdv_r2_kernel`, and `fused_bwd_kvfold_r2_kernel` under the `_kvsplit` token), 64 (the shipped `fused_bwd_dkdv_tiled_pf_kernel` geometry) or 32, read by the bare `_kvgrid` arm token (`_kvgrid_r32` / `_kvgrid_r64` force it). At 32 keys a thread holds 8 dk and 8 dv accumulators instead of 16 and 16, and the joint page is `(2 * 16 * 64 + 2 * 16 * keys) * 4` bytes (16,384 B at 64, 12,288 B at 32; a `_kvsplit` fold page is half that). The keys per block are a schedule, never a numeric term: every dk and dv chain keeps its terms and its order (heads of the kv group ascending, queries ascending over the key's visible range) at either value. AMD 32, MEASURED: DigitalOcean MI325X leg bench/results/e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv (commit 5cc3b8df), `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` against `baseline` lean step 1.623 / 1.633 -> 1.376 / 1.370 s (enwik8 / Pile GitHub, FLIP geomean 0.8436, every step witness equal), in-step dk/dv 169.8 ms (baseline) -> 21.2 ms; section 16 had read 32 from the tiled dk/dv thread state (32 accumulators and 10 operand registers per thread, twice the tiled dq fold's). The AMD shipped default (`attn_default_arm_for`) forces the same 32 with `_kvgrid_r32` (brief section 18), so this row and the default agree on AMD and a bare `_kvgrid` resolves to the default's instantiation there. Every other column 64, unmeasured. A shipped build reads this row only for a default carrying bare `_kvgrid`, which no column's default does."""
    if column == COLUMN_AMD:
        return 32
    if column == COLUMN_CPU:
        return 64  # the non-vendor default; no attention kernel runs here
    return 64


comptime ATTN_DEFAULT_WORD_BASELINE = 0
comptime ATTN_DEFAULT_WORD_STASH_TILED = 7
"""The attention arm word `stash_tiled`: bits 1 (fwd_sstash), 2 (bwd_stash) and 4 (bwd_tiled) of transformer/impl/llama/fused_attention.mojo (DEVIATIONS 2525 to 2527). The matrix cannot import that file (it imports this one), so the word is a literal here and fused_attention.mojo asserts at build time that it equals its own composition."""

comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF = 3175
"""The attention arm word `stash_tiled_fgrid_r32_qres_pf`: stash_tiled (7) | 64 (fwd_grid, DEVIATION 2531) | 2048 (forward rows 32) | 32 (fwd_qres, DEVIATION 2530) | 1024 (preflush, DEVIATION 2533) = 3175. fused_attention.mojo asserts at build time that it equals its own composition."""

comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32 = 52327
"""The attention arm word `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`: stash_tiled_fgrid_r32_qres_pf (3175) | 16384 (bwd_kvgrid, DEVIATION 2597) | 32768 (dk/dv keys per block 32) = 52327. fused_attention.mojo asserts at build time that it equals its own composition (`ATTN_ARM_R3_KVGRID_R32_DEFAULT`)."""

comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32 = 6343783
"""The attention arm word `stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32`: stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 (52327) | 2097152 (bwd_estash, DEVIATION 2650) | 4194304 (estash_dres, DEVIATION 2651) = 6343783. fused_attention.mojo asserts at build time that it equals its own composition (`ATTN_ARM_R3_KVGRID_R32_ESTASH_DRES_DEFAULT`)."""
comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32_BSWZ = 14732391
"""The word above plus DEVIATION 2900's `_bswz` bit (8388608): the causal
block-index map of the four kernels that arm runs. The matrix cannot import
transformer/impl/llama/fused_attention.mojo (that file imports this one), so
the word is a literal here and that file asserts at build time that it equals
its own composition (`ATTN_ARM_R3_KVGRID_R32_ESTASH_DRES_BSWZ_DEFAULT`)."""


def attn_default_arm_for[column: Int]() -> Int:
    """ROUTING row (DEVIATION 2534, 2026-09-11; brief docs/lanes/BRIEF_attention_step_2026-09-11.md sections 15 and 18): the attention arm word the SHIPPED build runs on this column (`ATTN_ARM_DEFAULT` in transformer/impl/llama/fused_attention.mojo; a `-D MOJOLEARN_ATTN_ARM_TRIAL=1` build runs it when MOJOLEARN_ATTN_ARM is unset and keeps every other arm selectable by name). Every arm is bit-equal to the eager oracle by the identity arguments of brief sections 4, 12, 14 and 16, so this row picks a schedule and never a result. NVIDIA `stash_tiled_fgrid_r32_qres_pf`, MEASURED: H100 leg bench/results/e1g/2026-09-11_154257-nvidia-h100-80gb-hbm3-attention-round3 (commit 5bcfa71d), lean LM step 0.3845 / 0.3819 s under stash_tiled against 0.3346 / 0.3340 s (enwik8 / Pile GitHub), every step witness equal, fwd+bwd on real activations 1.41x of stash_tiled; ENGINEERING_RULES 9 flips it. AMD `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` (ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32), MEASURED on the DigitalOcean MI325X against the previous AMD default `baseline`, every step witness equal (the comment in the body names the evidence and the verdict); a shipped build compiles its DEVIATION 2597 dk/dv kernel because the default carries it (brief section 18). Apple and every other column `stash_tiled` (unmeasured for the round 3 and 2597 arms as a price). `-D MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN=1` returns the NVIDIA word on every column, so a no-trial build on a Mac reaches the shipped round 3 branch; `-D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1` returns the previous NVIDIA word (now AMD's) on every column, so the same build reaches the shipped DEVIATION 2597 dk/dv branch; `-D MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN=1` returns the NVIDIA word as of the estash flip (DEVIATION 2657, without DEVIATION 2900's `_bswz` bit) on every column, so a no-trial build on a Mac reaches the shipped DEVIATION 2650 / 2651 estash branch (DEVIATION 2657, `ATTN_SHIPPED_BWD_ESTASH`; this is how the M4 gates that branch, since Apple's own default carries no estash bit). `-D MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN=1` returns the CURRENT NVIDIA word on every column, so a no-trial build on a Mac reaches the shipped DEVIATION 2900 branch (`ATTN_DEFAULT_BSWZ`; this is how the M4 gates a branch Apple's own default does not carry). Check knobs, the `MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL` pattern; never a shipped build; at most one of the four."""
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32_BSWZ
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF
    if column == COLUMN_NVIDIA:
        # DEVIATION 2657. Measured 2026-09-11 on a RunPod H100 80GB HBM3
        # against the previous NVIDIA default
        # stash_tiled_fgrid_r32_qres_pf_kvgrid_r32, commit 8dc33f00, every step
        # witness equal on both corpora, Apple vs NVIDIA identity trace
        # IDENTICAL over 60 stages
        # (bench/results/e1g/2026-09-11_215636-nvidia-h100-80gb-hbm3-attention-estash):
        #   verdict stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32 FLIP
        #   geomean=0.8207 enwik8=0.8226 pilegithub=0.8190 (same pod)
        # Lean step 0.2922 / 0.2916 -> 0.2403 / 0.2388 s (enwik8 / Pile GitHub).
        # In-step zdot 66.4 -> 14.9 ms; on real activations the backward is
        # 7.52 -> 3.23 ms (fwd+bwd 1.85x, the forward untouched at 1.00x).
        # The runner-up on the same leg, _estash without _dres, FLIP 0.8348:
        # ENGINEERING_RULES 9 takes the winner. The register lens (brief
        # section 20.2) reads the same: the shipped zdot kernel is 134 regs and
        # 1 block per SM, _estash 125 and 2, _estash_dres 64 and 4.
        # Before it: stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 (e1g/...185833,
        # FLIP geomean 0.9908); before that stash_tiled_fgrid_r32_qres_pf
        # (round 3, e1g/...154257).
        #
        # DEVIATION 2900 (brief section 22). Measured 2026-09-17 on a RunPod
        # H100 80GB HBM3, pod d7piefs556qlqe, commit de7d2063e, one heat
        # window, against the estash word above
        # (bench/results/e1g/2026-09-17_201140-nvidia-h100-attention-bswz):
        #   verdict stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32_bswz
        #   FLIP geomean=0.9592 enwik8=0.9583 pilegithub=0.9601
        #   witnesses_equal_baseline=True on both corpora
        # Lean step 0.20648 / 0.20651 -> 0.19786 / 0.19827 s (enwik8 / Pile
        # GitHub). It changes NO arithmetic: `_blk_map` is a bijection over
        # block_idx.x that hands the same (tile, head, batch) triples out
        # heaviest first instead of lightest first, so the hardware, which
        # dispatches in increasing block_idx.x, stops ending each causal
        # kernel in a tail of long blocks running alone.
        # In-step, per the lmtiming probes of the same leg (enwik8):
        #   forward r2   20.85 -> 17.19 ms  (0.825), 100 -> 96 regs, 2 -> 2 blocks/SM
        #   dq tiled     12.49 ->  9.68 ms  (0.775), 118 -> 128 regs, 2 -> 2
        #   dk/dv kvgrid 10.07 ->  7.10 ms  (0.706),  63 ->  63 regs, 4 -> 4
        #   zdot estash  14.79 -> 15.61 ms  (1.056),  64 ->  70 regs, 4 -> 3
        # The zdot kernel REGRESSES and the readback says why: the map's
        # index arithmetic costs it 6 registers, which crosses its occupancy
        # cliff from 4 resident blocks per SM to 3. The other three kernels
        # keep their occupancy and win on schedule alone. The net is -8.6 ms
        # of a 206.5 ms step; the zdot regression is a measured 0.83 ms left
        # on the table and brief section 22.7 names the follow-on.
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32_BSWZ
    if column == COLUMN_AMD:
        # DEVIATION 2657 ON AMD TOO. Measured 2026-09-12 on a Hot Aisle MI300X
        # (gfx942) against the previous AMD default
        # stash_tiled_fgrid_r32_qres_pf_kvgrid_r32, commit bb679f19, one VM and
        # one heat window, every step witness equal on both corpora
        # (bench/results/e1g/2026-09-12_133010-amd-mi300x-hotaisle-attn-estash):
        #   verdict stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32 FLIP
        #   geomean=0.9716 enwik8=0.9724 pilegithub=0.9709
        #   witnesses_equal_baseline=True on both corpora
        # Lean step 0.7573 / 0.7593 -> 0.7363 / 0.7372 s (enwik8 / Pile GitHub).
        # The gain is a third of NVIDIA's (0.8207 there) and the register lens
        # on this same VM says why. The shipped zdot kernel `zdot_stash_pf` is
        # 118 regs at a 17,696 byte page and ALREADY fits 3 blocks per CU here,
        # where on the H100 the same kernel was 134 regs at 1 block per SM. The
        # flipped kernel `zdot_estash_dres_pf` is 60 regs at 12,480 bytes and 5
        # blocks per CU (the plain `_estash` variant is 116 regs, 10,432 bytes,
        # 4 blocks). So AMD goes 3 -> 5 blocks where NVIDIA went 1 -> 4, and it
        # was never as starved to begin with, which is the whole difference in
        # the two gains. ENGINEERING_RULES 9 takes the win anyway: below 1 on
        # both corpora with the bits unmoved, and the rule sets no magnitude bar.
        # Before it: stash_tiled_fgrid_r32_qres_pf_kvgrid_r32, measured
        # 2026-09-11 on the DigitalOcean MI325X against baseline, commit
        # 5cc3b8df (e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv, FLIP
        # geomean=0.8436, in-step dk/dv 169.8 -> 21.2 ms); before that baseline
        # (e1g/2026-09-11_171959-amd-mi300x-runpod-attention-three).
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32
    if column == COLUMN_CPU:
        # The non-vendor default, the word the Apple fallthrough compiled into
        # the byte LM host binding. No fused attention kernel runs on the
        # host; the host step goes to the transformer oracles.
        return ATTN_DEFAULT_WORD_STASH_TILED
    return ATTN_DEFAULT_WORD_STASH_TILED


comptime STEP_GLUE_DEFAULT_WORD_SHIPPED = 0
comptime STEP_GLUE_DEFAULT_WORD_OPTSKIP_NOSHADOW_ROWS16 = 7
"""The step glue arm word `optskip_noshadow_rows16`: bits 1 (optskip, DEVIATION 2646), 2 (noshadow, DEVIATION 2647) and 4 (rows16, DEVIATION 2645) of core/step_glue.mojo = 7. The matrix cannot import that file (it imports this one), so the word is a literal here and core/step_glue.mojo asserts at build time that it equals its own composition (`STEP_GLUE_ARM_OPTSKIP_NOSHADOW_ROWS16`)."""


def step_glue_default_arm_for[column: Int]() -> Int:
    """ROUTING row (DEVIATION 2649, 2026-09-12; brief docs/lanes/BRIEF_step_glue_2026-09-11.md sections 2, 4 and 5): the step glue arm word the SHIPPED build runs on this column (`STEP_GLUE_ARM_DEFAULT` in core/step_glue.mojo; a `-D MOJOLEARN_STEP_GLUE_TRIAL=1` build runs it when MOJOLEARN_STEP_GLUE_ARM is unset and keeps every other arm selectable by name). Every arm is bit-equal to the shipped step by the identity arguments of brief sections 4.1, 4.2 and 4.3 and refuses the same inputs by section 5, so this row picks a schedule and never a result. NVIDIA `optskip_noshadow_rows16`, MEASURED (the body names the evidence and the verdict). Apple and every other column `shipped`, unmeasured as a price. `-D MOJOLEARN_STEP_GLUE_DEFAULT_EVERY_COLUMN=1` returns the NVIDIA word on every column, so a no-trial build on a Mac reaches the shipped glue branch; this is how the M4 gates that branch, since Apple's own default carries no glue bit. A check knob, the `MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL` pattern; never a shipped build."""
    comptime if is_defined["MOJOLEARN_STEP_GLUE_DEFAULT_EVERY_COLUMN"]():
        return STEP_GLUE_DEFAULT_WORD_OPTSKIP_NOSHADOW_ROWS16
    if column == COLUMN_NVIDIA:
        # DEVIATION 2649. Measured 2026-09-11 on a RunPod H100 80GB HBM3,
        # commit 030079af, one pod and one heat window, against the same
        # build's `shipped` arm
        # (bench/results/e1g/2026-09-11_215139-nvidia-h100-80gb-hbm3-step-glue):
        #   verdict optskip_noshadow_rows16 FLIP geomean=0.9723
        #   enwik8=0.9744 pilegithub=0.9703 witnesses_equal_shipped=True
        # Lean step 0.2911 / 0.2921 -> 0.2837 / 0.2834 s (enwik8 / Pile
        # GitHub). Attribution on the same box (timers build,
        # timing_witnesses_equal=True), enwik8, ms: fwd.norm1 3.068 -> 1.554,
        # fwd.norm2 3.077 -> 1.562, grad.norm1_kernels 2.026 -> 1.399,
        # grad.norm2_kernels 2.038 -> 1.393, and step.shadow_copy (2.309) and
        # step.opt_refuse_scan (2.175) gone; the optimizer, the scans and the
        # packing are unchanged to within 0.013 ms. The runners-up on the same
        # leg all FLIP too: optskip_noshadow_rows8 0.9734, optskip_noshadow
        # 0.9849, rows16 0.9852, rows8 0.9873. ENGINEERING_RULES 9 takes the
        # winner.
        return STEP_GLUE_DEFAULT_WORD_OPTSKIP_NOSHADOW_ROWS16
    if column == COLUMN_AMD:
        # DEVIATION 2649 ON AMD TOO. Measured 2026-09-12 on a Hot Aisle MI300X
        # (gfx942) against the SAME BUILD's `shipped` arm, commit bb679f19, one
        # VM and one heat window
        # (bench/results/e1g/2026-09-12_133013-amd-mi300x-hotaisle-step-glue):
        #   verdict optskip_noshadow_rows16 FLIP geomean=0.9854
        #   enwik8=0.9854 pilegithub=0.9853 witnesses_equal_shipped=True
        # Lean step 0.7634 / 0.7620 -> 0.7523 / 0.7509 s (enwik8 / Pile
        # GitHub), and against the separate SHIPPED build 0.9848 / 0.9839. The
        # drift control re-ran `shipped` last at 0.9971 of the first `shipped`
        # run, which is smaller than the 1.5 percent gain, so the win is not
        # the VM warming up. `step_glue_check: PASS` on the same box with the
        # reach lines for every arm (threads_per_block=16 first_moved_row=32
        # forward and backward, update first_moved_element=768).
        # The gain is half of NVIDIA's (0.9723 there) and the reason is the
        # same occupancy story read the other way: 2,048 token rows at
        # LLAMA_TPB 128 are 16 blocks, which starve an H100's 132 SMs harder
        # than they starve this board. ENGINEERING_RULES 9 sets no magnitude
        # bar, and both corpora are below 1 with the bits unmoved.
        return STEP_GLUE_DEFAULT_WORD_OPTSKIP_NOSHADOW_ROWS16
    if column == COLUMN_CPU:
        return STEP_GLUE_DEFAULT_WORD_SHIPPED  # the non-vendor default
    return STEP_GLUE_DEFAULT_WORD_SHIPPED


def knn_warpsort_select_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1922): whether the k-NN TILED path's selector is the implemented RAFT WARPSORT (`select_warpsort.mojo`, `warpsort_topk_block_kernel`) instead of the implemented RAFT radix (`select_radix.mojo`) for `2 < k <= 256`."""
    comptime if identical:
        return False
    if column == COLUMN_CPU:
        return False  # FAST row; a host build is IDENTICAL only
    return column == COLUMN_NVIDIA


def knn_auto_follows_their_dispatch_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 1923): whether the k-NN AUTO arm follows cuVS's dispatch UNCONDITIONALLY -- `k <= 64` + row-major + L2 goes to `fusedL2Knn`, x-split included (`knn_brute_force.cuh:443`) -- instead of DEVIATION 36's shape test (fused only when `launchConfigGenerator` picks `grid_x == 1`, tiled when it would engage the x-split)."""
    comptime if identical:
        return False
    if column == COLUMN_CPU:
        return False  # FAST row; a host build is IDENTICAL only
    return column == COLUMN_NVIDIA



comptime QUANTIZE_SEARCH_LINEAR = 0

comptime QUANTIZE_SEARCH_BINARY = 1

comptime QUANTIZE_SEARCH_TWO_LEVEL = 2


def quantize_search_for[column: Int]() -> Int:
    """SCHEDULING row: HOW the evaluator's quantize finds a value's bin."""
    if column == COLUMN_APPLE:
        return QUANTIZE_SEARCH_TWO_LEVEL
    if column == COLUMN_CPU:
        return QUANTIZE_SEARCH_LINEAR  # scheduling; never Apple's two-level
    return QUANTIZE_SEARCH_LINEAR


def _knn_identical_round_column(column: Int) -> Bool:
    """The columns whose IDENTICAL k-NN defaults were flipped 2026-09-09: small-k selector, transposed index layout with the register tile, and index-axis tiling. NVIDIA and AMD flipped on the H100 evidence; Apple flipped the same afternoon on the M4 four-arm price (100k x 32, k 10, 9 rounds, PRICE_MS medians baseline -> both: 20.5 -> 15.1 ms at 32 queries, 26.9 -> 14.9 at 128, 182.2 -> 66.6 at 1000; all four arms byte-equal to the NVIDIA baseline across 143,628 cells). On Apple the transpose-only arm was slightly faster still at 128 and 1000 queries (12.1, 60.2 ms) and slower at 32 (16.3 ms); the both column is the shipped one."""
    return (
        column == COLUMN_NVIDIA
        or column == COLUMN_AMD
        or column == COLUMN_AMD_RDNA
        or column == COLUMN_APPLE
        # the CPU column takes the 2026-09-09 IDENTICAL defaults too, so a
        # host k-NN twin restates what all four GPU columns run
        or column == COLUMN_CPU
    )


def knn_smallk_select_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row: whether the IDENTICAL tiled k-NN arm selects k <= KNN_SMALLK_MAX_K with the per-thread composite-key selector (`neighbors/checks/select_smallk_identical_candidate.mojo`) instead of the 64-bit radix. Both return the k smallest (distance, index) keys ascending, so the bits are equal by construction and the gate is the four-arm dispatch check. `-D MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT=1` forces the radix on every column; `-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1` forces the selector on every column."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL"]():
        return True
    return _knn_identical_round_column(column)


def knn_transposed_index_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row: whether the IDENTICAL tiled k-NN arm transposes the index once per request so the pinned distance tile reads it coalesced. Same per-cell fma chain, so the bits are equal. `-D MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT=1` forces the row-major layout; `-D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1` forces the transpose on every column."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL"]():
        return True
    return _knn_identical_round_column(column)


def knn_distance_register_tile_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row: whether the transposed IDENTICAL distance tile computes a column-selected register tile per thread (`pinned_distance_tile.mojo::pinned_distance_register_tile_kernel`) instead of one cell per thread. Every cell's chain is still one ascending serial fma chain over the feature axis (IDENTITY_PATHS row 24), so the bits are equal. Only reachable when `knn_transposed_index_for` is true. `-D MOJOLEARN_KNN_IDENTICAL_SCALAR_TILE=1` keeps one cell per thread."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_SCALAR_TILE"]():
        return False
    return knn_transposed_index_for[column, identical]()


def knn_selector_specialize_common_for[column: Int, identical: Bool]() -> Bool:
    """Compile-time k=10/15 removes dynamic insertion guards and threshold
    selection. NVIDIA 400k/4000q/k10: selector 20.3 -> 9.1 ms on the L40S
    (2026-09-09 selector resume). The same integer composite-key scan and
    block minimum are retained. Other columns can force the specialization
    for qualification; the generic capacity buckets remain the A/B arm.
    """
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_GENERIC_K"]():
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON"]():
        return True
    if column == COLUMN_CPU:
        return False  # NVIDIA's schedule, never the host's
    return column == COLUMN_NVIDIA


def knn_selector_shuffle_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (2026-09-09, lane/knn-selector): whether the IDENTICAL small-k selector (`select_smallk_identical_candidate.mojo::smallk_bucket_kernel`) takes each rank's block minimum through a lane-group butterfly (`shuffle_xor` over `column_lane_width` lanes, one shared slot per lane group, ONE barrier per rank, double-buffered slots) instead of the eight-level shared-memory tree (eleven barriers per rank). The reduced value is a UInt64 composite key and the fold is an integer minimum, which is associative, commutative and idempotent, so the winner is the same key under any tree and the bits are equal by construction; the gate is the four-arm dispatch check. Every column with a fixed lane width (`column_lane_width_is_fixed`) takes the butterfly; the Qualcomm and Intel columns, whose lane width the vendor's compiler chooses per kernel, keep the tree. `-D MOJOLEARN_KNN_IDENTICAL_TREE_SELECT=1` forces the tree on every column."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_TREE_SELECT"]():
        return False
    if column == COLUMN_CPU:
        return True  # the fixed-width reading, what every GPU column answers
    return column_lane_width_is_fixed(column)


def knn_selector_warpbound_guard_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (2026-09-11, DEVIATION 2523): whether the IDENTICAL small-k selector (`select_smallk_identical_candidate.mojo::smallk_bucket_kernel`) runs its per-lane insertion chain behind a warp-uniform ballot with a warp-scope admission bound (every second batch, each lane publishes its list head, aligned lane groups take their minimum through five xor shuffles, the bound is the maximum over groups, and a lane admits against min(own k-th smallest, bound)) instead of the always-issued predicated chain. The bound is at or above the k-th smallest of a subset of the union of the warp's lists, so at least k union keys are at or below it and no key above it can be in the row's top-k; keys carry their column so no equality; the union still holds the true top-k, the rank phase pops the same UInt64 minima with the same index tie rule, and the bits are equal by construction (the nine-arm dispatch check, 18 planted cases, M4 and H100). Measured on the H100 2026-09-11: the chain issued on 90 to 96 percent of warp-steps under the per-lane threshold and on 46 percent under the bound; full requests at 400k x 4k x d32 went 30.8 to 27.8 ms (k10) and 36.0 to 31.0 ms (k15), every pair in both orders (bench/results/e1g/2026-09-11_023138-nvidia). NVIDIA only until the Apple and AMD columns are timed (RUN OWED; both are fixed-lane-width columns and pass the identity check, so timing is the only gate). Requires the block-uniform trip count (DEVIATION 2497) and a fixed-lane-width column. `-D MOJOLEARN_KNN_IDENTICAL_INSERT_CHAIN=1` restores the predicated chain on every column."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_INSERT_CHAIN"]():
        return False
    comptime if not column_lane_width_is_fixed(column):
        return False
    if column == COLUMN_CPU:
        return False  # NVIDIA's schedule, never the host's
    return column == COLUMN_NVIDIA


def svm_block_solve_warp_folds_for[column: Int, width: Int]() -> Bool:
    """SCHEDULING row (2026-09-11, DEVIATION 2623): whether `svm/impl/smoblocksolve.mojo::smo_block_solve_kernel[width]` folds its three arg-reductions with `block_argext` warp butterflies (DEVIATION 2491) instead of the halving trees with a one-slot thread ballot that preceded it. Both select the same (value, key) element under a total order with unique keys, so no column's bits depend on this row. NVIDIA refuses the warp kernel at width 1024 (H100 80GB HBM3, driver 580.126.09, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, so every SVC fit above 512 training rows failed from 2491 through 0.8.2) and launches it at 512; the tree kernel launches at 1024 there. Fusing two of the warp folds or dropping the WSIZE threadgroup diagonal did not make the warp kernel launch. The Apple M4 and the AMD MI300X launch the warp kernel at 1024. `-D MOJOLEARN_SVM_TREE_FOLDS` takes the tree schedule on every column for an A/B."""
    comptime if is_defined["MOJOLEARN_SVM_TREE_FOLDS"]():
        return False
    if column == COLUMN_CPU:
        return True  # the every-width reading; the oracle spells the fold serially
    return not (column == COLUMN_NVIDIA and width > 512)


comptime SVM_SCHED_TREE = 0
comptime SVM_SCHED_WARP = 1
comptime SVM_SCHED_WARP_LANE0 = 2
comptime SVM_SCHED_FUSED_TREE = 3
comptime SVM_SCHED_RARY_TREE = 4


def svm_block_solve_tree_arity_for[column: Int, width: Int]() -> Int:
    """SCHEDULING row (2026-09-11, DEVIATION 2628): the arity R of SVM_SCHED_RARY_TREE's threadgroup tree (a power of two; levels = ceil(log_R(width))). Selection under a total order, so no bits depend on it. `-D MOJOLEARN_SVM_ARITY_16` / `_64` for an A/B; default 32."""
    comptime if is_defined["MOJOLEARN_SVM_ARITY_16"]():
        return 16
    comptime if is_defined["MOJOLEARN_SVM_ARITY_64"]():
        return 64
    return 32


def svm_block_solve_schedule_for[column: Int, width: Int]() -> Int:
    """SCHEDULING row (2026-09-11, DEVIATIONS 2627 and 2628): which schedule folds the three arg-reductions of `svm/impl/smoblocksolve.mojo::smo_block_solve_kernel[width]`. SVM_SCHED_TREE (0) is the pre-2491 halving trees with a thread ballot for `u` and `l` (42 barriers per inner iteration at width 1024); SVM_SCHED_WARP (1) is DEVIATION 2491's `block_argext` warp butterflies (8); SVM_SCHED_WARP_LANE0 (2) is `block_argext_lane0`, the same butterflies with the cross-warp fold on lane 0 in a runtime loop and a warp broadcast (DEVIATION 2627, 5); SVM_SCHED_FUSED_TREE (3) is one halving tree carrying the argmin, its thread and the argmax together plus a thread-carrying tree for `l`, no ballot (DEVIATION 2628, 26); SVM_SCHED_RARY_TREE (4) carries the same selections on a threadgroup tree of arity `svm_block_solve_tree_arity_for` (DEVIATION 2628's second shape, about 10). All five select the same (value, key) element under a total order with unique keys, so no column's bits depend on this row. Default: `svm_block_solve_warp_folds_for` (DEVIATION 2623) picks WARP or TREE. `-D MOJOLEARN_SVM_SCHED_TREE`, `_WARP`, `_WARP_LANE0`, `_FUSED_TREE` or `_RARY_TREE` forces one schedule on every column for an A/B (the `_RARY_TREE` define was missing from this row at 48f92b19, so that commit's R-ary kernel was unreachable)."""
    comptime if is_defined["MOJOLEARN_SVM_SCHED_TREE"]():
        return SVM_SCHED_TREE
    comptime if is_defined["MOJOLEARN_SVM_SCHED_WARP"]():
        return SVM_SCHED_WARP
    comptime if is_defined["MOJOLEARN_SVM_SCHED_WARP_LANE0"]():
        return SVM_SCHED_WARP_LANE0
    comptime if is_defined["MOJOLEARN_SVM_SCHED_FUSED_TREE"]():
        return SVM_SCHED_FUSED_TREE
    comptime if is_defined["MOJOLEARN_SVM_SCHED_RARY_TREE"]():
        return SVM_SCHED_RARY_TREE
    if svm_block_solve_warp_folds_for[column, width]():
        return SVM_SCHED_WARP
    # DEVIATION 2666 (2026-09-11): the column DEVIATION 2623 sends to the
    # halving trees -- NVIDIA above width 512, where CUDA refuses the warp
    # kernel -- takes the FUSED_TREE schedule instead. Measured on an NVIDIA
    # H200 (RunPod 4oih8bhjepzlmm, driver 570.211.01, ptxas 12.9.86), taxi
    # 10,000 x 11, five fits each: FUSED_TREE 771.1 ms, TREE 866.9 ms,
    # RARY_TREE at arity 16 1,324.4 ms and at 32 1,945.6 ms (1,937.3 ms
    # without its trailing and second update barriers), every arm giving the
    # same fits (n=400/600/2000 457e29b82bca9df9, 733a383c5699f427,
    # 2b66bc991a9c9ed0; taxi b0f91a7958162936) from five different binaries.
    # NVIDIA ONLY: Metal refuses the fused kernel's width-1024 pipeline
    # (threadgroup memory 36872 > 32768, Apple M4 gate 2026-09-11), so this
    # stays a row and never a global default. `-D MOJOLEARN_SVM_TREE_FOLDS`
    # still takes the pre-2491 trees on every column for an A/B.
    comptime if not is_defined["MOJOLEARN_SVM_TREE_FOLDS"]():
        comptime if column == COLUMN_NVIDIA:
            return SVM_SCHED_FUSED_TREE
    return SVM_SCHED_TREE


def umap_device_optimizer_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row (2026-09-09, lane/umap-optimizer): whether the IDENTICAL UMAP layout optimizer runs on the device (`umap/optimizer_identical_device.mojo`: one thread per vertex, one epoch snapshot, each vertex's update a fixed-order fold over its CSR row, negatives from Philox keyed by (seed, epoch, edge, slot), no atomics, no launch-geometry dependence) instead of the serial host loop (`umap/optimizer.mojo::optimize_layout_identical`, `umap/sparse_optimizer.mojo::optimize_sparse_layout_identical`). The two produce DIFFERENT bits (Jacobi versus Gauss-Seidel order); the device path is the IDENTICAL contract on every column and is gated against itself across launch widths and GPUs, not against the host loop. `-D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1` restores the host loop on every column (the pre-2026-09-09 cards). FAST and DETERMINISTIC never enter this row."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER"]():
        return False
    return True


comptime KNN_IDENTICAL_INDEX_TILE = 65536


def knn_index_tile_columns_for[column: Int, identical: Bool]() -> Int:
    """SCHEDULING row: the widest index-axis column tile the IDENTICAL tiled k-NN arm computes per query tile before merging partial top-k lists under the composite total order (`select_smallk_identical_candidate.mojo::partial_topk_merge_kernel`). 0 means the index axis is never split. Merging sorted (distance, index) lists is order-independent, so the bits are equal to the untiled path. `-D MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE=1` restores the untiled path."""
    comptime if not identical:
        return 0
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE"]():
        return 0
    if _knn_identical_round_column(column):
        return KNN_IDENTICAL_INDEX_TILE
    return 0


def gemm_wide_split_for[column: Int]() -> Bool:
    """Execution-only wide split-K tiles on the measured NVIDIA column. The CPU column answers False.

    The 128x128/KS16 tile reduces operand reloads for complete output tiles.
    Other columns keep their previous dispatcher pending local timings;
    every column's all-plan correctness gate still exercises the new tile.
    """
    if column == COLUMN_CPU:
        return False
    return column == COLUMN_NVIDIA


def knn_distance_zero_fma_repair_for[column: Int, identical: Bool]() -> Bool:
    """Repair Apple's pre-round FMA underflow only at the kNN register seam.

    The integer slow path runs only for a zero result and restores a rounded
    smallest-normal result when required. NVIDIA already uses round-then-FTZ.
    The exact integer oracle checks 396584 actual/simulated-underflow triples.
    The disable flag retains an explicit Apple before/after performance arm.
    """
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_ZERO_FMA_REPAIR"]():
        return False
    if column == COLUMN_CPU:
        return False  # a CPU FMA rounds once and does not pre-round underflow
    return identical and column == COLUMN_APPLE


@always_inline
def knn_distance_preflight_for[column: Int, identical: Bool]() -> Bool:
    """Exact whole-chain exponent admission avoids unnecessary Apple repairs."""
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_PREFLIGHT"]():
        return False
    if column == COLUMN_CPU:
        return False  # Apple's repair admission; nothing to repair on a CPU
    return identical and column == COLUMN_APPLE


@always_inline
def knn_distance_metadata_for[column: Int, identical: Bool]() -> Bool:
    """Force request-local exponent minima outside the measured default scope."""
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_PREFLIGHT_METADATA"]():
        return knn_distance_preflight_for[column, identical]()
    return False


@always_inline
def knn_distance_metadata_default_for[column: Int, identical: Bool]() -> Bool:
    """Apple capability; runtime dispatch additionally requires measured shapes."""
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_METADATA"]():
        return False
    return knn_distance_preflight_for[column, identical]()


@always_inline
def knn_distance_hardware_flush_for[column: Int, identical: Bool]() -> Bool:
    """Fully rounded NVIDIA FMA followed by exact hardware FTZ multiplication."""
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_SOFTWARE_FLUSH"]():
        return False
    if column == COLUMN_CPU:
        return False  # the software flush, NVIDIA's instruction is not here
    return identical and column == COLUMN_NVIDIA


@always_inline
def knn_distance_rows_for[column: Int, identical: Bool]() -> Int:
    """Measured NVIDIA eight-query register tile; each cell keeps its FMA chain.

    The 32/128-feature cases improve; the 8-feature coverage is flat to 0.9%
    slower. Other columns retain four query rows per thread.
    """
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_ROWS4"]():
        return 4
    if column == COLUMN_CPU:
        return 4  # scheduling; NVIDIA's eight-row tile is NVIDIA's
    return 8 if identical and column == COLUMN_NVIDIA else 4


@always_inline
def knn_distance_exact_chain_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 2629, 2026-09-11, lane/knn-speed): whether the transposed IDENTICAL register-tile distance admits a tile to the UNFLUSHED chain (`pinned_distance_tile.mojo::_rt_step_exact`) when request-local exponent metadata proves every product and every partial sum of every cell in the tile is zero or at least the smallest normal (minimum biased exponent sum >= 174, maximum sum plus ceil(log2 d) <= 376, no nonfinite input). Under that proof the per-step flush never sees a subnormal, so dropping it moves no bit; a tile that fails admission keeps the flushed chain. Kernel code is the same on every column; this row only decides where the admission runs. MEASURED NEUTRAL on the H100 2026-09-11 (pod 62dlwtf4amlt2s, 400k x 4k x d32, three interleaved before/after pairs of 7 rounds): request k10 23.65 -> 23.85 ms, k15 26.24 -> 26.19 ms, distance class 15.31 -> 15.41 ms; full distance and index dumps equal to origin/main, sabotage flips them. The flush is not what the distance class pays for, so the row is OFF on every column and stays opt-in: `-D MOJOLEARN_EXPERIMENTAL_KNN_EXACT_CHAIN=1` admits on every column; `-D MOJOLEARN_KNN_IDENTICAL_FLUSHED_CHAIN=1` forces the flushed chain."""
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_FLUSHED_CHAIN"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_EXACT_CHAIN"]():
        return identical
    # 2026-09-17 (lane/knn-tiled-distance, DEVIATION 3000): ON where the
    # shared-memory tile runs, which admits per BLOCK from the same
    # request-local metadata. There the flush IS what the distance class
    # pays for: on the RTX 4090 at 400k x 4k x d220 the smem tile's distance
    # class went 37.9 to 31.4 ms (k10) and the request 60.9 to 56.5 ms
    # (paired medians 0.754 to 0.695 of base), every bit equal on the cuda
    # column against the cpu host route and the sabotage arm DIVERGENT
    # (bench/results/knn_tiled_2026-09-17/). The register tile keeps the
    # 2026-09-11 reading (neutral, off).
    return knn_smem_distance_tile_for[column, identical]()


#: DEVIATION 2631's measured tile, FLIPPED 2026-09-11 on the H200 (pod
#: `zwmta1li2twxx2`): synthetic 400k x 4k x d32 request 23.53 -> 21.38 ms at
#: k10 (0.909) and 25.98 -> 23.52 at k15 (0.905) against the 512 default,
#: taxi 22.14 -> 19.92 ms (0.900) and Istella-S 100.29 -> 95.51 (0.952) in
#: interleaved races, geomean 0.926, recall@10 unchanged on both datasets and
#: every output bit equal. At 4,000 queries `plan_query_tile`'s query clamp
#: lowers it to 4,000, so the benchmark shape runs as ONE query tile.
comptime KNN_IDENTICAL_WIDE_QUERY_TILE = 4096


@always_inline
def knn_query_tile_for[column: Int, identical: Bool]() -> Int:
    """SCHEDULING row (DEVIATION 2631, 2026-09-11, lane/knn-finish): the default query tile of the IDENTICAL tiled k-NN arm (`neighbors/estimator.mojo::DEFAULT_QUERY_TILE`), and on the same column the radix scratch the small-k selector never reads is not allocated (`knn_brute_force.mojo::tiled_radix_scratch_len`). 0 means the estimator's historical rule (512 on NVIDIA IDENTICAL, 256 elsewhere). Tiling cannot move a bit: every cell's chain is a function of its query row and index column alone, every row's selection and partial merges run in the same column-tile order whatever the query tile, and the scratch is never read on a k <= 64 request. Arms for the A/B: `-D MOJOLEARN_KNN_QUERY_TILE_ARM_512`, `_1024`, `_2048`, `_4096` force that tile on every column; `-D MOJOLEARN_KNN_IDENTICAL_FULL_RADIX_SCRATCH=1` keeps the historical scratch."""
    comptime if not identical:
        return 0
    comptime if is_defined["MOJOLEARN_KNN_QUERY_TILE_ARM_512"]():
        return 512
    comptime if is_defined["MOJOLEARN_KNN_QUERY_TILE_ARM_1024"]():
        return 1024
    comptime if is_defined["MOJOLEARN_KNN_QUERY_TILE_ARM_2048"]():
        return 2048
    comptime if is_defined["MOJOLEARN_KNN_QUERY_TILE_ARM_4096"]():
        return 4096
    if column == COLUMN_CPU:
        return 0  # scheduling; the measured tile is NVIDIA's
    return KNN_IDENTICAL_WIDE_QUERY_TILE if column == COLUMN_NVIDIA else 0


@always_inline
def knn_radix_scratch_shrink_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 2631): whether a k <= 64 IDENTICAL tiled request allocates `k` radix scratch pairs per query row instead of `n_index // 8`. The small-k selector (`knn_smallk_select_for`) serves every column tile of such a request, so the radix kernel that reads the scratch is never launched and no output can depend on its size. Same column scope as `knn_query_tile_for`'s measured tile. `-D MOJOLEARN_KNN_IDENTICAL_FULL_RADIX_SCRATCH=1` keeps the historical size everywhere."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_FULL_RADIX_SCRATCH"]():
        return False
    if column == COLUMN_CPU:
        return False  # scheduling; NVIDIA's
    return column == COLUMN_NVIDIA


@always_inline
def knn_fused_distance_select_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 2667, 2026-09-11, lane/knn-finish): whether the transposed IDENTICAL tiled k-NN arm computes each column tile's distances INSIDE the small-k selector (`neighbors/checks/fused_distance_select_identical.mojo`), one launch per tile, instead of a register-tile distance launch that writes a `query_tile x index_tile` matrix and a selector launch that reads it back. The shape of cuVS's `fusedL2Knn` (`knn_brute_force.cuh:447-451`): no distance matrix is written. Each candidate's distance is the register tile's cell chain (`pinned_distance_tile.mojo::_rt_step` over the feature axis ascending, then the same epilogue, clamp and root), its key is the same composite (distance, index) key, and the selector's rank phase returns the k smallest keys of the whole tile, which do not depend on which thread saw which column, so neighbors and distances are the same bits. `-D MOJOLEARN_EXPERIMENTAL_KNN_FUSED_SELECT=1` forces it on every column; `-D MOJOLEARN_KNN_IDENTICAL_UNFUSED_SELECT=1` forces the two-launch form."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_UNFUSED_SELECT"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_FUSED_SELECT"]():
        return True
    return False


def forest_row_threads_for[column: Int]() -> Bool:
    """SCHEDULING row (DEVIATION 2964, 2026-09-17, lane/forest-groves-row-schedule): whether the `parallel_groves` forest inference kernels give one thread a whole row (its 32 lane sums and the 16/8/4/2/1 fold, the same graph) instead of 32 threads with a shared-memory fold. The launch site applies it only when a row is at most FOREST_ROW_THREADS_MAX_FEATURES floats (`core/forest_inference.mojo`); wider rows keep the 32-thread kernels."""
    comptime if is_defined["MOJOLEARN_FOREST_ROW_THREADS_OFF"]():
        return False
    comptime if is_defined["MOJOLEARN_FOREST_ROW_THREADS"]():
        return True  # every column, for an A/B
    if column == COLUMN_CPU:
        return False  # the host groves engine has its own schedule
    # FLIPPED ON NVIDIA 2026-09-18 for rows of at most 32 floats (RTX 4090,
    # bench/results/forest_groves_row_2026-09-18/): eight-call blocks, ms per
    # call, hashes equal to the 32-thread kernels' in every cell and the
    # sabotage arm (lane-order fold) divergent in every cell: taxi RF
    # classifier 4.6M rows x 16 190.7 to 125.7 (1.52x), taxi RF regressor
    # 5.75M x 16 239.0 to 157.4 (1.52x), HIGGS RF 500k x 28 24.0 to 19.9
    # (1.20x), HIGGS ET 66.5 to 68.0 (even, one side u). ABOVE 32 floats it
    # LOSES and stays off: Covtype 54 columns 27.0 to 32.0 (0.84x), Year 90
    # columns 56.5 to 63.8 (0.89x), Istella-S 220 columns 275.2 to 619.4
    # (0.44x) and its regressor 257.5 to 545.0 (0.47x): adjacent threads read
    # 32 different wide rows and the feature reads stop coalescing.
    # FLIPPED ON APPLE 2026-09-19 for the same <=32-feature bound (M4 Metal,
    # direct resident-kernel ABBA, all output bits equal): at 28 features the
    # row schedule improved scalar leaves by 1.28x (32 trees, depth8), 1.98x
    # (100 trees, depth10), and 2.10x (300 trees, depth10); vector leaves by
    # 1.53x at 2 outputs and 1.49x at 8 outputs (100 trees, depth10). The
    # IDENTICAL scalar check improved 1.72x. AMD remains untimed.
    return column == COLUMN_NVIDIA or column == COLUMN_APPLE


def knn_smem_distance_tile_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 3000, 2026-09-17, lane/knn-tiled-distance): whether the transposed IDENTICAL tiled k-NN arm computes each column tile's distances through the SHARED-MEMORY tile (`neighbors/checks/smem_distance_tile.mojo::smem_distance_tile_kernel`): a block of 256 threads owns 64 query rows x 128 index columns, stages each 16-feature slice of the query rows and of the transposed index columns into shared memory once (flushed at the store), and every thread advances its 8 x 4 accumulators from three 16-byte shared loads per feature step, instead of the register tile's twelve global loads and twelve flushes per step. Every cell is still one ascending `_rt_step` chain over the feature axis from +0.0 with the register tile's epilogue, clamp and root, and `ftz` is idempotent, so the bits are the register tile's; the gate is `tools/identity_break.py` on the knn lanes, cuda against the cpu host route, plus `-D MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1`. Needs the transposed layout and the register-tile row, and does not carry the Apple metadata or DEVIATION 2629 chains (a request on those keeps the register tile). ON on NVIDIA (measured, the comment in the body), OFF elsewhere; `-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1` forces it on any column, `-D MOJOLEARN_KNN_IDENTICAL_REGISTER_TILE_ONLY=1` forces the register tile."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_REGISTER_TILE_ONLY"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE"]():
        return True
    if column == COLUMN_CPU:
        return False  # scheduling; the measured schedule is NVIDIA's
    # FLIPPED ON NVIDIA 2026-09-17 (RTX 4090, bench/results/knn_tiled_2026-09-17/):
    # the distance class at 400k x 4k, 4,000 queries, 7 column tiles went
    # 59.4 to 37.9 ms on Istella-S (d 220, k 10) and 14.6 to 13.7 ms on taxi
    # (d 11); kneighbors paired medians 0.754 (Istella-S) and 0.938 (taxi)
    # of base. Every knn, radius and kde lane IDENTICAL on the cuda column
    # against the cpu host route (five fixtures, two repeats), the sabotage
    # arm DIVERGENT on every knn infer and batch cell. Apple and AMD are
    # untimed and keep the register tile.
    return column == COLUMN_NVIDIA


def knn_block_topk_select_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 3001, 2026-09-17, lane/knn-tiled-distance): whether the shared-memory tile (DEVIATION 3000, required) writes NO distance matrix and instead pops each of its rows' k smallest composite keys inside the block (`smem_distance_tile_kernel[TOPK=True]`, warp minima over the block's 128 columns), with `partial_keys_select_kernel` selecting the row's k smallest from the union of the per-block lists. The key is the small-k selector's `composite_key(distance, tile-local column)`, keys are unique, the row's k smallest keys are a subset of the union whatever the partition, and the rank phase pops the same UInt64 minima ascending, so neighbors and distances are the same bits (the distance is `twiddle_out` of the key's high half, the exact inverse of the key's `twiddle_in`); the gate is the same identity run as DEVIATION 3000. Replaces the `query_tile x index_tile` matrix write and the selector's read of it by a `query_tile x (index_tile / 128) x k` key buffer. Needs the small-k selector row and 1 <= k <= 64; the fused select (DEVIATION 2667) and selector trial builds keep the two-launch form. ON on NVIDIA for k <= KNN_BLOCK_TOPK_MAX_K (measured, the comment in the body), OFF elsewhere; `-D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1` forces it on any column, `-D MOJOLEARN_KNN_IDENTICAL_MATRIX_SELECT=1` keeps the matrix and the selector."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_MATRIX_SELECT"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK"]():
        return knn_smem_distance_tile_for[column, identical]()
    if column == COLUMN_CPU:
        return False  # scheduling; the measured schedule is NVIDIA's
    # FLIPPED ON NVIDIA 2026-09-17 for k <= KNN_BLOCK_TOPK_MAX_K (RTX 4090,
    # bench/results/knn_tiled_2026-09-17/): with the tile's rank loop paid
    # inside the distance class and the 1 GiB matrix never written, 4,000
    # queries against 400,000 rows went (paired medians of base) 0.649 on
    # Istella-S and 0.764 on taxi at k 10 against the tile alone at 0.695
    # and 0.982, and 42.5 against 55.1 ms (Istella-S) and 10.7 against 27.8
    # ms (taxi) at k 1. At k 64 the in-block rank loop costs more than the
    # selector it replaces (147 against 127 ms Istella-S, 116 against 97 ms
    # taxi), so above the bound the matrix and the small-k selector stay.
    return column == COLUMN_NVIDIA and knn_smem_distance_tile_for[column, identical]()


#: The widest k the block top-k (DEVIATION 3001) serves by default; larger
#: k keeps the matrix and the small-k selector (measured above).
#: `-D MOJOLEARN_KNN_BLOCK_TOPK_ALL_K=1` lifts it to the selector's 64.
comptime KNN_BLOCK_TOPK_MAX_K = 64 if is_defined["MOJOLEARN_KNN_BLOCK_TOPK_ALL_K"]() else 16


@always_inline
def knn_selector_bound_compact_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 3060, 2026-09-17, lane/knn-selector-speed): whether a column tile's top-k for KNN_SELECTOR_BOUND_MIN_K <= k <= 64 is taken by the bound-and-compact selector (`neighbors/checks/knn_selector_bound_compact.mojo`) instead of the small-k selector's per-thread k-deep lists and k rank rounds. Every thread keeps its C smallest composite keys (C = 8), the rank k - 1 of the 256 thread minima is a bound with at least k keys at or below it, every key at or below the bound is compacted into shared memory and ranked by counting; a row with a thread whose list is full below the bound, or with more than 256 candidates, is flagged and served by the UNCHANGED small-k kernel in a second launch that returns at once on every other row. Same composite key, unique keys, integer counts and compares only, the distance copied from the same tile cell, so neighbors and distances are the same bits; the gate is `neighbors/checks/knn_selector_bound_compact_check.mojo` (host oracle and the small-k selector, both paths reached), the 400,000 x 4,000 digests at k 32 and 64 against the small-k selector, `tools/identity_break.py` on the knn lanes, and `-D MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE=1`. Needs the small-k selector row. `-D MOJOLEARN_EXPERIMENTAL_KNN_SELECTOR_BOUND=1` forces it on any column, `-D MOJOLEARN_KNN_IDENTICAL_LIST_SELECT=1` keeps the small-k selector."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_LIST_SELECT"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_SELECTOR_BOUND"]():
        return knn_smallk_select_for[column, identical]()
    if column == COLUMN_CPU:
        return False  # scheduling; the measured schedule is NVIDIA's
    # FLIPPED ON NVIDIA 2026-09-17 for k >= KNN_SELECTOR_BOUND_MIN_K (RTX
    # 4090, pod 0btpza2l3g1plm, bench/results/knn_selector_2026-09-17/): the
    # selection class of 4,000 queries against 400,000 rows went 68.0 to 8.1
    # ms (Istella-S) and 69.5 to 8.2 ms (taxi) at k 64 and 18.8 to 8.0 and
    # 28.6 to 8.1 ms at k 32, with 0 of 28,000 row tiles flagged; the
    # interleaved race (5 rounds x 3 calls, digests equal across arms) read
    # 130.3 to 69.6 ms and 98.8 to 42.2 ms per call at k 64. Below 17 the
    # block top-k (DEVIATION 3001) serves the L2 metrics and this selector
    # is level with the small-k selector (7.7 to 8.1 against 7.1 to 13.6 ms),
    # so the bound stays at 17. Apple and AMD are owed at the next release.
    return column == COLUMN_NVIDIA and knn_smallk_select_for[column, identical]()


#: The smallest k the bound-and-compact selector (DEVIATION 3060) serves;
#: below it the small-k selector stays. `-D MOJOLEARN_KNN_SELECTOR_BOUND_ALL_K=1`
#: lowers it to 1 for an A/B.
comptime KNN_SELECTOR_BOUND_MIN_K = 1 if is_defined["MOJOLEARN_KNN_SELECTOR_BOUND_ALL_K"]() else 17


@always_inline
def knn_block_topk_bounded_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 3062, 2026-09-17, lane/knn-selector-speed): whether the block top-k (DEVIATION 3001, required) BOUNDS its rank loop on every column tile after a query tile's first: the tile kernel reads each row's k-th RUNNING distance (the merge of the earlier column tiles) and a thread row leaves the rank loop as soon as none of its rows has a remaining key whose distance half is below that, writing a sentinel terminator; `bound_compact_lists_launch` (`neighbors/checks/knn_selector_bound_compact.mojo`) then selects from the sentinel-terminated lists with DEVIATION 3060's bound-and-compact phases, and `partial_topk_merge_kernel` skips the ABSENT slots of a tile that offered fewer than k keys. Column tiles are taken in ascending column order, so a key at or above the k-th running distance has k smaller composite keys among the running entries alone: the merge would drop it, and dropping it earlier leaves the merged list the same bits. With the rank loop no longer k rounds deep on most blocks the block top-k serves every 1 <= k <= 64 (no distance matrix at any k). The first column tile is unbounded and may be narrower (`KNN_BOUNDED_FIRST_TILE`). The gate is the 400,000 x 4,000 digests against the matrix arm at every k, `tools/identity_break.py` on the knn lanes with a first tile narrow enough that their 4,096-row fixtures take bounded tiles, and `-D MOJOLEARN_KNN_BOUNDED_TOPK_SABOTAGE=1`. `-D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK_BOUNDED=1` forces it wherever the block top-k is on, `-D MOJOLEARN_KNN_IDENTICAL_UNBOUNDED_TOPK=1` keeps the full rank loop."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_UNBOUNDED_TOPK"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK_BOUNDED"]():
        return knn_block_topk_select_for[column, identical]()
    # MEASURED AND LOST on NVIDIA 2026-09-17 (RTX 4090, pod 0btpza2l3g1plm,
    # ~/mojolearn-evidence/knn-selector-speed/final_pull_pause/kss_out/bounded*):
    # every digest equal to the matrix arm's at k 1, 10, 17, 32 and 64 on
    # both blocks, but the Istella-S distance class went 42.4 to 49 ms at
    # k 10 and the k 64 call read 88 against 65 ms (taxi 64 against 44);
    # the bound and the early leave each alone cost the tile kernel 40
    # percent (137 to 164 registers against 128, one block of 256 per SM
    # instead of two). OFF on every column; docs/lanes/LANE_STATUS_knn-selector-speed.md.
    return False


#: The width of a query tile's FIRST (unbounded) column tile under DEVIATION
#: 3062; 0 means the index tile's own width. The defines are A/B and reach
#: arms (1024 makes the 4,096-row identity fixtures take bounded tiles).
comptime KNN_BOUNDED_FIRST_TILE = 1024 if is_defined["MOJOLEARN_KNN_BOUNDED_FIRST_TILE_1024"]() else (
    8192 if is_defined["MOJOLEARN_KNN_BOUNDED_FIRST_TILE_8192"]() else (
        16384 if is_defined["MOJOLEARN_KNN_BOUNDED_FIRST_TILE_16384"]() else 0
    )
)


@always_inline
def knn_block_topk_key32_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 3063, 2026-09-17, lane/knn-selector-speed): whether the block top-k's rank loop (`smem_distance_tile_kernel[TOPK=True]`, DEVIATION 3001) keeps each thread's 32 cells as 32-bit DISTANCE HALVES and one 32-bit mask of empty slots instead of 32 UInt64 composite keys. A key's low half is its tile-local column, which the slot names, so the written key is rebuilt from the half and the slot; a lane offers the smallest half among its live slots, the row's minimum is the same 32-bit warp minimum, the lowest lane holding it wins (column order) and pops its first live slot with that half (column order), which is the composite order; a lane with no live slot never enters the ballot, so a real half of 0xFFFFFFFF is still popped before the row is called empty. Same keys in the same ascending order into the same slots, so the bits are DEVIATION 3001's; the gate is the identity run of that deviation plus `-D MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1`. The point is register state: the 64-bit form holds about a hundred registers through the rank loop and any added state (DEVIATION 3062's bound or its early leave, each alone) cost the whole kernel 40 percent on the RTX 4090. `-D MOJOLEARN_EXPERIMENTAL_KNN_TOPK_KEY32=1` forces it on any column, `-D MOJOLEARN_KNN_IDENTICAL_TOPK_KEY64=1` keeps the UInt64 keys."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_TOPK_KEY64"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_TOPK_KEY32"]():
        return True
    # MEASURED AND LOST on NVIDIA 2026-09-17 (RTX 4090, pod 0btpza2l3g1plm,
    # ~/mojolearn-evidence/knn-selector-speed/final_pull_pause/kss_out/key32):
    # digests equal at every k, 118 registers against 128, and the
    # Istella-S distance class still read 42.3 against 32.8 ms at k 1 and
    # 49.2 against 42.4 at k 10 (the live-slot mask tests cost more than
    # the registers bought); with DEVIATION 3062 it split, taxi k 10 19.2
    # against 24.9 ms and Istella-S k 1 46.0 against 35.3. OFF on every
    # column; docs/lanes/LANE_STATUS_knn-selector-speed.md.
    return False


@always_inline
def knn_resident_derived_cache_for[column: Int, identical: Bool]() -> Bool:
    """SCHEDULING row (DEVIATION 3061, 2026-09-17, lane/knn-selector-speed): whether a RESIDENT k-NN index (`neighbors/resident_index.mojo`, DEVIATIONs 2921 and 3002) keeps, beside the uploaded index bytes, what every search derives from those bytes alone: the transposed layout, the index row norms of the metric, and DEVIATION 2629's per-row admission metadata. They are built on the first search that needs them by the SAME kernels over the SAME device bytes (`transpose_kernel`, `compute_norms_for_metric`, `vector_exponent_admission_kernel`) and read by every later search instead of being rebuilt; the device copy of a resident index is never written after its upload and a refit releases the handle, so a later search reads the values it would have computed and no output bit can move. Costs device memory for the life of the handle (the transposed layout is a second copy of the index) in place of a per-call allocation of the same size. The gate is `tools/identity_break.py` on the knn lanes (their infer and batch parts search a fitted index again) plus `-D MOJOLEARN_KNN_RESIDENT_CACHE_SABOTAGE=1`. `-D MOJOLEARN_EXPERIMENTAL_KNN_RESIDENT_CACHE=1` forces it on any column, `-D MOJOLEARN_KNN_IDENTICAL_NO_RESIDENT_CACHE=1` forces the per-call rebuild."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_KNN_IDENTICAL_NO_RESIDENT_CACHE"]():
        return False
    comptime if is_defined["MOJOLEARN_EXPERIMENTAL_KNN_RESIDENT_CACHE"]():
        return True
    if column == COLUMN_CPU:
        return False  # the host route has no device copy to derive from
    # FLIPPED ON NVIDIA 2026-09-17 (RTX 4090, pod 0btpza2l3g1plm,
    # bench/results/knn_selector_2026-09-17/): the per-call allocation of the
    # transposed layout (352 MB at 400,000 x 220), its transposition, the
    # index norms and the index admission scan were 6.5 of an 8.2 ms
    # one-query Istella-S call; the interleaved race read 8.20 to 1.64 ms at
    # one query and 53.7 to 47.7 ms at 4,000 queries (k 10), taxi 1.77 to
    # 1.13 and 26.9 to 24.3 ms, every digest equal. Apple and AMD are owed
    # at the next release.
    return column == COLUMN_NVIDIA


@always_inline
def umap_device_optimizer_live_row_for[column: Int, identical: Bool]() -> Bool:
    """ROUTING row (DEVIATION 2668, 2026-09-11, lane/knn-finish): whether the IDENTICAL UMAP device optimizer (`umap/optimizer_identical_device.mojo::umap_identical_epoch_kernel`) applies a vertex's own attractive and repulsive moves to its running position during its fold (cuML's per-vertex serial kernel, `optimize_batch_kernel.cuh:569-577, 608-616`) instead of summing every move from the epoch snapshot. The mirror edge's tail move stays deferred. MOVES UMAP BITS on every column (the IDENTICAL contract is one default for all columns); both forms are pure functions of the epoch snapshot with one writer per vertex, so each is independent of launch width and vendor, and the 2668 fold's 20,000-row fingerprint is one value (4040033352384472344) across launch widths 64, 128 and 256 with both UMAP identity checks passing. OFF BY DEFAULT: measured on the H200 2026-09-11 it SPLITS on the two datasets, which ENGINEERING_RULES section 9 gates per dataset. Sampled trustworthiness and 10-neighbor retention at 100,000 rows, 200 epochs: taxi 0.9062 / 0.3736 to 0.9323 / 0.3627 (trust up, retention down) and Istella-S 0.9737 / 0.4832 to 0.9636 / 0.4264 (both down), with the time flat on both (1.003 and 1.000). Quality worse on a dataset is a regression a user on that data sees, so the row stays opt-in through `-D MOJOLEARN_UMAP_IDENTICAL_LIVE_ROW=1`; `-D MOJOLEARN_UMAP_IDENTICAL_SNAPSHOT_FOLD=1` forces the snapshot fold even then. What the measurement DID establish is the cause of the cuML gap on taxi (the update order: our own serial host loop scores 0.9796 on the same graph and init against cuML's 0.9657), and that on Istella-S the shipped fold already beats cuML by a wide margin."""
    comptime if not identical:
        return False
    comptime if is_defined["MOJOLEARN_UMAP_IDENTICAL_SNAPSHOT_FOLD"]():
        return False
    comptime if is_defined["MOJOLEARN_UMAP_IDENTICAL_LIVE_ROW"]():
        return True
    return False


@always_inline
def lib_int8_matrix_unit_for[column: Int]() -> Bool:
    """Capability row (DEVIATION 2910, 2026-09-17, lane/int8-mma; contract clause L-9 of gemm/IDENTICAL_LOWBIT_CONTRACT.md): whether the column has an INTEGER matrix unit that takes int8 operands and accumulates in Int32, so `mojolearn.identical.gemm.int8i32.v1` may run its product on it (`gemm/checks/gemm_int8_mma.mojo`) instead of the one-thread-per-cell flat kernel. NVIDIA True: IMMA, `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, sm_80 and later. AMD True: CDNA MFMA, `v_mfma_i32_16x16x32_i8` on gfx942 and gfx950 (the k16 form of CDNA2 is not wired; a gfx90a build is OWED its own intrinsic). Apple False: Metal's simdgroup matrix takes half and float operands only, so the flat kernel serves it. CPU False: no kernel is launched on it and the host oracle is the answer. A True here CANNOT move a bit: an int8 product is exact and an Int32 sum of exact integers is order-free, so the unit's tile shape and internal summation are scheduling; the dequantization seam stays `dequant_int8_pinned` on either path, and `check_int8_mma_matches_flat` requires the two paths' bits to match on every shape. `-D MOJOLEARN_INT8_FORCE_FLAT=1` keeps the flat kernel everywhere without changing this row (the equality gate on a box that has the unit). RDNA, Qualcomm, Intel and the two graph columns answer False: none has been wired or measured."""
    if column == COLUMN_NVIDIA:
        return True
    if column == COLUMN_AMD:
        return True
    if column == COLUMN_CPU:
        return False  # the host runs no kernel; the oracle is the answer
    return False
