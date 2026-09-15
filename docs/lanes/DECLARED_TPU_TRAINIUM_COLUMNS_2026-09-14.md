# Declared TPU and Trainium columns (2026-09-14)

Andrew asked whether the identity contract, which already declares Qualcomm,
Intel and a portable spec baseline as columns nothing can build for yet,
should be future-proofed for AWS Trainium and Google's TPU the same way. This
lane does that on paper. It declares two columns, records what each vendor's
own documentation lets a kernel name, and refuses both by name where a
primitive the contract stands on is missing. Nothing was rented and no kernel
ran on either machine.

## What changed

- `checks/kernel_matrix.mojo` gains `COLUMN_TPU = 9` and `COLUMN_TRAINIUM = 10`
  (`COLUMN_COUNT = 11`). Neither has a `-D MOJOLEARN_COLUMN_*` define in
  `TARGET_COLUMN`, so no build compiles against them and no row they answer
  can reach a kernel.
- Four CONTRACT PRIMITIVE rows, answered for every column
  (`column_fma_instruction`, `column_float32_division`, `column_int32_exact`,
  `column_float_bits_readable`), plus `column_kernel_language`. Each answers
  whether a kernel on the column can NAME the primitive (`CAP_PRESENT`,
  `CAP_ABSENT`, or `CAP_UNAUDITED`). Rounding stays a measurement
  (`check-ieee-arith` with its built-to-separate FMA arm, `check-division`).
- `column_arithmetic_refusal_reason` names every documented ABSENT primitive.
  `column_meets_identity_floor` and `identity_refusal_reason` read it first.
  Only ABSENT refuses. UNAUDITED is a measurement owed and never a verdict.
  On 2026-09-14 Qualcomm, Intel and the spec baseline were UNAUDITED and
  resolved exactly as before; the 2026-09-15 audit below fills their rows.
- `column_has_threadgroup_int_atomics` answers False for both new columns
  (neither vendor documents an atomic add). It answered True for every column
  before, and still does for all nine older ones.
- `checks/hardware_matrix.mojo` answers the machine rows for both columns,
  transcribed where a vendor page states the number and labelled PLACEHOLDER
  where it does not.
- `checks/hardware_matrix_check.mojo` section 6 pins the primitive rows,
  pins that every buildable column names all four, pins that the unaudited
  declared columns are not refused on arithmetic, and pins that both new
  columns are refused with the arithmetic reason.
- `matrix_main.mojo` prints a primitives line under each column.

## Why the contract has primitives at all

IDENTICAL builds every float it produces from binary32 add, subtract and
multiply, one fused multiply-add (`numerics.identical_mul_add`, IDENTITY_PATHS
row 9, also the residual inside every portable transcendental and
`portable_sqrtf`), one hardware division (`portable_divf`, row 49), exact 32-bit
integer arithmetic (fixed-point accumulators, Philox), and reading a float's
bits (`ftz` flushes by bits, row 10). Square root, rsqrt, exp, log and the trig
functions are not primitives, because each is built from that list. A column
whose vendor lets a kernel name all of these can in principle run the
contract. A column that cannot name one of them cannot, until either the
vendor adds it or an exact-integer construction of it is written and gated.

## The audit

| primitive | tpu (Pallas) | trainium (NKI) |
|---|---|---|
| fused multiply-add | ABSENT | ABSENT |
| binary32 division | PRESENT as a name, rounding unmeasured | ABSENT (only `reciprocal`) |
| exact int32 | PRESENT (wraparound undocumented, integer reductions unsupported) | PRESENT (GpSimd "exact results for all 32-bit integer values") |
| float bits readable | UNAUDITED | UNAUDITED |
| atomic add | none documented (grid runs sequentially) | none documented |
| lanes | 128 (8x128 vector registers, TPU v6) | 128 (partition axis at most 128) |

Verdict. Both columns are refused on arithmetic. TPU lacks a fused multiply-add;
Trainium lacks a fused multiply-add and a division.

### Correction to the conversation that opened this lane

The first answers in that conversation described both machines as graph-only,
reachable only through StableHLO. That was wrong. Both vendors now document
kernel languages with explicit placement. Google's Pallas
(`jax.experimental.pallas.tpu`) exposes HBM, VMEM and SMEM memory spaces, a
grid of blocks run in lexicographic order, and 8x128 vector registers. AWS's
Neuron Kernel Interface (`nki.isa`) exposes a Tensor, Vector, Scalar and GpSimd
engine per NeuronCore, SBUF on-chip scratchpad and PSUM, and "keeps the
execution order and memory allocation that developers specify". Both look far
more like our GPU columns than like a graph compiler. The blocker is the
arithmetic they document, not the programming model. Mojo emits code for
neither.

### Hazards recorded while reading, not rows

- Pallas TPU matrix multiplication rounds 32-bit operands to bfloat16 "unless
  float32 precision is requested", the same trap as NVIDIA's TF32 product
  (IDENTITY_PATHS row 33).
- NKI Scalar Engine activations (sigmoid, gelu, exp, log, sqrt, rsqrt,
  reciprocal, sin and more) are "approximated with piece-wise polynomials".
  Outside the stated input ranges they return invalid results, and `sin(9.0)`
  returns about 27.65.
- NKI arithmetic instructions cast every input to float32 and compute "in
  float32 math". `tfloat32` (10-bit mantissa) is a separate data type.
- StableHLO leaves `reduce` order, `scatter` order and integer overflow
  implementation-defined, and defines add, multiply, divide and sqrt as the
  IEEE-754 operations (spec at commit 5a1e6d92).
- The NKI Python simulator "reproduces this hardware behavior for sin only", so
  a simulator run is not evidence about hardware rounding.

## What would flip a row

1. A vendor documents a fused multiply-add instruction (or a Pallas or NKI
   lowering of one) and `check-ieee-arith`'s separating arm, written in that
   kernel language, measures it fused on every separating pattern.
2. An exact-integer construction of fma (and, for Trainium, of division) from
   32-bit integer instructions and bit reinterpretation, gated against the host
   fma on 2^20 patterns including every subnormal class. Possible in principle
   on both machines, since both document exact int32. Unpriced, and almost
   certainly slow.
3. The float-bits rows need one reading of `bitcast_convert` lowering under
   Pallas and of float-to-integer tile reinterpretation under NKI.
4. The floor's atomics clause is written for concurrent blocks. A Pallas TPU
   grid is sequential, so an ordinary integer store already has a fixed order
   there. Restating the clause for sequential grids is a profile question owed
   at bring-up.

Not declared here. Qualcomm's Hexagon NPU, the Qualcomm target Modular has
announced for Mojo. Its kernel documentation was not researched in this lane.

## Verification (M4, one core, nice 19, `mojo build -j 1`)

- (2026-09-14 lane) Baseline at b449ffa78 and this lane, both numeric modes (default and
  `-D MOJOLEARN_NUMERIC_IDENTICAL=1`). `matrix_main` output with the new
  primitives lines and the two new columns removed is byte-identical to
  baseline in both modes, so every row printed for the nine older columns
  resolves as it did. `check_hardware_matrix` passes in both modes. Its only
  output change is the summary sentence.
- Sabotage, each in its own copy of the tree and each failing the check with the
  expected message. TPU fma set to PRESENT fails ("tpu is refused without its
  arithmetic reason"). Qualcomm fma set to ABSENT fails (Qualcomm refused on
  arithmetic). Apple division set to ABSENT fails (Apple refused on division).
- No kernel or binding was rebuilt. Every edited branch compares against column
  9 or 10, and `TARGET_COLUMN` can take neither value, so no build's comptime
  rows change. The printout comparison above is the observed half of that
  argument.

## Qualcomm, Intel and the spec baseline (audited 2026-09-15)

Andrew asked for the three older declared columns to be checked against the
same four primitives. Mojo emits code for none of them, so there is no vendor
kernel language to read. The audit reads the portable standards their rows
already cite (OpenCL, Vulkan through SPIR-V and GLSL, and SYCL for Intel's
oneAPI), and confirms each part ships a standard stack that has the primitive.
OpenCL runs on Adreno in phones and in Snapdragon X laptops (the llama.cpp
OpenCL backend, written for Adreno first, lists Adreno 750 through X2-90), and
Intel's compute runtime provides OpenCL and Level Zero for Xe.

| primitive | qualcomm | intel | spec-baseline |
|---|---|---|---|
| fused multiply-add | PRESENT (OpenCL C `fma`, correctly rounded by spec) | PRESENT (same, and `sycl::fma`) | ABSENT (Vulkan guarantees none) |
| binary32 division | PRESENT as a name, rounding a device flag | PRESENT as a name, rounding a device flag | PRESENT as a name, 2.5 ulp allowed |
| exact int32 | PRESENT | PRESENT | PRESENT |
| float bits readable | PRESENT (SPIR-V `OpBitcast`) | PRESENT (`OpBitcast`, `sycl::bit_cast`) | PRESENT |

What the specifications say, quoted.

- OpenCL C 3.0 `fma` "Returns the correctly rounded floating-point
  representation of the sum of c with the infinitely precise product of a and
  b. Rounding of intermediate products shall not occur." Table 65 lists fma as
  "Correctly rounded", and section 7.4 says addition, subtraction,
  multiplication and fused multiply-add "are IEEE 754 compliant and are
  therefore correctly rounded". SPIR-V OpenCL.std instruction 25 and SYCL 2020
  `sycl::fma` use the same sentence. `CL_FP_FMA` only says whether the builtin
  is done in hardware.
- The same OpenCL C spec says `#pragma OPENCL FP_CONTRACT` "DEFAULT value is
  ON", so a plain `a*b+c` may be fused or not at the compiler's choice. This is
  exactly the hazard the explicit fma exists to remove.
- Vulkan compute has no such guarantee. GLSL.std.450 `Fma` "Computes a * b +
  c", and the GLSL 4.60 precision table allows `a * b + c` as a "Correctly
  rounded single operation or sequence of two correctly rounded operations",
  with `fma()` "Inherited from a * b + c". SPIR-V's `NoContraction` decoration
  can forbid fusing, but nothing in Vulkan forces it. The spec baseline is what
  both standards guarantee, so its fma row is ABSENT, and it is now refused on
  arithmetic as well as on its 16 KB memory floor. The refusal names both.
- Division is the real risk on these columns. OpenCL C Table 65 allows
  single-precision `x / y` "<= 2.5 ulp" (the embedded profile "<= 3 ulp"), and
  correct rounding is the optional device flag
  `CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT`. GLSL allows "2.5 ULP". The row stays
  PRESENT because a division can be named, and `check-division` is the first
  gate at bring-up. A device that fails it still has a correctly rounded fma,
  which is enough to correct a quotient the way `portable_sqrtf` already
  corrects a square root. That repair is not built.
- OpenCL also makes subnormal support optional ("may be flushed to zero"),
  which `ftz` already normalizes.

Verification (M4, one core, both numeric modes, against origin/main 3ddf1f3c9).
`matrix_main` changes only on the primitives lines of the three columns and on
the refusal lines of the spec baseline, TPU and Trainium (each refusal now
also names its kernel-shaped reason). Every table row is unchanged.
`check_hardware_matrix` passes. Three sabotage arms fail it as required, namely
Qualcomm fma set to ABSENT, spec-baseline fma set to PRESENT ("spec-baseline
fma = 1, want 0"), and a refusal reason that drops the memory half.

## Sources (read 2026-09-14)

- Pallas TPU quickstart, https://docs.jax.dev/en/latest/pallas/tpu/quickstart.html
- Pallas TPU details (op list, data types, VMEM, vector registers, matmul precision), https://docs.jax.dev/en/latest/pallas/tpu/details.html
- NKI overview, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/index.html
- NKI ISA common fields (data types, math operators, activations, engine precision), https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.api.shared.html
- NKI ISA reference pages `tensor_tensor`, `scalar_tensor_tensor`, `tensor_scalar`, `tensor_reduce`, `tensor_scalar_reduce`, `activation`, `reciprocal`, `nc_matmul`, `core_barrier`, under https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/
- StableHLO specification, https://github.com/openxla/stablehlo/blob/5a1e6d92793b5f21551561e1628d86a24909ae49/docs/spec.md

Read 2026-09-15 for the Qualcomm, Intel and spec-baseline audit.

- OpenCL C 3.0 (math functions, FP_CONTRACT, chapter 7, Tables 65 and 66), https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html
- OpenCL API 3.0 (`CL_DEVICE_SINGLE_FP_CONFIG` flags), https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html
- SPIR-V OpenCL.std extended instructions, https://registry.khronos.org/SPIR-V/specs/unified1/OpenCL.ExtendedInstructionSet.100.html
- SPIR-V GLSL.std.450 extended instructions, https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
- SPIR-V specification (`OpBitcast`, `NoContraction`), https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
- GLSL 4.60 (section 4.7.1 precision table, `fma`, `precise`), https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html
- SYCL 2020 (`sycl::fma`, `sycl::bit_cast`, `fp_config`), https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
- llama.cpp OpenCL backend (Adreno support list), https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md
- Intel compute runtime (OpenCL and Level Zero for Xe), https://github.com/intel/compute-runtime
