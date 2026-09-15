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
  Only ABSENT refuses. UNAUDITED is a measurement owed and never a verdict,
  which is why Qualcomm, Intel and the spec baseline resolve exactly as before.
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

- Baseline at b449ffa78 and this lane, both numeric modes (default and
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

## Sources (read 2026-09-14)

- Pallas TPU quickstart, https://docs.jax.dev/en/latest/pallas/tpu/quickstart.html
- Pallas TPU details (op list, data types, VMEM, vector registers, matmul precision), https://docs.jax.dev/en/latest/pallas/tpu/details.html
- NKI overview, https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/index.html
- NKI ISA common fields (data types, math operators, activations, engine precision), https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.api.shared.html
- NKI ISA reference pages `tensor_tensor`, `scalar_tensor_tensor`, `tensor_scalar`, `tensor_reduce`, `tensor_scalar_reduce`, `activation`, `reciprocal`, `nc_matmul`, `core_barrier`, under https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/
- StableHLO specification, https://github.com/openxla/stablehlo/blob/5a1e6d92793b5f21551561e1628d86a24909ae49/docs/spec.md
