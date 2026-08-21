# Vendor columns: every GPU that exists, its minimums, and whether identity survives it

Written 2026-08-21, when three columns were declared that nothing can build
for. Read `mojo_only/kernel_matrix.mojo` first; this file is the evidence
behind its rows, and the row is the truth if the two ever disagree.

## Why declare a vendor before the toolchain has one

The bit-identical column used to be defined as the **intersection** of the
vendor columns present. That is correct arithmetic and a latent defect:

> Add a vendor with a smaller threadgroup budget and the intersection
> shrinks. A smaller budget gives a smaller block; a smaller block gives a
> different replication factor; a different replication factor sums a
> different set of partials. **Every model produced under `IDENTICAL` before
> the addition disagrees with every one produced after it**, on every device,
> with no error, no version, and no way for a user to find out except by
> keeping an old model file.

The product property is "the same fit gives the same model". A definition
under which supporting one more GPU rewrites that answer for everybody is not
a safe column, so the floor is now **frozen and versioned**
(`IDENTITY_PROFILE`), and a new vendor does exactly one of two things:

| | |
|---|---|
| meets the floor | joins `IDENTICAL`; **not one bit moves** |
| misses the floor | refused for `IDENTICAL` **by name**, runs `FAST` |

There is no "lower the floor a little". Lowering it is a different guarantee
about a different set of models and takes a profile bump with a migration.

Declaring the vendors now is what makes that a rule rather than a judgement
call taken later, under pressure, with shipped models already in the field.

## The floor (profile 1)

| row | value | why this row and not another |
|---|---|---|
| threadgroup bytes per block | 32,768 | decides the block size, which decides the replication factor, which decides which partials combine |
| logical replication lanes | 32 | `PINNED_REPLICATION_LANES`; the width of the private-copy group |
| hist_2 block | 512 | the block the safe column's accumulator runs |
| threadgroup Int32 atomics | required | the safe accumulator is integer *because* integer addition is associative; there is no substitute |

## The columns

`build` = Mojo emits code for it today (Mojo system requirements, read
2026-08-21: NVIDIA Turing→Blackwell driver 580+, AMD RDNA2→CDNA4 ROCm 6.3.3+,
Apple M1–M5 on macOS 15+). Everything else is the vendor's documented floor,
transcribed, never measured here.

| column | build | smem/block | lane width | max block | f32 atomics | i32 local atomics | real scratchpad | `IDENTICAL` |
|---|---|---|---|---|---|---|---|---|
| apple (Metal) | yes | 32 KB | 32 | 1024 | yes | yes | yes | **admitted** |
| nvidia (CUDA) | yes | 48 KB | 32 | 1024 | yes | yes | yes | **admitted** |
| amd (HIP) | yes | 64 KB | 64 | 1024 | yes | yes | yes | **admitted** |
| qualcomm (Adreno) | no | 32 KB | **8–128, compiler-chosen** | 1024 | extension only | yes | yes | **admitted** |
| intel (Xe/Arc) | no | 64 KB | **8/16/32, compiler-chosen** | 1024 | yes | yes | yes | **admitted** |
| arm (Mali) | no | 32 KB | 8 (Bifrost) / 16 (Valhall) | 512 | extension only | yes | **no** | **admitted** |

**The finding, and it was not the expected one: every declared vendor meets
the floor.** Adreno's per-workgroup local memory is 32 KB, exactly Metal's;
Mali advertises the same; Intel is above it. The design was already floored by
the most constrained mainstream GPU memory hierarchy, which is why freezing
the floor costs nothing today — and why now, while it is free, is the moment
to freeze it.

### The three cells that matter more than the rest

**Adreno's wave width is not a device constant.** Qualcomm's own OpenCL
guidance: the wave size "depends on Adreno GPU series and tiers as well as
**the compiler**; values could be 8, 16, 32, 64, 128". The Adreno X1 in
Snapdragon X Elite is 64- or 128-wide. Intel is the same shape of problem: the
sub-group is 8, 16 or 32 and their compiler picks from register pressure
unless a kernel demands one. **A design that read the hardware width would not
merely disagree across vendors on these two — it could disagree between two
builds of the same kernel for the same device.**

That this does not break us is a real property and not luck, but it is
narrow, so state it precisely: `PINNED_REPLICATION_LANES` is a **logical**
group width. Every kernel here syncs at `SYNC_BLOCK`, because Mojo exposes no
warp primitive on any vendor, so a 32-lane replication group is
threadgroup-synchronized arithmetic that happens to be 32 wide, and it stays
32 wide whether 8, 16, 64 or 128 lanes run in lockstep underneath. **The day
Mojo exposes lane primitives and `sync_granularity_for` stops returning
`SYNC_BLOCK`, that paragraph expires and these columns are unsafe until
re-argued.**

**Mali has no compute scratchpad at all.** Arm's GPU best-practices guide,
verbatim: "Arm GPUs do not implement dedicated on-chip shared memory for
compute shaders. The shared memory that is available to use is system RAM
that is backed up by the load-store cache." The bytes are there, so identity
is unaffected and the column is admissible; what is not there is the premise
of a replicated-histogram design, which is that private scratch is much
faster than the memory it spares. Whoever brings Mali up should expect to
re-measure the replication factor, not inherit it.

**Float atomics are an extension on the mobile columns.** `cl_ext_float_atomics`
needs OpenCL 2.0+ and is optional per device and per memory scope. A device
without it cannot run `FAST` as written and must take the fixed-point flush —
the vendor forcing the mode's hand. That is exactly what
`KernelSpec.flush_forced_by_vendor` was built for, and no founding column has
ever exercised it. The row is set conservatively (`False` for the mobile
columns) and should be replaced by a device query at bring-up, not by an
opinion.

## What is NOT a column, and cannot become one

Qualcomm acquired Modular in July 2026 and open-sourced Mojo and MAX at
ModCon in August; the platform's supported-accelerator list now includes
**AWS Trainium, Google TPUs, Qualcomm Cloud AI 100 and Qualcomm Dragonfly**.
None of those is a column here, and the reason is structural rather than a
matter of effort:

- They are **inference-serving targets on Modular Cloud**, not targets you
  write a Mojo threadgroup kernel for.
- They have **no threadgroup, no lane group, and no shared-memory
  scratchpad**. A TPU is a systolic array; Trainium and Cloud AI 100 are
  VLIW vector machines. Every row in the kernel matrix — block size, shared
  bytes, replication lanes, reduce width — names something that does not
  exist on them.

A learner on one of those is a **different backend**, not a column: the
histogram would have to be expressed as dense contractions, and whether the
result could be made bit-identical to the GPU column is an open question
nobody should answer from an armchair. Saying so here stops the next reader
from filing "add TPU to the matrix" as a small job.

## GPUs that exist and are not modeled

| family | why not |
|---|---|
| Imagination PowerVR | Real OpenCL/Vulkan compute, but tiny compute-market share and per-part local-memory limits that vary too much to pin honestly. Add a column when a target exists |
| Broadcom VideoCore (Raspberry Pi) | Vulkan compute only, no realistic training workload at this scale |
| Vivante, Moore Threads, Biren, Iluvatar | Either CUDA-alike with no public architecture document worth transcribing, or unavailable outside one region. A pinned column would be fiction |
| Software rasterizers (SwiftShader, llvmpipe) | Not hardware; would be a CPU path, and this tree has no CPU path |

## Adding a column, in order

1. Add the constant, `column_name`, and `column_is_buildable`.
2. Fill every row in `kernel_matrix.mojo` **with a citation per row**. Do not
   let a new column fall through to another vendor's value: the machine rows
   silently returned Apple's 10 cores for NVIDIA until 2026-08-19, and that
   is the failure this table exists to prevent.
3. Fill the machine rows in `hardware_matrix.mojo`, or mark them conservative
   placeholders. They are scheduling; no answer can move a bit; under-ask
   rather than over-ask.
4. Run `check_hardware_matrix`. It asserts every column answers every row,
   that the verdict and the refusal reason agree, and — the important one —
   that **the founding columns still resolve to exactly what they resolved to
   before**.
5. If the column misses the floor, do not touch the floor. Add it to `FAST`,
   give it an explicit exception naming it in the admission loop, and say so
   in the release notes.

## The hole this leaves

**The serialized model does not carry `IDENTITY_PROFILE`.** Until it does, a
profile-1 model and a future profile-2 model are distinguishable only by
provenance, which is exactly the sort of thing that gets lost. It is a header
field and about an hour's work, and it should be done before a second profile
exists rather than after. Tracked in `UNWIRED.md`.

## Sources

- Mojo system requirements (supported GPUs), read 2026-08-21.
- Modular, ModCon 2026 announcements: Trainium, TPU, Cloud AI 100, Dragonfly.
- Qualcomm, *Snapdragon Mobile Platform OpenCL General Programming and
  Optimization*; and *OpenCL Optimization and Best Practices for Qualcomm
  Adreno GPUs* (IWOCL 2018) — wave size varies by tier and compiler.
- Chips and Cheese, Adreno 640 and Snapdragon X Elite Adreno X1 analyses —
  32 KB local memory per workgroup; 64/128-wide waves.
- Intel oneAPI GPU Optimization Guide, *Shared Local Memory* (128 KB per
  Xe-core, 64 + 64 for two resident work-groups) and *Sub-groups and SIMD
  Vectorization* (compiler chooses 8/16/32).
- Arm GPU Best Practices Developer Guide, *Compute shading* — no dedicated
  on-chip shared memory; 64-thread baseline workgroup.
- Khronos OpenCL extension registry, `cl_ext_float_atomics`.
- NVIDIA CUDA C++ Programming Guide (48 KB default per block, 164 KB opt-in
  carveout at CC 8.0); AMD CDNA2 ISA reference (64 KB LDS per workgroup).
