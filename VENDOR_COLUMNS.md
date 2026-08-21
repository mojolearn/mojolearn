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
| **spec-baseline** | no | **16 KB** | **1 (none guaranteed)** | **128** | no | yes | not promised | **REFUSED** |

`spec-baseline` is not a vendor. It is what the portable specifications
GUARANTEE any conformant GPU provides — the intersection of Vulkan's required
limits (`maxComputeSharedMemorySize` 16,384; `maxComputeWorkGroupInvocations`
128) and WebGPU's defaults (`maxComputeWorkgroupStorageSize` 16,384;
`maxComputeInvocationsPerWorkgroup` 256). See "The floor has not moved" below
for why it is in the table when it can never be admitted.

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

**Arm Mali is the GPU in most non-Apple phones and tablets** — Arm licenses
it as IP and Samsung, MediaTek, Google (Tensor) and others ship it, so by
unit count it is one of the most numerous GPU families in existence, and by
training relevance one of the least. It is here because it is the family
whose *architecture* most contradicts this design, which makes it the useful
stress test even though nobody will train on a phone.

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

## How far has the lowest common denominator moved? It has not.

The question the vendor columns were added to answer, with the arithmetic
rather than a reassurance.

| row | before (apple, nvidia, amd) | after (+ qualcomm, intel, arm) | moved? |
|---|---|---|---|
| threadgroup bytes | min(32, 48, 64) = **32 KB** | min(32, 48, 64, 32, 64, 32) = **32 KB** | **no** |
| dispatch cap | min(1024, 1024, 1024) = 1024 | min(…, 1024, 1024, 512) = **512** | yes, and it costs nothing — see below |
| hardware lane width | min(32, 32, 64) = 32 | min(…, 8, 8, 8) = **8** | yes, and it is not a floor input: the replication group is LOGICAL |
| core float atomics | all three | **not universal** | yes — in `FAST`, not in `IDENTICAL` |
| threadgroup i32 atomics | all three | all six, and the baseline too | no |

**Apple was already the binding constraint and still is.** Adreno's 32 KB per
workgroup is exactly Metal's; Mali advertises the same; Intel is above both.
So the identity column does not change, and *that is the mechanism working*
rather than a lucky escape: it is frozen, so the only question a new vendor
raises is whether it joins or is refused.

### The 512-thread dispatch cap on Mali costs nothing

Resolved per column, straight out of the table rather than estimated:

| column | binary | half-byte | one-byte | hist_2 | vendor cap |
|---|---|---|---|---|---|
| apple | 512 | 512 | 256 | 512 | 1024 |
| nvidia / amd / intel | 768 | 768 | 384 | 384 | 1024 |
| qualcomm | 512 | 512 | 256 | 512 | 1024 |
| arm | 512 | 512 | 256 | 512 | **512** |
| spec-baseline | 128 | 128 | 128 | 128 | **128** |

On any 32 KB column the **shared-memory budget binds first and produces 512
anyway**: 32,768 / (16 floats × 4 bytes) = 512 for the binary and half-byte
kernels, and the shared-Int32 hist_2 arm is capped at 512 by design because
that is the block that fills 32 KB. Mali's cap sits exactly at the number the
memory budget already produced, so it removes nothing. The cap that would
hurt is the baseline's 128 — a quarter of the block, which is a different
profile, not a tuning loss.

### The dispatch cap was not being consulted at all

The baseline column found this on its first run and it is the argument for
declaring columns you cannot build for. `block_size_for` and
`hist2_block_size_for` bounded the block by the shared-memory budget and
**never by the vendor's maximum workgroup size**. On every buildable column
that was slack — all three caps are 1024, the largest block is 768 — so the
omission was invisible and would have stayed invisible. Against the baseline
it resolved a 256-thread block on a target that guarantees only 128
invocations: a grid the device is not required to be able to launch.

Both resolvers and the runtime one now clamp. `check_hardware_matrix` pins
apple at 512/512/256/512 and nvidia at 384 across the change, so the fix is
proved to have moved nothing on any column that runs today.

### What a portable profile would cost, now that it can be priced

If `IDENTICAL` ever has to reach a spec-minimum device, profile 2 would be
16 KB and a 128-thread block: **a quarter of the block, four times the
threadgroup slices, a different set of partial sums.** Feasible — 128 threads
still hosts four 32-lane replication groups, and the one instruction the
guarantee cannot do without (threadgroup `atomicAdd` on `i32`) is core in
both Vulkan and WGSL, so the refusal is a *size* and never a missing
capability. But it is a different guarantee about a different set of models,
which is exactly why it is a profile bump and not an edit.

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
