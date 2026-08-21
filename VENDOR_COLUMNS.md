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
| amd (CDNA: MI300X/MI355X) | yes | 64 KB | **64** | 1024 | yes | yes | yes | **admitted** |
| **amd-rdna (RX 9070/7900)** | yes | 64 KB | **32** | 1024 | yes | yes | yes | **admitted** |
| qualcomm (Adreno) | no | 32 KB | **8–128, compiler-chosen** | 1024 | extension only | yes | yes | **admitted** |
| intel (Xe/Arc) | no | 64 KB | **8/16/32, compiler-chosen** | 1024 | yes | yes | yes | **admitted** |
| **spec-baseline** | no | **16 KB** | **1 (none guaranteed)** | **128** | no | yes | not promised | **REFUSED** |

`spec-baseline` is not a vendor. It is what the portable specifications
GUARANTEE any conformant GPU provides — the intersection of Vulkan's required
limits (`maxComputeSharedMemorySize` 16,384; `maxComputeWorkGroupInvocations`
128) and WebGPU's defaults (`maxComputeWorkgroupStorageSize` 16,384;
`maxComputeInvocationsPerWorkgroup` 256). See "The floor has not moved" below
for why it is in the table when it can never be admitted.

**The finding, and it was not the expected one: every declared vendor meets
the floor.** Adreno's per-workgroup local memory is 32 KB, exactly Metal's;
Intel and both AMD families are above it. The design was already floored by
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

**AMD is two columns because AMD is two architectures.** CDNA (MI300X,
MI355X — the parts Mojo tests continuously) runs **wave64**. RDNA (RX 9070,
7900, 6900 — the cards a person buys to train on) runs **wave32**: RDNA's
native compute mode, and the only one HIP reaches, since the
`mwavefrontsize64` compiler option is experimental and unsupported by the HIP
runtime.

One `amd` column was therefore wrong for half the AMD parts Mojo supports,
and not abstractly: `lib_block_size_for` sizes the warpsort block as
`8 * lane_width`, so it resolved **512 threads for a device whose lane group
is 32 and whose correct answer is 256**. A scheduling row, so no bit ever
moved — but wrong on hardware people actually train on, which is exactly the
class of error this table exists to prevent. Both values are now pinned in
`check_hardware_matrix`.

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

| row | before (apple, nvidia, amd) | after (+ amd-rdna, qualcomm, intel) | moved? |
|---|---|---|---|
| threadgroup bytes | min(32, 48, 64) = **32 KB** | min(32, 48, 64, 64, 32, 64) = **32 KB** | **no** |
| dispatch cap | 1024 | 1024 | **no** |
| hardware lane width | min(32, 32, 64) = 32 | min(…, 32, 8, 8) = **8** | yes, and it is not a floor input: the replication group is LOGICAL |
| core float atomics | all three | **not universal** (Adreno) | yes — in `FAST`, not in `IDENTICAL` |
| threadgroup i32 atomics | all three | every column, baseline included | no |

**Apple was already the binding constraint and still is.** Adreno's 32 KB per
workgroup is exactly Metal's; Intel and both AMD families are above it.
So the identity column does not change, and *that is the mechanism working*
rather than a lucky escape: it is frozen, so the only question a new vendor
raises is whether it joins or is refused.

### A low dispatch cap would cost nothing until it is very low

Resolved per column, straight out of the table rather than estimated:

| column | binary | half-byte | one-byte | hist_2 | vendor cap |
|---|---|---|---|---|---|
| apple | 512 | 512 | 256 | 512 | 1024 |
| nvidia / amd / intel | 768 | 768 | 384 | 384 | 1024 |
| qualcomm | 512 | 512 | 256 | 512 | 1024 |
| spec-baseline | 128 | 128 | 128 | 128 | **128** |

On any 32 KB column the **shared-memory budget binds first and produces 512
anyway**: 32,768 / (16 floats × 4 bytes) = 512 for the binary and half-byte
kernels, and the shared-Int32 hist_2 arm is capped at 512 by design because
that is the block that fills 32 KB. Mali's cap sits exactly at the number the
memory budget had already chosen — so a vendor cap of 512, Mali's when it
was briefly a column, removes nothing at all. The cap that hurts is the
baseline's **128**: a quarter of the block, which is a different profile, not
a tuning loss.

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

### Which rows actually have to match, and which are free

The floor is short because most of the table does not need to agree. One
question sorts every row: **does it change the sequence or the precision of
the arithmetic?**

| free to differ per vendor | must match |
|---|---|
| grid shape, block count, which thread does which work | the shared-memory budget, because it sets the block, which sets how many private histogram slices exist |
| double buffering (RAFT's two smem pages) | the replication lane count |
| the quantize search strategy in the evaluator | the reduction stage width |
| launch and drain policy | the accumulator TYPE (float atomic vs fixed point) |

And one row moves between the columns depending on the kernel, which is the
part worth understanding:

- The **hist_2 family** (CatBoost's fused two-stat path, taken at every
  `maxBins <= 128`, so the common case) accumulates in **shared Int32**.
  Integer addition is associative, so the number of slices cannot change the
  total: **block size there is pure scheduling** and a vendor may pick any.
- The **binary / half-byte / wide one-byte families** accumulate in **shared
  float** (`smem[slot] = smem[slot] + stat`) and only flush through fixed
  point. There the block size decides which floats add to which, so it is
  **numeric** — and that is precisely what the identity floor's 32 KB is
  protecting.

So the floor is load-bearing for one half of the kernel set and redundant for
the other. **That is a live design lead, not a settled fact:** moving the
remaining families to shared fixed point would make the block scheduling
everywhere, which would let a much lower floor still be bit-identical — and
would make the `spec-baseline` column admissible instead of refused. Nobody
has costed it. It is written here rather than acted on because it is a profile
question, and profile questions are decided deliberately.

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

## Every other GPU that exists, and why none of them gets a column

The full enumeration, so that "are there others?" has an answer instead of a
recurrence. Anything shipping in volume is on this list.

**Desktop and datacenter**

| family | status |
|---|---|
| NVIDIA, AMD, Intel | columns |
| Moore Threads (MTT S80/S4000), Biren (BR100), MetaX, Iluvatar CoreX | CUDA-alike, no public architecture document worth transcribing, availability restricted to one region. A pinned column would be fiction |
| Zhaoxin / Glenfly Arise | negligible compute stack |

**Mobile and embedded IP**

| family | status |
|---|---|
| Apple, Qualcomm Adreno | columns |
| **Imagination PowerVR** | The strongest candidate on this list and probably **refused**. Imagination's own OpenCL guidance describes the shared local memory as living in a **common store of 4,096 words — 16 KB** per multiprocessor, with a 256-work-item group allocating 2,048 words leaving room for two resident groups. That is the baseline column's number, not ours. Worth a real column only if a toolchain target appears; until then `spec-baseline` predicts its verdict |
| **Arm Mali** | Struck as a column, kept as a warning — see below. No dedicated compute scratchpad; wave 8 (Bifrost) / 16 (Valhall) |
| Samsung Xclipse | **AMD RDNA IP in a phone.** Its numbers are the `amd-rdna` column's, scaled down. Not a distinct set of minimums |
| Huawei Maleoon | no public architecture documentation at all |
| VeriSilicon Vivante, Broadcom VideoCore (Raspberry Pi), Think Silicon NEMA | real OpenCL/Vulkan compute, no realistic training workload at this scale |

**Not hardware**

Software rasterizers (SwiftShader, llvmpipe) would be a CPU path, and this
tree has no CPU path.

### Why the tail does not need columns

**`spec-baseline` is the answer to all of them.** Every conformant GPU clears
16 KB and 128 invocations, so every device on this list sits somewhere between
the baseline and our floor, and its verdict follows without a column:

- clears the identity floor → joins, no bit moves
- clears only the baseline → refused for `IDENTICAL`, runs `FAST`

PowerVR is the demonstration: one sourced number (16 KB common store) places
it below the floor without anyone writing a `COLUMN_POWERVR`. Adding vendor
columns past this point buys precision exactly where we would never build, and
buys fiction everywhere else. **Add a real column when a toolchain target
exists, and not before.**

### Why `arm` was added and struck the same day

Mali is the GPU in most non-Apple phones — Arm licenses it and Samsung,
MediaTek and Google's Tensor ship it — so by unit count it is one of the most
numerous GPU families alive, and by **training** relevance one of the least.

It was added as a stress test (the family whose architecture most contradicts
this design) and struck on the rule that **the matrix holds the GPUs people
train on, plus the floor beneath them.** A column implies an intent, and there
is no world in which this library targets a phone GPU. `spec-baseline` covers
the stress-test job better anyway: it is refused, so it exercises the
admission gate, and it is grounded in a specification rather than in one
vendor's parts.

The Mali FACTS are kept — in the table above and in
`column_has_dedicated_shared_memory`'s docstring — because they are a warning
that outlives the column: **a conformant GPU can advertise the shared-memory
capacity and provide none of the speed.** Arm's own guide, verbatim: "Arm GPUs
do not implement dedicated on-chip shared memory for compute shaders. The
shared memory that is available to use is system RAM that is backed up by the
load-store cache." Any future column that answers `False` to that row loses
the bet a replicated histogram is making, and must re-measure the replication
factor rather than inherit ours.

### Honesty note: three of these rows have no reader

Audited by grep, 2026-08-21. `column_max_block_size` is genuinely load-bearing
— the floor gate and all three block resolvers call it. But
`column_has_dedicated_shared_memory`, `column_spec_guarantees_onchip_shared`
and `column_lane_width_is_fixed` are read by **nothing except the report and
this document**. They are declared facts for whoever brings a column up, not
inputs to a decision the code makes today, and calling them load-bearing would
be a claim this tree's own rules forbid. Tracked in `UNWIRED.md`.

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
