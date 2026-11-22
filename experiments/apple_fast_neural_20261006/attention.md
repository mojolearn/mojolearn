# Apple FAST attention implementation notes — source only

Branch: `ideas/apple-fast-neural-20261006`, forked from local `main`
`fd6cf8045`. A01–A12 are **not tested** in this campaign. No compiler, test,
lint, manifest checker, runtime, correctness check or measurement was run.
`attention.json` is authored recipe data, not an executed report. The source
changes and this document are the only evidence delivered by this lane.

## What is selectable

A01–A07 reuse existing implementations. Their switch declarations now say
`not tested in this campaign`; historical F07 component results and their
disabled defaults remain in place. Each card provides a standalone baseline
and candidate; A04 enables FLASH on both sides. A07 is the existing transformer
binding arena. Its allocation effect must be established through that binding,
including its use by Samba; ByteLM's internal block allocations are a separate
path and are not covered by an arena claim.

The new controls below are explicit compile-time defines. Each is guarded by
the existing Apple GPU + FAST + non-CPU-column condition. They default off and
do not turn on the required parent. The historical `AFN_ATTN_ALL` mechanism
bundle does not turn on any new geometry. There is no automatic selection by
input size, benchmark row or dataset name.

| Card | Bare define | Parent and implemented change |
| --- | --- | --- |
| A08 | `MOJOLEARN_AFN26_ATTN_NORM_TPB128` | `AFN_ATTN_NORM_SG` or `AFN_ATTN_FUSE_PRE`; four norm simdgroups per block |
| A08 | `MOJOLEARN_AFN26_ATTN_NORM_TPB512` | Same parents; sixteen norm simdgroups per block |
| A09 | `MOJOLEARN_AFN26_ATTN_FLASH_BK16` | `AFN_ATTN_FLASH` / `AFN_ATTN_GQA_TILE`; sixteen keys per tile |
| A10 | `MOJOLEARN_AFN26_ATTN_FLASH_TQ16` | Same parents; sixteen query rows and four simdgroups per block |
| A11 | `MOJOLEARN_AFN26_ATTN_PROJ_KB16` | `AFN_ATTN_FUSE_PRE` / `AFN_ATTN_FUSE_MLP`; sixteen reduction elements per projection window |
| A12 | `MOJOLEARN_AFN26_ATTN_PROJ_BM32` | Same parents; thirty-two output rows per projection block |

The parent spellings above omit their common `MOJOLEARN_` prefix for brevity;
the JSON contains complete bare define names. Recipe consumers must translate
each to the build system's define syntax. Defining a switch as `=0` does not
disable it because the source uses `is_defined`. Omit it from the baseline.
Selecting both A08 widths is a compile-time configuration error. No recipe
uses an `ALL` flag or silently adopts one as its baseline.

## RMSNorm geometry

`AFN_NORM_TPB` is separate from the old `AFN_TPB`. The kernel computes the row
as `block_index * (AFN_NORM_TPB / 32) + simdgroup`; both plain and residual
RMSNorm launchers use the same derived row count and block width. The row tail
returns a whole inactive simdgroup. All 32 lanes of a live row retain the
float4 loads, five-step shuffle fold, f32 accumulation and original epsilon.
Other kernel block sizes do not inherit a norm experiment.

This is an occupancy hypothesis, not a speed claim. A 512-thread group may
lose through scheduling/resource pressure even though the existing kernel has
no shared page. Narrow and wide rows, residual input, sums-only projection
staging, output materialization and chained next-norm callers all need future
coverage. No hardware launch-limit or compilation qualification is asserted.

## FLASH geometry and ownership

The tile now derives the following from `TQ` and `BK`:

- Block width is `TQ * 8`; there are eight softmax lanes per query row.
  Thus A10 uses 128 threads/four simdgroups, while the incumbent uses
  256 threads/eight simdgroups. A10 is a coherent query-tile experiment,
  including this necessary ownership change.
- Each softmax lane owns `BK / 8` keys: two for A09, four for the incumbent.
  The local score/probability vectors, visibility predicates and transposed
  probability writes use that derived count. The row shuffle tree is still
  eight lanes; no out-of-tile key columns are read or written by design.
- Score fragments, fragments per simdgroup, context fragments, K/V/Q/P
  strides, shared page sizes, key iterations, staging strides and every
  FLASH launch derive from the selected geometry. Grid query blocks use
  `ceil(sequence_length / (TQ / GROUP))`.
- Shared storage remains the sum of K, the larger of the V and initial Q
  pages, P, scores and three row-statistic pages. The existing static page
  limit remains 32 KB. Existing padded head classes 16, 32, 48, 64 and 80
  remain the only launcher classes; smaller geometry is not a head-coverage
  expansion in this delivery.
- With TQ16 and GROUP4, each query head owns four token rows. An eight-row
  Q fragment may contain rows from two query heads; both use the same KV
  head. The row-specific head/token output mapping and mask ranges remain
  derived per row. The assertion permits this integral four-row ownership.
  This combination is not tested and needs explicit future coverage.

Each geometry uses the same online max/sum rescaling and f32 matrix fragment
operations. BK16 introduces more online-softmax updates, so arithmetic order
can change even though the intended attention function is unchanged. TQ16
can increase K/V reloads and block count. Neither is assumed beneficial.

Retain separate A09 and A10 comparisons under plain FLASH and grouped FLASH.
Only after those should an interaction recipe compare the same enabled
FLASH/GQA parent against both new defines together. Cover the complete set of
existing padded head classes, partial sequence tiles, `n_rep` 1/2/4, unsupported
group fallback, absolute positions, causal/window bounds, visible-key tails,
empty visibility, `amax`, denominator and context outputs. Fresh prefill and
cache continuation are distinct workload boundaries, not interchangeable
samples.

## Fused projection geometry and shared memory

The fused neural projection keeps its 64-column tile and 256-thread block.
BM32 halves the row fragments per simdgroup; KB16 halves each K window. The
existing grid, accumulator extents, row-statistics page, load/store predicates,
RoPE row mapping and output epilogues read their derived constants. Generic
GEMM has not been edited.

Two source changes are necessary for these variants rather than optional
optimizations:

1. At KB16, the completed RoPE output tile can exceed the operand staging
   page. Shared allocation is therefore `max(A + 2*W, C_tile)`, plus row
   statistics. The existing barrier after the last multiply separates
   operand reads from reuse for the RoPE epilogue. The C tile can reuse the
   second W region because no operand is live during that epilogue. The
   static capacity assertion checks the actual complete page.
2. BM32+KB16 has fewer float4 A slots than threads. The staging loop uses a
   rounded-up slot count and predicates `row < BM` before reading input or
   row statistics and before shared writes. It still stages zero for every
   padded output row. The W tile remains a whole set of thread slots for
   both programmed K widths.

`afn_gemm_ok` continues to derive its full-window alignment from KB. This
means KB16 can serve aligned widths that KB32 previously sent to the fallback.
Future qualification must distinguish a same-route geometry comparison from
newly admitted alignment cases. This is alignment reasoning across neighboring
shapes; there is no exact benchmark-size rule. Head layout and the 64-column
RoPE partner ownership are unchanged.

The JSON gives separate FUSE_PRE, FUSE_MLP and combined-parent baselines for
A11 and A12. Their parent configuration is identical on both sides. Test the
two geometry defines together in a separate interaction cell, including all
epilogues, odd output-row tails, dimensions adjacent to legal K alignment,
fresh cache append, residual ordering and full-head/padded-head cases.

## Public reach and future binding builds

The existing `_afn_block_ok` contract only admits default-option,
forward-only, non-materialized, untraced, unsabotaged blocks with float4-aligned
model width and no int15 route. The programming leaves that guard in place.
Backward-visible/materialized training forwards retain their existing path.
Do not describe these source switches as training speedups without showing
the exact public training or held-out caller reaches them. No backward
intermediate has intentionally been removed from a training-capable route.

These are **future build targets only**, not commands to execute in this
assignment:

| Consumer | Existing binding/build entry | Scope |
| --- | --- | --- |
| TransformerBlock | `bindings/_mojolearn_transformer.mojo`, `bindings/build_transformer.sh` | A01–A12 where their parent and public forward guards allow them; arena is here |
| SambaStack attention layers | Same transformer binding, with unchanged training/Mamba dependencies | Full mixed-stack forward; prove public routing before claiming an affected cell |
| ByteLM | `bindings/_mojolearn_byte_lm.mojo`, `bindings/build_byte_lm.sh` | Internal transformer forward-only calls for A01–A06/A08–A12; no standalone arena claim |

Future Apple builds use the existing supported FAST/Apple build configuration
and `MOJOLEARN_BUILD_EXTRA_DEFINES` hook. Preserve the repository's guarded
build process, fresh arm output directories and binary provenance. Do not run
a build script under the assumption that it is source-only; scripts can build
and the transformer script also contains build-gate logic. Nothing here
requires a new compiler path, custom Metal code or unsupported toolchain mode.

## Pending qualification, not results

All new code, recipe data, combined configurations and other-target exclusion
are uncompiled and unverified. The authored files do not prove syntax,
linkability, routing, bounds, quality or speed. Historical source comments
refer to prior work only. No evidence has been promoted and no default has
been flipped.

For later work, map every affected operation to the full saved recipes in
`tools/bench_board_neural.py` and the locations indexed by
`experiments/performance_ideas/README.md`. Record exact dataset/corpus version
and hash, complete dimensions, layer configuration, numeric mode, arm defines,
seed, split and consumed outputs. Audit any intrinsic sequence/row caps before
calling a run full-dataset. Missing or ambiguous mappings remain pending.

Separate cold allocation/setup, repeated complete forward, prefill, decode and
held-out evaluation. The operation boundary must include required preparation,
uploads, synchronization, cache handling, output consumption and cleanup.
Quality coverage includes output/logit error, held-out NLL/perplexity, cache
contents/continuation and existing task bands without relaxation. Where a
changed inference caller participates in a training workflow, include its full
trajectory and held-out outputs while preserving optimizer settings/data order.

Include neighboring shapes and one non-board dataset; tiny kernels and public
smokes are insufficient performance evidence. Retain rejected and neutral
results as well as winners, along with failure/unsupported status and limited
sample counts. Record future failures rather than replacing their evidence.
Keep complete future build, test and GPU logs on disk, inspect exit status and
structured summaries first, and report evidence paths with bounded diagnostic
excerpts. None of that later qualification was performed here.
