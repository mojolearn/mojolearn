# Every pathway that can move a bit, and what `IDENTICAL` does about it

Opened 2026-08-21, when Andrew asked the question this document exists to
answer: *will the identical column actually be identical across every GPU?*

The answer was **no**, and the reason was not the histogram — the histogram is
the one part that was already right. The reason is that the enumeration was
incomplete, so pathways with no pinning at all sat outside a mechanism that
was designed correctly to hold them.

**This file is the enumeration.** It is the thing that makes the claim
provable rather than hopeful, and it is the artifact the paper is about.

## The rule

For a NUMERIC pathway, `IDENTICAL` has exactly three legitimate moves:

| move | when | example |
|---|---|---|
| **PIN** | the operation is deterministic given a parameter, and the parameter is machine-derived | the partition-stats chunk count |
| **REPLACE** | the operation is order-dependent in itself | float atomic → fixed-point integer accumulator |
| **REFUSE** | neither is available yet | a column that misses the identity floor |

There is no fourth move, and in particular there is no "usually fine". A
toggle that silently returns a non-identical model is worse than no toggle,
because it converts a checkable property into a belief.

## The ledger

| # | pathway | order-dependent? | what `IDENTICAL` does | status |
|---|---|---|---|---|
| 1 | global histogram flush | float `atomicAdd` | REPLACE — fixed-point Int32 | **closed** |
| 2 | hist_2 shared accumulation | float adds into per-warp slices | REPLACE — shared Int32 atomics | **closed** |
| 3 | histogram block size / replication factor | sets how many partials combine | PIN — the frozen floor bounds it | **closed** |
| 4 | replication lane count | a summation order | PIN — 32 on every vendor | **closed** |
| 5 | reduce stage width | a summation order | PIN — 512 | **closed** |
| 6 | library cross-lane folds (`PINNED_LIB_REDUCE_LANES`) | a summation order | PIN — 32 | **closed** |
| 7 | **partition stats → LEAF VALUES** | **chunk count = f(core count)** | **PIN — `partition_chunks_sm_for`** | **closed 2026-08-21** |
| 8 | **fixed-point scale magnitude** | **device float reduce + float atomic** | REPLACE — needed, not written | **OPEN** |
| 9 | **FMA contraction** | codegen decision | **CONSTRUCTION LANDED 2026-08-21**: `numerics.identical_mul_add` -- explicit `fma` under IDENTICAL (one rounding, identical on Metal/PTX/AMDGPU), the naive chain under FAST. Apple's FAST baseline measured UNFUSED (`check-ieee-arith`: fused 0 / unfused 1,046,394 of 2^20), so IDENTICAL differs from FAST on Apple BY DESIGN and is the same bits everywhere. | **helper landed; APPLICATION CHECKLIST OPEN** -- route every enumerated multiply-add seam through it: score gain arithmetic, cursor update, estimator derivations, leaf rescale. Each site converted must cite this row. |
| 10 | division and `sqrt` in scoring | IEEE-correct on normals everywhere measured; the REAL hazard is DENORMAL POLICY | **CONSTRUCTION LANDED 2026-08-21**: `numerics.ftz` -- flush-to-signed-zero under IDENTICAL, comptime no-op under FAST. MEASURED, not designed: the identical model reproduced ALL 53,041 observed Metal divergences bit for bit (`check-ieee-arith`'s ftz-model arm), so on FTZ backends the helper is bitwise inert and on denormal-honoring backends it aligns them to Metal. | **Apple column CLOSED (IEEE+FTZ, no fast-math); helper landed; APPLICATION CHECKLIST OPEN** -- flush every cross-kernel float seam and store pinned expressions' intermediates through `ftz`; run `check-ieee-arith` first on every new backend column. |
| 11 | k-NN tie handling (`select_radix`) | `atomicAdd` placement, no index tie-break | REFUSE — out of scope for GBDT | documented in `UNWIRED.md` |

Six closed, three open, one out of scope, one refused. **That ratio is the
honest state of the guarantee**, and it is the first time it has been written
down in one place.

## The three findings that produced this file

### 1. The toggle was not reachable

`NUMERIC_IDENTICAL` existed, was documented, and had a well-argued design.
**It could not be selected.** Five kernel files each declared their own
`comptime BUILD_MODE = NUMERIC_FAST`, so building the identical arm meant
editing five files and knowing which five. Every statement anyone had made
about "the IDENTICAL mode" was a statement about a configuration that had
never been built.

Fixed: `mojo_only/numerics.GLOBAL_NUMERIC_MODE` is one line, every site reads
it, the default is unchanged bit for bit, and flipping it rebuilds the tree in
the other mode.

### 2. A scheduling row was feeding a float sum

`partition_stats_chunks` is CatBoost's `CeilDivide(2 * SMCount(), statCount)`.
It decides how many chunks a leaf's rows are split into before `block.sum`
reduces each one **in float**. So the machine's core count decides the last
bits of every per-leaf stat, and per-leaf stats **are the leaf values**.

Apple's 10 cores and an A100's 108 SMs give different chunk counts and
therefore different models — not rarely, not probabilistically, but on every
tree. `hardware_matrix.gpu_cores_for` declares itself SCHEDULING and is right
about every other reader it has.

`numerics.mojo` opens by warning about exactly this: *"a block count is a
summation order."* The mistake was one indirection away from that sentence and
survived a matrix, a check and two audits.

Fixed by pinning inside `partition_stats_chunks` itself — the single place the
formula lives — because the launch AND the `stat_partials` buffer sizing both
read it, and pinning only the launch would have sized a buffer from one count
and indexed it with another.

### 3. The scale is derived from a float atomic

Item 8, still open. `choose_scale` snapping to a power of two contains it to
roughly 1e-6 per boosting round; when it fires the whole histogram shifts by a
factor of two. Small probability, large blast radius, and no reason to keep it:
the fix is the accumulator the histogram already uses.

## What has to be true before the claim is made

1. Items 8, 9 and 10 closed.
2. `check_identity_paths` — a test that FAILS if a new float reduction appears
   without a matrix row. This ledger is a document today, which means it rots.
3. The cross-vendor run (E1), which is now the LAST step rather than the first.

Running E1 before 1 would produce a failure that teaches nothing: we already
know what it would find.
