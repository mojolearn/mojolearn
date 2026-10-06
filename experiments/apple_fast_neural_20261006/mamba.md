# Mamba Apple FAST source campaign

**Not tested.** No build, verification, lint, test, smoke, numerical execution,
or measurement was run for this assignment. Code and A/B definitions are
proposals. Neither speed nor quality has been established. New mechanisms are
off by default, guarded by FAST + Apple GPU, and marked `not tested` inline.
Existing historical evidence and existing defaults are retained.

The source-only branch is `ideas/apple-fast-neural-20261006`, forked from
`main` at `fd6cf8045`. The machine-readable recipes are [mamba.json](mamba.json).
No umbrella `ALL` define occurs in a recipe. Geometry recipes enable the same
parent on both sides; all other compilation settings must stay fixed.

## Implemented source changes

- **M01–M05:** Added current-campaign qualification notes beside existing
  chunk scan, input fusion, SSD MMA, SISO fusion and arena switches. Recipes
  independently select each existing arm. Historical measurements are kept
  explicitly separate from this untested campaign.
- **M06:** Added `MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4`. Each thread scans groups
  of four contiguous floats using the existing SIMD load operation, selects
  the minimum integer failure code locally, and joins the incumbent integer
  block/name reductions. Remaining one to three elements use scalar loads.
  Grid-stride groups retain full coverage. Refusal names, ordering, messages,
  finish readback and the mandatory device route remain intact by design.
- **M07:** Added `MOJOLEARN_AFN26_MAMBA1_CHUNKS16` and
  `MOJOLEARN_AFN26_MAMBA1_CHUNKS64`, each requiring the existing chunk-scan
  switch. Summary buffers contain 16/32/64 logical chunks; launches use
  32/32/64 physical threads. Surplus threads participate in both threadgroup
  barriers and never index a nonexistent summary. Carry folds only occupied
  chunks; the last occupied lane writes the state it walked for the final
  output. Zero-length sequences retain incoming state without a launch.
  Shared summaries use `3 * chunks * DSTATE * sizeof(f32)` bytes: at the
  caller's supported `DSTATE=16`, 3,072/6,144/12,288 bytes. The number of chunks
  is explicit experiment configuration, never selected by sequence length.
- **M08:** Added `MOJOLEARN_AFN26_MAMBA2_SSD_K16`, requiring SSD MMA. K staging,
  B stride, staging trip counts and 8-wide fragment loops derive from the
  chosen window. Shared staging is
  `4 * (K * (64 + 4) + BN * (K + 4))` bytes: with `BN=64`, 9,472 bytes at
  K=16 versus 17,920 at K=32; with `BN=32`, 6,912 versus 13,312 bytes. Full
  dot products, f32 accumulation, causal clipping and the existing legal-tile
  fallback are kept. More barriers can offset any occupancy gain.
- **M09:** Added `MOJOLEARN_AFN26_MAMBA3_THREADS64` and
  `MOJOLEARN_AFN26_MAMBA3_THREADS256`, requiring SISO fusion. All four fused
  launch grids and block sizes follow the same selected width; kernels map
  cells through `block_dim.x`. Their arithmetic, report ownership and serial
  angle chain are retained. Conflicting chunk or thread choices are refused
  at compile time when the parent is active on Apple FAST.
- **M10:** Added `MOJOLEARN_AFN26_MAMBA_REFUSAL_THREADS128` and
  `MOJOLEARN_AFN26_MAMBA_REFUSAL_GRID4`. Reduction storage, reduction tree,
  scan strides and fold launch dimensions follow the block size. GRID4
  targets four elements per physical thread before the same 32-block cap;
  the scan's grid stride handles all excess work. Every unused partial
  column stays initialized to the NONE sentinel. Single changes, their
  combination and combinations with vector4 have explicit recipes.

The shared refusal helper is used outside Apple FAST. Its new switches all
include `AFN_APPLE_FAST`, so their default values leave IDENTICAL, NVIDIA,
AMD and host settings at the incumbent geometry. No new host arithmetic,
compiler workaround, reduced precision, quality relaxation or dataset-name
dispatch is introduced. `afn_mamba_switches()` reports active AFN26 variants
for later provenance collection.

## Corrections discovered while reading incumbent source

The original M06 proposal was host refusal versus batched device refusal.
That distinction no longer exists for a GPU caller: current source requires
device refusal on every device, and the legacy AFN enable define is inert.
Restoring a host scan would violate the runtime policy. M06 therefore became
the explicit **device scalar versus device vector4** proposal above.

The original M10 proposal assumed 32 partial blocks were always launched.
Current source already launches `min(ceil(n / threads), 32)` and fills unused
partials. The new grid candidate instead changes the target to four elements
per thread: `min(ceil(n / (threads * 4)), 32)`. This is a work/overhead
hypothesis, not a benchmark-size threshold. The vector4 variant keeps that
grid policy independent from its four-cell load grouping.

## Future binding and workload qualification, not performed here

The implementation is reachable through existing Mamba call sites; no new
Python API or binding edit was needed. A later authorized frozen build would
target `bindings/_mojolearn_mamba.mojo` via the existing
`bindings/build_mamba.sh`, with FAST explicitly selected and the Apple column.
That script accepts the experiment flags through `MOJOLEARN_MAMBA_DEFINES`.
It defaults to IDENTICAL, so failing to set FAST would leave these candidates
inactive. A script that builds a binary may also perform checks; none was run
here. Mixed Samba public callers require the relevant neural sibling bindings
and the same frozen configuration, not a standalone Mamba component binary.

Future full-workload recipes start from the Mamba forward/inference and Samba
forward/training definitions in `tools/bench_board_neural.py`. Audit actual
caps, sequence corpus/version/hash, dimensions, model settings, seeds, steps,
numeric mode and call-site reachability before timing. Training applicability
must be established for each selected block and route; an inference-only
cell does not cover its training consumer. Missing full-dataset coverage
remains pending. The component fixtures and historical F10 ratios preserved
in source are not substitutes.

Each future whole operation must include required preparation, upload,
forward/training, synchronization and output consumption. Separate cold,
repeated, continuation and training use. Run all affected Mamba families for
refusal and arena changes, plus each applicable complete Samba configuration.
Quality gates must retain the existing tolerances and include output/logit
error, held-out loss, training trajectories, backward consumers and every
returned recurrent/convolution/angle state. Changing bits is permitted;
degrading quality is not.

Later edge coverage should include empty and shorter-than-chunk sequences,
ragged ends, neighboring supported dimensions, a non-board dataset, long
recurrence, continuation from nonzero state, buffer resize/lifetime and
failed-call rollback. Refusal coverage must place NaNs/infinities in each
vector lane and scalar tail, after the first grid pass, and in multiple
names/indices to establish unchanged precedence. These are pending work,
not claims that source reasoning has verified behavior.

No build/test/GPU evidence files exist for this lane because those actions
were expressly forbidden. The retained deliverables are the source edits,
the idea list, this note and the machine-readable recipe file. Any future
validation must retain full logs separately and report exit status, coverage,
failures and evidence paths without dumping logs into context.
