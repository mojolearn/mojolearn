# Current-main depthwise bridge/scan experiment

Baseline: `61ea51757` (fresh origin/main, 2026-10-04).
Branch: `lane/apple-fast-dwcurrent`.

`MOJOLEARN_GBDT_DW_BRIDGE_SCAN` is opt-in, FAST + Apple, Depthwise only,
quantized one-byte/two-stat layouts, non-root levels. A parallel GPU kernel
reads and clears dense integer accumulators, performs the existing bridge
conversion, and performs the same serial Float32 fold per feature in shared
memory. Sixteen features per threadgroup, 256 threads, 16 KiB shared memory.
It removes one kernel launch and the float histogram write/read intermediate
per non-root level. Root mode-bin selection continues to consume raw bins.
One-hot features and zero/one-fold features retain the original semantics.
No host fallback, fitted-value change or bit change is intended; FAST run
nondeterminism still requires the actual prediction quality gate.

Manager builds/times; no local compilation or tests were run. `git diff
--check` is clean. Required bindings: FAST base and gbdt. Build both arms
from this branch, arm A no define (identical source behavior to baseline),
arm B `-D MOJOLEARN_GBDT_DW_BRIDGE_SCAN`. Build entrypoint is
`bindings/build_gbdt.sh`, with `MOJOLEARN_NUMERIC_MODE=fast`,
`MOJOLEARN_EXTRA_DEFINES` as above and `MOJOLEARN_COMPILE_JOBS=1`.

Exact proposed M3 CMD (not queued by the lane):

```sh
AFT_OUT="$HOME/aft-ab/dwcurrent-taxi" MOJOLEARN_COMPILE_JOBS=1 bash tools/aft_ab.sh gbdt gbdt-depthwise taxi 1 "" "-D MOJOLEARN_GBDT_DW_BRIDGE_SCAN"
```

If taxi passes, verify the other affected board row, reusing the same
compiled arms and still one timing run per arm:

```sh
AFT_OUT="$HOME/aft-ab/dwcurrent-taxi" AFT_SKIP_BUILD=1 MOJOLEARN_COMPILE_JOBS=1 bash tools/aft_ab.sh gbdt gbdt-depthwise istella 1 "" "-D MOJOLEARN_GBDT_DW_BRIDGE_SCAN"
```

The driver's `FSPEED-ACC` reports **held-out prediction AUC and logloss**.
Accept only a meaningful speed gain with neither metric declining beyond
observed baseline noise; any material AUC loss or logloss increase is a hold.
Taxicab historical depthwise AUC spread is about .6322-.6325, so changes at
that scale need manager judgment, not automatic approval from a single run.
Do not rerun opponents or enable this default before manager approval.

Prior options ruled out by the existing ledger: SM_X8 regressed, SM_X4 was
noise; SEG_SUMS_BLOCK -0.3%, FAST_FUSED_Q +1%, FAST_SKIP_FINAL_STATS -0.5%,
FAST_DEV_SCALE -0.3% were all dropped. FUSED_CHAIN, NO_LEVEL_SYNC, MODE_SKIP,
DW2_PART_VEC4 and DW2_SCAN_SMEM are already main defaults, so none is a new
current-main candidate. This experiment fuses the bridge with the scan;
it is distinct from the failed quantization-pass fusion.

CTR_INDEX_FUSED assessment: it is absent on current main. Its old-batch
implementation is in `origin/lane/apple-fast-batch` (`gbdt/ctrs/fast_prep.mojo`
and `gbdt/train.mojo`) and keeps CTR columns on the device through compressed
index construction. The historical 27240 -> 25566 ms with AUC .630808 ->
.630766 is promising but not current-main evidence. The switch also implies
CTR_FAST_PREP and routes the permutation-dependent calculation through
CtrPrepFast; this is not an isolated index-copy switch. A safe current-main
port requires the prep context, device-column lifetime wiring, uniform-border
min/max, and current permutation handling, rather than copying its define.
It also contains a non-Uniform border fallback that copies the whole column
to the host; a new GPU-only port must restrict the optimized path to supported
Uniform borders or supply a GPU border builder. No old code/default was
copied into this narrow depthwise candidate. CTR needs a separate current-main
A/B on gbdt-categorical taxicat, with both prediction AUC and logloss, before
any adoption; its prior 6.1% gain is not assumed to transfer.
