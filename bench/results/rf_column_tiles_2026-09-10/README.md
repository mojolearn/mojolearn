# Bounded RF timing follow-up — September 10, 2026

No timing window passed the pre-existing canary max/min limit of 1.1.
RF column tiles remain opt-in on every vendor and numeric mode. The raw
medians below are diagnostic observations, not established speedups.

| Mode | Attempt | Reference ms | Tile2 ms | Tile4 ms | Canary max/min | Timing gate |
|---|---|---:|---:|---:|---:|---|
| fast | attempt1 | 1189.6 | 1028.2 | 1015.4 | 1.148 | invalid |
| identical | attempt1 | 1852.2 | 1576.8 | 1669.0 | 2.447 | invalid |
| fast | attempt2 | 1786.3 | 1477.0 | 1570.7 | 2.158 | invalid |
| identical | attempt2 | 1751.2 | 1490.8 | 1407.9 | 1.725 | invalid |

Both attempts per mode completed under the shared build lock. No remote
jobs were used and no other processes were interrupted. The initial process
scan found no active benchmark/build job. A later command-name-only process
snapshot records substantial system/application CPU activity; it does not
establish the cause of the observed timing drift.

Each attempt measured a synthetic binary 262,145-row, 28-column input,
20 trees, depth limit 12, 128 bins. Each binary first completed a discarded
five-fit invocation. Measurement then ran reference/tile2/tile4/tile4/tile2/
reference; each process ran five canary warmups and five fits, discarding
its first fit. Thus each arm contributes eight retained fit samples per
attempt. All measured mode/tile readbacks and 120 full-model fingerprints
passed; candidate and reference hashes match. No prediction gate was added
here; the prior September 9 complete-model/prediction gates remain unchanged.

Six fresh binaries were built for attempt1; attempt2 reused those exact
SHA256-verified executables. Executables remain under
`build/rf-column-tiles-sep10/`, outside versioned evidence. Per-attempt
provenance records binary hashes, git head and command options. Source
hashes and toolchain version accompany the raw build, warmup and timing logs.
The final retry remained noisy, so the bounded experiment stopped.

Reproduction (change output directory to preserve these samples):

```sh
tools/with_build_lock.sh python3 ensemble/bench/rf_column_tiles_bench.py \
  --mode fast --rows 262145 --prewarm \
  --output-dir bench/results/rf-column-tiles-new/attempt1 \
  --binary-dir build/rf-column-tiles-new
```

Use `--mode identical` for the second mode; add `--skip-build` with a new
output directory for a retry against the same binaries. The harness always
retains raw timing data; successful execution does not imply a valid timing
window. Read `timing_valid` and the canary spread in each JSON summary.
