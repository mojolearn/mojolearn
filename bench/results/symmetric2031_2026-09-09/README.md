# Symmetric GBDT row-index-only split candidate — Apple M4

`MOJOLEARN_2031_SYM_RIDX_SPLITS` leaves gradient/statistic planes in document
order and permutes only row indices. This validation does **not** enable the
candidate by default: its largest tested workload regressed.

The native raw-input fit probe checks learned splits, every leaf value, bias,
quantization borders/fold counts, training-loss bits and predictions. All
four FAST workloads had equal fingerprints in all baseline/candidate fits.
Timed regions include raw-input native `train()` and device synchronization;
data generation, fingerprints and prediction materialization are untimed.
This is not a Python end-to-end timing.

Each case ran baseline/candidate/candidate/baseline, with one warmup plus
three measured fits per process. The shared build lock was held across the
entire sequence, and the benchmark lock identified the timing window.
Baseline pass medians drifted at most 2.25%.

| Rows / columns | Loss | Trees / depth / borders | Baseline ms | Candidate ms | Ratio |
|---|---|---|---:|---:|---:|
| 65537 / 8 | Logloss | 12 / 6 / 15 | 149.729 | 145.371 | 1.030x |
| 262145 / 28 | Logloss | 12 / 8 / 32 | 244.689 | 236.4195 | 1.035x |
| 262145 / 28 | RMSE | 12 / 8 / 64 | 149.088 | 144.8485 | 1.029x |
| 1000003 / 28 | Logloss | 8 / 8 / 32 | 334.8625 | 360.030 | 0.930x |

The million-row candidate takes approximately 7.5% longer. The smaller
2.9–3.5% gains do not justify globally enabling this schedule.

Reproduce by building `checks/gbdt_sym_ridx_fit_check.mojo` with and without
`-D MOJOLEARN_2031_SYM_RIDX_SPLITS=1`, then invoking
`tools/with_build_lock.sh python3 checks/gbdt_sym_ridx_ab.py BASELINE CANDIDATE OUTPUT`.
`summary.json` records the individual timings and complete parameter sets.

## NVIDIA single-pass partition

A pre-existing leased SSH endpoint refused the read-only connection attempt.
No remote process, service, training job or resource was changed. The user
subsequently restricted remote work to read-only diagnostics; no remote
validation was attempted after that instruction.

`checks/gbdt_large_partition_check.mojo` supplies a host-oracle check for
1,000,003-row, 777-row and empty leaves, permuted leaf-slot IDs, ragged
chunk tails, sentinel gaps, and all-zero/all-one/mixed flags. The launch goes
through the actual stable-partition router and prints whether the
single-pass path was reached. M4 uses the existing three-launch fallback;
a passing M4 run does **not** validate NVIDIA single-pass execution.
The large-leaf oracle passed all three patterns in FAST, DETERMINISTIC and
IDENTICAL on M4; every banner correctly reports `single_pass_reached False`.
The IDENTICAL and DETERMINISTIC builds carried the single-pass opt-in define
to ensure it does not escape the NVIDIA vendor gate. NVIDIA IDENTICAL
enablement remains opt-in and unvalidated on live hardware in this task.

Additional 65,537-row full-model checks passed in DETERMINISTIC and IDENTICAL
for baseline/candidate. Candidate readback is True in DETERMINISTIC and False
in IDENTICAL, as the existing route matrix requires. These checks are
correctness only; performance was measured only in FAST.
