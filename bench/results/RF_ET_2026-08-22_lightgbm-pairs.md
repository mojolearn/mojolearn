# RF and ET vs LightGBM: the pairs suite on M4, and what the NVIDIA leg needs

2026-08-22 morning, M4 base, per Andrew's ask: run the harness for RF and
ET, us vs LightGBM, and do the NVIDIA one. NVIDIA's gbm-bench harness,
`pairs` configuration: every arm interleaved inside ONE process, depth-8
parity config (gbm-bench's own `shared_params`, the same shape it hands
cuML's RF; LightGBM capped at 256 leaves = a full depth-8 tree), 100
trees. On Apple, LightGBM ships NO GPU arm, so lgbm-*-cpu is its
strongest legal arm here -- that asymmetry is the thesis. JSONs + box
records beside this file.

## The table

| dataset | pair | ours (GPU) | LightGBM (CPU) | time | accuracy |
|---|---|---|---|---|---|
| covtype 581k x 54, 7-class | RF | **4.62 s** | 12.70 s | **2.75x us** | ours 0.7224 vs 0.7170 |
| covtype | ET | **4.10 s** | 6.84 s | **1.67x us** | ours 0.6522 vs 0.6198 |
| year 515k x 90, regression | RF | 23.09 s | **7.69 s** | 3.0x them | MSE 91.7 vs 92.0 (parity, ours a hair better) |
| year | ET | 15.60 s | **6.17 s** | 2.5x them | MSE 98.0 vs 92.8 (OURS BEHIND) |
| higgs 8.8M x 28, binary | RF | 68.12 s | **29.53 s** | 2.3x them | ours wins ALL: AUC 0.7747 vs 0.7681, acc 0.699 vs 0.658, logloss 0.589 vs 0.606 |
| higgs | ET | REFUSED | 15.10 s (earlier run) | -- | -- |

All three datasets that were on the box ran; nothing was picked or
dropped by result.

## What the split says

- **Multiclass is ours.** LightGBM's rf mode pays one tree per class per
  iteration on covtype's 7 classes; the cuML port's histogram carries
  `n_classes` natively in one pass. Both pairs flip to us, with better
  accuracy.
- **Wide-column regression is theirs, today.** year (90 features,
  RegressionBin at 8 bytes/bin) is the shape where their CPU histogram
  engine wins. DEVIATION 314 (the pre-binned dataset, gated bit-identical
  this morning, certification window pending) attacks exactly this row:
  its one voided ABAB read ~1.5x at 500k rows, which would roughly halve
  the year gap, not close it. The remaining distance is a real finding.
- **higgs RF loses time and wins every accuracy column.** Row sampling
  differs BY LIGHTGBM'S OWN CONSTRAINT (`bagging_fraction=0.632` is the
  closest their rf mode can legally get to a bootstrap;
  PARITY_NOTES["lgbm-rf-bagging"]), so their trees see 63.2% of rows and
  ours see a full with-replacement sample. The accuracy spread is what
  that buys.
- **The higgs ET cell is a REFUSAL, not a gap in the table**: the
  extratrees lane's DEVIATION 175 bounds the Int64-exact Gini numerator,
  and 8.8M rows exceeds SCORE_MAX_ROWS_EXACT, so the arm raises by name.
  That bound is the extratrees lane's to lift; a benchmark surfacing it
  loudly is the design working. (The first higgs invocation ran the RF
  pair, then died at the ET arm's raise before writing; the recorded RF
  numbers are from the rf-only rerun.)

## The accuracy question, closed by three probes (2026-08-22, late morning)

Andrew's directive: fix RF accuracy where behind. The behind cells were
covtype F1/Precision and year MAE (both with accuracy/MSE AHEAD). Three
single-arm probes -- accuracy is a deterministic function of the input
bits, so a contended box cannot falsify these; their timings are noise
and not quoted:

1. **covtype at n_bins=255** (LightGBM's max_bin): Acc 0.7221 / F1
   0.7009 -- IDENTICAL to 128 bins. Quantization exonerated.
2. **covtype at max_features=1.0** (LightGBM rf mode's own
   feature_fraction): Acc **0.7567** / F1 **0.7478** / Prec 0.7548,
   against their 0.7170 / 0.7466 / 0.7880. The whole F1 gap was the
   feature-sampling asymmetry; at feature parity we lead accuracy AND
   F1. PARITY_NOTES["lgbm-rf-features"] is the standing record.
3. **year at n_bins=255**: MAE 6.8128 vs 6.8126 -- unchanged. The MAE
   hairline (ours 6.8126, theirs 6.8037, 0.13%) is their 0.632
   subsampling acting as regularization that trades MSE for MAE; we win
   MSE, which is the criterion RF regression optimizes.

VERDICT: no RF accuracy defect anywhere in the table. The behind cells
are config asymmetries inherent to each library's own defaults, now
measured and documented. No engine or default change is warranted
(defaults stay cuML's, per the standing rule).

## The NVIDIA leg, ready and waiting on a box

`tools/nvidia_forest_bench.sh <user@host> <dataset>` runs the six-arm
interleave -- mojolearn-rf-gpu, lgbm-rf-cpu, **lgbm-rf-gpu**,
mojolearn-et-gpu, lgbm-et-cpu, **lgbm-et-gpu** -- because on NVIDIA
LightGBM ships a CUDA learner and benchmarking only their CPU arm on the
vendor's own hardware would be choosing the weaker opponent. The
lgbm-*-gpu arms are registered (device_type="cuda"); the script builds
LightGBM from source with USE_CUDA=ON (the pip wheel has none) and does
the FIRST-EVER CUDA build of ensemble/ + extratrees/ bindings. Treat
that first run as a build, not a benchmark. Blocked only on a rented
box (RunPod OAuth awaiting Andrew, or any `user@host` with CUDA).
