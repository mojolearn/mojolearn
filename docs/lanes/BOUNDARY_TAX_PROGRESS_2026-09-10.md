# Boundary-tax implementation progress

Scope: WP0–WP5 and WP8. WP6 and WP7 are owned by another lane and are
excluded here. Follow the original brief for mechanisms and vendor gates.

## WP0: current-source attribution

Instrumented the ET binding's X/y List construction, native fit/context and
Python-object forest return; the dataset upload has its own nested timer.
The Python wrapper records the binding call and Array packing separately.
All clocks are behind `MOJOLEARN_STAGE_TIMES=1`, off by default. These host
clocks add no queue drains; the existing trainer's stage clocks do drain.

Reproduce using the current source's FAST trees extension:

```sh
MOJOLEARN_NUMERIC_MODE=fast nice -n 19 sh bindings/build_trees.sh
PYTHONPATH=python nice -n 19 python3 tools/bench_boundary_tax.py \
  --data /path/to/HIGGS.f32.npy --mode fast
```

The retained run is `bench/results/boundary_tax_2026-09-10/wp0_itemized_m4.txt`.
It uses the real HIGGS prefix, 1M rows × 28 features, 100 trees, depth 16,
and records source-data/model hashes. One warm-up and two alternating pairs
compare instrumentation on/off. This attributes current-source costs; it
does not recreate September 1's exact source or certify an optimization.

| Slice | Instrumented fit 1 ms | Instrumented fit 2 ms |
|---|---:|---:|
| X List construction | 17.177 | 34.455 |
| y List construction | 0.560 | 0.725 |
| Dataset allocation/staging/upload (inside native fit) | 113.470 | 100.832 |
| Native fit/context, including upload | 34375.894 | 34527.000 |
| Forest → Python objects | 1021.390 | 1024.048 |
| Python lists → Array | 1462.784 | 1463.458 |
| Native trainer phase total (inside native fit) | 34258.323 | 34422.141 |

The native binding remainder outside its trainer phase total is 1156.722
and 1164.109 ms. Python-object return accounts for 88.3% and 88.0% of that
remainder; X List staging 1.5%/3.0%, y staging under 0.1%, and dataset upload
9.8%/8.7%. Context/setup/rounding covers the small remainder. Python Array
packing is OUTSIDE the binding remainder, adding another 1.46s per fit.
These measurements support WP2's target without attributing the historical
1.9s difference to a mechanism that was not measured then.

Current trainer staging consumes 17.1–17.3s in these instrumented runs,
much more than the historical profile. WP8 must isolate its own contribution
with same-process A/B; this table alone cannot identify that whole cost.
All five fits have identical model hashes (1,823,474 nodes). Uninstrumented
measured fits take 35.628/35.734s (0.30% endpoint spread); instrumented fits
take 37.332/37.507s. This is two attribution pairs, not the five-pair
optimization gate. NVIDIA/AMD attribution remains owed. No competitor ratio is inferred here.

## Packages

- WP1a: committed as `630ae89b` (parent `ade09042`). ET borrows X and
  bulk-copies pinned storage through the existing trainer. Labels still use
  their original validation/quantization path and an O(rows) List; the removed
  X List was O(rows × features). Borrowed/List dataset bytes and complete
  classifier/regressor forests match with bootstrap on/off in all three modes
  on Metal. RF forwards the live X address to WP4 and bulk-copies inputs.
  Float64 direct-to-pinned fusion is a follow-on, with a benchmark prepared
  against the actual transpose implementation; a flat-cast proxy is not
  enough evidence. Binding integration and large vendor timing are separate.
- WP2: caller-buffer export is now the default, with typed native ownership
  and finally-release cleanup shared by RF/ET. All three Metal modes pass
  same-fit five-array and saved-NPZ byte equality for four estimators plus
  weighted RF classification (15 cases). HIGGS 1M × 28, 100 trees, 1,823,474
  nodes: five interleaved export-only pairs gave minimum 2394.039 ms for
  List export/Array packing and 3.663 ms for caller-buffer export/allocation,
  with 1.424% baseline endpoint drift and matching model hashes. This is not
  a whole-fit speedup claim. Evidence is `bench/results/boundary_tax_2026-09-10/wp2/`;
  CUDA/HIP gates and device-resident fit handoff remain follow-ons.
- WP3: the borrowed-pointer, reusable-I/O path remains the default. The six-cell
  Metal matrix passes FAST/DETERMINISTIC/IDENTICAL with separate-array and
  packed-sibling layouts. Complete RF/ET outputs match the retained List path
  at widths 1/2/3/5/8/9, including ragged forests, subnormal inputs and row/tree
  tails; workspace resize/reuse/empty/lifecycle and packed-leaf negative controls
  pass. Compiled mode/vendor/layout readbacks and source/binary hashes are in
  [WP3 evidence](../../bench/results/boundary_tax_2026-09-10/wp3/RESULTS.md).
  These are correctness checks, not new timing or cross-vendor qualification.
- WP4: device identity-row fill and optional borrowed-X OOB path pass native
  checks in FAST, DETERMINISTIC and IDENTICAL on Metal. Both input layouts,
  signed zero/subnormal inputs, row-count tails and full forest/OOB fingerprints
  are covered. IDENTICAL borrowed reads retain the device FTZ convention.
  Existing native callers may omit the address and keep the download path;
  WP1 wires the binding's caller-owned address. Weighted sampling is unchanged.
  Large RF OOB-on/off timing and NVIDIA/AMD execution remain owed; commands
  and source/artifact hashes are in the WP1/WP4 evidence directory.
- WP5: eight-slot flat-input cindex staging implemented, retaining eval/predict
  NaN refusal. FAST/DETERMINISTIC/IDENTICAL per-cell oracle and columns-twin
  gates pass at rows/features 1/19, 257/35, 8193/67, with ring wraps, skipped
  constants, mixed layouts, source-mutation reach, and late NaN refusal.
  Existing IDENTICAL NaN fit/predict/save-load integration passes (4000
  predictions; Min/Max negative control moves 3979 rows). See
  [WP5 evidence](../../bench/results/boundary_tax_2026-09-10/wp5/README.md).
  NVIDIA/AMD correctness and large narrow/wide NVIDIA timing remain owed;
  these small correctness fixtures establish no speed gain.
- WP8: vectorized full-capacity byte comparison and memcpy candidate prepared.
  Scattered-byte, vector-boundary, scalar-tail and actual skip checks pass on
  Metal FAST. Same-process large-fit scalar-reference/candidate timing remains
  required before committing the candidate default.
