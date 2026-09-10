# Performance and executable identity continuation — September 10

The previous pass was pushed through 2419895f before this work. This pass
adds exact performance improvements to Mamba3, kNN and transformer, completes
the missing embedding plan and log2 gates, and builds actual RDNA targets.
Tree implementations were not edited. FAST/DETERMINISTIC behavior is preserved;
this lane ran no FAST/DETERMINISTIC surface or kernel comparisons.

## Accepted performance changes

All before/after figures below are synchronized request medians from
the same physical H100 80GB HBM3, driver 580.126.09. They are own-versus-own
comparisons, not newly timed opponents. Each sequence shape is the original
seed-7 public fixture. kNN uses the existing dyadic-v1 fixture.

| Workload | Before ms | Final default ms | Less time |
|---|---:|---:|---:|
| Mamba3 B8/L4096/D512 |72.118117|65.868374|8.7%|
| Mamba3 B8/L1024/D2048 |126.097558|120.395977|4.5%|
| kNN400k index/4000 queries/D32/k10 |39.304458|35.473274|9.7%|
| kNN400k index/4000 queries/D32/k15 |44.808886|40.942684|8.6%|
| Transformer B8/L4096/D512, reversed order |255.330876|245.223133|4.0%|
| Transformer B8/L1024/D2048, reversed order |260.435883|245.318148|5.8%|

Mamba3 shares independently rounded K/V operands in the state-increment
kernel while preserving each ascending 64-term FMA chain and ragged zero
terms. NVIDIA defaults use the tile at or above 1,048,576 increment cells, with
256-thread/10 KiB capability guards; the threshold is an occupancy guard,
not a tuned crossover across unmeasured intermediate shapes. Tiny calls and
Apple retain the previous default. First forced-tile prices also win, by
7.2%/3.2%; final guard-enabled prices use five rounds. Native traces match,
2,752,512 direct increment cells pass on Apple/NVIDIA, and NVIDIA final
public gates pass 39,087,232 output/report cells, 102 surface checks and all
three complete output SHA256 values. Apple forced integration passes native
continuation/decode/refusal checks, 146,560 cells and 102 public checks.
Evidence: bench/results/mamba3/2026-09-10-increment-tile/.

kNN's NVIDIA register tile reuses each index load across eight query rows,
preserving every feature's FMA order. The unchanged logical-width extraction
is separately measured performance-neutral. Native exact oracle 396,584×3,
24 layouts,36 long selector cases, all 16 grid fingerprints, and final
flag-absent oracle/layout gates pass. Seven-round prices include both orders
for width extraction, broader D8/D32/D128 coverage, and final default rows.
D8 coverage is flat to 0.9% slower; this is a measured tradeoff. Apple retains
four rows and passes the final integrated logical-width gate. No prior
redux/chunk4/rounding-relaxation experiment was reintroduced.
Evidence: bench/results/knn/2026-09-10-logical-width/.

Transformer removes pinned host staging copies from IDENTICAL transfers,
with synchronization preserving caller lifetimes. All 82 full-array records,
including all backward gradients and unused cache bytes, agree on Apple and
NVIDIA. Both large complete-output SHA256 values remain unchanged. Final
flag-absent bindings pass on both platforms. Seven-round H100 timings in both
orders improve; Apple small-HD64 pairs improve consistently, HD128 host
samples vary. No arithmetic, tolerance or Torch comparator was changed.
Evidence: bench/results/transformer_transfer_2026-09-10/.

## Missing identity machinery now executes

Embedding PLAN_SORT packs unique(id,position)UInt64 keys, sorts them on
device, derives run arrays, and reuses the existing FP32 backward fold.
Clause 11(d) passes 102 fixture comparisons on each Apple/NVIDIA device plus
56 edge configurations × 2 plans × 3 geometries. An actual device reverse-tie
negative control fails by the expected metadata mismatch and is registered
in the sabotage banner. Apple/NVIDIA baseline cards match. The H100 shipped
V128256/D4096/T4096 gate compares every 525,336,576 gradient cell, counts,
begins and used permutation across both plans and 32/96/160-thread launches.
Scan remains the default. Single-call plan timings are diagnostic, not an
automatic crossover policy. The separate existing nonfinite W/dY refusal
gap remains open; clause (f) is not certified.
Run: pixi run check-embedding-plan-sort.
Evidence: bench/results/embedding_plan_sort_2026-09-10/.

portable_log2_64 now has its 262,144-hashed-input full-range gate, all 2,098
powers of two, 82 boundary neighbors and special values. Apple ARM64 and
Linux x86-64 both stay within 1 ULP of host libm and return the same result
hash 18138053008657378164. All powers are exact; the admitted bound remains
2 ULP, not a correctly-rounded-for-all-inputs claim.
Run: pixi run check-portable-log2-64.
Evidence: bench/results/portable_log2_2026-09-10/.

## Fourth-column scope

The exact UInt64 logical minimum now supports widths 1–128 with a shared
fallback when the physical subgroup width is unknown or insufficient.
Apple/NVIDIA native gates pass. Intel, Qualcomm, RDNA and spec-baseline policy
simulations pass on Apple, explicitly labelled simulations. Spec-baseline's
existing 128-thread/16 KiB limits are preserved; it is not admitted to the
library's stronger identity floor. RDNA detection now precedes generic AMD,
selecting 32 lanes rather than incorrectly assuming CDNA 64.

Actual compiler target builds pass for gfx1100 and gfx1201, including the
production kNN selector for gfx1100, with an assertion requiring a real
RDNA target rather than a simulated column. No RDNA physical device ran;
no Intel/Qualcomm backend or generic FP32-reduction certification is claimed.
The omitted item 3 was requested from the user; work proceeds with this
explicitly bounded integer-reduction prerequisite.
Run: pixi run check-lane-minimum.
Details: docs/lanes/KNN_LOGICAL_WIDTH_2026-09-10.md.

## References, provenance and remaining gaps

bench/OPPONENT_REFERENCE.md now contains all 16 latest own kNN rows, three
Mamba3 rows and transformer own-versus-own measurements with admission
status. Existing Torch/cuML prices were reused; no opponent was retimed.
Historical resident-device Torch versus our host-to-host Mamba3 scope remains
explicit. Residual ratios are about 4.02×/6.21× for Mamba3 and 3.47×/3.79×
for the main kNN rows. Transformer's original Torch comparison still fails
admission. Dense GEMM's prior 5.2–5.7× gap was not changed in this pass.

Raw logs retain original lane commit IDs. Equivalent changed-source commits
reachable from main are:

| Source | Lane commit | Main commit |
|---|---|---|
| Mamba3 initial tile |041bc46a|5fc7d4a1|
| Mamba3 final guarded default |eded801f|ba4e3257|
| kNN eight-row probe |8fdfd85f|edada38d|
| kNN final default |c03bbcfc|0e213d65|
| Embedding shipped gate |0250d63b|af63fef3|
| Embedding portable wrapper |c0d34cb0|b2b525b9|
| Transformer final default |f31508bb|f31508bb|
| Log2 final gate |fd71584a|fd71584a|

One guarded H100 pod, ujlg5ytky38cvv, served the serialized GPU jobs. It was
deleted at 05:33 EDT: DELETE 204, subsequent GET 404 verified. All owned evidence
was fetched before deletion. Cleanup logs and final integrated Apple lane
gate are in bench/results/performance_identity_2026-09-10/.
