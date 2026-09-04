# The three-vendor board, 2026-08-28: what is bitwise identical, and what everything costs

One day, three columns, two questions kept apart on purpose.

* **IDENTICAL** is the `-D MOJOLEARN_NUMERIC_IDENTICAL=1` build. The question
  is whether the same source produces THE SAME BITS on Apple, NVIDIA and AMD.
* **FAST** is the default build. It promises SPEED ONLY. `mojo_only/numerics.mojo`
  documents it as "per-vendor speed everywhere, and results that move in the
  last bits RUN TO RUN", and `gemm/IDENTICAL_FP32_CONTRACT.md` clause 4 says
  the profile "does not promise that IDENTICAL equals FAST, on any vendor".
  **No timing on this page is an identity claim and no identity result on this
  page is a timing claim.**

Hardware: Apple M4 (this desk), NVIDIA H100 80GB SXM (RunPod), AMD Instinct
MI325X (DigitalOcean, `tor1`).

---

# PART 1 — BITWISE IDENTITY

## 1.1 What IS bit-identical, measured today

Identity round E3 **round 13**, commit `a0a0eee`, all three columns recording
that same commit and each reporting `identical set: 10 of 10 binaries`.

| family | cells / stages | Apple ↔ NVIDIA | Apple ↔ AMD | NVIDIA ↔ AMD |
|---|---|---|---|---|
| decision trees (E2) | 116 cells | 109 IDENTICAL, 2 no-card, 5 refused alike | same | — |
| unsupervised + linear (E2U) | 93 cells | 71 IDENTICAL, 22 refused alike | same | — |
| E1U cards (k-means 77, k-NN 6, DBSCAN 3) | 86 stages | IDENTICAL | IDENTICAL | IDENTICAL |
| phase-8 lanes | 358 stages | IDENTICAL | IDENTICAL | IDENTICAL |

**Zero divergent cells or stages anywhere.**

The nine phase-8 lanes and their card sizes:

| lane | stages | | lane | stages |
|---|---|---|---|---|
| isolation forest | 123 | | cd (coordinate descent) | 20 |
| metrics | 61 | | mamba (SSM block) | 17 |
| gemm | 60 | | linkage (single-linkage) | 8 |
| svm | 32 | | kde | 7 |
| transformer (Llama decoder layer) | 30 | | | |

The E2 tree matrix's 116 cells are gbdt 79 (losses, bootstraps, scorers,
estimators, searchers, bin counts, categoricals, NaN policy, plus 4 depthwise
and 4 lossguide), extra trees 19, random forest 18.

**NVIDIA ↔ AMD was diffed DIRECTLY for the first time.** Every previous round
took Apple as the reference and compared outward, so the two vendor columns
had never been put against each other. They agree on all 358 phase-8 stages
and all 86 E1U stages. Transitivity over exact hashes made that predictable;
it had never been checked.

## 1.2 What is NOT covered, stated plainly

| not covered | why |
|---|---|
| **train-here-infer-there this round** | `no cross_infer file` on both boxes — the legs did not carry the Mac's models. The 106/106 result stands from E3 round 8 at `fe00e8a`, not from today. |
| **transformer clauses (b), (c)** | opt-in in the driver, never yet run in a round. |
| **transformer clause (e)** | the section 8 planted audit ABORTS the driver on its first plant: `LlamaDeviceWeights` refuses at UPLOAD while the clause's `try` wraps only the later forward call. A defect in the clause, not in the block. Deliberately NOT allowlisted. |
| **mamba's FAST arm** | has never been built on any vendor. Documented at the site; the judge allowlists it BY NAME. |
| **phase 6 linalg IDENTICAL pass** | a finding on all three columns. Open. |
| **anything outside a card** | a matching hash means the buffers agreed AT EACH CHECKPOINT, not that the computation was identical, and nothing is known about what the trace did not checkpoint. This is `identity_trace_diff.py`'s own stated limit and it is inherited whole. |

## 1.3 What the flag actually buys, measured on the same box in the same run

| probe | FAST | IDENTICAL |
|---|---|---|
| gemm vs its oracle, Apple M4 | 96 OK, **31 shapes differ** | **128 OK, 0 differ** |
| gemm vs its oracle, AMD MI325X | 87 OK, **41 shapes differ** | **128 OK, 0 differ** |
| transformer clause (d), decode == prefill | **FAILS** — 91 stage-tokens, first at token 0 `q_proj.out` on 26 of 32 cells | **PASSES** — 4 decode steps bit-identical on all 11,632 compared cells |
| k-NN tie set, one box, one fixture, three runs | **three different sorted index sets** | one |

The two vendors' FAST GEMM misses the oracle on *different* shape sets (31 vs
41). None of those are defects: under FAST both sides are the unpinned
spelling and the check says so at the seam — "a measurement and not an
assertion".

## 1.4 The one thing that failed, and it is NOT a FAST arithmetic wobble

The judge's verdict for round 13 is **NOT-CLOSED**, on one unexpected finding
present on NVIDIA and AMD and absent on Apple:

    gemm [fast] check FAILED: gemm_device_check [FAST]: 1 of 6 gates FAILED

That gate is `check_device_is_batch_invariant` — a cell whose BITS depend on
how many OTHER cells shared the launch. It is asserted in BOTH modes on
purpose, and the file's own header gives the reason: launch and batch
invariance "are properties of the kernel's SHAPE — no float crosses a thread
boundary, and the leaf partition comes from `k` alone — not of the arithmetic
pins, so FAST has no excuse for failing them and a FAST failure is a real
defect."

**OPEN.** It holds on Apple under FAST and fails on both other vendors.

---

# PART 2 — FAST TIMINGS AGAINST PEERS

## 2.1 NVIDIA H100 — the only vendor where every opponent is legal

**Classical, 21 lanes vs cuML / cuVS / torch — 13 wins, 4 losses, 6 no opponent**

| faster | | slower | |
|---|---|---|---|
| metrics | **18.37x** | knn | 2.76x |
| ols | **13.89x** | cholesky | 2.11x |
| pca | **12.90x** | kmeans | 1.29x |
| krr | 6.08x | holtwinters | 1.27x |
| hdbscan | 3.80x | | |
| svm | 3.65x | | |
| kpss | 3.35x | | |
| linkage | 2.41x | | |
| ivf | 2.04x | | |
| kde | 1.89x | | |
| dbscan | 1.60x | | |
| cd | 1.13x | | |

gmm, gp, nystroem, rbfsampler, resample and spectral have no RAPIDS opponent
at all; the table says so rather than substituting one.

**Trees, higgs 1M / 2M, 5 rounds**

| lane | opponent | 1M | 2M |
|---|---|---|---|
| gbdt-symmetric | catboost-gpu | **1.01x FASTER** | 1.32x slower |
| gbdt-depthwise | xgboost-gpu | 1.86x | 1.90x |
| gbdt-depthwise | catboost-gpu | 1.21x | 1.66x |
| gbdt-lossguide | xgboost-gpu | 2.27x | 2.27x |
| gbdt-lossguide | catboost-gpu | 1.14x | 1.44x |
| rf | cuml-rf-gpu | 1.67x | 1.64x |
| iforest | cuml-iforest-gpu | 2.73x | — |
| et | lightgbm-cuda | **36.69x FASTER** | **30.53x FASTER** |

The Aug 28 round moved every rung: depthwise 2M went 2.73x → **1.90x** behind
xgboost-gpu and 1.94x → **1.66x** behind catboost-gpu; lossguide 2.83x →
**2.27x**; rf 1.81/1.71 → **1.67/1.64x**. gbdt-symmetric meeting CatBoost's own
GPU learner at 1M is a first.

**gemmseq, 130 comparison rows, 55 wins / 75 losses.** attention wins every
arm at the small lane shape (1.04-1.40x) and loses 1.5-4.2x at the llama8b
shapes; gemm is 1.53x behind cuBLAS-fp32 and 2.47x behind cuBLAS-tf32.

## 2.2 Apple M4 — our GPU against the only arms that run here

**Trees, shipped fixtures, 3 rounds**

| lane | fixture | opponent | verdict |
|---|---|---|---|
| rf | covtype | lightgbm-cpu | **16.89x FASTER** |
| gbdt-symmetric | year | catboost-cpu | **1.14x FASTER** |
| et | covtype | sklearn-et-cpu | **1.12x FASTER** |
| et | covtype | lightgbm-cpu | 1.48x slower |
| gbdt-depthwise | year | catboost-cpu | 1.08x slower |
| rf | covtype | sklearn-rf-cpu | 1.23x slower |
| gbdt-lossguide | year | catboost-cpu | 1.47x slower |
| iforest | anomaly | sklearn-iforest-cpu | 1.49x slower |
| gbdt-lossguide | year | lightgbm-cpu | 2.23x slower |

**gemmseq vs torch on MPS, 116 rows** — rows that did not exist before today,
because the seq torch arm could not see Apple's GPU:

| lane | best case | worst case |
|---|---|---|
| rmsnorm | **1.85x FASTER** | 14.48x slower |
| gemm | **1.08x FASTER** | 2.62x slower |
| mlp | 1.05x slower | 2.31x slower |
| attention | 1.86x slower | 13.27x slower |
| transformer | 2.17x slower | 8.02x slower |

Competitive at the small lane fixture, falling away badly at the llama8b
shapes, worst against torch's flash and SDPA paths.

**classical: 6 of 21 lanes have a legal Apple opponent.** resample is 4.58x
FASTER than scipy; the other five are FIXED-COST rows at 24x2..48x4 fixtures
where sklearn's launch-free CPU path wins 12-29x — a statement about per-call
overhead at those shapes, not about kernels. The other fifteen lanes' opponents
are cuML and cuVS, both CUDA-only, and the table says NO OPPONENT ON THIS BOX.
The comparison those lanes want is the 2026-08-20 algorithm-matched board: at
4M x 32 we are 4.29x faster than sklearn on ols, 3.52x on pca, 2.71x on kmeans.

## 2.3 AMD MI325X — no legal opponent exists, so this column is absolutes

Every GPU library benchmarked here — cuBLAS, cuML, cuVS, CatBoost-GPU,
XGBoost-GPU, LightGBM-CUDA — is CUDA-only. The leg runs OURS ONLY by design
and each opponent arm prints its own `FSPEED-REFUSED` rather than falling back
to sklearn on the host CPU. What the column gives is our arm's milliseconds on
a third vendor beside the same arm elsewhere.

| lane | shape | NVIDIA H100 | AMD MI325X | |
|---|---|---|---|---|
| rf | higgs 1M | 5309 ms | **4525 ms** | AMD 1.17x faster |
| rf | higgs 2M | 7308 ms | **6268 ms** | AMD 1.17x faster |
| et | higgs 1M | **4160 ms** | 18294 ms at 4f6a17a; 61772 ms at 9c8ffc23 (DEVIATION 1945) | AMD **4.40x slower** on the row's own source; see below |
| et | higgs 2M | **7661 ms** | 35521 ms at 4f6a17a; 88751 ms at 9c8ffc23 (DEVIATION 1945) | AMD **4.64x slower** on the row's own source; see below |
| iforest | anomaly 500k | 232 ms | **155 ms** | AMD 1.50x faster (Apple 220 ms) |

Random forest and isolation forest are FASTER on the MI325X than on the H100.
Extra trees was profiled on 2026-08-29 (`extratrees/DEVIATIONS.md` 1943 and
1945, legs `e1/2026-08-29_202227-mojolearn-e2-amd` and
`e1/2026-08-29_204736-mojolearn-e2-amd`, `lanes/et_profile/`). Two of the
nineteen kernels, the range and score passes, held 99% of the device time:
17.0 s at 1M with cuML's 128-thread block, one row per thread, a two-wavefront
workgroup on CDNA that leaves the MI325X dispatch-bound. DEVIATION 1943 keys
the block width on `WARP_SIZE` (512 on a 64-lane wavefront, 128 elsewhere),
and the device time at 1M is 6.2 s at 9c8ffc23 with every identical-tier ET
hash unchanged. The whole-fit AMD number did not follow it: the same shipping
surface now spends 53.9 s at 1M in the host-side `NodeQueue.push`, a cost the
18294 ms row cannot have contained, on push code that has not changed since
4f6a17a. That is DEVIATION 1945, OPEN, and until it is closed this column's
ET rows are two measurements that disagree, not a verdict.

**The three gbdt lanes are absent from this board because the AMD FAST gbdt
SPEED rows are unrun.** FAST gradient boosting builds and runs on the MI325X
since DEVIATIONS 1906 and 1910 (commits `f853e8df` and `19b319c7`;
`bench/results/e1/2026-08-29_093711-mojolearn-e2-amd/p9_fast_build_gbdt.log`
and `stability/fast.txt`, gbdt lanes STABLE 6/6).

---

# PART 3 — WHAT THIS ROUND COST TO GET

Nine defects, and the pattern across them matters more than any one: with a
single exception they were all SILENT. None of them made anything go red.

| # | defect | how it hid |
|---|---|---|
| 1 | `gemm_nt_gram`'s IDENTICAL arm has not compiled since 2026-08-25 (DEVIATION 1873): an immutable pointer handed to a `MutPointer` kernel | a failed binding build is a PHASE3-FINDING, and a finding is not an abort — so phases 3, 4, 6-identical and 7 stopped running on EVERY column for three days while rounds reported green |
| 2 | phase 3 built 5 identical bindings; `_backend._MODULES` lists 10 | the other five became `_MissingIdentical` stubs — the designed graceful degradation — until `_solver_impl.py`'s import-time guard, which a stub cannot pass, turned one missing binding into an unimportable package |
| 3 | `e2_matrix_fit.py` / `e1_traced_fit.py` / `e2u_matrix_fit.py` died on `git rev-parse HEAD` | the RunPod leg ships a `git archive` — no `.git` on the box. A provenance STRING killed the NVIDIA matrix. The DigitalOcean leg ships a bundle and clones it, so the two legs disagreed about something neither measures |
| 4 | the phase-8 finding line quoted an expected FAST oracle REPORT as the cause of a real failure | it made the FAST GEMM look like it was drifting from an oracle it never promised to match, while hiding an actual batch-invariance defect |
| 5 | `iforest` built a 123-stage card and wrote it to a scratch path nobody collects | its own gate went green nine checks at a time while the lane had zero cells in every round |
| 6 | `transformer` was never listed in phase 8 | its SCOPE line said "nothing cross-vendor until a leg runs" and no leg could run |
| 7 | `einops` imported at module scope in `mamba/corpus/gen_corpus.py` for five lines inside one function | took the attention, mlp, rmsnorm, transformer and gemm torch arms down with it; the Apple gemmseq board came home NO OPPONENT on every row |
| 8 | `speed_torch_seq.py` refused any box without CUDA | six Apple lanes had no peer, while its sibling arm in the same family has always driven `torch.mps` |
| 9 | the AMD speed leg had no dataset download step | every higgs rung refused and every lane fell back to a synthetic fixture — 60 REAL timed rounds, none on the dataset the comparison needs |

Numbers 5, 6, 8 and 9 all produce boards full of `NO OPPONENT` or plausible
rows rather than errors. That is the failure mode this harness has to be
watched for: it reads as a result.

---

## Provenance

Identity: `archive/evidence/E3_RESULTS.md` round 13, `bench/results/e1/2026-08-28_130918-MacBook-Air-1-terrabyte`
(Apple), `2026-08-28_131651-runpod-nvidia`, `2026-08-28_173933-mojolearn-e2-amd`.
Judge: `bash tools/e3_round_judge.sh <apple> <nvidia> <amd>`.

Timings: `bench/results/fast_speed/2026-08-28-*.md`, built by
`tools/fast_speed_table.py` from the fetched arm logs. Every `ours` header on
every board reads `mode=FAST`; `ours_headers_identical` is 0 on every leg.
