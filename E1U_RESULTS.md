# E1U RESULTS — cross-vendor bit-identity for k-means, k-NN and DBSCAN

**Claim demonstrated 2026-08-23**: at ONE commit, from ONE source tree,
under `NUMERIC_IDENTICAL`, with byte-identical inputs, an **Apple M4
(Metal)** and an **AMD MI300X (HIP/CDNA3)** produce **bit-identical outputs
AND bit-identical per-stage training certificates** for k-means, k-NN and
DBSCAN.

`E1_RESULTS.md` is the tree-ensemble half of this claim (ExtraTrees, Random
Forest, symmetric GBDT). This is the unsupervised half, which
`IDENTITY_PATHS.md` rows 19-26 opened on 2026-08-23 and which had **never
been to a second vendor** until this run — the Python bindings for `cluster/`,
`neighbors/` and `dbscan/` do not emit a per-stage certificate, so
`tools/e1_traced_fit.py` could not produce a card for them and none existed to
compare.

## Protocol

One driver (`tools/e1_unsupervised.sh`), one traced main
(`bench/unsupervised_trace_main.mojo`), one arm per process. Fixtures are
integer-exact functions of a constant seed — every coordinate is a small
integer over a power of two, exact in Float32 on any backend — and the input
hashes are printed **before** the fit and compared first.

| | Apple leg | AMD leg |
|---|---|---|
| device | Apple M4, Metal | AMD Instinct MI300X, ROCm 6.4.1 |
| column resolved | `1` / lane_width 32 | `3` / lane_width 64 |
| toolchain | Mojo 1.0.0 (`ed45d567`) | Mojo 1.0.0 (`ed45d567`) |
| commit | `3d0a842` | `.git` absent (tarball) |
| **mojo source sha256** | `665d04ebefbca0f4…` | `665d04ebefbca0f4…` |
| findings | 0 | 0 |
| artifacts | `bench/results/e1u/2026-08-23_073546-MacBook-Air-1-terrabyte/` | `bench/results/e1u/2026-08-23_113251-ac94b5a09591/` |

**The source hash is the commit-parity evidence, not the commit line.** The
repository is private, so the AMD box received a tarball and had no `.git`;
`commit.txt` there would have read "unknown", which is a belief about what
was shipped rather than a comparison. Hashing every `.mojo` file makes it a
measurement. `tools/e1_unsupervised.sh` now records it on every leg.

## The result

| arm | stages | Apple | AMD | card verdict |
|---|---|---|---|---|
| kmeans (4,096 x 8, k=8, 10 iters) | 77 | centroids `1636775608130736985`, labels `9543594618727214305` | same | **IDENTICAL, all 77 stages** |
| knn (4,096 index, 512 queries, k=10) | 6 | distances `9794987834769335813`, indices `16793617586120664034` | same | **IDENTICAL, all 6 stages** |
| dbscan (2,048 x 4) | 3 | labels `14389807861238588709` | same | **IDENTICAL, all 3 stages** |

86 matched stage pairs, zero divergences, and the caller-visible outputs
agree as well. Diffs in `bench/results/e1u/diff_apple_vs_amd_*.txt`.

**The k-NN row is the one that matters most.** Its fixture plants a
43-member tie class and puts half the queries exactly on the planted point,
so the answer is decided by the selector's tie rule and not by the
distances. Under FAST, that same fixture returned **three different sorted
index sets in three consecutive runs on one M4**
(`bench/results/column_invariance/2026-08-23_063334/RESULT.md`). Under
IDENTICAL it returns one value on two vendors.

## What the AMD leg measured that nothing else could

### 1. AMD contracts too — row 9's remaining justification narrows

    contraction: a*b+c is FUSED -- read off the 1629 patterns that
    SEPARATE the two spellings, never the tie-dominated totals

The tie-dominated totals in the same output still read `unfused 1048576
fused 0`, which is exactly the artifact IDENTITY_PATHS row 9 was corrected
for on 2026-08-23. **The 2026-08-22 AMD leg's "a\*b+c is UNFUSED on this
backend" was that artifact**, now superseded by a measurement off the
built-to-separate arm.

The consequence is that all three measured columns contract (1629/1629, `bootstrap.log`
of `bench/results/e1/2026-08-28_131651-runpod-nvidia`), so `identical_mul_add`
is bit-inert on **Apple, NVIDIA and AMD**. Row 9's remedy stands for a future
column that does not contract.

### 2. AMD honors denormals; Metal flushes — and the pins absorb it

    div 0 wrong; sqrt 0 wrong; L2 score shape 0 wrong; cosine shape 0 wrong
    ftz model: reproduces 0 of 0 div/sqrt divergences bit-for-bit
    ieee arith check OK: fully IEEE including denormals on this backend

Exactly what `E1_RUNBOOK.md` predicted for CDNA, and the reason
`numerics.ftz` exists. `0 of 0` is not a weak result here: it says there was
nothing to reproduce because this backend produced no divergences, where
Metal produced 53,041.

Note for the NVIDIA column: **AMD's `sqrt` is correctly rounded**, while the
E2 lane measured the H100's `std.math.sqrt` at 180,714 of 2^20 patterns one
ulp off (176,577 of them normal). Three vendors, two behaviours.

### 3. Row 12's certificate line holds on a second vendor

    translog device hash = 8705486125800438413

The same number Apple prints. `portable_expf` and `portable_logf` are both
max 1 ulp against the float64 reference on the MI300X.

### 4. Thirty-one gates green on AMD, including the ball cover

`check_dbscan_arms_agree_on_the_border OK (IDENTICAL): 1020 labels identical
through the ball-cover index and through brute force` — on a 64-wide
wavefront, which is what DEVIATION 515 was written for and is the only place
that fix could be validated.

## The defect this run found, and could only have found here

**DEVIATION 515.** The first build on the MI300X did not produce a wrong
number, it produced no number:

    LLVM ERROR: Cannot select: i32 = AMDGPUISD::SETCC <i1 CopyFromReg>, 0,
    setne   In function: neighbors_ported_neighbors_ba..._658cdd32df991420

`ball_cover/registers.mojo` asked for `vote[DType.uint32]` — a 32-bit ballot
of a 64-lane wavefront, which has no instruction. It aborted the whole
compile, so `cluster/` and `dbscan/` could not be built on AMD **at all**,
even though neither calls that kernel: one module, one codegen. The fix is
the one that file's own DEVIATION 2 banner had already prescribed for a
future vendor.

It also exposed a false pass: before 515 the constant was the literal 32, so
a local `-D MOJOLEARN_COLUMN_AMD=1` build silently compiled a *32-lane* ball
cover and the arms-agree check passed on it — a green check testing the
Apple kernel under an AMD header. `column_is_simulated()` now makes such a
build report NOT ANSWERABLE.

## What is NOT claimed

- **Three vendors, measured.** The H100 ran this leg the same day
  (`bench/results/e1/2026-08-28_131651-runpod-nvidia/e1u/`), and k-means,
  k-NN and DBSCAN are identical on all three columns from E3 round 8
  (`fe00e8a`) through round 13 (`a0a0eee`). `E1_RESULTS.md`'s own honesty
  note is why the third column mattered. For the tree ensembles, Apple↔AMD
  agreed through every stage while an H100 diverged at
  `tree001.winners.scores`. Two backends agreeing closes nothing.
- **One configuration per algorithm**, not the config matrix. k-means runs
  `INIT_ARRAY` (k-means++'s float scan is not exercised); k-NN runs the
  default arm at `k <= 64`; DBSCAN runs brute-force eps at the default
  memory budget.
- **Nothing is claimed for FAST**, which is deliberately per-vendor — and on
  this k-NN fixture is not even reproducible across runs on one device.
- The MI300X leg ran once. The Apple side was reproduced (three separate
  runs, plus a detached-worktree rebuild that came out byte-identical); the
  same cards have since reproduced on an MI325X in six E3 legs
  (`2026-08-23_132856`, `141817`, `150003`, `163350`, `172650` and
  `2026-08-28_173933` under `bench/results/e1/`).
