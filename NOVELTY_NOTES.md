# Candidate-novel work and findings, with evidence and honesty flags

Drawn up 2026-08-21 on request. RULES OF THIS FILE: every entry carries
its evidence pointer and a PRIOR-ART flag -- nothing here is a claim
until the literature check runs. Portability itself is NEVER claimed
(inherited from Mojo, standing rule). The two paper repos
(bitwise-gbdt, mlsys) draw from this; this file is the feeder, not the
paper.

## A. Likely novel results (check prior art before claiming)

1. **Cross-vendor bit-identity as a designed property of GBDT training,
   with a complete order-dependence enumeration.** The IDENTITY_PATHS
   discipline: every float pathway classified PIN / REPLACE / REFUSE,
   guarded by checks. Incumbents' own positions are collected: CatBoost
   tags its GPU non_deterministic, LightGBM's deterministic flag is
   CPU-only, XGBoost scopes reproducibility to same-platform, and
   nobody compares trained tree MODELS across devices. Evidence:
   IDENTITY_PATHS.md, the frozen identity floor, `check-ieee-arith`.

2. **The cross-vendor hazard for tree learners is DENORMAL POLICY, not
   fast-math.** Measured 2026-08-21 on Metal through MAX
   (`mojo_only/ieee_arith_check.mojo`, 2^20 hashed patterns): division
   and sqrt correctly rounded on every normal input, ZERO fast-math
   substitution, no FMA contraction -- and 100.0% of divergence is
   documented flush-to-zero, operands, intermediates and results. CUDA
   default honors denormals; Metal flushes. A GBDT identity design
   therefore needs a source-level denormal policy, and nothing else
   from the "GPU float folklore" list. We have not seen this stated
   anywhere for tree learners.

3. **Exact sibling subtraction as an instrument, not just an
   optimization.** With the Int32 fixed-point histogram, sibling
   subtraction is EXACT integer arithmetic, so subtraction-on vs
   subtraction-off must agree bit-for-bit -- turning a correctness
   property into a one-run REACH PROOF for the biggest optimization in
   the learner (measured: 1.96x, mse bit-identical,
   SHAPE_SWEEP_2026-08-21). Using bit-equality contracts as
   sabotage-grade instrumentation appears methodologically fresh.

4. **The four-axes floor-amortization law, measured.** On unified-memory
   consumer silicon, GPU-vs-CPU GBDT outcome is decided by one constant
   (the per-tree launch/drain floor, a platform price) against four
   independent work axes -- rows, features, depth, CLASSES -- each
   measured to move the ratio monotonically our way (parity at 800k x
   100 -> 2-4x wins on every enlarged axis; covtype MultiClass closes a
   2x RMSE deficit to 3-8% with zero code change). The crossover is
   arithmetic, not mystery. Evidence: EPSILON_*, COVTYPE_MULTICLASS_*,
   PERF decompositions.

5. **The density cliff, quantified per level, with its fixes refuted.**
   Indexed histogram reads degrade 11.5 -> 27.8 ms/level at CONSTANT
   built rows as leaves fragment cache lines (depth differencing), the
   M4's indexed-read amplification curve is measured
   (104 -> 5.6 GB/s useful across densities 1 -> 1/64), and both
   candidate remedies (their GatherBins arm; level compaction) are
   refuted by the same numbers on this device. A micro-architecture
   result GBDT papers do not usually carry. Evidence:
   SHAPE_SWEEP_2026-08-21, `mojo_only/density_probe.mojo`.

10. **A measured, bit-exact model of a GPU's denormal behavior as the
   portability construction itself.** Instead of designing a denormal
   policy, the policy was MEASURED: a source-level flush-op-flush model
   reproduced all 53,041 observed Metal divergences bit for bit across
   2^20 hashed patterns (`check-ieee-arith`'s ftz-model arm), and that
   model IS the IDENTICAL-mode helper (`numerics.ftz`) -- provably
   inert on FTZ hardware, alignment on denormal-honoring hardware. We
   are not aware of prior work deriving a cross-vendor float contract
   for a training system by bit-exact behavioral modeling of one
   vendor. PRIOR-ART CHECK NEEDED.

## B. Methodological contributions (novel-ish practice, not theorems)

6. **A benchmark protocol that catches its own lies.** Both arms
   alternate in one process; every rep prints loss parity; cross-window
   comparisons are refused outright. It has now caught THREE false
   findings that conventional benchmarking would have shipped: a 1.79x
   "win" that was 1.31x interleaved, a +30-50 ms "Bayesian overhead"
   that was thermal, and the loaded-window epsilon medians that
   flattered us because the CPU arm suffers memory pressure more than
   the GPU arm. Evidence: results files' correction addenda -- the
   protocol's paper section writes itself from them.

7. **Oracle-compiled gates for ported numerics**: compiling the
   incumbent's own source (their MT19937-64, their CityHash 1.0 -- a
   variant public vectors would MISgate) as the test oracle for a port,
   including the (ui64)(int) sign-extension chain. The CityHash gate
   caught a real Mojo cast-chain zero-extension on first contact.

## C. Toolchain findings (report upstream, never paper claims)

8. **The basename lottery, now with a mechanism profile.** Mojo 1.0
   AOT GPU-kernel emission for a CPython shared-lib tracks the SOURCE
   FILE'S BASENAME in stable name-keyed buckets: `copyml*` names emit
   73 of ~100 GBDT kernels (invariant under 12 content paddings), 30+
   other names emit 0, and a GBDT-only split emits 17 -- so emission is
   not a budget, not content-hashed, not monotone in module size.
   Current source exceeds every known bucket: the Python extension is
   unbuildable at HEAD by any local means. Reproducer data ready for a
   Modular issue (third in the series after #6932/#6933).

9. The Mojo numeric-trap family (recorded across sessions, all
   measured): `String(float)` 1-ulp round-trip failures, `log` 5e-8
   absolute error re-deciding DP ties, cross-expression FMA
   contraction, SIMD cast chains zero-extending where C++ sign-extends.
   The pattern -- "assume stdlib numerics approximate until measured
   against an external oracle" -- is a transferable discipline.

## D. Explicitly NOT novel (kept here so nobody re-claims them)

* Portability across vendors: inherited from Mojo/MAX, never ours.
* GPU GBDT training itself, histograms, symmetric trees: CatBoost's.
* The prep-bill finding (train()'s 24 s quantization): an engineering
  hole, interesting only for how it hid -- every benchmark correctly
  quantized outside the timed region, so the user-facing path was the
  one place nobody timed.
