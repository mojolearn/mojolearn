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

3. **Bit-equality contracts as an instrument, not just a gate.** The
   subtraction-on vs subtraction-off mse equality was used as a one-run
   REACH PROOF for the biggest optimization in the learner (measured:
   1.96x, mse bit-identical, SHAPE_SWEEP_2026-08-21). The original
   derivation -- Int32 fixed point makes the subtraction EXACT, so the
   arms MUST agree -- is FALSE (DEVIATION 136 / PORTING.md 136a: cells
   convert through `Float32(Int(q))/fixed_scale` BEFORE the float32
   subtraction, and derived cells measurably differ by up to 3 ulp), so
   the equality is an empirical outcome at the measured shapes, not a
   theorem. Using bit-equality contracts as sabotage-grade
   instrumentation still appears methodologically fresh, but the claim
   must be stated as a per-shape measurement.

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

11. **The cost of a determinism contract derived FROM the contract,
   rather than measured as a slowdown.** Because the profiles are frozen
   versioned documents that enumerate every seam, the identity tax is
   countable by reading source: a transcendental term (ops in the
   portable polynomial times call sites per stage), a flush term (the
   17 enumerated seams' `ftz` insertions, provably ZERO on flushing
   hardware), a fold-depth term (the profile's tree against the native
   warp primitive), a refused-hardware term (datasheet peaks), and an
   occupancy term (the pinned launch geometry). The prediction is
   FALSIFIABLE: `tools/lanes_price.sh` becomes the falsification of the
   static estimate rather than its source, and a disagreement is a
   finding about the toolchain. Needs no measurement, which is why it
   can proceed under the standing measurement freeze. Evidence and
   method: IDENTITY_COST_PLAN.md Part 1. PRIOR-ART CHECK NEEDED, and
   the check is narrow: has anyone derived determinism cost from a
   numerical contract instead of reporting a measured slowdown?

12. **Cross-vendor cost comparison as a controlled counterfactual.**
   Under a certified identity contract the step count is shared BY
   CONSTRUCTION, so the "does the other device reach the same outcome"
   term is not modeled, it is eliminated, and choosing hardware becomes
   arithmetic over a measured per-step time. Two corollaries with no
   incumbent: mid-run migration between vendors without perturbing the
   trajectory (spot arbitrage DURING a run), and a cheaper run that
   carries stage-by-stage evidence it performed the identical
   computation, which is a compliance object and not only an
   optimization. HEAVILY CAVEATED: both need identity closed for
   TRAINING (backward, optimizer state, RNG, data order, checkpoint
   serialization, collectives) and today the repo has classical lanes
   plus ONE forward-only Mamba-1 block. Evidence: IDENTITY_COST_PLAN.md
   Parts 2 and 3. PRIOR-ART CHECK NEEDED.

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
   THE PROTOCOL'S OWN FAILURE MODE, found 2026-08-21 (PREP_BILL step
   7): when the two arms' combined resident set crowds physical RAM,
   the shared process inflates BOTH arms (ours 3.5-4x, theirs 2-3x at
   200k x 2000 on 16 GB) and the interleaved ratio is garbage --
   pageouts barely move, so the mechanism is compression pressure and
   a pageout check does not clear the window. Validity condition:
   combined footprint well under RAM; otherwise isolated per-arm runs
   with the cross-window caveat stated. A protocol whose validity
   envelope is measured and stated is part of the contribution.

7. **Oracle-compiled gates for ported numerics**: compiling the
   incumbent's own source (their MT19937-64, their CityHash 1.0 -- a
   variant public vectors would MISgate) as the test oracle for a port,
   including the (ui64)(int) sign-extension chain. The CityHash gate
   caught a real Mojo cast-chain zero-extension on first contact.

7b. **Scale decides WHICH of the incumbent's designs to port.**
   CatBoost quantizes two ways -- GPU `ComputeBorders` (device
   RadixSort, full data) and CPU (100k subsample, host sort inside the
   DP) -- and the faithful port of each is fastest in a different
   regime: at 400k values/column the device sort cut the border build
   4x, but at the subsample's 100k the SAME device sort is
   launch-floor-bound (Metal ~21 us/launch + per-column sync; 500
   columns = 1.8 s of pure control plane) and the host path wins.
   "Copy, do not improve" therefore needs a scale annotation: the
   mechanism to copy is scale-dependent, and the honest port carries
   BOTH of their paths with the incumbent's own switch (their
   subsample bound) deciding. Measured 2026-08-21, prep bill
   3.4 -> 1.2-1.6 s, both paths gated bit-exact. Related: the
   four-axes floor-amortization law (A.2) -- this is its quantization
   corollary.

13. **An output-only equality gate is BLIND to fold-order violations, measured.**
   The mamba block's `S1_FOLD_DESCENDING` sabotage (reverse the RMSNorm sum-of-
   squares fold) was run at two widths on one Apple M4. At d_model 8 it moves
   `norm.sumsq` on 1 of 4 rows and NOTHING else: the divide and the reciprocal
   square root absorb a 1 ulp change and `norm.out` is identical again. At
   d_model 16 it moves 3 of 4 rows and 13 of 16 stages, with `out_proj.out`
   differing on 23 of 64 cells -- and `residual.out` is STILL bit-identical,
   because the residual add puts an out_proj of order 1e-3 beside an input of
   order 1. So at the shape where the arm is STRONGEST, a gate comparing only
   the block's output calls it inert and licenses any fold order at S1. Four of
   the block's six arms move no cell of the final output at all. The claim is
   therefore not that per-stage traces are convenient: it is that output
   equality is the WRONG INSTRUMENT for a determinism contract, and cannot
   falsify the clauses such a contract is made of. Evidence:
   `mamba/mojo_only/mamba_check.mojo`'s sabotage ledger, IDENTITY_PATHS row 55.
   **INDEPENDENTLY REPRODUCED THE SAME NIGHT IN AN UNRELATED ALGORITHM**, which
   is what moves this off "a mamba quirk". holtwinters, 9 sabotage arms run on
   one M4: `LS_TIE` moves 9 of 37 stages and 64 of 2800 cells and leaves the
   FINAL FORECAST bit-identical. Stronger still, `CRIT_ORDER` moves ZERO cells
   of 2800 -- it changes one recorded LABEL, the stop criterion -- so a gate
   comparing parameters and fitted components passes it for ever, and it is
   caught only because the criterion is itself a recorded stage. That extends
   the claim: it is not only per-STAGE recording that is required, it is
   recording the non-numeric DECISIONS too, or a whole class of divergence has
   no instrument at all. A third arm, `ROTATE_CONV`, is reached by one fixture
   in six because the other five run a single block where the rotation is the
   identity -- an arm can look inert for a reason that has nothing to do with
   absorption, so reach must be measured per fixture as well.
   Evidence: `holtwinters/mojo_only/hw_check.mojo` arm table, IDENTITY_PATHS
   row 57.
   **AND THE CONVERSE, MEASURED IN A THIRD LANE 2026-08-24, which is the
   practical payoff rather than the warning:** recording ONE decision stage in
   spectral (`spectral.lanczos.config`) converted its `MAXITER` sabotage from a
   recorded REACH FAILURE into a live arm biting on all six fixtures, moving 1
   stage and ZERO cells, and thereby RETIRED an owed heavy fixture (n > 100
   with slow convergence) rather than deferring it. A decision stage did what a
   heavy fixture could not. Recording decisions is not only how you SEE a class
   of divergence, it is how you make clauses reachable at all.
   **AND THE LIMIT OF THAT, measured the same day, which keeps the claim
   honest:** holtwinters' DEVIATION 699 added eight recorded decisions and
   promoted NONE of its two inert arms, because a decision stage promotes an
   arm only when the arm is DECISION-SHAPED. `STD_SQRT` is a sqrt spelling and
   `HW_MAX_CLAMP` is an LLVM constant fold; neither is a branch the algorithm
   takes, and no amount of decision recording makes a vendor-arithmetic
   question answerable on one vendor.
   **AND IT HAS NOW BEEN EXHIBITED WITHOUT A SABOTAGE AT ALL, ON A SHIPPING
   FIXTURE, WHICH IS QUALITATIVELY STRONGER EVIDENCE THAN ANY ARM.** The first
   run of the widened metrics card, 2026-08-24: `metrics.mi.terms` and
   `metrics.mi_swapped.terms` differ, their accumulators differ on 36 of 36
   cells (a row-major fold against a column-major fold of the same 36 terms),
   and the two RETURNED SCORES are equal BIT FOR BIT at `522cc1ae6bc734b9`. No
   injected fault, no perturbation, just two legitimate orderings of one sum
   agreeing in the answer and disagreeing everywhere behind it. Under the
   previous card, which recorded only the returned score, that entire
   divergence was invisible, and `MI(pred, truth)` was not recorded in any form.
   And the coincidence is NOT a law: on a second fixture the same two folds
   DIFFER in the returned score. So the old card was not merely weak, it was
   weak in a way that depended on which fixture ran.
   PRIOR-ART CHECK NEEDED: reproducibility work generally reports final-output
   equality, so the question is whether anyone has shown it INSUFFICIENT.

14. **A determinism gate is likelier to be VACUOUS than to be wrong, and
   vacuity has a cheap general remedy.** In one night, FOUR independent lanes
   each found a gate of their own that would have passed for ever, on every
   vendor, while testing nothing. mamba: a row-slice comparison where a slicer
   returning row 0 for every request would compare each row to itself.
   holtwinters: a launch-invariance check spelled as a chain of
   `_first_diff(..) == ""`, green if the bit collector returned empty or the
   fit ignored tpb/pad/poison. arima: a sabotage that came back null because
   the gate planted `-0.0` where the filter's first `0.0 + x` washes the sign
   out -- the ARM was fine and the GATE was blind. spectral: gates whose
   helpers were not yet checked for the same property. mamba a second time, and
   this is the sharpest case: its clause (e) planted 26 NaN and infinity values
   and all 26 were refused by name, but an UNCONDITIONALLY raising refusal
   would have produced exactly that result while gating nothing. The agent had
   already seen the 26 come back green before it noticed, and said so rather
   than quietly adding the control. The control is a CLEAN call that must NOT
   raise and must record all 17 stages. None of these is a bug
   in the code under test; all are gates that cannot fail.
   The remedy is a NEGATIVE CONTROL beside each gate, asserting that inputs
   which MUST differ do differ, and raising a distinct VACUOUS verdict rather
   than FAILED so the two are never confused in a log. Cost is one extra
   comparison per gate. The general lesson for a determinism contract: an
   equality gate's default failure mode is silence, so every clause needs both
   a sabotage (does a violation move the number) and a control (can this gate
   move at all). Evidence: the four lanes' check files and IDENTITY_PATHS rows
   55, 57, 58, 59. PRIOR-ART CHECK NEEDED -- mutation testing and metamorphic
   testing are the neighbouring literatures and the claim must be positioned
   against them, not presented as new in kind.

15. **A determinism clause's coverage is PER FIXTURE, and can be exactly one
   fixture wide without anyone noticing.** Measured in the spectral lane: of
   seven live sabotage arms, THREE are caught by a single fixture each, and
   deleting that one fixture would silently disarm a contract clause while
   every gate stayed green. The three mechanisms are all different, which is
   what makes this a class rather than an anecdote. `SIGN_FLIP` is inert on
   four of five because after a restart the projected matrix built from
   `-beta_k` is exactly `D T D`, so the sign pin RE-CANONICALIZES the flip
   away. `LAPLACIAN_SEAM` is inert on four for two unrelated reasons: two
   fixtures never execute the seam, and two carry power-of-two weights where
   the reassociation is exact. `SPMV_ROTATE` is inert on four for a purely
   mechanical reason -- they fit in one block, so the rotation is the
   identity. holtwinters hit that third mechanism independently the same
   night. A fourth arm, `NCV`, was inert on EVERY fixture and only became live
   when a fixture was built for it alone.
   The consequence for practice: a sabotage table reporting pass/fail per ARM
   is not enough. Reach must be reported per ARM PER FIXTURE, and an arm
   carried by one fixture is a fixture that may not be deleted. Note also that
   an inert arm is NOT one thing: spectral's `STD_SQRT` is inert because this
   host's sqrt is correctly rounded, which is a MEASUREMENT the arm exists to
   take, while `MAXITER` is inert because nothing reaches it, which is a
   REACH FAILURE. Conflating them turns a hole into a result.
   A FOURTH mechanism, and the sharpest, because it makes the SMALLEST
   fixture the wrong one: a fixture can be STRUCTURALLY BLIND to the property
   under test. The mamba corpus cross-check compares stages stored
   channel-major against ours stored token-major, so the reindexing is the
   thing most likely to be wrong. At L=1 THE TWO LAYOUTS ARE THE SAME BYTES,
   so the smallest corpus case passes identically whether the reindexing is
   right or wrong. The orchestrator had explicitly instructed 'smallest case
   first', which for that question was exactly backwards, and the lane said so
   rather than reporting the green. Certified instead at L=4 with a
   transposed-dump control that fails exactly the 7 channel-major stages and
   no others -- a control that bit EVERYTHING would have been weaker than one
   that bites exactly what it corrupts.
   A FIFTH mechanism, from arima 2026-08-24, and it inverts the obvious way to
   widen a fixture. A LONGER CHAIN IS NOT MORE COVERAGE IF THE CHAIN IS OVER
   EXACT VALUES. Its fold-order sabotage moves 0.3% on the order with the
   LONGEST state chain (rd=8) and 25% on one of the shortest (rd=2), because a
   differenced or seasonal transition matrix is mostly structural exact 0.0 and
   1.0, and a fold over exact values is order-independent however long it runs.
   The driver is the count of genuinely arithmetic terms, not the chain length.
   The lane had predicted the opposite in writing and left the wrong prediction
   in the docstring beside the outcome. Widening a fixture means widening the
   part that carries ARITHMETIC, not the part that carries STRUCTURE.
   Evidence: IDENTITY_PATHS rows 55, 58 and 59, `spectral/mojo_only/spectral_check.mojo`,
   `mamba/mojo_only/mamba_check.mojo`.
   PRIOR-ART CHECK NEEDED, against mutation-testing coverage literature, where
   the neighbouring notion is mutant-killing test minimality.

## C. Toolchain findings (report upstream, never paper claims)

8. **Metal AOT suppression + cache poisoning (replaces the retracted
   "basename lottery", 5cd37db).** `MACOSX_DEPLOYMENT_TARGET` set to
   ANY value makes `mojo build` write an empty 134-byte metallib per
   kernel (0 AIR blobs vs 141 unset, one variable, fresh cache both
   sides), and `$MODULAR_HOME/cache/.mojo_cache` is content-addressed
   WITHOUT keying on the deployment target, so one poisoned build
   serves empty metallibs to every later build whatever its flags.
   The name-keyed buckets of the earlier claim were cache attrition,
   not emission behavior. Workaround shipped: pass the floor via
   `-Xlinker -platform_version`, unset the env var, verify the Mach-O
   header. Two upstream issues (suppression + cache keying), data in
   5cd37db (third in the series after #6932/#6933).

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
* Predicting a network's runtime on another GPU from measurements on
  one you have, explicitly for cost-efficient selection: Habitat
  (arXiv 2102.00527, 11.8% average error over six architectures).
  Recommending cost-efficient cloud GPU configurations: Srifty (MLSys
  2022). The cost formula C = steps * time * price is NOT ours and must
  never be presented as a contribution. See IDENTITY_COST_PLAN.md Part 2.
* Deterministic training on ONE fixed hardware and software
  configuration: ordinary engineering, widely done. Likewise noticing
  that cuDNN and PyTorch decline to guarantee bitwise reproducibility
  across architectures; the absence of that guarantee is THEIR
  documented position, and citing it is context, not a finding.
