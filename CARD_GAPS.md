# What the cards cannot see

Audited 2026-08-24 by a read-only pass over every lane, on Andrew's question
"are there more hashes we can do?". Nothing was compiled, run, timed or edited
to produce this. Every claim below is a source read with a `file:line`.

The question is not decoration. Three findings measured 2026-08-23 say a stage
that is not recorded is a divergence nobody can see:

- An output-only equality gate is BLIND. mamba's fold sabotage moves 13 of 16
  stages and 23/64 cells of the second-to-last stage while the FINAL OUTPUT
  stays bit-identical. Four of six arms move no output cell at all.
- A DECISION that changes no number still needs a stage. holtwinters'
  `CRIT_ORDER` moves ZERO of 2800 cells and is caught only because DEVIATION
  665 records the stop criterion.
- Reach is PER FIXTURE, and a fixture can be structurally blind to the property
  under test.

See NOVELTY_NOTES entries 13, 14, 15.

---

## THE URGENT ONE: metrics is instrument-weak in a lane we have CERTIFIED

`metrics/metrics_main.mojo` records inputs, then a final scalar, with nothing
in between, for fourteen scores. This lane is part of the three-vendor
certification at E3 round 11 (`a32e304`). The certification is not thereby
wrong, but **the instrument backing it is weaker than the claim implies, and
the lane has already measured that it is.**

`metrics/ported/stats/detail/scores.mojo:287-291` says, in the lane's own
words, that `r2 = 1 - sse/ssto` **ABSORBS** a last-bit move in either sum
whenever `sse << ssto`, measured on the 4099-row fixture, "so the checks gate
the sums, not only the ratio." The CHECKS gate `(y_bar, sse, ssto)`. **THE CARD
DOES NOT.** `r2_score_launch` (`scores.mojo:274-275`) computes all four parts
and returns `parts[3]`, discarding the other three, and the card
(`metrics_main.mojo:161`) records only that one scalar.

So the absorption is documented, measured, defended in the local checks, and
still invisible to the cross-vendor card. Same shape for
`metrics.kl_divergence` (`:170`, a fold of n log terms with nothing between
input and answer) and for the eight cluster scores at `:128-147`, whose only
recorded intermediate is `metrics.contingency` (`:120`) while ARI, homogeneity,
completeness and v_measure are each ratios of differences of large numbers.

**The fix is nearly free and rule-legal.** `chunk_count` is a pure function of
n (`metrics/mojo_only/pinned_sum.mojo:72-74`), so the chunk partials are NOT
the machine-sized scratch that `core/identity_trace.mojo` rule 3 forbids.

Groups C and D of that lane are fine and need nothing: silhouette records
per-sample values (`:183`), trustworthiness routes `knn.*`, `trust.emb_ind` and
`trust.rank_sum` through the trace (`:199`).

## THE CROSS-LANE ONE: the build mode is provenance, not a record

`_mode_name()` reaches a trace only through `header()`, which writes a `#`
comment, and BOTH readers drop comments (`core/identity_trace.mojo:262-267`,
`tools/identity_trace_diff.py:175`). **An IDENTICAL trace and a FAST trace
align stage-for-stage with no record saying they were different builds.**

Andrew's standing rule is that a `numerics.mojo` mode flip with no lock gives
correctly labelled measurements of the WRONG arm, and that the mode must be
READ BACK from the run. One recorded integer at seq 0 turns that entire failure
class into a first-stage divergence with a name. Cost: 4 bytes per card.

Care required: adding a record changes every card's record list, so old and new
cards do not compare. A leg builds both ends from one pinned SHA, so that is
survivable, but it must not be spelled as an arithmetic STAGE in any profile
whose contract freezes its stage list (mamba section 7 is explicit that
changing the stage list creates a v2).

## Three more cards that hash on the far side of a washer

- **spectral**: `spectral.ritz.vectors` is recorded AFTER `pin_column_signs`
  (`symmetric_eig_host.mojo:326-340`), so a whole-column sign divergence
  between two vendors is RE-CANONICALIZED AWAY BEFORE IT IS HASHED. The
  contract already concedes the consequence: the device-side instrument for
  DEVIATION 770 rests on ONE fixture (`IDENTICAL_SPECTRAL_CONTRACT.md:405-412`).
  Two integers per column fix it, and they catch a second thing: the pivot scan
  skips on `== zero`, and `+0.0 == -0.0`, so under the Metal flush asymmetry
  two vendors can pick DIFFERENT PIVOT ROWS and still emit identical vectors.
- **holtwinters**: `hw.params` (`runner.mojo:386`) is recorded AFTER
  `bound_device`'s hard clamp to [0,1] (`hw_utils.mojo:168-191`), so any two
  unbounded values that both land outside the bound hash identically. The
  unbounded values appear only in the per-iteration stages, and only while
  `it < trace_iters`; past that the runner writes a HEADER COMMENT and
  truncates (`runner.mojo:363-368`), and comments are dropped by both readers.
  On a long fit the final unbounded parameters are recorded NOWHERE.
- **solver/cd**: `cd.sweepNNN.coef` (`cd.mojo:496`) is recorded after the soft
  threshold maps everything inside the L1 band to exactly `+0.0`
  (`cd.mojo:195-207`) and after the `sq <= CD_SQUARED_GUARD` zeroing
  (`:208-212`). For a lasso fit that is most coordinates, and the pre-threshold
  gradient is recorded nowhere.

## THE RULE THAT KEEPS THIS PROGRAM FROM BREAKING WHAT WE ALREADY HAVE

Added 2026-08-24 after the holtwinters lane pushed back on a suggestion in
this audit's own brief, correctly.

**A decision worth hashing is one the ALGORITHM makes, not one the SCHEDULER
makes.**

The brief listed "chosen block or launch geometry" as a candidate decision.
It must NOT go in an identity card. The card is ASSERTED LAUNCH-INVARIANT: a
stage recording launch geometry would differ between two tpb settings BY
CONSTRUCTION and would break the exact property `check_hw_launch_invariance`
exists to prove. Launch geometry belongs in a separate assertion, never in the
identity trace.

The same test applies to every candidate in this file. `chunk_count` is a pure
function of n, so metrics' chunk partials are the ALGORITHM's structure and are
legal. `contract_partition` is pure in k and `choose_gemm_plan` is pure in
(m, n, k), so gemm's plan id is the algorithm's and is legal. Anything whose
value depends on the machine, the occupancy or the dispatch is not.

Corollary on wording, also from that lane: **"ungated" is the wrong word for a
branch nobody can reach.** It invites someone to gate it. holtwinters'
`get_num_blocks` 65535 cap binds only above 33.5 million series, which is more
than 6 GB of input at n=48, so it is recorded UNPORTED-IN-EFFECT and CLOSED
rather than left on an owed list for ever.

### Corollary: an instrument that is itself an identity hazard is not an improvement

From the spectral lane, 2026-08-24. Two candidate decision-stages were HANDED
BACK rather than added, because each needs a COUNT OVER A DEVICE BUFFER, which
is a cross-thread reduction, which is a new fold whose own order would have to
be pinned before the stage could be trusted. Adding an unpinned fold in order
to instrument a pinned one makes the card worse. The two are: whether the `u`
clamp at 1e-7 fired and on how many entries (`u` is the one Lanczos seam with
no instrument at all), and whether `zero_to_one_functor` substituted a 1.0 for
a zero degree (`spectral.diag` records only the post-substitution value, so a
1.0 there is ambiguous). Both remain open and both are correctly open.

### And the payoff, measured

The program is not theoretical. Recording ONE decision stage
(`spectral.lanczos.config`: n, k, ncv, max_iterations, which) converted the
`MAXITER` sabotage from a RECORDED REACH FAILURE into a live arm that bites on
all six fixtures, moving 1 stage and ZERO cells. That RETIRED an owed heavy
fixture (n > 100 with slow convergence) instead of deferring it. Eight of nine
arms in that lane now bite. A decision stage did what a heavy fixture could
not.

## The top five, if only five land

Every one is a scalar or a handful of integers. None adds a large buffer.

1. `metrics.r2.{y_bar,sse,ssto}`. The absorption is already measured and the
   function already returns the values.
2. `build.mode` as a recorded integer in every lane's card.
3. spectral's `pin_column_signs` `(pivot row, negated)` per column, plus
   `sweeps_used`, which `symmetric_eig_host.mojo:192-208` already returns, whose
   own docstring calls it "an integer a card can carry", and which every caller
   discards (`lanczos.mojo:630,652`). That is the card instrument `SWEEP_CAP`
   currently lacks.
4. arima's `d_info` (the degeneracy verdict, written at `batched_kalman.mojo:314`
   and `:460-462`, readable at `:602`, recorded never) and `d_piv` (the LU pivot
   vector, `matrix.mojo:92-108`, first-wins `>` tie-break, in the stage the
   card's own header calls "the stage most likely to move between vendors").
5. isolation_forest's `min_val`, `max_val`, `rand_frac` per split node
   (`isolation_tree_builder.mojo:318-338`): `threshold = fma(rand_frac,
   ftz(max_val - min_val), min_val)`, so when `(max-min)` is small against
   `|min|` the product is ABSORBED and a divergence in either bound yields the
   identical recorded threshold.

## Lanes that are already fine

- **kde**: per-cell `kde.dists` and `kde.logk` mean every kernel term including
  the support-cutoff sentinel is hashed before the log-sum-exp compresses it.
  Nothing to add.
- **mamba**: 17 buffer stages, both state buffers, and the residual add
  bracketed on both sides. Best buffer coverage in the repo. Its gaps are all
  DECISIONS, chiefly the softplus guard arm, which the contract itself
  (`IDENTICAL_MAMBA_CONTRACT.md:112-119`) measured to be a decision that
  provably changes no number near the boundary in FP32.
- **gemm**: output-only in FORM (`bench/gemm_card_main.mojo:499` records only
  C) but defended in SUBSTANCE, since `contract_partition` is pure in k and
  `choose_gemm_plan` is pure in (m,n,k). Recording the plan id and (L,P), three
  integers, would convert "C differs" into a named divergence.

## Two audit notes that are not hashes

- **isolation_forest, possible defect**: `isolation_tree_builder.mojo:296-297`,
  `if node_idx >= max_nodes_per_tree: continue` silently DROPS a node, leaving
  `node_feature/threshold/left/right` at the poison fill while a parent still
  points at it, and `structure.*` records only `n_used` entries so the dangling
  child is never hashed. Needs a lane verdict.
- **hierarchy/linkage**: the card RE-RUNS `pairwise_distances`
  (`linkage_main.mojo:85-89`) and `build_sorted_mst` (`:108-110`) to produce
  `linkage.dists` and `linkage.mst.*`, rather than recording the buffers the
  `single_linkage` call at `:94` actually consumed. If the two runs ever
  disagreed the card would be internally inconsistent and nothing checks it.

The full ranked per-lane list, including spectral A1-A8, arima C1-C5,
isolation_forest D1-D6, holtwinters E1-E5, svm J1-J4 and linkage I1, is in this
round's audit report and is reproduced lane by lane above where it ranks.
