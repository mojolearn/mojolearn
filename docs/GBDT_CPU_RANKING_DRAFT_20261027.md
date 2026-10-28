# CPU pointwise ranking implementation

**Affected CPU binding compilation passed on 2026-10-28.**
The complete host entry compiled at O0 for Apple M1 with ten compiler workers.
The retained native source is commit 8dc73e4c130425bc9a60a8004b9d9ef2e5343a04.
See [compile receipt](plans/gbdt-cpu-ranking-20261028-compile.json). No runtime, identity or vendor campaign was run.
This is an engineering handoff, not a published support claim.

Branch: `lane/gbdt-cpu-ranking-20261027`.
Base: `b3239afc49478f01ff338b35121ee749396417aa`.
Worktree: `/Users/andrewhendel/mojolearn-wt/gbdt-cpu-ranking-20261027`.

## What changed

The CPU already has QueryRMSE, PairLogit and YetiRank derivatives and grouped
leaf estimation for greedy symmetric trees. The missing route is their
`use_pointwise_searcher=True` doc-parallel structure search. The draft adds
that route through `gbdt/host/gbdt_oracle_pointwise_rank.mojo` and the existing
native `gbdt_fit` binding, retaining greedy dispatch (the shared PairLogit leaf repair also affects it).

- Plain SymmetricTree on numeric columns, without sample/class weights or
  evaluation rows. The shared binding continues to refuse categorical columns,
  NaNs, feature subsampling and non-GreedyLogSum borders on this route.
- Cosine, NewtonCosine, L2 and NewtonL2 scores. The new single-fold cosine
  scorer starts its denominator at `1e-10`, as the GPU calcer does; the
  ordered paired-fold scorer remains separate and retains `1e-20`.
- PairLogit uses pair weights for first-order search and Hessians for Newton
  search. Both generated group pairs and explicit pairs are handled.
- QueryRMSE and YetiRank reuse their existing host derivative owners.
  YetiRank retains the separate search/leaf RNG draws, ten sampled
  permutations and decay `0.85`, as the current GPU implementation does.
- Score-noise deviation uses the unbootstrapped planes, the pinned 512-thread
  reduction shape and row-count denominator. Bootstrap follows that step.
  Existing No/Bayesian/Bernoulli/Poisson native bootstrap implementations are
  reused; the usual binding option checks still apply.
- Native leaf estimation, cursor updates and model conversion remain native.
  PairLogit leaf regularization and Gradient leaves use per-leaf pair-weight
  sums, including zero-weight leaves, rather than row counts. This shared
  estimator repair also applies to the existing greedy PairLogit route.
  YetiRank retains
  Newton-only leaves; QueryRMSE/PairLogit accept Newton or Gradient.

No multi-GPU CPU equivalent is proposed. Ranking evaluation sets are also
refused by the current GPU owner; that is a shared limitation, not missing
CPU parity. Sample-weight support for this route remains separate work.

## Compilation and integration

The full public host binding entry compiled successfully with
MOJOLEARN_COLUMN_CPU and MOJOLEARN_NUMERIC_IDENTICAL=1. Its native call graph
includes the grouped pointwise owner and shared greedy PairLogit estimator.
This is compilation evidence only; no parameter case was executed.
The source archive, manifest, compiler hash, command, resource allocation,
complete log and binary are retained under
`~/mojolearn-evidence/gbdt-cpu-ranking-compile-20261028/`.
Verifier registration changed no native code; later native edits update
comments only, so the successful compilation is reused.

Main integration must preserve the separate numeric Logloss sample-weight
and CTR evaluation repairs, and pass required source/push guards.
Runtime, identity and negative-control execution remain pending.

## Verifier registration and evidence still pending

At Andrew's follow-up request, a separate commit registers CPU verifier
selection for the existing `gbdt-pair-logit-pointwise` and
`gbdt-yeti-rank-pointwise` lanes. They leave `GPU_ONLY_LANES`, join the gbdt
family's training list and remain annotated `unwatched`. That permits the
verifier to run and report real outcomes; it does not claim a passing CPU
result or replace an existing GPU reference. The new
`gbdt-query-rmse-pointwise` lane uses the same grouped fixture with the actual
pointwise searcher and is marked `no reference`. The existing greedy
`gbdt-query-rmse` lane and its records stay unchanged.

All three lanes hash prediction and loss curves, retain the inference/model
probe and register batch prediction. The existing `MOJOLEARN_HOST_SABOTAGE`
negative control reaches their shared native leaf solver. Source reachability
is checked statically; a caught negative-control execution is still pending.
No new GPU lanes duplicate the two existing ranking lanes, and no numerical
hashes, admission files or release evidence are created.

The remaining explicit CPU exclusions are `optim-resident-io`,
`sequence-device-release` and `training-device-cross-entropy`, all device
resource/pool tests. Multi-GPU lanes have a separate applicability rule.
Removing the two algorithm exclusions establishes declared selection only;
it does not prove every parameter combination or all retained evidence exists.

Do not advertise verified CPU coverage on the strength of this declaration.
Current work instructions prohibit a runtime or cross-vendor campaign; any
later qualification must retain its actual scope. Compilation before merge
remains mandatory for the implementation and actual public binding.

When runtime qualification is separately authorized, the affected cases include
the existing `gbdt-pair-logit-pointwise` and `gbdt-yeti-rank-pointwise` lanes,
QueryRMSE pointwise, explicit weighted pairs, Gradient leaves, zero-weight
leaves, repeated use, and the existing Logloss/ordered users of the shared
structure helper. Greedy PairLogit Gradient leaves and zero-weight leaves are
also affected by the shared estimator repair. Arithmetic/reduction equivalence is a source hypothesis
until measured.

Source inspection found that the shared PairLogit leaf helper constructed leaf
weights from row counts, while GPU pair setup replaces row weights with pair
weights. This draft repairs that native shared helper for both CPU searchers.
It compiled but must not be described as runtime-qualified Gradient-leaf
parity until the affected evidence exists.

Retained local evidence directory:
`~/mojolearn-evidence/gbdt-cpu-ranking-draft-20261027/`.
It contains source hashes, the bounded review diff and commit/push guard logs.
The later local compilation retained its binary in the compile evidence directory.
No rented machine or background watcher was used for ranking.
