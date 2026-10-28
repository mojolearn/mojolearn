# CPU pointwise ranking implementation draft

**Uncompiled. Do not merge this branch before successful affected compilation.**
Andrew authorized this implementation in its own worktree on 2026-10-27,
with compilation deferred. No runtime, identity or vendor campaign was run.
This is an engineering handoff, not a published support claim.

Branch: `lane/gbdt-cpu-ranking-20261027`.
Base: `b3239afc49478f01ff338b35121ee749396417aa`.
Worktree: `/Users/andrewhendel/mojolearn-wt/gbdt-cpu-ranking-20261027`.

## What changed

The CPU already has QueryRMSE, PairLogit and YetiRank derivatives and grouped
leaf estimation for greedy symmetric trees. The missing route is their
`use_pointwise_searcher=True` doc-parallel structure search. The draft adds
that route through `gbdt/host/gbdt_oracle_pointwise_rank.mojo` and the existing
native `gbdt_fit` binding, keeping the original greedy route unchanged.

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

## Required before merge

1. Compile the new owner with calls covering generated and explicit PairLogit
   pairs, both search-weight planes and both leaf methods; QueryRMSE; YetiRank;
   single-fold cosine and L2; bootstrap/no-bootstrap; and the existing greedy
   PairLogit caller of the repaired pair-weight estimator. An import alone does
   not establish that these functions compiled.
2. Compile the affected actual host binding, including `gbdt_fit_binding`,
   using the supported local Mojo toolchain and `MOJOLEARN_COLUMN_CPU` /
   `MOJOLEARN_NUMERIC_IDENTICAL=1`. Retain full output, exit status, compiler,
   flags, source closure and binary hashes. The repository entry is
   `bindings/entry/entry__mojolearn_gbdt_host.mojo`; the supported family build
   script is `bindings/build_gbdt_host.sh`.
3. Inspect any integration overlap with the parent's separate numeric Logloss
   `sample_weight` repair and CTR evaluation draft. Do not copy another
   worktree's uncommitted files or merge this branch just because those other
   routes compiled.
4. Run required source/push guards on the integrated commit. Compilation is
   development merge evidence only, never evidence of runtime correctness or
   bitwise identity.

## Evidence still pending

The two pointwise ranking entries in `GPU_ONLY_LANES` remain unchanged because
this draft has neither compilation nor runtime evidence. Do not remove those
coverage exclusions or advertise this in public documentation on the strength
of the implementation alone. Current work instructions prohibit a runtime or
cross-vendor campaign; any later qualification must retain its actual scope.

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
It remains uncompiled and should not be described as qualified Gradient-leaf
parity until the affected evidence exists.

Retained local evidence directory:
`~/mojolearn-evidence/gbdt-cpu-ranking-draft-20261027/`.
It contains source hashes, the bounded review diff and commit/push guard logs.
No rented machine, background watcher or generated binary was used.
