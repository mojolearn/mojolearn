# CPU CTR held-out evaluation handoff — 2026-10-27

**UNCOMPILED. MUST COMPILE THE AFFECTED PATH BEFORE MERGE.** This branch is
source work authorized by Andrew's request to leave larger CPU configuration
repairs in separate worktrees, with their compilation requirement explicit.
No runtime, identity or vendor campaign was run. Source checks do not establish
that Mojo accepts this change or that the public operation runs correctly.

Branch: `lane/gbdt-cpu-ctr-eval-20261027`.
Base: `b3239afc49478f01ff338b35121ee749396417aa`.
Worktree: `/Users/andrewhendel/mojolearn-wt/gbdt-cpu-ctr-eval-20261027`.
Evidence: `/Users/andrewhendel/mojolearn-evidence/gbdt-cpu-ctr-eval-20261027/`.

The change connects evaluation sets to the existing CPU categorical CTR fit.
It does not introduce another CTR algorithm. The GPU fit already calls the
GPU-free `gbdt.models.ctr_value_table.expand_raw_columns` on held-out inputs.
The CPU fit now calls that same helper using its final learn-pool tables, then
quantizes the expanded columns against its training grid and compressed-index
layout. Its existing `gbdt_host_boost` receives the held-out index and resolved
`GbdtHostEval`, so it owns the curve, detector and best-model shrink as it does
on numeric SymmetricTree Logloss. The binding returns the resulting held-out
curve instead of refusing a CTR evaluation set.

The new route is limited to the existing categorical configuration: Plain
SymmetricTree Logloss, Cosine score, Newton leaves, no bootstrap or score noise,
no sample/class weights and no boost-from-average. Existing categorical,
dimension, NaN and other parameter guards remain in place. The shared detector
accepts the existing resolved None/Iter/IncToDec settings; the binding resolves
`use_best_model` and `best_model_min_trees` before the owner call.

Only the existing default **SkipTest** CTR behavior is represented. Held-out
rows and targets never enter CTR counts, target borders or permutation
histories. Unseen categories use the shared apply-time table's prior. This
change does not expose `counter_calc_method=Full`, new CTR types, tensor CTR
training, weighted categorical training, Ordered categorical training or
non-Logloss categorical objectives. Those remain outside this batch.

Changed files:

- `gbdt/host/gbdt_oracle_ctr.mojo`: optional held-out record, shape admission,
  final-table expansion, shared-grid binarization and existing boost hookup.
- `bindings/_mojolearn_gbdt_host.mojo`: CTR-only eval preparation and result
  conversion. This hunk is separate from the numeric Logloss weight repair and
  the pointwise-ranking work in other branches.
- `python/mojolearn/tests/test_cpu_training_gbdt_ordered.py`: replace the old
  CTR-eval refusal assertion with unexecuted regression fixtures. They check
  that unseen eval categories and changed eval targets leave the learn model
  unchanged, that the held-out loss agrees with final-model prediction, and
  that best-model shrink honors its minimum-tree threshold.

`git diff --check` and Python AST parsing of the modified fixture passed.
The fixture was not imported or executed. Required source guards run through
the normal pre-push hook; its complete output and exit status are retained in
the evidence directory. `receipt.json` records the final commit and guard
result after pushing. There is no binary or compilation receipt for this lane.

Before integration, compile the actual host binding with the supported local
Mojo toolchain, `MOJOLEARN_NUMERIC_IDENTICAL=1`, `MOJOLEARN_COLUMN_CPU`, and the
Mac CPU target. The repository entry is
`bindings/entry/entry__mojolearn_gbdt_host.mojo`, used by
`bindings/build_gbdt_host.sh`. A focused entry that actually instantiates
`gbdt_fit_binding` and its CTR owner is also acceptable. Use the actual CPU
allocation for compiler workers; do not inherit the builder's two-worker
default. Retain command, source closure, compiler, flags, exit status, complete
log and binary hash. Repair compile failures and replace the inline
compilation-pending notes with that receipt before merging.

Runtime behavior and identity remain pending after any successful compile.
Future affected-path checks should cover no-eval preservation, explicit/default
best-model behavior, Iter and IncToDec stopping, constant eval targets, more
than one CTR permutation, unseen categorical codes, a mixture of numeric,
one-hot and CTR columns, and malformed/NaN categorical inputs. This handoff
does not authorize a new runtime or cross-vendor campaign; follow the current
owner instruction when that work is scheduled. Do not change public support
claims or identity evidence on the strength of this uncompiled branch.
