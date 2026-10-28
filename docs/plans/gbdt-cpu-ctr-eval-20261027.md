# CPU CTR held-out evaluation handoff — updated 2026-10-28

**Full CPU GBDT binding compiled successfully. Runtime and identity remain
pending.** Source commit `2d5c5081a83eab132a56a2ef0e199e23c9bf7ed6` was built on
the authorized Peer2 RunPod CPU allocation on 2026-10-28. This replaces the
initial uncompiled handoff and satisfies the affected-path compilation
prerequisite for that source. It does not establish runtime behavior or
bitwise identity. This follow-up changes comments and evidence only.

Branch: `lane/gbdt-cpu-ctr-eval-20261027`.
Base: `b3239afc49478f01ff338b35121ee749396417aa`.
Worktree: `/Users/andrewhendel/mojolearn-wt/gbdt-cpu-ctr-eval-20261027`.
Evidence: `/Users/andrewhendel/mojolearn-evidence/gbdt-cpu-ctr-eval-20261027/`.
Compile evidence:
`/Users/andrewhendel/mojolearn-evidence/peer2-native-boundary-20261027/cpu-batch-7/remote/leg_out/gbdt-ctr/`.
Tracked receipt: [gbdt-cpu-ctr-eval-20261028-compile.json](gbdt-cpu-ctr-eval-20261028-compile.json).

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

`git diff --check`, Python AST parsing of the modified fixture and all four
source guards passed for the implementation commit. The fixture was not
imported or executed. The normal pre-push hook remains required for this
comments/evidence follow-up, with complete logs retained in the evidence
directory. Its original `receipt.json` records the source-only handoff as it
stood then; the tracked compile receipt above records the subsequent build.

The full entry `bindings/entry/entry__mojolearn_gbdt_host.mojo` compiled with
Mojo `1.0.0 (ed45d567)`, `--optimization-level 0`, `--target-cpu x86-64-v3`,
`MOJOLEARN_NUMERIC_IDENTICAL=1` and `MOJOLEARN_COLUMN_CPU`; exit status was 0.
Affinity provided 32 CPUs, the cgroup CPU quota was unlimited, and the compiler
used `-j 32`. CPU saturation was not measured. The retained binary is
18,735,176 bytes, SHA-256
`5cc38580b1dc644a3164943b2bc374e75c6b7d113f8f0b25d504723d85bd4c30`.
The source archive and binary hashes were checked against the compile records,
and both changed Mojo files in the archive matched the implementation commit.

Source bytes, complete compiler output and binary are retained on the Mac
under the compile evidence path above and in R2's complete archive:
`measurements/runpod-cpu/mojolearn-cpu-peer2-native-20261027-20261028-040353/results-complete.tgz`.
The owner recorded verified termination of pod `b8o973bpjz3685` after retention.
No build or runtime test was repeated for this comments/evidence update.
Main integration remains serialized by the coordinating thread.

Runtime behavior and identity remain pending after this successful compile.
Future affected-path checks should cover no-eval preservation, explicit/default
best-model behavior, Iter and IncToDec stopping, constant eval targets, more
than one CTR permutation, unseen categorical codes, a mixture of numeric,
one-hot and CTR columns, and malformed/NaN categorical inputs. This handoff
does not authorize a new runtime or cross-vendor campaign; follow the current
owner instruction when that work is scheduled. Do not promote this compilation
to runtime success or identity evidence, or extend it to another target.
