# Tombstone gaps: records that cannot be reconstructed

Found by lane tombstone-guard-20261028 (2026-10-28) while checking every `docs/TOMBSTONES.md` index row against its
section, its patch under `experiments/removed/` and an inline tombstone marker. What could be repaired was repaired by
ADDING (anchors, markers); nothing was removed. The rows below could not be reconstructed and are listed here instead.
`tools/tests/test_tombstones_preserved.py` reads the table: a listed row is excused only from the check named in its
row. Like the rest of the record, this file is append-only.

| key | check excused | why it cannot be reconstructed |
|---|---|---|
| `MOJOLEARN_RESAMPLE_FAST_TAKE` | unpatched row names a recoverable sha | The A/B ran on `lane/apple-fast-s-shap`, which was never pushed (EXPERIMENTS.md "Speed round 2" row; TOMBSTONES.md section). No ref with that name exists locally or on origin, and no commit in any ref adds or removes a line naming `RESAMPLE_FAST_TAKE` (`git log --all -S RESAMPLE_FAST_TAKE -- '*.py' '*.mojo'` is empty). The code (a `numpy.take` host gather that the no-host-routes hook refused) is lost; the verdict is DROPPED-semantics and stands. |

## Repaired by addition (2026-10-28)

- Seven index links pointed at anchors no heading produced (`=` in a heading is dropped by the anchor rule, so
  `MOJOLEARN_CLASSICAL_PCA_COV=23` is `#mojolearn_classical_pca_cov23`, not the `-arm23` the rows link). An explicit
  `<a id>` was added above each heading: `mojolearn_classical_pca_cov-arm23`, `-arm4`, `mojolearn_idn_attn_softmax-arm1`,
  `mojolearn_idn_neural_gemm_epilogue-arm1`, `mojolearn_idn_neural_gemm_schedule-arm3`, `-arm8`, and
  `gbdt-oracle-multi-dimensional-host-walk` (the section is titled "GPU oracle multi-dimensional host walk"). The rows
  themselves are unchanged.
- "Framework adapter per-operation surfaces" (removed 2026-10-24) had only `//` notes in `adapters/torch/registration.cpp`
  and `adapters/native/abi.h` and plain comments in the Mojo files. An inline tombstone marker with its `Restore:` line was
  added to `adapters/native/operation.mojo` and `adapters/max_ops/__init__.mojo`. It has no patch by design (removed API,
  not an experiment define; recoverable at `b022e2ea5`).

## Entries before the inline-marker rule

45 index rows dated 2026-10-03 to 2026-10-09 have no inline marker in the tree: their deletions predate the 2026-10-10
rule. They are listed in `experiments/tombstones/pre_rule_entries.tsv`, which only shrinks. 11 of them are still named
in other comments or records (column 4); the rest have only their TOMBSTONES.md section, patch and guard refusal.
