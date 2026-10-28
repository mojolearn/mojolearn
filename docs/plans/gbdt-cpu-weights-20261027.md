# CPU GBDT weights and coverage comments, 2026-10-27

Branch: `fix/cpu-coverage-comments-20261027`, based on `b3239afc49478f01ff338b35121ee749396417aa`.

Plain numeric Logloss now admits `sample_weight`, alone or combined with
`class_weights`. The native entry copies the sample column and constructs the
same `sample_weight[row] * class_weights[y[row] > 0.5]` weight column as the GPU
entry. With no class weights it preserves the sample word; with neither option
the existing unweighted route remains selected. Negative sample weights retain
the native training error.

The change reuses the existing weighted search, leaf-estimation and learn-loss
owners. SymmetricTree retains its admitted numeric bootstrap, score-noise and
evaluation options. Depthwise and Lossguide retain the weighted owner's Newton
leaves/no-bootstrap restriction. Other losses, Ordered boosting and categorical
weighting remain explicitly refused. Multi-GPU operations are outside this work.

Coverage comments now distinguish the existing public CPU training API, existing
CTR training with permutations, and the remaining configuration-specific gaps.
The generated support matrix was refreshed. Its generator also corrected the
citation date to the existing dated release heading; no version changed.

## Validation and evidence

- Local CPU GBDT host binding compilation passed (exit 0), including the native
  public fit entry. The source closure still matches the retained archive. See
  `gbdt-cpu-weights-20261027.compile.json` and the complete `compile-result.json`
  in the evidence directory below. This satisfies the development compile gate.
- Static checks: `tools/docs_facts.py --check`, `git diff --check`, and AST parsing
  of the four changed Python sources. Complete outputs and exits are retained.
- Public probe controls were added for all three policies, combined sample/class
  weights, symmetric evaluation/stopping, and retained weighted-bootstrap refusals.
  These fixtures have **not** been executed in this batch.
- Runtime behavior, GPU/CPU agreement and within-release vendor identity have
  **not** been checked. Compilation does not establish any of them.

Evidence: `~/mojolearn-evidence/gbdt-cpu-weights-20261027/` contains the exact
native source archive and hashes, compiler hash, command, resource settings,
complete compile output, exit receipt and resulting shared library. The build
uses the local Mac allocation of 10 active/logical CPUs and `-j 10`; utilization
is not measured. No rented machine is used for this batch.
