# Reviewed source-freeze equivalence for comparisons

The default remains exact `source_sha` equality. Historical results are never
rewritten or rerun to manufacture missing provenance. An optional case-level
`freeze_equivalence: {path, sha256}` references a reviewed
`mojolearn.freeze-equivalence/1` JSON document. The comparator validates it;
it does not create or approve it. This is comparison evidence only, with
`accepted=false` and `promoted=false`. Original source IDs remain in each
column, each board input and each observation; the comparison anchor is separate.

Version 1 accepts exactly `nvidia-native` and `amd`, with explicit native targets.
It requires:

* `status: REVIEWED`, `review: {reviewer, reviewed_at, rationale}`, matching
  `case_id`, `configuration_id`, `anchor_source_sha`, and `scope_sha256`.
  The latter is SHA-256 of canonical JSON for the comparator's normalized
  expected scope, excluding only `source_sha` (sorted keys, compact separators).
* `repository`, resolved relative to the attestation, containing both original
  Git commits. Every column records its original `source_sha` and the exact
  sorted `reviewed_changed_files` relative to the anchor. Git computes the
  actual list. Numerical/API/worker/capture/lockfile changes are rejected.
  Only the fixed registration/preparation allowlist in the validator and
  documentation/evidence changes are eligible. Broad caller-supplied exclusions
  are not supported. A new allowlist entry needs a reviewed harness change.
* `columns` containing both routes. Each entry has `receipt: {path, sha256}` for
  its selected original full result, `execution_closure: {path, sha256}`, and
  `compile_receipts: {A: {deployed_path: {path, sha256}}, B: {...}}` covering every
  declared loaded binding exactly. Evidence paths resolve from the attestation.
* Raw successful compile receipts matching the deployed binary, numerical
  source, compiler executable hash, source closure, target and defines. Every
  numerical source-file hash, binding set, compiler hash and compile argument
  must match within the same arm across vendors. The only argument differences
  accepted are known source/output/compiler path relocation, the single explicit
  native `sm_*`/`gfx*` accelerator and NVIDIA/AMD column define. Parallelism,
  optimization, includes and arithmetic switches are preserved. There is no
  arbitrary flags waiver and no cross-arm equality requirement.
* Execution closure schema `mojolearn.captured-execution-closure/1`, `status:
  CAPTURED`, `method: observed_deployed_files`, `complete: true`, `missing: []`,
  original `source_sha`, original selected `receipt_sha256`, `capture_version`,
  and `sections: {harness: {logical_path: sha256}, api: {...}, runtime: {...},
  capture: {...}}`. Every section must be nonempty and exactly match the other
  route. Captures must cover actual deployed code and runtime dependencies,
  including supplemental native libraries; package versions alone are not
  sufficient. Version 1 has no runtime-library exclusion. Different or unproven
  vendor runtime closures remain incomplete. Reconstructing a historical
  installed package from current Git is not an observed closure.

The reviewer attests completeness and provenance of these retained inventories;
the validator checks the immutable evidence, source diff and exact equality. A
hash-bound JSON claim is not independently a deployment observation. No genuine
historical attestation is supplied with this change: existing runtime summaries
contain versions and binding hashes, but do not establish complete observed
Python/API/runtime/capture closure. Those comparisons remain `INCOMPLETE`.

All pre-existing dataset, settings, logical arm, timed boundary, harness hash,
scored-sample, typed output/model-state and repeated-use checks remain required.
In particular an attestation cannot fill absent fold models, selection state or
other missing capture scope. Capture inventories can be added outside the clock
in a future actual scored worker; no standalone numerical verification is needed.

`observed_output_bits` separately reports literal same-arm scored output
`AGREE`, `DIFFER` or `INCOMPLETE` for matching full data/settings/configuration
and typed output scope. It preserves each source and harness hash and always
sets `qualified_identity=false`. It can report an observed output agreement
while full model identity remains incomplete because provenance or state is
missing. It never changes identity admission, quality, votes or default settings.
