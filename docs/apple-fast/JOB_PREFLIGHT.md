# Future pinned-job metadata preflight

`python3 tools/apple_fast_job_preflight.py SPEC.json` runs read-only on M3.
Exit0 emits a READY JSON receipt; exit2 emits INFRASTRUCTURE_ERROR. It changes
no queue, results, artifacts or worktrees, imports no native library, and runs
no build, numerical test or timing. Existing/in-flight jobs are unaffected.
Manager review/deployment is separate; do not replace a running manager.

Policy is shared with `apple_fast_pinned_job.py` through
`apple_fast_job_policy.py`. The added resident timing policy requires its
first argument to equal the exact harness source; its helper still verifies
the quality report/hash internally. Native helper second argument must match
the submission tag. Preflight never executes the allowed script.

Example spec (replace SHA/hash placeholders with full recorded values):

```json
{
  "version": 1,
  "tag": "resident-catalog-t-v1",
  "harness_source": "FULL_HARNESS_COMMIT",
  "script": "tools/catalog_resident_timing.py",
  "args": ["FULL_HARNESS_COMMIT", "resident-catalog-t-v1", "/Users/ec2-user/mq/out/resident-catalog-q-v1-quality/report.json", "REPORT_SHA256"],
  "artifacts": [{
    "id": "probe", "kind": "pair", "compiled_source": "FULL_HARNESS_COMMIT",
    "binding": "resident_gemm_probe", "numeric_mode": "fast",
    "defines_A": "", "defines_B": "-D MOJOLEARN_APPLE_GEMM_RESIDENT_PROBE"
  }],
  "prerequisites": [{
    "id": "quality", "kind": "file",
    "path": "~/mq/out/resident-catalog-q-v1-quality/report.json",
    "sha256": "REPORT_SHA256",
    "json_equals": {"status": "PASS", "fixture": "catalog-resident-matrix-v1"}
  }],
  "cases": [{"name": "six-fixed-matrix-shapes", "requires": ["probe", "quality"],
    "source_files": ["tools/catalog_resident_quality.py", "experiments/apple_fast/gemm/mma.mojo"]}]
}
```

Every case explicitly lists prerequisite IDs and additional source files.
All declarations are required and checked, including unused ones. Missing
prerequisites are infrastructure errors, never numeric HOLD or an empty queue.
Sources must exist in `~/mojolearn.git`; script/source files are read at the
pin. Pairs live under `~/mq/verified-arms/COMMIT/BINDING/` and require exact
manifest source/binding/mode/defines plus both A.so/B.so hashes. Each native
case must name an artifact and the helper's compiled source must have an
artifact declaration. No artifact installation occurs.

Single prerequisites such as the IDENTICAL helper binding use:

```json
{"id":"ibase", "kind":"single", "compiled_source":"FULL_COMMIT",
 "binding":"ibase", "numeric_mode":"identical", "defines":"",
 "artifact":"_mojolearn.so", "manifest_equals":{
   "contract":"shared-gemm-prerequisite-v1", "module":"_mojolearn",
   "install_path":"identical/_mojolearn.so", "required_exports":["all_finite_f32"]}}
```

That directory's manifest must contain source_sha, binding, numeric_mode,
defines, artifact and sha256. Optional manifest_equals adds exact top-level
metadata expectations. This supports the reviewed ibase schema; declaring an
export in metadata does **not** prove that symbol exists or imports successfully.
The final case helper owns native import, capability and caller-reach checks.

Tag checks inspect all queue/results text and top-level output names matching
the tag or its `-`/`.` artifacts. Missing queue.txt/results.txt/out is an error.
The receipt hashes queue snapshots, policy, scripts, manifests and dependencies.
It is not a reservation: concurrent activity can invalidate readiness. The sole
manager rechecks immediately before submission and queues only the reviewed
command/policy. File hashes detect changed inputs, not ABI compatibility,
complete undeclared dependencies, numerical quality or runtime route reach.

Local verification: `python3 tools/test_apple_fast_job_preflight.py` uses only
small temporary metadata fixtures, fake mirror responses, and fake binary
bytes. It never loads them or invokes GPU/model runtime.
