# Fresh AMD kernel qualification checkpoint

MI300X gfx942, guarded HotAisle single-GPU 8-core VM, source `f771338c7`.
All six kernel lanes, nine fixtures, two repeats, full retained property flags.
Every one of the **216 numerical train/infer/model/batch parts** agrees with
both retained independent CPU and Apple columns. The remaining 216 admitted
parts are explicit N/A (stepfull/batchgrad/batchscale/ragged), not numerical
comparisons; these classical lanes do not declare rlpair.

All **54 GPU saved-model recordings and 54 independent CPU inference replays**
passed against the fresh AMD inference column. Raw columns, models, receipts,
logs and source/binary provenance are retained under captures-kernels. This
checkpoint was fetched while independent later families continued on the VM;
the final rental teardown receipt is separate and not claimed by this record.

`admission.json` pins all 18 CPU/Apple/AMD source columns and records strict
scoped admission. Exactly 54 lane/fixture cells changed; all **2,070 unrelated
cells remained byte-equivalent as decoded JSON**. Each selected cell carries
all eight required parts and three agreeing device classes. No old property was
dropped. `admit.py` reproduces the merge from the pinned pre-admission table.

The six routes now have real diagnostic verifier references. They remain
qualification-pending, **not default promoted**: NVIDIA property/model evidence
and installed CPU verifier replay remain owed. This is source qualification,
not a final 0.8.7 wheel or a full current four-vendor certificate.

## Independent native inference control

A compiled CPU estimators sabotage binding was used to load the **unchanged AMD
GPU saved models**. `classical_host_gate.py check --expect-mismatch
--every-fixture` observed changed numerical CPU outputs in **54/54 fixtures**,
with zero unmoved cases. Runtime 0.61 seconds under the local slot, one worker.
The binding SHA, exact command, report and log are under cpu-saved-model-fault.
This is in addition to the prior 54/54 native CPU training controls. No fresh
AMD GPU fault build or complete release-artifact fault certificate is claimed.
