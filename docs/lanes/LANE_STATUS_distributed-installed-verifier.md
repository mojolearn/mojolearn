# Installed distributed verification

2026-09-18, `lane/distributed-installed-verifier`, based on fe2ea048b.

The implementation now ships as `mojolearn._verify_distributed`; the existing
`tools/distributed_classical_check.py` is a CLI-compatible thin wrapper with no
source-path injection. Root integration supplies the public `verify-distributed`
command. Module and wrapper accept:

```
--devices 0,1 --out new-capture.json [--require-installed]
--compare first.json second.json
```

The tiny profile fits ARIMA(5 series x32), Holt-Winters(5x24), multiclass GPC(9x3,
3 classes), and IVF(65x17,4 lists,3 fit iterations). It compares five operations
(ARIMA/Holt-Winters prediction, GPC prediction/fit, distributed IVF search)
against canonical one-GPU numerical results, on one/two/reversed device layouts,
twice:30 clean cells. Cells checkpoint atomically as each completes. A failure
retains completed cells and a failed final receipt, never an implicit pass.

Ten real transport controls drop/reverse returned result batches across the five
operations; they must actually trigger and be detected numerically or refused.
They are not native arithmetic sabotage controls. Native faults and physical
kernel traces remain explicitly OWED.

Every actual worker pool is inventoried through CUDA/HIP driver UUID/PCI/PID
queries, using the unchanged `_gpu_witness.py` and tests ported from the CV lane.
Repeated physical devices (including same-PCI MIG instances) are refused. Every
worker must receive the operation appropriate to its numerical case; inventory
alone or an unused allocated worker does not satisfy the gate. This is numerical
plus placement evidence, not a claim that an external GPU trace was captured.

`--require-installed` rejects editable installs, source-shadowed imports, source
modules differing from wheel RECORD and native bindings outside/differing from
wheel RECORD. Standard `_verify.environment` metadata, source/module hashes,
binding hashes/vendor, checkout commit, and separately validated guarded archive
`MOJOLEARN_COMMIT` are retained. An installed wheel with no checkout does not
invent a source commit. NumPy is required for these tiny verification fixtures.

44 binding-free tests pass, including driver ABI failures/aliases, missing cells,
unused workers, wrong operation, missing controls, digest drift, installed source/native RECORD mismatches, transport fault
restoration and strict cross-capture comparison. Actual NVIDIA/AMD execution of
this new shipped orchestration remains owed; no rentals started in this lane.

Fixture input byte digests are required and compared independently of outputs.
The profile requires the complete eight-module source manifest; truncated
manifests cannot make two partial captures appear equivalent.
