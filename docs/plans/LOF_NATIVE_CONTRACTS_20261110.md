# LOF native plan contracts

Dependency: product producer `a9acd91a2c62aa0d3fa73178f85498e2a575fc2d`.
Entry point: `checks/native_tests/lof_plan_contract.mojo`.
Suite: `checks/native_tests/lof_plan_contract_suite.mojo`.

Twenty-eight independent fixtures are authored and source-reviewed, uncompiled
and unrun. `host_parallelize` schedules fixture units; each owns its buffers and
error outputs, with no shared mutable fixture state. There is no GPU dependency,
timing, identity run or box job.

The descriptor witness checks ABI fields and writes fixed markers. It does not
implement k-NN, sorting, densities, scores or classification. It cannot establish
numerical correctness of those operations; the separate `lof_native_cpu` suite
owns actual provider execution and independent numerical references.

| Units | Contract |
| --- | --- |
| 0–5 | Exact tiled/legacy/unfused operation IDs, exclusion flags, unfused priority, native status propagation, fresh scratch, matrix lifetime, release of squared scratch after roots while indices remain owned. |
| 6–7 | Fused/unfused staged query buffers; late allocation; independent reference-row and fitted-row metadata; density/score/decision address order and owner lifetime. |
| 8–10 | Both sign modes and their threshold words; actual shared native int32-to-int64 conversion; empty labels skip the descriptor and accept a null empty destination. |
| 11 | Empty scratch keeps address zero; zero-row shape remains empty. |
| 12–17 | Six hand-calculated percentiles of `[-9, -3, 1, 5]`, including endpoints, exact knots and both interpolation branches. The witness supplies those sorted values; it does no sorting. |
| 18 | Contamination `0.1` produces the once-rounded Float32 `-7.2` word `0xc0e66666`, then widens that rounded value. |
| 19–21 | Negative dimensions, element-count overflow and byte-count overflow retain the existing exact native Array refusal text. |
| 22–25 | Root, density, percentile and unfused-first-op errors propagate before later stages; earlier owners remain live and later scratch remains unallocated. |
| 26–27 | Repeated selection initializes fresh private scratch; moving a plan keeps both buffer owners alive through the following native call. |

No product source changed in this fixture commit. No existing Python boundary
case or independent numerical oracle was removed: these contract witnesses do
not replace numerical evidence. The new product's eight definitions / 18 public
boundary cases remain separate and unrun. Numerical LOF Python test migration
belongs to the actual-provider fixture handoff.

Evidence:
`~/mojolearn-evidence/python-mojo-migration-20261110/lof-native-contracts/` retains
the full source snapshot, patch and commit/push logs. Root must compile this
entry and run it once with its bundled native correctness batch before reporting
a runtime result. The upstream shared MAX/Metal concurrent-context failure is
unrelated to this pure native suite and remains parked without a workaround.
