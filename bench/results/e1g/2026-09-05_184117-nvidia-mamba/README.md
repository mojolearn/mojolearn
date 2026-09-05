# NVIDIA fixed UMAP: PASS; Mamba3 L65 intermediate gate: RED

RTX 4090, source `d88c788374a06597c9ebd457666afc7e4054d0d9`.
The baseline certificate passes all five cases and retains 54 gradient tensors.
All 24 follow-up rows except the final long-sequence row passed: the targeted
self-neighbor regression, UMAP stage/transform/sparse checks, public UMAP
API suites, all six expanded quality fixtures in each numeric mode, both kNN
dispatch checks, and the Mamba forward/state API.

Mamba1 L64 passed. Mamba3 L65 has a complete ten-leaf public inventory, and
none of those leaves is reported as failing by the strict comparison.
However, 13 intermediate tensors fail the existing float32 oracle tolerances;
maximum absolute discrepancies range from 2.503e-6 to 4.578e-5. The complete
long-sequence certificate is RED and must not be promoted from byte capture
success alone. See `remote/followup/mamba-long-cert/mamba3-l65/gate.log`.
No threshold was relaxed and no failing tensor was removed from the gate.

The native S14/S15/S16 kernels use explicit serial arithmetic; the corresponding
oracle uses PyTorch reductions/contractions. Reduction order is a candidate
explanation, not yet a demonstrated cause. Public byte capture did not retain
the failing temporary intermediate arrays on this run, so the matching AMD
leg explicitly retains them. Future certificate tooling retains all diagnostics
on either numerical failure or inventory failure.

The controller fetched artifacts, returned failure, and deleted pod
ea3wnaz73kr7nr with verified HTTP 404. See the
[controller log](../../resume/2026-09-05-next-certification/nvidia-fixed-controller.log).
