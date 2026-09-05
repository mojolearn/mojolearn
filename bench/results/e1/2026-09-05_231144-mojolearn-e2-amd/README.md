# Matching AMD UMAP qualification and Mamba3 defect localization

AMD MI325X, source `d88c788374a06597c9ebd457666afc7e4054d0d9`.
All UMAP checks and 18 expanded mode/fixture cases passed. With the matching
NVIDIA run, both native fixtures match (186 and 690 uint32 cells), all six
held-out IDENTICAL inputs and training/query embeddings match, and all
36 vendor/mode/quality cases passed. See the
[comparison](../../resume/2026-09-05-next-certification/fixed-cross-device.json).

The five baseline rows were GREEN under the then-current oracle, with 54
native gradient tensors matching NVIDIA exactly. Mamba1 L64 was GREEN;
Mamba3 L65 was RED on 13 intermediate tensors. This is not a complete passing
backward certificate, and the historical Mamba3 baseline gate is superseded
by the independent-gradient correction described below.

Full temporary native/oracle arrays are retained under
`diag/followup/mamba-long-cert/mamba3-l65/diagnostic-{actual,oracle}`.
Their investigation found more than reduction rounding: the native
`mamba3_beta_join_kernel` omitted the scale->gamma->dt/trap_raw branch,
and the staged float32 oracle repeated that omission. The independently
differentiated float64 forward rejects four old public leaves: x,
block_norm.weight, in_proj.weight and dt_bias. The old native bytes therefore
cannot certify the full Mamba3 gradient merely because they match NVIDIA.
The new exact two-token regression and additional whole-forward gate target
that defect. Neither drops the failing intermediate checks nor widens tolerances.
See the [diagnostic comparison](../../resume/2026-09-05-next-certification/mamba3-long-diagnostics.json)
and [strengthened gate on old bytes](../../resume/2026-09-05-next-certification/mamba3-old-bytes-strengthened-gate.log).

All diagnostics were collected before droplet 598112072 was deleted;
HTTP 404 verified its absence. See the
[controller log](../../resume/2026-09-05-next-certification/amd-matching-controller.log).
