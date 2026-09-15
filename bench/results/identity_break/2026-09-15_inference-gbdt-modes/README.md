# Public CPU inference for the newest GBDT modes (2026-09-15)

lane/inference-gbdt-modes. The held-out predictions of saved models of the
gbdt-ordered-rmse, gbdt-feature-freq, gbdt-pointwise-l2-bayesian-eval and
gbdt-categorical-ctr lanes, predicted through `mojolearn.host_model` (HostGBDT
on the shipped forest host binding), against the three committed GPU columns
of `bench/results/identity_break/2026-09-14_166-lanes/` (Apple M4 Metal,
NVIDIA H100 sm_90a, AMD MI325X gfx942).

How the column was taken: `tools/identity_break.py` with
`MOJOLEARN_IDENTITY_HOST_INFER` naming the four lanes. Each fit runs on the
CPU reference path; the model is saved; the infer cell is the lane's held-out
probe asked of `host_model(<saved file>)`; the model cell is the saved file's
hash (so the file predicted from is byte for byte the file the GPU columns
saved); the reload cell is the class's own `load`.

## Apple M4, one core (commit 479a9575e)

Bindings built from that commit with one job (`bindings-apple-m4.sha256`).

    diff --require-columns 4 --owed-json, lanes scoped to the four:
    summary: IDENTICAL=36
    summary (infer/model): IDENTICAL=72
    summary (batch): IDENTICAL=36
    require-columns 4 over the four lanes: OK (0 OWED)

Files: `cpu-apple-m4-host-infer.json`, `diff-apple-m4-host-infer.txt`,
`owed-apple-m4-host-infer.json`.

The x86 column, the forest sabotage column and the installed-wheel check ran
on one RunPod CPU pod; their results are in the sections below when present.
