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

## x86, RunPod CPU pod (AMD EPYC 4564P, commit b2ac52fa2)

Pod 2bbnf9dh15m8kg, 8 vCPU, bindings built there (`bindings-x86-pod1.sha256`),
verified deleted, $0.03. Same command, diffed on the Mac:

    summary: IDENTICAL=36
    summary (infer/model): IDENTICAL=72
    summary (batch): IDENTICAL=36
    require-columns 4 over the four lanes: OK (0 OWED)

Files: `cpu-x86-epyc-4564p-host-infer.json`, `diff-x86-epyc-4564p-host-infer.txt`,
`owed-x86-epyc-4564p-host-infer.json`, `pod1-timings.tsv`.

## Installed test wheel, x86 (commit b2ac52fa2 source, pod hikr33gb9v4fnz)

Verified deleted, $0.016. `tools/inf_gbdt_wheel_models.py dump` saved the 36
CPU-trained models (their bytes are the GPU columns' saved files, by the model
cells above) and held-out rows. A wheel was built from `python/` in the pixi
`pkg` env with only `_mojolearn_forest_host.so` under `mojolearn/host/`
(`wheel-contents-x86.txt`) and unpacked into an isolated target (no pixi env
carries pip). `check`, importing `mojolearn` from that target, predicted each
model through `host_model` and hashed it as identity_break does:

    summary: IDENTICAL=36 DIFFER=0      (wheel-check-x86.txt)

The same installed package with the forest sabotage binary
(`-D MOJOLEARN_FOREST_HOST_SABOTAGE=1`, sha256 3a05b7ff...):

    summary: IDENTICAL=0 DIFFER=36      (wheel-check-x86-forest-sabotage.txt)

## Negative control on the identity column

That pod's forest sabotage identity column read IDENTICAL and is NOT
evidence: identity_break's host_record had already loaded the production forest
binding from `MOJOLEARN_HOST_DIR` under the module name `_forest_host` reuses,
so the sabotage binary never predicted. The column was removed, and
`_probe_fit_host` now refuses a host model whose bound module is not
`MOJOLEARN_FOREST_HOST_BINARY`. The rerun is below when present.
