# Handoff 2026-11-21 (wrap-up of the device=/check/boundary wave)

Read first in a new session. Live log of the wave: `~/mojolearn-evidence/briefs-20261118/FOLLOWUPS.md`.
Future work (held projects, categorical/CatBoost plan, owed small items): `docs/plans/FUTURE_WORK_2026-11-21.md`.

## State

- Integration branch `preinteg/wave2-20261113` (worktree `~/mojolearn-wt/preinteg-wave2`), head at this commit,
  pushed. All lanes of the wave are merged into it. No lanes running.
- main has NOT moved: it receives the integration branch whole, by fast-forward, only after a green CPU-box
  compile check.
- Running: CPU compile check `cpc1-0001` on the RunPod CPU box (pod `w9siah2val466b`, 32 vCPU, idle self-delete),
  of `1058399518`, which is one commit before the dw-causal merge (`0ebb1cec6e`). Read it with
  `bash tools/cpu_central.sh summary cpc1-0001` from the integration worktree.

## What we are waiting on, and the next steps

1. `cpc1-0001` result.
   - Green: run one more check of the integration head (it adds dw-causal and these docs), then fast-forward main
     to it and push.
   - Red: hand the error lines to the compile-fix lane (`fx-wave-compile`, brief in the briefs folder, Opus), merge
     its fix, check again on the CPU box. Repeat until green. Never compile on the laptop.
2. After green: one GPU run of `tools/ops/feature_proof_job.sh` (only with Andrew's go).
3. Nothing else starts without Andrew: see FUTURE_WORK for the held projects.

## Done in this wave (all on integration, compile pending)

- `device=None|"gpu"|"cpu"` through one route (`core/device_route.mojo`,
  `providers/python/device_route_boundary_door.mojo`), admitted centrally; generated class table
  `tools/device_class_table.py` (94 classes run cpu beside gpu, 9 not connected, 6 take no device=).
- `model.check(X)` on both devices in one process.
- Receipts: `model.receipt(X, y, path=...)`, `mojolearn.reproduce(...)`, `python -m mojolearn reproduce` (exit 0
  same, 1 different, 3 fast refused).
- Python boundary split (UNPACK / RUN / PACK) across linear, SVM, kernel, PLS, decomposition, clustering,
  neighbors, GP/metrics/manifold, prep/encoders, forests, boosting, forecasting, sequence, Samba, generation,
  cross-vendor and parallel doors; CausalLM engine/generate/stream/snapshot/resume (partial).
- One member ABI defined (`core/member_abi.mojo`); rollout held.
- `fast` tf32 winner promoted on every vendor (AGENTS.md rule, 2026-11-20).
- Paper: reproduce feature and related work, mlsys main `b75e2d2`.

## Behaviour changes to tell Andrew (in FOLLOWUPS)

NaN-mean search candidates rank last, tied; CausalLM checkpoint fingerprint hashed once per model generation
(in-place weight edits after the first fingerprint are not caught; recommended to accept); AutoARIMA error order;
gbdt non-str `model_` refused; split fits publish only after fit; manifest float spelling; refusal ordering in
several doors.
