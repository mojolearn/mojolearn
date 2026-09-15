# lane/ties-sabotage status

Goal: every hashed cell of knn-cosine, knn-rbc, radius, radius-manhattan, ivf
and ivf-euclidean moves under `-D MOJOLEARN_HOST_SABOTAGE=1` on every fixture,
`ties` included, in the identity sabotage column and in the classical host gate
recordings; then the saved-model sabotage step requires every fixture.

## Done (on the branch)

- `core/knn_host_predict.mojo`: `host_sabotage_value_flip` (bits always differ)
  on every L1, Lp and cosine cell distance, every ball cover edge distance and
  every ball cover k-NN distance, sabotage builds only.
- `ivf/host/ivf_host.mojo`: `ivf_sabotage_value_flip` on every distance
  `host_ivf_search` returns, after the root, sabotage builds only.
- `tools/classical_host_gate.py`: `sabotage_verdict`, `--every-fixture` and
  `--lane-rule-only LANE` (the looser rule kept for a named lane).
  `tools/test_cpu_identity_gate.py` `SabotageVerdictTests`.
- `.github/workflows/cpu-identity-gate.yml`: the saved-model sabotage step
  uses `--every-fixture`.

## Owed

- One RunPod CPU pod (`tools/runpod_cpu_leg.sh`, lane tag `ties-sabotage`):
  old arms (origin/main sources rebuilt on the pod) then new arms; per lane and
  fixture counts; production diff against the committed records; owed check;
  gate on the new rule. Command file in the session scratchpad
  (`ties_leg_cmd.sh`); results under `bench/results/identity_break/2026-09-15_ties-sabotage`.
- If a recording lane other than the six fails `--every-fixture`, name it with
  `--lane-rule-only` in the workflow and list it here.
- Merge to main: docs_facts --check, wheel_ci pins, test_cpu_identity_gate.
