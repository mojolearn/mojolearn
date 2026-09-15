# lane/ties-sabotage status

Goal: every hashed cell of knn-cosine, knn-rbc, radius, radius-manhattan, ivf
and ivf-euclidean moves under `-D MOJOLEARN_HOST_SABOTAGE=1` on every fixture,
`ties` included, in the identity sabotage column and in the classical host gate
recordings; then the saved-model sabotage step requires every fixture.

## Done

- `core/knn_host_predict.mojo`: `host_sabotage_value_flip` (bits always differ)
  on every L1, Lp and cosine cell distance, every ball cover edge distance and
  every ball cover k-NN distance, sabotage builds only.
- `ivf/host/ivf_host.mojo`: `ivf_sabotage_value_flip` on every distance
  `host_ivf_search` returns, after the root, sabotage builds only.
- `tools/classical_host_gate.py`: `sabotage_verdict`, `--every-fixture` and
  `--lane-rule-only LANE`. `tools/test_cpu_identity_gate.py` `SabotageVerdictTests`
  (fail against a copy without the every-fixture branch).
- `.github/workflows/cpu-identity-gate.yml`: the saved-model sabotage step
  uses `--every-fixture`, no exemption.
- Evidence: `bench/results/identity_break/2026-09-15_ties-sabotage/README.md`
  (one RunPod CPU pod, DELETE verified). Old arms 166 of 216 parts moved, new
  arms 180 of 216; the 36 left are the neighbor lanes' saved-file hashes, which
  no host arm reaches. Production IDENTICAL to the records on all nine fixtures.

## Open (not changed here)

- The owed check fails on the 18 radius and radius-manhattan model file hashes
  under both the old and new arms (no host sabotage arm can move a saved index
  file). Needs a decision: a GPU record carrying those cells, or an owed rule
  for file hashes.

## Remaining

- Merge to main: docs_facts --check, wheel_ci pins, test_cpu_identity_gate; push
  HEAD:main; remove the worktree.
