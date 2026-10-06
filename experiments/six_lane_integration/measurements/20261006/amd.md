# Candidate A/B measurements

Complete full-workload executions are retained separately from quality and identity admission. Failed attempts preserved at controller-qualified paths; no default promotion.
Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Default decisions are recorded beside source toggles; this board does not change them.

| Candidate | Mode | Measurement status | Captured pairs |
|---|---|---|---:|
| I.X.complete-proposed | identical | PENDING_MEASUREMENT | 0 |

## Captured evidence

| Candidate | Vendor / route | Case | Scope | Status | B/A time | Evidence |
|---|---|---|---|---|---:|---|

## Campaign notes

- A=candidate; B=incumbent. Timed evidence is pending admission, not a default promotion.
- These are combined-configuration full workloads, not completed individual constituent experiments.
- Initial 12-pair quality review: all12 preserve baseline metrics; 4 task-metric gates pass, 6 taxi opponent comparisons pending (historical4m vs current5.25m rows), Apple Istella KMeans fails best-opponent gate, NVIDIA inherits opponent-quality deficit. Additional saved assessments are retained in next-quality-review.json.
- One excluded warmup and one scored sample per arm. Original failed attempts are retained.
- NVIDIA PTX and AMD have no compatible retained artifacts; missing-only build question remains pending.
- IDENTICAL compares each same arm across vendors; unavailable typed complete model state remains incomplete.
- Scored output and partial/public-save model hashes are retained separately; partial hashes do not prove complete state identity.
- Apple first four PCA/OLS pairs overlapped shared external-storage data transfer; KMeans overlap unestablished. No quiet-storage or promotion claim.
- No compilation or separate numerical verification rerun. Full provider and worker logs remain under /Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006

## Recorded source decisions

| Candidate / arm | Decision | Source commit | Evidence |
|---|---|---|---|
| AF.X.complete-proposed/classical/kmeans@dataset=istella/attempt-0001 | NOT PROMOTED: Candidate and baseline both have worse inertia than retained same-data sklearn; deficit is inherited, not introduced by candidate. | 51a3eb11bd99b921e775fa5fc6f6dbedca125382 | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--kmeans-repair2--runs/69a08c873718b0d710ad/attempt-0001/receipt.json |
| AF.X.complete-proposed/algos/huber@dataset=istella/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | 47301d12b14859e81cadc9ab6a0cd4f728d0e206 | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--expanded-reg--runs/4fa48ab7783acc171486/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:enet-cv@dataset=taxi/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | 1c773404b24dcb05b9fd4684d5f4c8654c13f780 | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-expanded-reg/3392ecb27af9675963d9/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:lasso-cv@dataset=taxi/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | 1c773404b24dcb05b9fd4684d5f4c8654c13f780 | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-expanded-reg/96ac4d3d170baa9b8709/attempt-0001/receipt.json |
