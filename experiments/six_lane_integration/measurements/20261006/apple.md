# Candidate A/B measurements

Complete full-workload executions are retained separately from quality and identity admission. Failed attempts preserved at controller-qualified paths; no default promotion.
Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Default decisions are recorded beside source toggles; this board does not change them.

| Candidate | Mode | Measurement status | Captured pairs |
|---|---|---|---:|
| AF.X.complete-proposed | fast | FAILED_OR_INCOMPLETE, PENDING_ADMISSION, QUALITY_FAILED | 0 |
| I.X.complete-proposed | identical | PENDING_MEASUREMENT | 0 |

## Captured evidence

| Candidate | Vendor / route | Case | Scope | Status | B/A time | Evidence |
|---|---|---|---|---|---:|---|
| AF.X.complete-proposed | apple/apple-fast | classical/ols@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--runs/440956ba5f3a721465b8/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | classical/ols@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--runs/4a650353219f94bd483f/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | classical/kmeans@dataset=istella/attempt-0001 | full_workload | FAILED_OR_INCOMPLETE | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--runs/69a08c873718b0d710ad/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | classical/kmeans@dataset=istella/attempt-0002 | full_workload | FAILED_OR_INCOMPLETE | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--runs/69a08c873718b0d710ad/attempt-0002/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | classical/pca@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--runs/75455971e66712016083/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | classical/pca@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--runs/918eace39f801d37fc21/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | classical/kmeans@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--kmeans-repair2--runs/449522fb6bd97f700c04/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | classical/kmeans@dataset=istella/attempt-0001 | full_workload | QUALITY_FAILED | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--kmeans-repair2--runs/69a08c873718b0d710ad/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | algos/resample@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--resample-full--runs/047898b203cef197f8fa/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple/apple-fast | algos/resample@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--resample-full--runs/661a766183dd2fb07d65/attempt-0001/receipt.json |

## Campaign notes

- A=candidate; B=incumbent. Timed evidence is pending admission, not a default promotion.
- These are combined-configuration full workloads, not completed individual constituent experiments.
- Saved independent quality review: all12 preserve baseline metrics; 4 task-metric gates pass, 6 taxi opponent comparisons pending (historical4m vs current5.25m rows), Apple Istella KMeans fails best-opponent gate, NVIDIA inherits opponent-quality deficit.
- One excluded warmup and one scored sample per arm. Original failed attempts are retained.
- NVIDIA PTX and AMD have no compatible retained artifacts; missing-only build question remains pending.
- IDENTICAL compares each same arm across vendors; unavailable typed complete model state remains incomplete.
- Scored output and partial/public-save model hashes are retained separately; partial hashes do not prove complete state identity.
- Apple first four PCA/OLS pairs overlapped shared external-storage data transfer; KMeans overlap unestablished. No quiet-storage or promotion claim.
- No compilation or separate numerical verification rerun. Full provider and worker logs remain under /Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006
