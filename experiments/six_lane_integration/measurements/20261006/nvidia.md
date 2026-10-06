# Candidate A/B measurements

Complete full-workload executions are retained separately from quality and identity admission. Failed attempts preserved at controller-qualified paths; no default promotion.
Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Default decisions are recorded beside source toggles; this board does not change them.

| Candidate | Mode | Measurement status | Captured pairs |
|---|---|---|---:|
| I.X.complete-proposed | identical | FAILED_OR_INCOMPLETE, PENDING_ADMISSION | 0 |

## Captured evidence

| Candidate | Vendor / route | Case | Scope | Status | B/A time | Evidence |
|---|---|---|---|---|---:|---|
| I.X.complete-proposed | nvidia/native-sm90 | more:gmm@dataset=taxi/attempt-0001 | full_workload | FAILED_OR_INCOMPLETE | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/03f523ab5951d08a295b/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | more:elasticnet@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/14941dbe10b9bd78a1e4/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | more:lasso@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/1b07971bfe465c3ca93f/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | more:lasso@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/1e17ae43f1c2de4a7171/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | more:ridge@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/4238c569b4eada2d0199/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | more:linearsvr@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/49c28f55539ef700a59f/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | more:linearsvr@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/55b2b5ab6491c0d7e9fe/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | more:elasticnet@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/71d06cebb7470b76364a/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | more:ridge@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-next-reg/80deb286d318651dcdf8/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | classical:ols@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements/201c5234acc7b0ac4689/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | classical:ols@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements/5df5c5720c0d6d872c16/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | classical:kmeans@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements/8af58db47601e372485e/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | classical:pca@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements/aaee5fbb8a39edad255c/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | classical:pca@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements/d93173a97b8d0e0dd273/attempt-0001/receipt.json |
| I.X.complete-proposed | nvidia/native-sm90 | classical:kmeans@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | — | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements/fb6aa398a4fe2f787945/attempt-0001/receipt.json |

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
