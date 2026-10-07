# Candidate A/B measurements

Complete full-workload executions are retained separately from quality and identity admission. Failed attempts preserved at controller-qualified paths; no default promotion. Hash receipts alone do not establish retained array/model bytes; see artifact-retention.json.
Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Default decisions are recorded beside source toggles; this board does not change them.

| Candidate | Mode | Measurement status | Captured pairs |
|---|---|---|---:|
| AF.X.complete-proposed | fast | FAILED_OR_INCOMPLETE, PENDING_ADMISSION, QUALITY_FAILED | 86 |
| I.X.complete-proposed | identical | PENDING_MEASUREMENT | 0 |

## Full-workload coverage

Counts are retained complete A/B pairs, not individually decided experiment switches. Execution completion does not establish quality or identity admission.

| Vendor / mode | Complete pairs | Original failed attempts | Quality-rejected pairs | Remaining scope |
|---|---:|---:|---:|---|
| Apple / FAST | 86 | 3 | 5 | See unrun scope for additional FAST pairs; earlier array-preservation and quality limitations remain. |

## Individual experiment coverage

[Itemized coverage ledger](REMAINING.md): 375 catalog entries and 128 interaction plans; only 2 exact selections have receipts in this campaign. Combined timings do not qualify individual members.

Unrun, source-rejected and previously decided work remain distinct. This ledger is not a claim that every listed entry has runnable binaries.

## Captured evidence

Observed ratios retain complete scored pairs even while quality or identity is pending. They are not admitted gains or default decisions. A is candidate; B is baseline.

[Exact toggle profiles and per-attempt quality details](#experiments-run-toggles-timing-and-quality) are at the bottom. Times below are seconds; A is candidate and B is incumbent.

| Candidate | Vendor / route | Case | Scope | Status | A (s) | B (s) | Admitted A/B | Observed A/B | Quality (A/B) | Toggles / details | Evidence |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| AF.X.complete-proposed | apple/apple-fast | classical/ols@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.534309 | 0.531459 | — | 1.0054 | BASELINE_NONREGRESSION_OPPONENT_EVIDENCE_PENDING; finite A/B=true/true; r2 A/B=0.908775/0.908775; rmse A/B=4.69807/4.69807 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-a32782c39b49d75b) | [retained receipt](<receipts/apple/apple--captured--runs/440956ba5f3a721465b8/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical/ols@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.58948 | 1.59958 | — | 0.9937 | TASK_METRIC_GATE_PASSED; finite A/B=true/true; r2 A/B=0.331943/0.331943; rmse A/B=0.682027/0.682027 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-c6dd0878e8d6b977) | [retained receipt](<receipts/apple/apple--captured--runs/4a650353219f94bd483f/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical/kmeans@dataset=istella/attempt-0001 | full_workload | FAILED_OR_INCOMPLETE | — | — | — | — | NOT_ASSESSED; metrics not recorded | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-322ccbf903df0c22) | [retained receipt](<receipts/apple/apple--captured--runs/69a08c873718b0d710ad/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical/kmeans@dataset=istella/attempt-0002 | full_workload | FAILED_OR_INCOMPLETE | — | — | — | — | NOT_ASSESSED; metrics not recorded | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-c8a6ac85f8cf9cb3) | [retained receipt](<receipts/apple/apple--captured--runs/69a08c873718b0d710ad/attempt-0002/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical/pca@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.498951 | 0.500522 | — | 0.9969 | BASELINE_NONREGRESSION_OPPONENT_EVIDENCE_PENDING; explained_variance_ratio_sum A/B=0.999997/0.999997 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-91aaa1753ba2f73e) | [retained receipt](<receipts/apple/apple--captured--runs/75455971e66712016083/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical/pca@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.7872 | 1.14723 | — | 1.5578 | TASK_METRIC_GATE_PASSED; explained_variance_ratio_sum A/B=1/1 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-c1ffb3d13c13dc5e) | [retained receipt](<receipts/apple/apple--captured--runs/918eace39f801d37fc21/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical/kmeans@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 3.31906 | 1.66506 | — | 1.9934 | PENDING; inertia A/B=4.01988e+08/4.01988e+08; n_iter A/B=84/84 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-288af0845518e7db) | [retained receipt](<receipts/apple/apple--captured--kmeans-repair2--runs/449522fb6bd97f700c04/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical/kmeans@dataset=istella/attempt-0001 | full_workload | QUALITY_FAILED | 2.38934 | 2.36248 | — | 1.0114 | FAILED_FAST_OPPONENT_GATE; inertia A/B=6.05072e+17/6.05072e+17; n_iter A/B=33/33 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-3cb7b17e87b2b93c) | [retained receipt](<receipts/apple/apple--captured--kmeans-repair2--runs/69a08c873718b0d710ad/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/resample@dataset=istella/attempt-0001 | full_workload | QUALITY_FAILED | 2.29268 | 2.29334 | — | 0.9997 | QUALITY_FAILED; max_mean_shift_over_std A/B=0.00194692/0.00194692 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-e10674a2304dcc24) | [retained receipt](<receipts/apple/apple--captured--resample-full--runs/047898b203cef197f8fa/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/resample@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.75579 | 0.740297 | — | 1.0209 | TASK_METRIC_GATE_PASSED; max_mean_shift_over_std A/B=0.000818294/0.000818294 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-62c30d7b110ec4fe) | [retained receipt](<receipts/apple/apple--captured--resample-full--runs/661a766183dd2fb07d65/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical2/linearsvr@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.27067 | 1.3157 | — | 0.9658 | PENDING; finite A/B=true/true; r2 A/B=-0.107028/-0.107028; rmse A/B=0.87796/0.87796 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-454ec65777dc4431) | [retained receipt](<receipts/apple/apple--captured--reg-full--runs/05207b357bcbe067ca6a/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical2/ridge@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.64688 | 1.64271 | — | 1.0025 | TASK_METRIC_GATE_PASSED; finite A/B=true/true; r2 A/B=0.332534/0.332534; rmse A/B=0.681726/0.681726 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-74b2f6f2bb84e966) | [retained receipt](<receipts/apple/apple--captured--reg-full--runs/6d77306d88d72aab7b44/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical2/linearsvr@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.606728 | 0.607164 | — | 0.9993 | PENDING; finite A/B=true/true; r2 A/B=0.899284/0.899284; rmse A/B=4.93641/4.93641 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-4557bd5b84099a1e) | [retained receipt](<receipts/apple/apple--captured--reg-full--runs/71dfc119e59c074352bb/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical2/ridge@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.608297 | 0.588781 | — | 1.0331 | TASK_METRIC_GATE_PASSED; finite A/B=true/true; r2 A/B=0.908772/0.908772; rmse A/B=4.69815/4.69815 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-60094147f639ef4f) | [retained receipt](<receipts/apple/apple--captured--reg-full--runs/94c802682d07720f1572/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical2/gmm@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 2.67696 | 1.84056 | — | 1.4544 | TASK_METRIC_GATE_PASSED; bic A/B=-1.88789e+08/-1.88789e+08; mean_log_likelihood A/B=11.6016/11.6016; n_iter A/B=24/24 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-7927b956ba0b72c1) | [retained receipt](<receipts/apple/apple--captured--reg-full--runs/d4f24411b4f604c84997/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/bayesian-ridge@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.67635 | 1.65577 | — | 1.0124 | PENDING; finite A/B=true/true; r2 A/B=0.33211/0.332103; rmse A/B=0.681943/0.681946 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-de5212f72587e628) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/11f34b0bacc6a216cbf5/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/bayesian-ridge@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.602563 | 0.600014 | — | 1.0042 | PENDING; finite A/B=true/true; r2 A/B=0.908772/0.908772; rmse A/B=4.69813/4.69813 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-c5e9f9f7b828c55f) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/121d2bf7022b542e4d6e/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/lars@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.9917 | 2.00181 | — | 0.9950 | PENDING; finite A/B=true/true; r2 A/B=0.332579/0.332252; rmse A/B=0.681703/0.68187 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-158387d3db293254) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/1ffab8882758257c9f23/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/sgd-reg@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 7.24818 | 7.19279 | — | 1.0077 | PENDING; finite A/B=true/true; r2 A/B=0.331677/0.331677; rmse A/B=0.682163/0.682163 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-5b6dea355b89327d) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/22c62826ae044e079978/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/lasso-lars@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.4024 | 1.39462 | — | 1.0056 | PENDING; finite A/B=true/true; r2 A/B=0.314676/0.314669; rmse A/B=0.690785/0.690789 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-beeef7a464f96181) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/3a40940bae1961555f78/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/lasso-cv@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.69191 | 1.63021 | — | 1.0378 | PENDING; finite A/B=true/true; r2 A/B=0.329438/0.329438; rmse A/B=0.683305/0.683305 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-5f43961bc834421b) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/3cfd7d59ee9c8efa1497/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/huber@dataset=istella/attempt-0001 | full_workload | QUALITY_FAILED | 2.40918 | 2.40432 | — | 1.0020 | QUALITY_FAILED; finite A/B=true/true; r2 A/B=-0.00708934/-0.00595219; rmse A/B=0.837393/0.83692 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-f3328cbf23ffeea8) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/4fa48ab7783acc171486/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/pa-reg@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 10.2012 | 10.1981 | — | 1.0003 | PENDING; finite A/B=true/true; r2 A/B=0.310359/0.310359; rmse A/B=0.692958/0.692958 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-541855c4db4d9e40) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/6fcf310a716926c2d853/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/lasso-lars@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.60632 | 0.602449 | — | 1.0064 | PENDING; finite A/B=true/true; r2 A/B=0.908794/0.908794; rmse A/B=4.69757/4.69757 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-5be9ee86049e552d) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/791dbcdb69bd137c8ef4/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/ridge-cv@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 25.2837 | 25.1195 | — | 1.0065 | PENDING; finite A/B=true/true; r2 A/B=0.332506/0.332506; rmse A/B=0.68174/0.68174 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-ff24a097f693b2ad) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/805874e87463199882f2/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/lars@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.599341 | 0.605434 | — | 0.9899 | PENDING; finite A/B=true/true; r2 A/B=0.908772/0.908772; rmse A/B=4.69813/4.69813 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-35e944e0962689e1) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/980247052f23e672228d/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/enet-cv@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.628106 | 0.62274 | — | 1.0086 | PENDING; finite A/B=true/true; r2 A/B=0.908812/0.908812; rmse A/B=4.69711/4.69711 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-a2e3080382d098f0) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/9f64ff019913cdb9b802/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/enet-cv@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.63416 | 1.65814 | — | 0.9855 | PENDING; finite A/B=true/true; r2 A/B=0.330799/0.330799; rmse A/B=0.682612/0.682612 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-2d655de64969c832) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/b1e39171e9fbf74e3e34/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/ridge-cv@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.680501 | 0.681994 | — | 0.9978 | PENDING; finite A/B=true/true; r2 A/B=0.908772/0.908772; rmse A/B=4.69813/4.69813 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-ac28476b0071d590) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/ba6202e41915aa8b651c/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/sgd-reg@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 13.7573 | 13.8481 | — | 0.9934 | PENDING; finite A/B=true/true; r2 A/B=0.908771/0.908771; rmse A/B=4.69817/4.69817 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-f0865758948fef12) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/d3adb34325456bfba29c/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/pa-reg@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 8.70751 | 8.70015 | — | 1.0008 | PENDING; finite A/B=true/true; r2 A/B=0.90038/0.90038; rmse A/B=4.90947/4.90947 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-8e3c6a1a864efe0a) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/d74a68acdb33bdd77426/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/huber@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 2.27818 | 1.38404 | — | 1.6460 | PENDING; finite A/B=true/true; r2 A/B=0.899614/0.899614; rmse A/B=4.92832/4.92832 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-9a79641563131b6c) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/f0c4179bfa4fda6936f2/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/lasso-cv@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.777047 | 0.699416 | — | 1.1110 | PENDING; finite A/B=true/true; r2 A/B=0.908831/0.908831; rmse A/B=4.69662/4.69662 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-bef083c3b647560e) | [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/f2b21063131028aae994/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | classical2/gmm@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 151.587 | 152.32 | — | 0.9952 | PENDING; bic A/B=-8.53062e+08/-8.53062e+08; mean_log_likelihood A/B=210.071/210.071; n_iter A/B=33/33 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-be9c56d14626b19f) | [retained receipt](<receipts/apple/apple--captured--gmm-istella-full--runs/e54f3cfdf88e2e305a71/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/qn-reg@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.565719 | 0.565016 | — | 1.0012 | PENDING; finite A/B=true/true; r2 A/B=0.90876/0.90876; rmse A/B=4.69845/4.69845 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-6013393bb5e97f57) | [retained receipt](<receipts/apple/apple--captured--pls-qn-full--runs/177e591a0085ff705250/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/pls@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 4.05227 | 3.81507 | — | 1.0622 | PENDING; finite A/B=true/true; r2 A/B=0.294361/0.29436; rmse A/B=0.700949/0.700949 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-3f1ba240a3e2e646) | [retained receipt](<receipts/apple/apple--captured--pls-qn-full--runs/1f0fc8e14b88c7e0588c/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/pls@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 1.51965 | 1.18645 | — | 1.2808 | PENDING; finite A/B=true/true; r2 A/B=0.905574/0.905574; rmse A/B=4.77978/4.77978 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-47df8a4a5f60750d) | [retained receipt](<receipts/apple/apple--captured--pls-qn-full--runs/258012ade2a00a9645a9/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/qn-reg@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.72767 | 1.73084 | — | 0.9982 | PENDING; finite A/B=true/true; r2 A/B=0.3314/0.3314; rmse A/B=0.682305/0.682305 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-1efdeca8d0da4aaa) | [retained receipt](<receipts/apple/apple--captured--pls-qn-full--runs/e93854e1cf11c7c7adfb/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/randomized-svd@dataset=taxi@input=tsvd-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.37831 | 1.33345 | — | 1.0336 | PENDING; relative_reconstruction_error A/B=0.0272095/0.0272095 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-baa89f17be306cf6) | [retained receipt](<receipts/apple/apple--captured--tsvd-full-v1--runs/aad81edca8ccf47b9ecc/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/randomized-svd@dataset=istella@input=tsvd-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.78753 | 1.55219 | — | 1.1516 | PENDING; relative_reconstruction_error A/B=0.000229582/0.000229582 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-0911613709b5f581) | [retained receipt](<receipts/apple/apple--captured--tsvd-full-v1--runs/c8188af9b06dceb11c17/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/select-f-regression@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.537282 | 0.549767 | — | 0.9773 | PENDING; n_selected A/B=5/5 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-9421af25cd249ea1) | [retained receipt](<receipts/apple/apple--captured--selectors-full--runs/3bb7fcd8cc23a07afee7/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/select-r-regression@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.04811 | 1.04659 | — | 1.0015 | PENDING; n_selected A/B=110/110 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-27030ad779657eb1) | [retained receipt](<receipts/apple/apple--captured--selectors-full--runs/ab68cf756738987139b1/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/select-r-regression@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.54312 | 0.556438 | — | 0.9761 | PENDING; n_selected A/B=5/5 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-7d1d9f8dfabff0fe) | [retained receipt](<receipts/apple/apple--captured--selectors-full--runs/b9e816c552496edb13ef/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/select-f-regression@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 1.05952 | 1.05421 | — | 1.0050 | PENDING; n_selected A/B=110/110 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-1acc7d37a06507d4) | [retained receipt](<receipts/apple/apple--captured--selectors-full--runs/bceac841eda6c1e6cc53/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/power-transformer@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.32201 | 1.31524 | — | 1.0051 | PENDING; output_shape A/B=500000x11/500000x11 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-e320bd1611c96447) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/0f6d07d5f7a9fa8e86ec/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/simple-imputer@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 4.5063 | 4.52148 | — | 0.9966 | PENDING; masked_rmse A/B=348517/348517 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-fdd957e4ecd00693) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/10caf828bd0b75b5e3ca/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/ridge-clf@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.594427 | 0.597923 | — | 0.9942 | PENDING; accuracy A/B=0.764828/0.764828 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-630a225f29927d08) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/186156f500974335532d/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/nmf@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 6.20225 | 3.99489 | — | 1.5525 | PENDING; relative_reconstruction_error A/B=0.101238/0.101238 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-b9198196ba07ffb9) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/1e141f7a4b0c299d303b/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/sgd-clf@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 10.738 | 10.7216 | — | 1.0015 | PENDING; accuracy A/B=0.75694/0.75694 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-9082213724626923) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/21217534b9ec24fdec13/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/ridge-clf@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 4.91783 | 4.92755 | — | 0.9980 | PENDING; accuracy A/B=0.91056/0.91056 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-8a57f086dc9f2da5) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/2da7ce7cebbe7ae16fd0/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/factor-analysis@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 9.46465 | 9.44507 | — | 1.0021 | PENDING; mean_log_likelihood A/B=98.2948/98.2948 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-12eb27621ba6bf16) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/31a029c7a61734cee829/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/pls-canonical@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.29328 | 1.02896 | — | 1.2569 | PENDING; mean_canonical_corr A/B=0.557279/0.557279 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-9821b20e14150fa2) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/366d7ef45f2e1ab38840/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/cca@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.4188 | 1.17771 | — | 1.2047 | PENDING; mean_canonical_corr A/B=0.574828/0.574828 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-4e3ed81580ca885c) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/43f0120c502ae1170075/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/power-transformer@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 4.2317 | 4.2365 | — | 0.9989 | PENDING; output_shape A/B=500000x220/500000x220 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-34adcad1c2d9bfd1) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/50e1b5764b66cafe829d/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/sgd-ocsvm@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.70548 | 0.704507 | — | 1.0014 | PENDING; fraction_flagged A/B=0.066736/0.066736 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-ab70a58c367cac51) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/5b496fffda8d73e5cc95/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/maxabs-scaler@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.555044 | 0.539559 | — | 1.0287 | PENDING; output_shape A/B=500000x11/500000x11 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-2a21e87a2423c72e) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/7012c5ed73f2ea3883ad/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/pa-clf@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 10.1992 | 10.2073 | — | 0.9992 | PENDING; accuracy A/B=0.922242/0.922242 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-15b4ecb49cddb1d1) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/706c6c218583c2556f05/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/qda@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.870554 | 0.876028 | — | 0.9938 | PENDING; accuracy A/B=0.727134/0.727134; logloss A/B=1.0698/1.0698 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-1cf44a8254f4654b) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/83311ee493aec7159caa/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/pa-clf@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 6.9313 | 6.93296 | — | 0.9998 | PENDING; accuracy A/B=0.76288/0.76288 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-f9bccd951073f002) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/871ec7c64114eeac953f/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/factor-analysis@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.741826 | 0.74082 | — | 1.0014 | PENDING; mean_log_likelihood A/B=-14.8357/-14.8357 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-0b3bc089cb244b34) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/8aced037a81f1cc42846/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/maxabs-scaler@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.30746 | 1.27944 | — | 1.0219 | PENDING; output_shape A/B=500000x220/500000x220 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-db5834c9c27efd57) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/95058e058d6c35a7f5aa/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/nmf@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 27.9172 | 22.6487 | — | 1.2326 | PENDING; relative_reconstruction_error A/B=0.326038/0.326038 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-1ee4c1fbc8012177) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/9c048f5b19f91cda3a34/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/lda-clf@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.66971 | 0.634366 | — | 1.0557 | PENDING; accuracy A/B=0.76362/0.763624; logloss A/B=0.538246/0.538246 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-8d6e9a07d67df7b4) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/ac6458a148bfb4f9c011/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/categorical-nb@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.595268 | 0.597673 | — | 0.9960 | PENDING; accuracy A/B=0.767572/0.767572; logloss A/B=0.536612/0.536612 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-8e4baeb2c2495265) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/acff99fbeb6c301c3fc0/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/minibatch-kmeans@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.1869 | 1.19839 | — | 0.9904 | PENDING; n_clusters A/B=8/8; silhouette A/B=0.076493/0.076493 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-3927c5ee2b7a6960) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/ad3d45e08cb8ab6c00e7/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/pls-canonical@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 5.8376 | 4.63671 | — | 1.2590 | PENDING; mean_canonical_corr A/B=0.876301/0.876301 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-8e4ca1617bad6848) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/b0667caada66668f51aa/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/cca@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 41.4825 | 22.6234 | — | 1.8336 | PENDING; mean_canonical_corr A/B=0.989054/0.989054 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-b397bfed2fe659b4) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/b17b6e514c1006f88c5f/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/minibatch-kmeans@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.673498 | 0.654529 | — | 1.0290 | PENDING; n_clusters A/B=8/8; silhouette A/B=0.17146/0.17146 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-854651a37285160d) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/ba1006a727002dda6a90/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/perceptron@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 10.0619 | 10.0527 | — | 1.0009 | PENDING; accuracy A/B=0.874976/0.874976 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-1517e260d7d8b58f) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/bcdd603c81cc1f7814d2/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/multinomial-nb@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | FAILED_OR_INCOMPLETE | — | — | — | — | NOT_ASSESSED; metrics not recorded | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-56dabedb926fcb84) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/dc3d41d16eac72f9f13f/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/categorical-nb@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.553744 | 0.552733 | — | 1.0018 | PENDING; accuracy A/B=0.840176/0.840176; logloss A/B=0.411866/0.411866 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-98caf060dd85817f) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/de43e5d216cb8ccb8197/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/perceptron@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 6.6747 | 6.67849 | — | 0.9994 | PENDING; accuracy A/B=0.493462/0.493462 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-7e5e1e28baf80d16) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/df871bd8e0d9c37c890f/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/lda-clf@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.94996 | 1.91052 | — | 1.0206 | PENDING; accuracy A/B=0.913552/0.913612; logloss A/B=0.233257/0.233327 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-42a608e0ac18b8d1) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/e9508d14d3370dfade6c/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/qda@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.80208 | 1.70604 | — | 1.0563 | PENDING; accuracy A/B=0.879716/0.879682; logloss A/B=3.41997/3.41964 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-0d0b5594fa0a8d2f) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/fa0b1575c2b165e284aa/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/sgd-clf@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 7.22751 | 7.26677 | — | 0.9946 | PENDING; accuracy A/B=0.92238/0.92238 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-b13a046d3cf309c0) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/fad0e0840b7714d0907f/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/simple-imputer@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.843326 | 0.836752 | — | 1.0079 | PENDING; masked_rmse A/B=6.08788/6.08788 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-912ad3119709ba21) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/fbe0df53cb628d25849b/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/sgd-ocsvm@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.64304 | 1.62791 | — | 1.0093 | PENDING; fraction_flagged A/B=0.123182/0.123182 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-cd7c19865a39cfae) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/fd25410fa324ee1d2da7/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/gaussian-nb@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.576743 | 0.580559 | — | 0.9934 | PENDING; accuracy A/B=0.720538/0.720538; logloss A/B=1.14088/1.14088 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-f4706d9f505de501) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/0d1e6d8e4700016071f0/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/target-encoder@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.598636 | 0.580283 | — | 1.0316 | PENDING; output_shape A/B=500000x8/500000x8 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-190f057efefd762d) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/27ef5795de5e53b035e7/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/select-f-classif@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.05409 | 1.05979 | — | 0.9946 | PENDING; n_selected A/B=110/110 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-16f51e15b7f26080) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/6fc9ae71c2c1506ffb73/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/random-trees-embedding@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.82841 | 1.80447 | — | 1.0133 | PENDING; nonzeros_per_row A/B=10/10; output_columns A/B=219/219 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-116c12bf79353864) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/786c65354ed40e98b838/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/gaussian-nb@dataset=istella@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.2474 | 1.23192 | — | 1.0126 | PENDING; accuracy A/B=0.868346/0.868346; logloss A/B=3.51944/3.51944 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-8174c832fc88ba19) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/93e940c30f835c0f74ce/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/random-trees-embedding@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 1.42955 | 1.35096 | — | 1.0582 | PENDING; nonzeros_per_row A/B=10/10; output_columns A/B=283/283 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-278ee34473875268) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/b18d2fcc10233c33faf1/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/target-encoder@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.689413 | 0.643027 | — | 1.0721 | PENDING; output_shape A/B=500000x5/500000x5 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-79d26a9761586603) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/be8c92995859ff28b0ea/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/select-f-classif@dataset=taxi@input=classification-full-v1/attempt-0001 | full_workload | PENDING_ADMISSION | 0.55521 | 0.556425 | — | 0.9978 | PENDING; n_selected A/B=5/5 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-8c82b1970257ac99) | [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/d1c0ff06e93ce2c5d6e4/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/qr@dataset=istella/attempt-0001 | full_workload | QUALITY_FAILED | 17.6231 | 4.20427 | — | 4.1917 | QUALITY_FAILED; relative_gram_difference A/B=2.35619e-06/1.60183e-07 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-c12c57c7c2220909) | [retained receipt](<receipts/apple/apple--captured--qr-svd-full--runs/2268b1fe4abee7294d5e/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/qr@dataset=taxi/attempt-0001 | full_workload | QUALITY_FAILED | 1.13008 | 0.686172 | — | 1.6469 | QUALITY_FAILED; relative_gram_difference A/B=2.44216e-06/7.3759e-07 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-09b759b2c9f3e9fd) | [retained receipt](<receipts/apple/apple--captured--qr-svd-full--runs/6df9e7692fcc24a89c0c/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/svd@dataset=istella/attempt-0001 | full_workload | PENDING_ADMISSION | 4.67355 | 4.68002 | — | 0.9986 | PENDING; max_rel_singular_value_error A/B=2210.04/2210.04; relative_reconstruction_error_100k_rows A/B=3.43081e-05/3.43503e-05 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-79b79ad3b31ac9c5) | [retained receipt](<receipts/apple/apple--captured--qr-svd-full--runs/92b2b8e22831f6a2b014/attempt-0001/receipt.json>) |
| AF.X.complete-proposed | apple/apple-fast | algos/svd@dataset=taxi/attempt-0001 | full_workload | PENDING_ADMISSION | 0.732211 | 0.744609 | — | 0.9834 | PENDING; max_rel_singular_value_error A/B=9.44616e-07/9.44616e-07; relative_reconstruction_error_100k_rows A/B=1.77648e-06/1.77648e-06 | [toggles](#toggles-b5f3f72f7b33aadb) / [details](#attempt-7f7a1c3b05abd44c) | [retained receipt](<receipts/apple/apple--captured--qr-svd-full--runs/cd159d14eb49f9042c45/attempt-0001/receipt.json>) |

## Campaign notes

- A=candidate; B=incumbent. Timed evidence is pending admission, not a default promotion.
- Complete-proposed receipts measure combined configurations. Additional exact selection IDs identify isolated or interaction/dropout profiles; no result automatically credits its members.
- Initial 12-pair quality review: all12 preserve baseline metrics; 4 task-metric gates pass, 6 taxi opponent comparisons pending (historical4m vs current5.25m rows), Apple Istella KMeans fails best-opponent gate, NVIDIA inherits opponent-quality deficit. Additional saved assessments are retained in next-quality-review.json.
- One excluded warmup and one scored sample per arm. Original failed attempts are retained.
- The original AMD campaign used accepted retained artifacts and then stopped compilation. That dated stop is historical; later targeted build authorization and readiness belong to their own source freeze. No publisher action compiles, launches jobs or promotes defaults.
- IDENTICAL compares each same arm across vendors; unavailable typed complete model state remains incomplete.
- Scored output and partial/public-save model hashes are retained separately; partial hashes do not prove complete state identity.
- Apple first four PCA/OLS pairs overlapped shared workspace storage data transfer; KMeans overlap unestablished. No quiet-storage or promotion claim.
- Apple teardown preservation failed: the workspace was on the internal SSD, not retained EBS. Logs, timings, metrics and hash receipts survive; some raw array bytes remain unrecovered. See artifact-retention.json for exact recovery coverage and provenance. Original receipts are unchanged.
- Races reuse accepted binaries without separate numerical verification reruns. Earlier separately authorized AMD builds are historical artifact evidence, not measurements. Full provider and worker logs remain under /Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006
- R2 storage reconciliation: 0 new current-campaign measurements found in the recorded search. Storage locations, fetched archive hashes, historical comparisons and search limitations are retained in storage-reconciliation.json. Older medium/component results do not fill full-workload candidate gaps.
- Historical AMD missing-artifact compiler stop: see compilation-stopped.json. Accepted completed binaries remain reusable; interrupted/unbuilt jobs do not count as ready. Later targeted compilation has its own authorization and freeze.
- scored-hash-coverage.json is a dated, hash-bound metadata audit of captured scored outputs and model states. Complete output hashes do not establish complete fitted-model identity; missing model state remains pending.
- Source-only review of NVIDIA GaussianNB Taxi and LDA Istella retains both combined-configuration quality failures: no implementation or harness bug established. Changed C55 reduction order is a source-supported explanation, not isolated causal proof. Unexercised controls and individual alternatives remain pending; see nvidia-nb-lda-source-diagnosis.json.
- Apple source-only review retains QR combined-configuration quality failures; no concrete implementation defect or isolated L09 regression was established. Resampling A and B have equal saved quality and both fail the opponent quality requirement; this does not establish a new P10 regression. Original evidence and missing-array limitations remain explicit. See apple-qr-resample-source-diagnosis.json.
- Full Taxi GMM refused both candidate and incumbent before scoring. Source-only review found no established implementation or harness defect; retain the incomplete pair and original failures, with zero scored samples and no candidate win/loss decision. See nvidia-gmm-taxi-source-diagnosis.json.
- LassoCV and ElasticNetCV Taxi retain combined-configuration quality failures on NVIDIA and AMD. Source-only review does not establish an implementation defect or isolate a control; saved quality failures cannot promote these defaults. See cv-taxi-source-diagnosis.json.
- Saved same-arm AMD/NVIDIA output comparison: 84 matched workloads; primary counts {"AGREE": 168}, repeated counts {"AGREE": 168}. Full identity counts {"INCOMPLETE": 84, "MATCH": 0, "MISMATCH": 0, "NOT_REQUIRED": 0}. These are saved-signature comparisons, not new model runs or default admission; unmatched and failed arms are retained in same-arm-output-comparison.json and its snapshot.
- Targeted queue uses one immutable source freeze with deduplicated builds; 106 profiles and 1540 planned paired vendor/workload cells, all pending execution. Missing recipes and control blockers remain explicit; no claim of all-algorithm coverage.
- CPU/GPU workers use sixty-minute idle retention after all assigned queued work finishes and verified off-machine preservation. No numerical default was changed by this planning publication.
- Targeted continuations: 106 additional exact profiles; 0 have receipts. See continuation-coverage.json for members and pending scope, and publication-continuations.json for exact plan/review snapshots and missing inputs. The catalog coverage ledger counts authored catalog IDs only; it does not relabel targeted profile IDs as catalog receipts.

## Recorded source decisions

| Candidate / arm | Decision | Source commit | Evidence |
|---|---|---|---|
| RESAMPLE_GPU_GATHER, RESAMPLE_FAST_WAIT_PAIR, RESAMPLE_FAST_TILED_GATHER | PREVIOUSLY PROMOTED ON (Apple FAST): Six full dataset pairs, one excluded warmup+score each arm; gather-off old default not retimed; explicit owner decision. | 0f779ed5d3f0a2ab2f418e054d3af950a766e5f1 | resample/estimator.mojo; existing-default-decisions.json |
| KDE_FAST_DIRECT_PREP | PREVIOUSLY PROMOTED ON (Apple FAST): Historical scoped509x3/521x7/997x13, not this full campaign. | 709f43abb0f5e18973e2027dfbda9a34ea40d505 | kde/resident_fit.mojo; existing-default-decisions.json |
| HDB_LINKAGE_DEVICE | PREVIOUSLY PROMOTED ON (Apple FAST): Historical scoped509x7/997x13/1031x67, not this full campaign. | 429622bef670f9f3e4c0e8f7110bd3b716c2c3e0 | hdbscan/impl/detail/fast_apple.mojo; existing-default-decisions.json |
| AFN_LM_BWD_NOSYNC | PREVIOUSLY PROMOTED ON (Apple FAST): Historical12-step single-shape train/checkpoint/refusal task, not this full campaign. | b953bc9a2759c175181069bf447719fdc7c93a61 | training/byte_lm_afn.mojo; existing-default-decisions.json |
| AF.X.complete-proposed/classical/kmeans@dataset=istella/attempt-0001 | NOT PROMOTED: Candidate and baseline both have worse inertia than retained same-data sklearn; deficit is inherited, not introduced by candidate. | 51a3eb11bd99b921e775fa5fc6f6dbedca125382 | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--kmeans-repair2--runs/69a08c873718b0d710ad/attempt-0001/receipt.json |
| AF.X.complete-proposed/algos/resample@dataset=istella/attempt-0001 | NOT PROMOTED: Saved candidate task metrics are worse than at least one exact-full-input/settings opponent under existing tolerance; no default admission. | 55a815e13728392be41903769c33ece8948cad4a | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--resample-full--runs/047898b203cef197f8fa/attempt-0001/receipt.json |
| AF.X.complete-proposed/algos/huber@dataset=istella/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | 47301d12b14859e81cadc9ab6a0cd4f728d0e206 | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--expanded-reg--runs/4fa48ab7783acc171486/attempt-0001/receipt.json |
| AF.X.complete-proposed/algos/qr@dataset=istella/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | db59bb9557035da8fd11b0a020a84e33b0581c30 | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--qr-svd-full--runs/2268b1fe4abee7294d5e/attempt-0001/receipt.json |
| AF.X.complete-proposed/algos/qr@dataset=taxi/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | db59bb9557035da8fd11b0a020a84e33b0581c30 | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--qr-svd-full--runs/6df9e7692fcc24a89c0c/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:enet-cv@dataset=taxi/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | 1c773404b24dcb05b9fd4684d5f4c8654c13f780 | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-expanded-reg/3392ecb27af9675963d9/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:lasso-cv@dataset=taxi/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | 1c773404b24dcb05b9fd4684d5f4c8654c13f780 | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-expanded-reg/96ac4d3d170baa9b8709/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:gaussian-nb@dataset=taxi@input=classification-full-v1/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | db59bb9557035da8fd11b0a020a84e33b0581c30 | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-classification-full-v1/3643733e60c05588ed87/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:lda-clf@dataset=istella@input=classification-full-v1/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | db59bb9557035da8fd11b0a020a84e33b0581c30 | experiments/six_lane_integration/measurements/20261006/receipts/nvidia/nvidia-native--capture-attempt-02--artifacts--measurements-classification-full-v1/da04cf28502e3d16dc99/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:lasso-cv@dataset=taxi/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | cf442daed63f8d2c9d99947c3a7a4e50fb442319 | experiments/six_lane_integration/measurements/20261006/receipts/amd/amd--capture-attempt-01--artifacts--measurements-expanded-reg/6e33969bb9b101021356/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:enet-cv@dataset=taxi/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | cf442daed63f8d2c9d99947c3a7a4e50fb442319 | experiments/six_lane_integration/measurements/20261006/receipts/amd/amd--capture-attempt-01--artifacts--measurements-expanded-reg/f3260b9ad432706adba9/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:gaussian-nb@dataset=taxi@input=classification-full-v1/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | 137caf2704fe0139c6a08a39e63869f74a827a8b | experiments/six_lane_integration/measurements/20261006/receipts/amd/amd--capture-attempt-01--artifacts--measurements-classification-full-v1/659b9236920edb90c199/attempt-0001/receipt.json |
| I.X.complete-proposed/expanded:lda-clf@dataset=istella@input=classification-full-v1/attempt-0001 | NOT PROMOTED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | 137caf2704fe0139c6a08a39e63869f74a827a8b | experiments/six_lane_integration/measurements/20261006/receipts/amd/amd--capture-attempt-01--artifacts--measurements-classification-full-v1/896a91284fe76be2f130/attempt-0001/receipt.json |

## Unrun or blocked scope

These are not completed measurements and have no inferred timing. This summary does not imply every individual catalog experiment has been run.

| Scope | Status / reason | Evidence |
|---|---|---|
| Apple FAST LogReg/LinearSVC × Taxi/Istella | 0/4 pairs completed; BLOCKED_ALLOCATE_HOSTS_UNCONDITIONAL_SCP_DENY; freeze 779cd5453425139e00229fb99cfa4ec852d7677c. Original launch-tag error and released-host evidence are retained. | campaign-coverage.json: apple_pending |
| Individual candidates, alternative arms and other affected workloads | Only exact recorded selections have receipts. Missing recipes, incompatible artifacts and untested interactions remain pending; combined results do not decide constituents. | experiments/six_lane_integration/catalog.json |
| Apple FAST MLP classifier/regressor × Taxi/Istella | 4 additional pending pairs; No retained accepted Apple FAST x_sequence pair; IDENTICAL cannot substitute | campaign-coverage.json: apple_readiness |
| Continuation targeted-ab-20261007 | Pending sources/inputs: /Users/andrewhendel/mojolearn-evidence/targeted-ab-20261007/quality-review/targeted-quality.json | publication-continuations.json |

## Failed or quality-rejected attempts

Original attempts remain visible after repairs. A quality failure may have complete timings; those are observations, not admitted gains. Interrupted or failed executions have no valid pair timing.

| Candidate | Vendor / case | Outcome / reason | Worker exits | Samples A; B (warmup/scored) | Observed A/B time | Evidence |
|---|---|---|---|---|---:|---|
| AF.X.complete-proposed | apple / classical/kmeans@dataset=istella/attempt-0001 | FAILED_OR_INCOMPLETE: Workload failed or did not write result JSON | [1] | A: 0/0; B: 0/0 | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--runs/69a08c873718b0d710ad/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple / classical/kmeans@dataset=istella/attempt-0002 | FAILED_OR_INCOMPLETE: Workload failed or did not write result JSON | [1] | A: 0/0; B: 0/0 | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--runs/69a08c873718b0d710ad/attempt-0002/receipt.json |
| AF.X.complete-proposed | apple / classical/kmeans@dataset=istella/attempt-0001 | QUALITY_FAILED: Candidate and baseline both have worse inertia than retained same-data sklearn; deficit is inherited, not introduced by candidate. | [0, 0, 0, 0] | A: 1/1; B: 1/1 | 1.0114 (2.3893s / 2.3625s) | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--kmeans-repair2--runs/69a08c873718b0d710ad/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple / algos/resample@dataset=istella/attempt-0001 | QUALITY_FAILED: Saved candidate task metrics are worse than at least one exact-full-input/settings opponent under existing tolerance; no default admission. | [0, 0, 0, 0] | A: 1/1; B: 1/1 | 0.9997 (2.2927s / 2.2933s) | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--resample-full--runs/047898b203cef197f8fa/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple / algos/huber@dataset=istella/attempt-0001 | QUALITY_FAILED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | [0, 0, 0, 0] | A: 1/1; B: 1/1 | 1.0020 (2.4092s / 2.4043s) | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--expanded-reg--runs/4fa48ab7783acc171486/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple / algos/multinomial-nb@dataset=taxi@input=classification-full-v1/attempt-0001 | FAILED_OR_INCOMPLETE: Workload failed or did not write result JSON | [1] | A: 0/0; B: 0/0 | — | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--classification-full-v1--runs/dc3d41d16eac72f9f13f/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple / algos/qr@dataset=istella/attempt-0001 | QUALITY_FAILED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | [0, 0, 0, 0] | A: 1/1; B: 1/1 | 4.1917 (17.6231s / 4.2043s) | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--qr-svd-full--runs/2268b1fe4abee7294d5e/attempt-0001/receipt.json |
| AF.X.complete-proposed | apple / algos/qr@dataset=taxi/attempt-0001 | QUALITY_FAILED: Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules. | [0, 0, 0, 0] | A: 1/1; B: 1/1 | 1.6469 (1.1301s / 0.6862s) | experiments/six_lane_integration/measurements/20261006/receipts/apple/apple--captured--qr-svd-full--runs/6df9e7692fcc24a89c0c/attempt-0001/receipt.json |

## Experiments run: toggles, timing and quality

A is the candidate; B preserves the frozen incumbent. These are recorded arm configurations, not proof that every requested switch was compiled or reached at runtime. An omitted define is not OFF. Combined timings do not establish individual toggle winners.

Expand a toggle profile or an attempt below. Metric values come from the saved scored results; their presence does not imply quality acceptance, complete model identity or promotion.

<a id="toggles-b5f3f72f7b33aadb"></a>
<details>
<summary>Recorded toggle profile b5f3f72f7b33aadb</summary>

| Recorded control | A: candidate | B: incumbent |
|---|---|---|
| define MOJOLEARN_AFCL_G01 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G03 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G04 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G05 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G06 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G07 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G08 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G09 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G10 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G11 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G12 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G13 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_G14 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L01 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L02 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L03 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L04 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L05 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L06 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L07 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L08 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L09 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L10 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L11 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L12 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L13 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_L14 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P01 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P02 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P03 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P04 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P05 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P06 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P07 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P08 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P09 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P10 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P11 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P12 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P13 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_P14 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T01 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T02 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T03 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T04 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T05 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T06 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T07 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T08 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T09 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T10 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T11 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFCL_T12 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_ATTN_FLASH_BK16 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_ATTN_FLASH_TQ16 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_ATTN_NORM_TPB128 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_ATTN_PROJ_BM32 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_ATTN_PROJ_KB16 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_CNN_CHANNELS16 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_CNN_COLS_ONCE | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_CNN_K16 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_CNN_ROWS32 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_EMB_GATHER8 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_EMB_RESIDENT | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_EMB_SCRATCH | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_EMB_THREADS64 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_LM_BWD_FUSE | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_LM_CE_BLOCK128 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_LM_HEAD_FUSE | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_LM_NOSYNC | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_LM_PARAM_VIEWS | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_LM_WGRAD_MIN128 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_LM_WGRAD_SPLIT | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_MAMBA1_CHUNKS16 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_MAMBA2_SSD_K16 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_MAMBA3_THREADS64 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_MAMBA_REFUSAL_THREADS128 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_MLP_FUSED_STEP | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_MLP_RESIDENT | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_MLP_ROWS32 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_OPT_BLOCK128 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN26_OPT_FUSE_SCAN | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_ATTN_ARENA | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_ATTN_FLASH | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_ATTN_FUSE_MLP | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_ATTN_FUSE_PRE | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_ATTN_GQA_TILE | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_ATTN_NORM_SG | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_ATTN_ROPE_CACHE | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_CNN_DIRECT | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_EMB_ATOMIC_BWD | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_MAMBA1_CHUNKSCAN | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_MAMBA1_FUSE_IN | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_MAMBA2_SSD_MMA | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_MAMBA3_SISO_FUSED | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFN_MAMBA_ARENA | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F01 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F02 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F03 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F04 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F05 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F06 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F07 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F08 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F09 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F10 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F11 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_F12 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G01 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G02 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G04 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G05 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G07 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G08 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G09 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G10 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G11 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_G12 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N01 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N02 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N03 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N04 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N05 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N06 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N07 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N08 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N09 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N10 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N11 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_N12 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P01 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P03 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P04 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P05 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P06 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P07 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P08 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P09 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P10 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P11 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_AFT_P12 | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_DECOMP_FAST_GEMM_TILED | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_GBDT_PREDICT_PACKED | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_HDB_CORE_TILE | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_NB_CAT_ATOMIC | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_PCA_FAST_COMPENSATED_COV | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_QN_FAST_COALESCED_OFF | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_QR_FAST_DEV | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_RESAMPLE_FAST_GATHER | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_RESAMPLE_FAST_TILED_GATHER | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR | 1 | incumbent default (no explicit override) |
| define MOJOLEARN_SYM_RESOLVE_BLOCK | 1 | incumbent default (no explicit override) |
| environment.MOJOLEARN_NUMERIC_MODE | fast | incumbent default (no explicit override) |
| environment.MOJOLEARN_X_LINEAR_ENETCV_FAST | 1 | incumbent default (no explicit override) |
| environment.MOJOLEARN_X_PREP_FAST_II_CONV | 1 | incumbent default (no explicit override) |

</details>

<a id="coverage-7f7db161711c3a3c"></a>
<details>
<summary>Recorded implementation IDs and source coverage 7f7db161711c3a3c</summary>

These are recorded source disclosures, not proof of runtime reach.

Implementation IDs: ["AF.C.AFCL-G01", "AF.C.AFCL-G03", "AF.C.AFCL-G04", "AF.C.AFCL-G05", "AF.C.AFCL-G06", "AF.C.AFCL-G07", "AF.C.AFCL-G08", "AF.C.AFCL-G09", "AF.C.AFCL-G10", "AF.C.AFCL-G11", "AF.C.AFCL-G12", "AF.C.AFCL-G13", "AF.C.AFCL-G14", "AF.C.AFCL-L01", "AF.C.AFCL-L02", "AF.C.AFCL-L03", "AF.C.AFCL-L04", "AF.C.AFCL-L05", "AF.C.AFCL-L06", "AF.C.AFCL-L07", "AF.C.AFCL-L08", "AF.C.AFCL-L09", "AF.C.AFCL-L10", "AF.C.AFCL-L11", "AF.C.AFCL-L12", "AF.C.AFCL-L13", "AF.C.AFCL-L14", "AF.C.AFCL-P01", "AF.C.AFCL-P02", "AF.C.AFCL-P03", "AF.C.AFCL-P04", "AF.C.AFCL-P05", "AF.C.AFCL-P06", "AF.C.AFCL-P07", "AF.C.AFCL-P08", "AF.C.AFCL-P09", "AF.C.AFCL-P10", "AF.C.AFCL-P11", "AF.C.AFCL-P12", "AF.C.AFCL-P13", "AF.C.AFCL-P14", "AF.C.AFCL-T01", "AF.C.AFCL-T02", "AF.C.AFCL-T03", "AF.C.AFCL-T04", "AF.C.AFCL-T05", "AF.C.AFCL-T06", "AF.C.AFCL-T07", "AF.C.AFCL-T08", "AF.C.AFCL-T09", "AF.C.AFCL-T10", "AF.C.AFCL-T11", "AF.C.AFCL-T12", "AF.T.F01", "AF.T.F02", "AF.T.F03", "AF.T.F04", "AF.T.F05", "AF.T.F06", "AF.T.F07", "AF.T.F08", "AF.T.F09", "AF.T.F10", "AF.T.F11", "AF.T.F12", "AF.T.G01", "AF.T.G02", "AF.T.G04", "AF.T.G05", "AF.T.G07", "AF.T.G08", "AF.T.G09", "AF.T.G10", "AF.T.G11", "AF.T.G12", "AF.T.N01", "AF.T.N02", "AF.T.N03", "AF.T.N04", "AF.T.N05", "AF.T.N06", "AF.T.N07", "AF.T.N08", "AF.T.N09", "AF.T.N10", "AF.T.N11", "AF.T.N12", "AF.T.P01", "AF.T.P03", "AF.T.P04", "AF.T.P05", "AF.T.P06", "AF.T.P07", "AF.T.P08", "AF.T.P09", "AF.T.P10", "AF.T.P11", "AF.T.P12", "AF.N.A01", "AF.N.A02", "AF.N.A03", "AF.N.A04", "AF.N.A05", "AF.N.A06", "AF.N.A07", "AF.N.A08", "AF.N.A09", "AF.N.A10", "AF.N.A11", "AF.N.A12", "AF.N.T01", "AF.N.T02", "AF.N.T03", "AF.N.T04", "AF.N.T05", "AF.N.T06", "AF.N.T07", "AF.N.T08", "AF.N.T09", "AF.N.T10", "AF.N.T11", "AF.N.T12", "AF.N.M01", "AF.N.M02", "AF.N.M03", "AF.N.M04", "AF.N.M05", "AF.N.M06", "AF.N.M07", "AF.N.M08", "AF.N.M09", "AF.N.M10", "AF.N.E01", "AF.N.E02", "AF.N.E03", "AF.N.E04", "AF.N.E05", "AF.N.E06", "AF.N.E07", "AF.N.E08", "AF.N.E09", "AF.N.E10"]

Source coverage pending:

None recorded.

</details>

<a id="attempt-a32782c39b49d75b"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical/ols@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--runs/440956ba5f3a721465b8/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5343086250359192 s; B: 0.5314585829619318 s.

Quality assessment: BASELINE_NONREGRESSION_OPPONENT_EVIDENCE_PENDING. Identity: NOT_REQUIRED.

Current full taxi has 5,250,086 fit rows. Historical board opponents used4,000,000; incomparable dataset size, so no opponent quality admission.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9087746820210518 | 0.9087746820210518 |
| rmse | 4.698070243856422 | 4.698070243856422 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9087746820210518, 0.9087746820210518, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.698070243856422, 4.698070243856422, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |

Source: ad4214558e482c1391e7e7e59579bddbf54204bc
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Resource limitations: ["Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established."]

</details>

<a id="attempt-c6dd0878e8d6b977"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical/ols@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--runs/4a650353219f94bd483f/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.5894755000481382 s; B: 1.5995837500086054 s.

Quality assessment: TASK_METRIC_GATE_PASSED. Identity: NOT_REQUIRED.

Candidate matches/improves baseline and retained same-data independent opponents within existing board tolerances; scope is recorded task metrics only.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.3319434511617213 | 0.3319434511617213 |
| rmse | 0.682027419998501 | 0.682027419998501 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| baseline_vs_opponents.sklearn-cpu.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| baseline_vs_opponents.sklearn-cpu.metrics.r2 | ["BETTER", 0.3319434511617213, 0.0018812499493927604, 0.9943326191771253] |
| baseline_vs_opponents.sklearn-cpu.metrics.rmse | ["BETTER", 0.682027419998501, 0.8336549427116416, 0.18188283298595886] |
| baseline_vs_opponents.sklearn-cpu.unknown | [] |
| baseline_vs_opponents.sklearn-cpu.verdict | BETTER |
| baseline_vs_opponents.sklearn-cpu.worst | 0.0 |
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.3319434511617213, 0.3319434511617213, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.682027419998501, 0.682027419998501, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| candidate_vs_opponents.sklearn-cpu.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_opponents.sklearn-cpu.metrics.r2 | ["BETTER", 0.3319434511617213, 0.0018812499493927604, 0.9943326191771253] |
| candidate_vs_opponents.sklearn-cpu.metrics.rmse | ["BETTER", 0.682027419998501, 0.8336549427116416, 0.18188283298595886] |
| candidate_vs_opponents.sklearn-cpu.unknown | [] |
| candidate_vs_opponents.sklearn-cpu.verdict | BETTER |
| candidate_vs_opponents.sklearn-cpu.worst | 0.0 |
| opponent_metrics.sklearn-cpu.finite | true |
| opponent_metrics.sklearn-cpu.r2 | 0.0018812499493927604 |
| opponent_metrics.sklearn-cpu.rmse | 0.8336549427116416 |

Source: ad4214558e482c1391e7e7e59579bddbf54204bc
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Resource limitations: ["Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established."]

</details>

<a id="attempt-322ccbf903df0c22"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical/kmeans@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--runs/69a08c873718b0d710ad/attempt-0001/receipt.json>)

Status: FAILED_OR_INCOMPLETE. Scope: full_workload. Observed A: — s; B: — s.

Quality assessment: NOT_ASSESSED. Identity: NOT_REQUIRED.

Scored quality metrics not recorded for this attempt.

Source: ad4214558e482c1391e7e7e59579bddbf54204bc
Samples (warmup/scored): {"A": {"scored": 0, "warmup": 0}, "B": {"scored": 0, "warmup": 0}}
Worker exits: [1]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Failures: ["Workload failed or did not write result JSON"]
Resource limitations: ["Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established."]

</details>

<a id="attempt-c8a6ac85f8cf9cb3"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical/kmeans@dataset=istella/attempt-0002</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--runs/69a08c873718b0d710ad/attempt-0002/receipt.json>)

Status: FAILED_OR_INCOMPLETE. Scope: full_workload. Observed A: — s; B: — s.

Quality assessment: NOT_ASSESSED. Identity: NOT_REQUIRED.

Scored quality metrics not recorded for this attempt.

Source: ad4214558e482c1391e7e7e59579bddbf54204bc
Samples (warmup/scored): {"A": {"scored": 0, "warmup": 0}, "B": {"scored": 0, "warmup": 0}}
Worker exits: [1]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Failures: ["Workload failed or did not write result JSON"]
Resource limitations: ["Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established."]

</details>

<a id="attempt-91aaa1753ba2f73e"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical/pca@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--runs/75455971e66712016083/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.49895141704473644 s; B: 0.5005222079344094 s.

Quality assessment: BASELINE_NONREGRESSION_OPPONENT_EVIDENCE_PENDING. Identity: NOT_REQUIRED.

Current full taxi has 5,250,086 fit rows. Historical board opponents used4,000,000; incomparable dataset size, so no opponent quality admission.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| explained_variance_ratio_sum | 0.9999965982735889 | 0.9999966062452422 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.explained_variance_ratio_sum | ["SAME", 0.9999965982735889, 0.9999966062452422, -7.97168039136347e-09] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |

Source: ad4214558e482c1391e7e7e59579bddbf54204bc
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Resource limitations: ["Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established."]

</details>

<a id="attempt-c1ffb3d13c13dc5e"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical/pca@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--runs/918eace39f801d37fc21/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.7872002499643713 s; B: 1.1472330000251532 s.

Quality assessment: TASK_METRIC_GATE_PASSED. Identity: NOT_REQUIRED.

Candidate matches/improves baseline and retained same-data independent opponents within existing board tolerances; scope is recorded task metrics only.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| explained_variance_ratio_sum | 1.0000000601776293 | 1.0000000568180731 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| baseline_vs_opponents.sklearn-cpu.metrics.explained_variance_ratio_sum | ["SAME", 1.0000000568180731, 0.9999999491312511, 1.0768681589656098e-07] |
| baseline_vs_opponents.sklearn-cpu.unknown | [] |
| baseline_vs_opponents.sklearn-cpu.verdict | SAME |
| baseline_vs_opponents.sklearn-cpu.worst | 0.0 |
| candidate_vs_baseline.metrics.explained_variance_ratio_sum | ["SAME", 1.0000000601776293, 1.0000000568180731, 3.3595559866276815e-09] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| candidate_vs_opponents.sklearn-cpu.metrics.explained_variance_ratio_sum | ["SAME", 1.0000000601776293, 0.9999999491312511, 1.1104637152140878e-07] |
| candidate_vs_opponents.sklearn-cpu.unknown | [] |
| candidate_vs_opponents.sklearn-cpu.verdict | SAME |
| candidate_vs_opponents.sklearn-cpu.worst | 0.0 |
| opponent_metrics.sklearn-cpu.explained_variance_ratio_sum | 0.9999999491312511 |

Source: ad4214558e482c1391e7e7e59579bddbf54204bc
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Resource limitations: ["Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established."]

</details>

<a id="attempt-288af0845518e7db"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical/kmeans@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--kmeans-repair2--runs/449522fb6bd97f700c04/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 3.3190562500385568 s; B: 1.665058874990791 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

New full opponent output retained but comparable settings, complete opponent roster, or established metric directions remain unresolved.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| inertia | 401987663.1286732 | 401987663.1286732 |
| n_iter | 84 | 84 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.inertia | ["SAME", 401987663.1286732, 401987663.1286732, 0.0] |
| candidate_vs_baseline.metrics.n_iter | ["INFO", 84.0, 84.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| new_full_opponent_review.expected_arms | ["sklearn-cpu", "torch-gpu"] |
| new_full_opponent_review.matched_arms | ["torch-gpu"] |
| new_full_opponent_review.qualified | [{"arm": "torch-gpu", "receipt": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/apple/opponent-quality/captured/run-01/cells/cell-04/board/raw/classical/rows-full/kmeans-taxi.json", "receipt_sha256": "43216fc65acd803cda3036337b5053d2885e491da7f3d600ec3947b93b99d81a", "input_npz_sha256": "0db47891038cb000a320470f8f6f7172285118e9c0697bc3e118dab2bc4921e0", "parameter_mismatches": {}, "recorded_parameter_differences": {"random_state": [7, null], "init_centroids": [null, null], "oversampling_factor": [0.0, null]}, "semantic_equivalences": [{"field": "init_centroids", "recorded_values": [null, null], "reason": "Explicit-centroid buffer is inactive under k-means++ in frozen KMeans source.", "source_sha256": {"python/mojolearn/cluster.py": "81c1ea6ea4bc86596e3fc4dcf79b28c26f816580988852ed50be11b28d52f04f", "tools/classical_two_datasets.py": "9c7b8f09f11ed471f047233cde951b11402dc1d7d88a7c5d72a8f354289b5209"}, "audit_manifest": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/quality-review/semantic-source-audit/source-manifest.json", "scope": "Task/objective comparison only; no numerical-identity or equal solver trajectory claim"}, {"field": "random_state", "recorded_values": [7, null], "reason": "Frozen TorchKMeans creates a per-fit torch.Generator and manual_seed(7); declared seed7 matches our random_state7, without claiming identical RNG draws.", "source_sha256": {"python/mojolearn/cluster.py": "81c1ea6ea4bc86596e3fc4dcf79b28c26f816580988852ed50be11b28d52f04f", "tools/classical_two_datasets.py": "9c7b8f09f11ed471f047233cde951b11402dc1d7d88a7c5d72a8f354289b5209"}, "audit_manifest": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/quality-review/semantic-source-audit/source-manifest.json", "scope": "Task/objective comparison only; no numerical-identity or equal solver trajectory claim"}, {"field": "oversampling_factor", "recorded_values": [0.0, null], "reason": "Frozen classical lane explicitly pairs our sequential k-means++ (oversampling0) with reference greedy k-means++ on the same Lloyd/inertia task. Initialization algorithms intentionally differ and are recorded; independent inertia quality remains mandatory.", "source_sha256": {"python/mojolearn/cluster.py": "81c1ea6ea4bc86596e3fc4dcf79b28c26f816580988852ed50be11b28d52f04f", "tools/classical_two_datasets.py": "9c7b8f09f11ed471f047233cde951b11402dc1d7d88a7c5d72a8f354289b5209"}, "audit_manifest": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/quality-review/semantic-source-audit/source-manifest.json", "scope": "Task/objective comparison only; no numerical-identity or equal solver trajectory claim"}], "parameter_record_schema": "nested get_params or direct declared function kwargs", "metrics": {"inertia": 401662517.47630227, "n_iter": 69}, "candidate": {"verdict": "SAME", "metrics": {"inertia": ["SAME", 401987663.1286732, 401662517.47630227, -0.0008088448531985248], "n_iter": ["INFO", 84.0, 69.0, null]}, "unknown": [], "worst": 0.0}, "baseline": {"verdict": "SAME", "metrics": {"inertia": ["SAME", 401987663.1286732, 401662517.47630227, -0.0008088448531985248], "n_iter": ["INFO", 84.0, 69.0, null]}, "unknown": [], "worst": 0.0}}] |
| new_full_opponent_review.scope | Saved task-quality only; no opponent ratios or default promotion |
| new_full_opponent_review.unqualified | [] |

Source: 51a3eb11bd99b921e775fa5fc6f6dbedca125382
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Resource limitations: ["Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established."]

</details>

<a id="attempt-3cb7b17e87b2b93c"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical/kmeans@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--kmeans-repair2--runs/69a08c873718b0d710ad/attempt-0001/receipt.json>)

Status: QUALITY_FAILED. Scope: full_workload. Observed A: 2.389335541985929 s; B: 2.3624788330635056 s.

Quality assessment: FAILED_FAST_OPPONENT_GATE. Identity: NOT_REQUIRED.

Candidate and baseline both have worse inertia than retained same-data sklearn; deficit is inherited, not introduced by candidate.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| inertia | 6.050717116872483e+17 | 6.050717116872483e+17 |
| n_iter | 33 | 33 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| baseline_vs_opponents.sklearn-cpu.metrics.inertia | ["WORSE", 6.050717116872483e+17, 5.958591227360614e+17, -0.015225615035773897] |
| baseline_vs_opponents.sklearn-cpu.metrics.n_iter | ["INFO", 33.0, 36.0, null] |
| baseline_vs_opponents.sklearn-cpu.unknown | [] |
| baseline_vs_opponents.sklearn-cpu.verdict | WORSE |
| baseline_vs_opponents.sklearn-cpu.worst | -0.015225615035773897 |
| baseline_vs_opponents.torch-gpu.metrics.inertia | ["WORSE", 6.050717116872483e+17, 5.991151862715049e+17, -0.009844329689671878] |
| baseline_vs_opponents.torch-gpu.metrics.n_iter | ["INFO", 33.0, 65.0, null] |
| baseline_vs_opponents.torch-gpu.unknown | [] |
| baseline_vs_opponents.torch-gpu.verdict | WORSE |
| baseline_vs_opponents.torch-gpu.worst | -0.009844329689671878 |
| candidate_vs_baseline.metrics.inertia | ["SAME", 6.050717116872483e+17, 6.050717116872483e+17, 0.0] |
| candidate_vs_baseline.metrics.n_iter | ["INFO", 33.0, 33.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| candidate_vs_opponents.sklearn-cpu.metrics.inertia | ["WORSE", 6.050717116872483e+17, 5.958591227360614e+17, -0.015225615035773897] |
| candidate_vs_opponents.sklearn-cpu.metrics.n_iter | ["INFO", 33.0, 36.0, null] |
| candidate_vs_opponents.sklearn-cpu.unknown | [] |
| candidate_vs_opponents.sklearn-cpu.verdict | WORSE |
| candidate_vs_opponents.sklearn-cpu.worst | -0.015225615035773897 |
| candidate_vs_opponents.torch-gpu.metrics.inertia | ["WORSE", 6.050717116872483e+17, 5.991151862715049e+17, -0.009844329689671878] |
| candidate_vs_opponents.torch-gpu.metrics.n_iter | ["INFO", 33.0, 65.0, null] |
| candidate_vs_opponents.torch-gpu.unknown | [] |
| candidate_vs_opponents.torch-gpu.verdict | WORSE |
| candidate_vs_opponents.torch-gpu.worst | -0.009844329689671878 |
| opponent_metrics.sklearn-cpu.inertia | 5.958591227360614e+17 |
| opponent_metrics.sklearn-cpu.inertia_over_ours | 0.9847743849642261 |
| opponent_metrics.sklearn-cpu.n_iter | 36 |
| opponent_metrics.torch-gpu.inertia | 5.991151862715049e+17 |
| opponent_metrics.torch-gpu.inertia_over_ours | 0.9901556703103281 |
| opponent_metrics.torch-gpu.n_iter | 65 |

Source: 51a3eb11bd99b921e775fa5fc6f6dbedca125382
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Resource limitations: ["Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established."]

</details>

<a id="attempt-e10674a2304dcc24"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/resample@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--resample-full--runs/047898b203cef197f8fa/attempt-0001/receipt.json>)

Status: QUALITY_FAILED. Scope: full_workload. Observed A: 2.292678292025812 s; B: 2.2933388750534505 s.

Quality assessment: QUALITY_FAILED. Identity: NOT_REQUIRED.

Saved candidate task metrics are worse than at least one exact-full-input/settings opponent under existing tolerance; no default admission.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| max_mean_shift_over_std | 0.001946923534707672 | 0.001946923534707672 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.max_mean_shift_over_std | ["SAME", 0.001946923534707672, 0.001946923534707672, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.max_mean_shift_over_std | lower |
| new_full_opponent_review.expected_arms | ["sklearn-cpu"] |
| new_full_opponent_review.matched_arms | ["sklearn-cpu"] |
| new_full_opponent_review.qualified | [{"arm": "sklearn-cpu", "receipt": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/apple/opponent-quality/captured/run-01/cells/cell-12/board/raw/algos/rows-full/resample-istella.json", "receipt_sha256": "4d41392e14bd5cdf54a1ee1f41763b4a20fe8ce3d1f841db14b40fce1281b070", "input_npz_sha256": "f1708d9b22fa874af0ee9b2dfb00de1f5d2dea848c910332ee496775dfc2f724", "parameter_mismatches": {}, "recorded_parameter_differences": {}, "semantic_equivalences": [], "parameter_record_schema": "nested get_params or direct declared function kwargs", "metrics": {"max_mean_shift_over_std": 0.0015666546468173469}, "candidate": {"verdict": "WORSE", "metrics": {"max_mean_shift_over_std": ["WORSE", 0.001946923534707672, 0.0015666546468173469, -0.19531783406553865]}, "unknown": [], "worst": -0.19531783406553865}, "baseline": {"verdict": "WORSE", "metrics": {"max_mean_shift_over_std": ["WORSE", 0.001946923534707672, 0.0015666546468173469, -0.19531783406553865]}, "unknown": [], "worst": -0.19531783406553865}}] |
| new_full_opponent_review.scope | Saved task-quality only; no opponent ratios or default promotion |
| new_full_opponent_review.unqualified | [] |

Source: 55a815e13728392be41903769c33ece8948cad4a
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-62c30d7b110ec4fe"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/resample@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--resample-full--runs/661a766183dd2fb07d65/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.755789791001007 s; B: 0.7402970410184935 s.

Quality assessment: TASK_METRIC_GATE_PASSED. Identity: NOT_REQUIRED.

Saved candidate preserves baseline and meets all selected supported same-full-input/settings opponent task metrics under existing tolerances.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| max_mean_shift_over_std | 0.0008182939412305245 | 0.0008182939412305245 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.max_mean_shift_over_std | ["SAME", 0.0008182939412305245, 0.0008182939412305245, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.max_mean_shift_over_std | lower |
| new_full_opponent_review.expected_arms | ["sklearn-cpu"] |
| new_full_opponent_review.matched_arms | ["sklearn-cpu"] |
| new_full_opponent_review.qualified | [{"arm": "sklearn-cpu", "receipt": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/apple/opponent-quality/captured/run-01/cells/cell-11/board/raw/algos/rows-full/resample-taxi.json", "receipt_sha256": "0e341c5d52134a4fe3082e83d67bc55fe0fc6b882f884f47b963616ee2a83e7d", "input_npz_sha256": "ec1b39f38b116f78dff51d67f41c7bbdfcd56508562445274d4c4dfb7bb2387f", "parameter_mismatches": {}, "recorded_parameter_differences": {}, "semantic_equivalences": [], "parameter_record_schema": "nested get_params or direct declared function kwargs", "metrics": {"max_mean_shift_over_std": 0.0010264748022140665}, "candidate": {"verdict": "BETTER", "metrics": {"max_mean_shift_over_std": ["BETTER", 0.0008182939412305245, 0.0010264748022140665, 0.20281146749487075]}, "unknown": [], "worst": 0.0}, "baseline": {"verdict": "BETTER", "metrics": {"max_mean_shift_over_std": ["BETTER", 0.0008182939412305245, 0.0010264748022140665, 0.20281146749487075]}, "unknown": [], "worst": 0.0}}] |
| new_full_opponent_review.scope | Saved task-quality only; no opponent ratios or default promotion |
| new_full_opponent_review.unqualified | [] |

Source: 55a815e13728392be41903769c33ece8948cad4a
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-454ec65777dc4431"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical2/linearsvr@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--reg-full--runs/05207b357bcbe067ca6a/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.2706715419190004 s; B: 1.3157004999229684 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | -0.10702817996248681 | -0.10702817996248681 |
| rmse | 0.8779596576997942 | 0.8779596576997942 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", -0.10702817996248681, -0.10702817996248681, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.8779596576997942, 0.8779596576997942, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 55a815e13728392be41903769c33ece8948cad4a
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-74b2f6f2bb84e966"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical2/ridge@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--reg-full--runs/6d77306d88d72aab7b44/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.6468826250638813 s; B: 1.6427078749984503 s.

Quality assessment: TASK_METRIC_GATE_PASSED. Identity: NOT_REQUIRED.

Saved candidate preserves baseline and meets all selected supported same-full-input/settings opponent task metrics under existing tolerances.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.3325338446311392 | 0.3325338446311392 |
| rmse | 0.6817259832876175 | 0.6817259832876175 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.3325338446311392, 0.3325338446311392, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6817259832876175, 0.6817259832876175, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |
| new_full_opponent_review.expected_arms | ["sklearn-cpu"] |
| new_full_opponent_review.matched_arms | ["sklearn-cpu"] |
| new_full_opponent_review.qualified | [{"arm": "sklearn-cpu", "receipt": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/apple/opponent-quality/captured/run-01/cells/cell-06/board/raw/classical2/rows-full/ridge-istella.json", "receipt_sha256": "506857f987f0c54233374a77ab3ef2149a8220e12cd1f9ec3962cf8d254906d6", "input_npz_sha256": "f1708d9b22fa874af0ee9b2dfb00de1f5d2dea848c910332ee496775dfc2f724", "parameter_mismatches": {}, "recorded_parameter_differences": {"solver": ["eig", "cholesky"], "normalize": [false, null]}, "semantic_equivalences": [{"field": "solver", "recorded_values": ["eig", "cholesky"], "reason": "Frozen board explicitly compares eig and Cholesky implementations of min \|\|y-Xw\|\|²+alpha\|\|w\|\|²; same alpha/intercept. Existing task metric gate judges numerical differences.", "source_sha256": {"python/mojolearn/linear_model.py": "e494ef22cffe6398fdc563b4b000d1096b214cb162343ba043ec9af962cc555c", "tools/bench_board_more.py": "ceaa630cfd8a7f0c09cc6ec366667686493454cc59fc9627ddec1abda9b117bf", "sklearn/linear_model/_ridge.py": "a0a8c5e26611c1bc91a97ae85d6add59b98a2fa5c92ff7c25467ea88bd6e4ce3", "sklearn/linear_model/_base.py": "1d2ffdba0cb4509d11983fa87ba4187d4469c9c1ae9fd9193461281b20b35133"}, "audit_manifest": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/quality-review/semantic-source-audit/source-manifest.json", "scope": "Task/objective comparison only; no numerical-identity or equal solver trajectory claim"}, {"field": "normalize", "recorded_values": [false, null], "reason": "Our normalize=False adds no variance scaling; installed sklearn _preprocess_data only centers when fitting intercept and sets X_scale=ones. The removed option is semantically false here.", "source_sha256": {"python/mojolearn/linear_model.py": "e494ef22cffe6398fdc563b4b000d1096b214cb162343ba043ec9af962cc555c", "tools/bench_board_more.py": "ceaa630cfd8a7f0c09cc6ec366667686493454cc59fc9627ddec1abda9b117bf", "sklearn/linear_model/_ridge.py": "a0a8c5e26611c1bc91a97ae85d6add59b98a2fa5c92ff7c25467ea88bd6e4ce3", "sklearn/linear_model/_base.py": "1d2ffdba0cb4509d11983fa87ba4187d4469c9c1ae9fd9193461281b20b35133"}, "audit_manifest": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/quality-review/semantic-source-audit/source-manifest.json", "scope": "Task/objective comparison only; no numerical-identity or equal solver trajectory claim"}], "parameter_record_schema": "nested get_params or direct declared function kwargs", "metrics": {"finite": true, "r2": 0.33252128046410945, "rmse": 0.6817323995521324}, "candidate": {"verdict": "SAME", "metrics": {"finite": ["SAME", 1.0, 1.0, 0.0], "r2": ["SAME", 0.3325338446311392, 0.33252128046410945, 3.778312262833969e-05], "rmse": ["SAME", 0.6817259832876175, 0.6817323995521324, 9.411705412760572e-06]}, "unknown": [], "worst": 0.0}, "baseline": {"verdict": "SAME", "metrics": {"finite": ["SAME", 1.0, 1.0, 0.0], "r2": ["SAME", 0.3325338446311392, 0.33252128046410945, 3.778312262833969e-05], "rmse": ["SAME", 0.6817259832876175, 0.6817323995521324, 9.411705412760572e-06]}, "unknown": [], "worst": 0.0}}] |
| new_full_opponent_review.scope | Saved task-quality only; no opponent ratios or default promotion |
| new_full_opponent_review.unqualified | [] |

Source: 55a815e13728392be41903769c33ece8948cad4a
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-4557bd5b84099a1e"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical2/linearsvr@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--reg-full--runs/71dfc119e59c074352bb/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.606728250044398 s; B: 0.6071637080749497 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

New full opponent output retained but comparable settings, complete opponent roster, or established metric directions remain unresolved.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.8992839609513043 | 0.8992839609513043 |
| rmse | 4.936408958120795 | 4.936408958120795 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.8992839609513043, 0.8992839609513043, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.936408958120795, 4.936408958120795, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |
| new_full_opponent_review.expected_arms | ["sklearn-cpu"] |
| new_full_opponent_review.matched_arms | [] |
| new_full_opponent_review.qualified | [] |
| new_full_opponent_review.scope | Saved task-quality only; no opponent ratios or default promotion |
| new_full_opponent_review.unqualified | [{"arm": "sklearn-cpu", "receipt": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/apple/opponent-quality/captured/run-01/cells/cell-13/board/raw/classical2/rows-full/linearsvr-taxi.json", "receipt_sha256": "b26afbafc8df89f5accf2195d4ac5e2a712ee63dba58f05ece0001b00a1c7548", "input_npz_sha256": "ec1b39f38b116f78dff51d67f41c7bbdfcd56508562445274d4c4dfb7bb2387f", "parameter_mismatches": {"penalty": ["l2", null], "penalized_intercept": [false, null], "linesearch_max_iter": [100, null], "lbfgs_memory": [5, null]}, "recorded_parameter_differences": {"penalty": ["l2", null], "penalized_intercept": [false, null], "linesearch_max_iter": [100, null], "lbfgs_memory": [5, null]}, "semantic_equivalences": [], "parameter_record_schema": "nested get_params or direct declared function kwargs", "reasons": ["Constructor settings differ or are missing: {\"lbfgs_memory\": [5, null], \"linesearch_max_iter\": [100, null], \"penalized_intercept\": [false, null], \"penalty\": [\"l2\", null]}"]}] |

Source: 55a815e13728392be41903769c33ece8948cad4a
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-60094147f639ef4f"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical2/ridge@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--reg-full--runs/94c802682d07720f1572/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.6082973750308156 s; B: 0.5887810839340091 s.

Quality assessment: TASK_METRIC_GATE_PASSED. Identity: NOT_REQUIRED.

Saved candidate preserves baseline and meets all selected supported same-full-input/settings opponent task metrics under existing tolerances.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9087715577966976 | 0.9087715577966976 |
| rmse | 4.698150691368868 | 4.698150691368868 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9087715577966976, 0.9087715577966976, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.698150691368868, 4.698150691368868, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |
| new_full_opponent_review.expected_arms | ["sklearn-cpu"] |
| new_full_opponent_review.matched_arms | ["sklearn-cpu"] |
| new_full_opponent_review.qualified | [{"arm": "sklearn-cpu", "receipt": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/apple/opponent-quality/captured/run-01/cells/cell-05/board/raw/classical2/rows-full/ridge-taxi.json", "receipt_sha256": "a2ae38c0d1472c2d2b9245073bc97a97f77656eeefdf2baff2afb5440e475ed7", "input_npz_sha256": "ec1b39f38b116f78dff51d67f41c7bbdfcd56508562445274d4c4dfb7bb2387f", "parameter_mismatches": {}, "recorded_parameter_differences": {"solver": ["eig", "cholesky"], "normalize": [false, null]}, "semantic_equivalences": [{"field": "solver", "recorded_values": ["eig", "cholesky"], "reason": "Frozen board explicitly compares eig and Cholesky implementations of min \|\|y-Xw\|\|²+alpha\|\|w\|\|²; same alpha/intercept. Existing task metric gate judges numerical differences.", "source_sha256": {"python/mojolearn/linear_model.py": "e494ef22cffe6398fdc563b4b000d1096b214cb162343ba043ec9af962cc555c", "tools/bench_board_more.py": "ceaa630cfd8a7f0c09cc6ec366667686493454cc59fc9627ddec1abda9b117bf", "sklearn/linear_model/_ridge.py": "a0a8c5e26611c1bc91a97ae85d6add59b98a2fa5c92ff7c25467ea88bd6e4ce3", "sklearn/linear_model/_base.py": "1d2ffdba0cb4509d11983fa87ba4187d4469c9c1ae9fd9193461281b20b35133"}, "audit_manifest": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/quality-review/semantic-source-audit/source-manifest.json", "scope": "Task/objective comparison only; no numerical-identity or equal solver trajectory claim"}, {"field": "normalize", "recorded_values": [false, null], "reason": "Our normalize=False adds no variance scaling; installed sklearn _preprocess_data only centers when fitting intercept and sets X_scale=ones. The removed option is semantically false here.", "source_sha256": {"python/mojolearn/linear_model.py": "e494ef22cffe6398fdc563b4b000d1096b214cb162343ba043ec9af962cc555c", "tools/bench_board_more.py": "ceaa630cfd8a7f0c09cc6ec366667686493454cc59fc9627ddec1abda9b117bf", "sklearn/linear_model/_ridge.py": "a0a8c5e26611c1bc91a97ae85d6add59b98a2fa5c92ff7c25467ea88bd6e4ce3", "sklearn/linear_model/_base.py": "1d2ffdba0cb4509d11983fa87ba4187d4469c9c1ae9fd9193461281b20b35133"}, "audit_manifest": "/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006/quality-review/semantic-source-audit/source-manifest.json", "scope": "Task/objective comparison only; no numerical-identity or equal solver trajectory claim"}], "parameter_record_schema": "nested get_params or direct declared function kwargs", "metrics": {"finite": true, "r2": 0.9088773731335543, "rmse": 4.695425222063106}, "candidate": {"verdict": "SAME", "metrics": {"finite": ["SAME", 1.0, 1.0, 0.0], "r2": ["SAME", 0.9087715577966976, 0.9088773731335543, -0.00011642421737477817], "rmse": ["SAME", 4.698150691368868, 4.695425222063106, -0.0005801153442713471]}, "unknown": [], "worst": 0.0}, "baseline": {"verdict": "SAME", "metrics": {"finite": ["SAME", 1.0, 1.0, 0.0], "r2": ["SAME", 0.9087715577966976, 0.9088773731335543, -0.00011642421737477817], "rmse": ["SAME", 4.698150691368868, 4.695425222063106, -0.0005801153442713471]}, "unknown": [], "worst": 0.0}}] |
| new_full_opponent_review.scope | Saved task-quality only; no opponent ratios or default promotion |
| new_full_opponent_review.unqualified | [] |

Source: 55a815e13728392be41903769c33ece8948cad4a
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-7927b956ba0b72c1"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical2/gmm@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--reg-full--runs/d4f24411b4f604c84997/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 2.676956541952677 s; B: 1.8405577089870349 s.

Quality assessment: TASK_METRIC_GATE_PASSED. Identity: NOT_REQUIRED.

Recorded candidate task metrics preserve baseline and meet same-input independent sklearn quality under existing tolerances.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| bic | -188789016.44902077 | -188789016.44902077 |
| mean_log_likelihood | 11.60158091121872 | 11.60158091121872 |
| n_iter | 24 | 24 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.bic | ["SAME", -188789016.44902077, -188789016.44902077, 0.0] |
| candidate_vs_baseline.metrics.mean_log_likelihood | ["SAME", 11.60158091121872, 11.60158091121872, 0.0] |
| candidate_vs_baseline.metrics.n_iter | ["INFO", 24.0, 24.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.bic | lower |
| metric_directions.mean_log_likelihood | higher |
| metric_directions.n_iter | info |
| opponent_metrics.bic | -184832994.344408 |
| opponent_metrics.mean_log_likelihood | 10.690672192719653 |
| opponent_metrics.n_iter | 24 |

Source: 55a815e13728392be41903769c33ece8948cad4a
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-de5212f72587e628"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/bayesian-ridge@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/11f34b0bacc6a216cbf5/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.676354791969061 s; B: 1.6557661250699311 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.33210975276942234 | 0.3321027270626662 |
| rmse | 0.6819425250025867 | 0.6819461117562676 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.33210975276942234, 0.3321027270626662, 2.1154773979267214e-05] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6819425250025867, 0.6819461117562676, 5.259585206393523e-06] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-c5e9f9f7b828c55f"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/bayesian-ridge@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/121d2bf7022b542e4d6e/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.6025629590731114 s; B: 0.6000144169665873 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9087724154739473 | 0.9087723671098267 |
| rmse | 4.698128606664048 | 4.698129852015602 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9087724154739473, 0.9087723671098267, 5.321917762221498e-08] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.698128606664048, 4.698129852015602, 2.6507388953964247e-07] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-158387d3db293254"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/lars@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/1ffab8882758257c9f23/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.9916996250394732 s; B: 2.0018063749885187 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.332578894167445 | 0.3322521267368259 |
| rmse | 0.6817029769093841 | 0.6818698363849675 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.332578894167445, 0.3322521267368259, 0.0009825260602814513] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6817029769093841, 0.6818698363849675, 0.00024470869171746846] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-5b6dea355b89327d"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/sgd-reg@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/22c62826ae044e079978/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 7.248178375069983 s; B: 7.1927933329716325 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.33167739764062487 | 0.33167739764062487 |
| rmse | 0.6821632151900116 | 0.6821632151900116 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.33167739764062487, 0.33167739764062487, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6821632151900116, 0.6821632151900116, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-beeef7a464f96181"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/lasso-lars@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/3a40940bae1961555f78/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.4024035409092903 s; B: 1.3946241249796003 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.31467648366016077 | 0.3146686790811021 |
| rmse | 0.6907852244891691 | 0.6907891578669239 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.31467648366016077, 0.3146686790811021, 2.4801913914605373e-05] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6907852244891691, 0.6907891578669239, 5.69403516244192e-06] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-5f43961bc834421b"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/lasso-cv@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/3cfd7d59ee9c8efa1497/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.691912499954924 s; B: 1.6302146660163999 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.3294381008459595 | 0.3294381008459595 |
| rmse | 0.6833050952198324 | 0.6833050952198324 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.3294381008459595, 0.3294381008459595, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6833050952198324, 0.6833050952198324, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-f3328cbf23ffeea8"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/huber@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/4fa48ab7783acc171486/attempt-0001/receipt.json>)

Status: QUALITY_FAILED. Scope: full_workload. Observed A: 2.4091830409597605 s; B: 2.404323499999009 s.

Quality assessment: QUALITY_FAILED. Identity: NOT_REQUIRED.

Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | -0.007089342908900065 | -0.005952189726329049 |
| rmse | 0.8373928001448464 | 0.836919896299495 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["WORSE", -0.007089342908900065, -0.005952189726329049, -0.16040318506012977] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.8373928001448464, 0.836919896299495, -0.000564733593684552] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | WORSE |
| candidate_vs_baseline.worst | -0.16040318506012977 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-541855c4db4d9e40"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/pa-reg@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/6fcf310a716926c2d853/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 10.201159375021234 s; B: 10.198122665984556 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.3103594532257662 | 0.3103594532257662 |
| rmse | 0.6929575264611235 | 0.6929575264611235 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.3103594532257662, 0.3103594532257662, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6929575264611235, 0.6929575264611235, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-5be9ee86049e552d"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/lasso-lars@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/791dbcdb69bd137c8ef4/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.6063201669603586 s; B: 0.6024489999981597 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9087942179988827 | 0.9087941572937263 |
| rmse | 4.6975671690935314 | 4.697568732407261 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9087942179988827, 0.9087941572937263, 6.679747213934124e-08] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.6975671690935314, 4.697568732407261, 3.3279209271619687e-07] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-ff24a097f693b2ad"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/ridge-cv@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/805874e87463199882f2/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 25.283735082950443 s; B: 25.11953295895364 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.33250616129806376 | 0.33250616129806376 |
| rmse | 0.6817401205226135 | 0.6817401205226135 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.33250616129806376, 0.33250616129806376, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6817401205226135, 0.6817401205226135, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-35e944e0962689e1"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/lars@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/980247052f23e672228d/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5993408750509843 s; B: 0.6054343750001863 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.908772372822577 | 0.9087723166792013 |
| rmse | 4.6981297049152 | 4.698131150578255 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.908772372822577, 0.9087723166792013, 6.177936004022338e-08] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.6981297049152, 4.698131150578255, 3.0771023811589143e-07] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-a2e3080382d098f0"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/enet-cv@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/9f64ff019913cdb9b802/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.6281062089838088 s; B: 0.6227400419302285 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9088121238319825 | 0.9088121238319825 |
| rmse | 4.69710602518004 | 4.69710602518004 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9088121238319825, 0.9088121238319825, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.69710602518004, 4.69710602518004, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-2d655de64969c832"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/enet-cv@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/b1e39171e9fbf74e3e34/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.634162375004962 s; B: 1.6581437500426546 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.3307987047808765 | 0.3307987047808765 |
| rmse | 0.6826115129513999 | 0.6826115129513999 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.3307987047808765, 0.3307987047808765, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6826115129513999, 0.6826115129513999, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-ac28476b0071d590"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/ridge-cv@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/ba6202e41915aa8b651c/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.6805006669601426 s; B: 0.6819942910224199 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9087723799414855 | 0.9087723115061564 |
| rmse | 4.698129521606933 | 4.6981312837814775 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9087723799414855, 0.9087723115061564, 7.530524758544137e-08] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.698129521606933, 4.6981312837814775, 3.7507988551647665e-07] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-f0865758948fef12"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/sgd-reg@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/d3adb34325456bfba29c/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 13.757316125091165 s; B: 13.848073333036155 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9087707711113641 | 0.9087707711113641 |
| rmse | 4.698170947980658 | 4.698170947980658 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9087707711113641, 0.9087707711113641, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.698170947980658, 4.698170947980658, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-8e3c6a1a864efe0a"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/pa-reg@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/d74a68acdb33bdd77426/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 8.707507833023556 s; B: 8.700152582954615 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9003800254969431 | 0.9003800254969431 |
| rmse | 4.909474697616844 | 4.909474697616844 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9003800254969431, 0.9003800254969431, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.909474697616844, 4.909474697616844, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-9a79641563131b6c"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/huber@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/f0c4179bfa4fda6936f2/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 2.278183583985083 s; B: 1.384035750059411 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.8996135814350751 | 0.8996135799847971 |
| rmse | 4.928324471168913 | 4.9283245067685515 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.8996135814350751, 0.8996135799847971, 1.6121121765226127e-09] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.928324471168913, 4.9283245067685515, 7.22347701404659e-09] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-bef083c3b647560e"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/lasso-cv@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--expanded-reg--runs/f2b21063131028aae994/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.7770473749842495 s; B: 0.6994155829306692 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9088308665972334 | 0.9088308665972334 |
| rmse | 4.696623278550089 | 4.696623278550089 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9088308665972334, 0.9088308665972334, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.696623278550089, 4.696623278550089, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-be9c56d14626b19f"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — classical2/gmm@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--gmm-istella-full--runs/e54f3cfdf88e2e305a71/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 151.5873012499651 s; B: 152.32042204099707 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| bic | -853061803.057183 | -853061803.057183 |
| mean_log_likelihood | 210.07076004476 | 210.07076004476 |
| n_iter | 33 | 33 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.bic | ["SAME", -853061803.057183, -853061803.057183, 0.0] |
| candidate_vs_baseline.metrics.mean_log_likelihood | ["SAME", 210.07076004476, 210.07076004476, 0.0] |
| candidate_vs_baseline.metrics.n_iter | ["INFO", 33.0, 33.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.bic | lower |
| metric_directions.mean_log_likelihood | higher |
| metric_directions.n_iter | info |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-6013393bb5e97f57"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/qn-reg@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--pls-qn-full--runs/177e591a0085ff705250/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5657192079816014 s; B: 0.5650157920317724 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9087597449000081 | 0.9087597449000081 |
| rmse | 4.698454856226518 | 4.698454856226518 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9087597449000081, 0.9087597449000081, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.698454856226518, 4.698454856226518, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-3f1ba240a3e2e646"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/pls@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--pls-qn-full--runs/1f0fc8e14b88c7e0588c/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 4.0522691670339555 s; B: 3.815072833094746 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.2943605458845713 | 0.2943604369384434 |
| rmse | 0.700949370518026 | 0.7009494246290285 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.2943605458845713, 0.2943604369384434, 3.701111763110923e-07] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.700949370518026, 0.7009494246290285, 7.719672870295655e-08] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-47df8a4a5f60750d"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/pls@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--pls-qn-full--runs/258012ade2a00a9645a9/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.5196539169410244 s; B: 1.186451124958694 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.9055737344925392 | 0.9055738104686821 |
| rmse | 4.779783436866216 | 4.779781513939471 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.9055737344925392, 0.9055738104686821, -8.389834385030486e-08] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 4.779783436866216, 4.779781513939471, -4.023041567353584e-07] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-1efdeca8d0da4aaa"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/qn-reg@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--pls-qn-full--runs/e93854e1cf11c7c7adfb/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.7276659170165658 s; B: 1.7308373750420287 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| finite | true | true |
| r2 | 0.33139950764380466 | 0.33139950764380466 |
| rmse | 0.6823050229274745 | 0.6823050229274745 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.finite | ["SAME", 1.0, 1.0, 0.0] |
| candidate_vs_baseline.metrics.r2 | ["SAME", 0.33139950764380466, 0.33139950764380466, 0.0] |
| candidate_vs_baseline.metrics.rmse | ["SAME", 0.6823050229274745, 0.6823050229274745, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.finite | higher |
| metric_directions.r2 | higher |
| metric_directions.rmse | lower |

Source: 47301d12b14859e81cadc9ab6a0cd4f728d0e206
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-baa89f17be306cf6"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/randomized-svd@dataset=taxi@input=tsvd-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--tsvd-full-v1--runs/aad81edca8ccf47b9ecc/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.3783062079455703 s; B: 1.3334462499478832 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| relative_reconstruction_error | 0.027209519221198897 | 0.02720951922113826 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.relative_reconstruction_error | ["SAME", 0.027209519221198897, 0.02720951922113826, -2.2285948286437018e-12] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.relative_reconstruction_error | lower |

Source: c5e3a037b69adc74ef5a193d6c1f563e68b812a9
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-0911613709b5f581"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/randomized-svd@dataset=istella@input=tsvd-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--tsvd-full-v1--runs/c8188af9b06dceb11c17/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.787532874965109 s; B: 1.5521876660641283 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| relative_reconstruction_error | 0.00022958200131431916 | 0.00022958200159608837 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.relative_reconstruction_error | ["SAME", 0.00022958200131431916, 0.00022958200159608837, 1.2273140219405484e-09] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.relative_reconstruction_error | lower |

Source: c5e3a037b69adc74ef5a193d6c1f563e68b812a9
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-9421af25cd249ea1"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/select-f-regression@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--selectors-full--runs/3bb7fcd8cc23a07afee7/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5372816659510136 s; B: 0.5497665000148118 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| n_selected | 5 | 5 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.n_selected | ["INFO", 5.0, 5.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.n_selected | info |

Source: c5e3a037b69adc74ef5a193d6c1f563e68b812a9
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-27030ad779657eb1"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/select-r-regression@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--selectors-full--runs/ab68cf756738987139b1/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.0481146249221638 s; B: 1.0465909579070285 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| n_selected | 110 | 110 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.n_selected | ["INFO", 110.0, 110.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.n_selected | info |

Source: c5e3a037b69adc74ef5a193d6c1f563e68b812a9
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-7d1d9f8dfabff0fe"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/select-r-regression@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--selectors-full--runs/b9e816c552496edb13ef/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5431199170416221 s; B: 0.5564376250840724 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| n_selected | 5 | 5 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.n_selected | ["INFO", 5.0, 5.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.n_selected | info |

Source: c5e3a037b69adc74ef5a193d6c1f563e68b812a9
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-1acc7d37a06507d4"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/select-f-regression@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--selectors-full--runs/bceac841eda6c1e6cc53/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.0595173330511898 s; B: 1.0542124159401283 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| n_selected | 110 | 110 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.n_selected | ["INFO", 110.0, 110.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.n_selected | info |

Source: c5e3a037b69adc74ef5a193d6c1f563e68b812a9
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-e320bd1611c96447"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/power-transformer@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/0f6d07d5f7a9fa8e86ec/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.3220051670214161 s; B: 1.3152390000177547 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| output_shape | 500000x11 | 500000x11 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.output_shape | ["INFO", "500000x11", "500000x11", null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.output_shape | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-fdd957e4ecd00693"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/simple-imputer@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/10caf828bd0b75b5e3ca/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 4.506303416914307 s; B: 4.5214825419243425 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| masked_rmse | 348517.29107358545 | 348517.29107358545 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.masked_rmse | ["SAME", 348517.29107358545, 348517.29107358545, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.masked_rmse | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-630a225f29927d08"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/ridge-clf@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/186156f500974335532d/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5944271250627935 s; B: 0.5979228341020644 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.764828 | 0.764828 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.764828, 0.764828, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-b9198196ba07ffb9"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/nmf@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/1e141f7a4b0c299d303b/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 6.20225400000345 s; B: 3.994893666007556 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| relative_reconstruction_error | 0.10123801123001107 | 0.10123802053308077 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.relative_reconstruction_error | ["SAME", 0.10123801123001107, 0.10123802053308077, 9.18930423026335e-08] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.relative_reconstruction_error | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-9082213724626923"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/sgd-clf@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/21217534b9ec24fdec13/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 10.737975417054258 s; B: 10.721622625016607 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.75694 | 0.75694 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.75694, 0.75694, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-8a57f086dc9f2da5"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/ridge-clf@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/2da7ce7cebbe7ae16fd0/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 4.917833624989726 s; B: 4.927551500033587 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.91056 | 0.91056 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.91056, 0.91056, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-12eb27621ba6bf16"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/factor-analysis@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/31a029c7a61734cee829/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 9.464645790983923 s; B: 9.445072249975055 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| mean_log_likelihood | 98.29477253771475 | 98.29477253771475 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.mean_log_likelihood | ["SAME", 98.29477253771475, 98.29477253771475, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.mean_log_likelihood | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-9821b20e14150fa2"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/pls-canonical@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/366d7ef45f2e1ab38840/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.2932769579347223 s; B: 1.0289566669380292 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| mean_canonical_corr | 0.5572790084785727 | 0.5572789925559 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.mean_canonical_corr | ["SAME", 0.5572790084785727, 0.5572789925559, 2.857217389066752e-08] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.mean_canonical_corr | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-4e3ed81580ca885c"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/cca@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/43f0120c502ae1170075/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.4188017920823768 s; B: 1.1777119589969516 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| mean_canonical_corr | 0.5748276455462062 | 0.5748276588628063 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.mean_canonical_corr | ["SAME", 0.5748276455462062, 0.5748276588628063, -2.3166247899818558e-08] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.mean_canonical_corr | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-34adcad1c2d9bfd1"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/power-transformer@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/50e1b5764b66cafe829d/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 4.231699166004546 s; B: 4.236502042040229 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| output_shape | 500000x220 | 500000x220 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.output_shape | ["INFO", "500000x220", "500000x220", null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.output_shape | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-ab70a58c367cac51"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/sgd-ocsvm@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/5b496fffda8d73e5cc95/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.7054802079219371 s; B: 0.7045067499857396 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| fraction_flagged | 0.066736 | 0.066736 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.fraction_flagged | ["INFO", 0.066736, 0.066736, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.fraction_flagged | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-2a21e87a2423c72e"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/maxabs-scaler@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/7012c5ed73f2ea3883ad/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5550444170366973 s; B: 0.5395590830594301 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| output_shape | 500000x11 | 500000x11 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.output_shape | ["INFO", "500000x11", "500000x11", null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.output_shape | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-15b4ecb49cddb1d1"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/pa-clf@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/706c6c218583c2556f05/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 10.199209874961525 s; B: 10.207266291952692 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.922242 | 0.922242 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.922242, 0.922242, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-1cf44a8254f4654b"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/qda@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/83311ee493aec7159caa/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.87055395799689 s; B: 0.876027874997817 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.727134 | 0.727134 |
| logloss | 1.069802155174614 | 1.069803487393698 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.727134, 0.727134, 0.0] |
| candidate_vs_baseline.metrics.logloss | ["SAME", 1.069802155174614, 1.069803487393698, 1.2452932708079158e-06] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |
| metric_directions.logloss | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-f9bccd951073f002"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/pa-clf@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/871ec7c64114eeac953f/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 6.93129941704683 s; B: 6.932964749983512 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.76288 | 0.76288 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.76288, 0.76288, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-0b3bc089cb244b34"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/factor-analysis@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/8aced037a81f1cc42846/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.7418258750112727 s; B: 0.7408203750383109 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| mean_log_likelihood | -14.835702629768983 | -14.835702629768983 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.mean_log_likelihood | ["SAME", -14.835702629768983, -14.835702629768983, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.mean_log_likelihood | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-db5834c9c27efd57"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/maxabs-scaler@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/95058e058d6c35a7f5aa/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.307464167010039 s; B: 1.279437833931297 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| output_shape | 500000x220 | 500000x220 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.output_shape | ["INFO", "500000x220", "500000x220", null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.output_shape | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-1ee4c1fbc8012177"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/nmf@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/9c048f5b19f91cda3a34/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 27.917224499979056 s; B: 22.648736208095215 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| relative_reconstruction_error | 0.32603817030763943 | 0.32603831370171243 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.relative_reconstruction_error | ["SAME", 0.32603817030763943, 0.32603831370171243, 4.398074305329766e-07] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.relative_reconstruction_error | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-8d6e9a07d67df7b4"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/lda-clf@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/ac6458a148bfb4f9c011/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.6697097499854863 s; B: 0.6343664590967819 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.76362 | 0.763624 |
| logloss | 0.5382462521088273 | 0.5382462794010144 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.76362, 0.763624, -5.238180046729805e-06] |
| candidate_vs_baseline.metrics.logloss | ["SAME", 0.5382462521088273, 0.5382462794010144, 5.070576095253126e-08] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |
| metric_directions.logloss | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-8e4baeb2c2495265"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/categorical-nb@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/acff99fbeb6c301c3fc0/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5952678329776973 s; B: 0.5976734579307958 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.767572 | 0.767572 |
| logloss | 0.5366124582756454 | 0.5366124582756454 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.767572, 0.767572, 0.0] |
| candidate_vs_baseline.metrics.logloss | ["SAME", 0.5366124582756454, 0.5366124582756454, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |
| metric_directions.logloss | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-3927c5ee2b7a6960"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/minibatch-kmeans@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/ad3d45e08cb8ab6c00e7/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.186901375069283 s; B: 1.1983942080987617 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| n_clusters | 8 | 8 |
| silhouette | 0.07649299931207186 | 0.07649299931207186 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.n_clusters | ["INFO", 8.0, 8.0, null] |
| candidate_vs_baseline.metrics.silhouette | ["SAME", 0.07649299931207186, 0.07649299931207186, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.n_clusters | info |
| metric_directions.silhouette | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-8e4ca1617bad6848"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/pls-canonical@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/b0667caada66668f51aa/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 5.837604667060077 s; B: 4.63670854200609 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| mean_canonical_corr | 0.8763007936505436 | 0.8763008127460923 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.mean_canonical_corr | ["SAME", 0.8763007936505436, 0.8763008127460923, -2.1791088653670855e-08] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.mean_canonical_corr | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-b397bfed2fe659b4"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/cca@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/b17b6e514c1006f88c5f/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 41.48247150005773 s; B: 22.62344487500377 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| mean_canonical_corr | 0.9890539742649506 | 0.989053929353883 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.mean_canonical_corr | ["SAME", 0.9890539742649506, 0.989053929353883, 4.540810582652929e-08] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.mean_canonical_corr | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-854651a37285160d"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/minibatch-kmeans@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/ba1006a727002dda6a90/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.6734981659101322 s; B: 0.6545286249602214 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| n_clusters | 8 | 8 |
| silhouette | 0.17145988449367378 | 0.17145988449367378 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.n_clusters | ["INFO", 8.0, 8.0, null] |
| candidate_vs_baseline.metrics.silhouette | ["SAME", 0.17145988449367378, 0.17145988449367378, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.n_clusters | info |
| metric_directions.silhouette | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-1517e260d7d8b58f"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/perceptron@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/bcdd603c81cc1f7814d2/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 10.061868375050835 s; B: 10.05267620808445 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.874976 | 0.874976 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.874976, 0.874976, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-56dabedb926fcb84"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/multinomial-nb@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/dc3d41d16eac72f9f13f/attempt-0001/receipt.json>)

Status: FAILED_OR_INCOMPLETE. Scope: full_workload. Observed A: — s; B: — s.

Quality assessment: NOT_ASSESSED. Identity: NOT_REQUIRED.

Scored quality metrics not recorded for this attempt.

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 0, "warmup": 0}, "B": {"scored": 0, "warmup": 0}}
Worker exits: [1]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)
Failures: ["Workload failed or did not write result JSON"]

</details>

<a id="attempt-98caf060dd85817f"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/categorical-nb@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/de43e5d216cb8ccb8197/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5537436249433085 s; B: 0.5527334589278325 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.840176 | 0.840176 |
| logloss | 0.41186649035415707 | 0.41186649035415707 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.840176, 0.840176, 0.0] |
| candidate_vs_baseline.metrics.logloss | ["SAME", 0.41186649035415707, 0.41186649035415707, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |
| metric_directions.logloss | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-7e5e1e28baf80d16"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/perceptron@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/df871bd8e0d9c37c890f/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 6.674702542019077 s; B: 6.67849483306054 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.493462 | 0.493462 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.493462, 0.493462, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-42a608e0ac18b8d1"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/lda-clf@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/e9508d14d3370dfade6c/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.949959292076528 s; B: 1.9105208329856396 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.913552 | 0.913612 |
| logloss | 0.2332565092210709 | 0.23332692474017572 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.913552, 0.913612, -6.56733930814711e-05] |
| candidate_vs_baseline.metrics.logloss | ["SAME", 0.2332565092210709, 0.23332692474017572, 0.0003017890849211559] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |
| metric_directions.logloss | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-0d0b5594fa0a8d2f"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/qda@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/fa0b1575c2b165e284aa/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.802077459054999 s; B: 1.7060410000849515 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.879716 | 0.879682 |
| logloss | 3.41997239311957 | 3.4196421677151796 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.879716, 0.879682, 3.864883667011799e-05] |
| candidate_vs_baseline.metrics.logloss | ["SAME", 3.41997239311957, 3.4196421677151796, -9.655791522018286e-05] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |
| metric_directions.logloss | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-b13a046d3cf309c0"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/sgd-clf@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/fad0e0840b7714d0907f/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 7.227511374978349 s; B: 7.266766374930739 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.92238 | 0.92238 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.92238, 0.92238, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-912ad3119709ba21"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/simple-imputer@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/fbe0df53cb628d25849b/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.843325916910544 s; B: 0.8367519580060616 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| masked_rmse | 6.08788405636742 | 6.08788405636742 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.masked_rmse | ["SAME", 6.08788405636742, 6.08788405636742, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.masked_rmse | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-cd7c19865a39cfae"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/sgd-ocsvm@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1--runs/fd25410fa324ee1d2da7/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.643035624991171 s; B: 1.627905625035055 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| fraction_flagged | 0.123182 | 0.123182 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.fraction_flagged | ["INFO", 0.123182, 0.123182, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.fraction_flagged | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-f4706d9f505de501"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/gaussian-nb@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/0d1e6d8e4700016071f0/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5767430829582736 s; B: 0.5805591250536963 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.720538 | 0.720538 |
| logloss | 1.1408800529708094 | 1.1408800529708094 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.720538, 0.720538, 0.0] |
| candidate_vs_baseline.metrics.logloss | ["SAME", 1.1408800529708094, 1.1408800529708094, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |
| metric_directions.logloss | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-190f057efefd762d"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/target-encoder@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/27ef5795de5e53b035e7/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5986357920337468 s; B: 0.5802826250437647 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| output_shape | 500000x8 | 500000x8 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.output_shape | ["INFO", "500000x8", "500000x8", null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.output_shape | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-16f51e15b7f26080"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/select-f-classif@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/6fc9ae71c2c1506ffb73/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.0540899590123445 s; B: 1.0597869580378756 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| n_selected | 110 | 110 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.n_selected | ["INFO", 110.0, 110.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.n_selected | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-116c12bf79353864"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/random-trees-embedding@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/786c65354ed40e98b838/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.8284146660007536 s; B: 1.8044732910348102 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| nonzeros_per_row | 10.0 | 10.0 |
| output_columns | 219 | 219 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.nonzeros_per_row | ["INFO", 10.0, 10.0, null] |
| candidate_vs_baseline.metrics.output_columns | ["INFO", 219.0, 219.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.nonzeros_per_row | info |
| metric_directions.output_columns | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-8174c832fc88ba19"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/gaussian-nb@dataset=istella@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/93e940c30f835c0f74ce/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.2473996250191703 s; B: 1.2319236249895766 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| accuracy | 0.868346 | 0.868346 |
| logloss | 3.5194426723515444 | 3.5194426723515444 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.accuracy | ["SAME", 0.868346, 0.868346, 0.0] |
| candidate_vs_baseline.metrics.logloss | ["SAME", 3.5194426723515444, 3.5194426723515444, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.accuracy | higher |
| metric_directions.logloss | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-278ee34473875268"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/random-trees-embedding@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/b18d2fcc10233c33faf1/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 1.4295469589997083 s; B: 1.3509567499859259 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| nonzeros_per_row | 10.0 | 10.0 |
| output_columns | 283 | 283 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.nonzeros_per_row | ["INFO", 10.0, 10.0, null] |
| candidate_vs_baseline.metrics.output_columns | ["INFO", 283.0, 283.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.nonzeros_per_row | info |
| metric_directions.output_columns | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-79d26a9761586603"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/target-encoder@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/be8c92995859ff28b0ea/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.6894132080487907 s; B: 0.6430268329568207 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| output_shape | 500000x5 | 500000x5 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.output_shape | ["INFO", "500000x5", "500000x5", null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.output_shape | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-8c82b1970257ac99"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/select-f-classif@dataset=taxi@input=classification-full-v1/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--classification-full-v1-remaining--runs/d1c0ff06e93ce2c5d6e4/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.5552098749903962 s; B: 0.5564248750451952 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Saved metrics lack an established shared comparison; no quality threshold invented.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| n_selected | 5 | 5 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.n_selected | ["INFO", 5.0, 5.0, null] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | UNKNOWN |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.n_selected | info |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-c12c57c7c2220909"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/qr@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--qr-svd-full--runs/2268b1fe4abee7294d5e/attempt-0001/receipt.json>)

Status: QUALITY_FAILED. Scope: full_workload. Observed A: 17.623082916019484 s; B: 4.2042725830106065 s.

Quality assessment: QUALITY_FAILED. Identity: NOT_REQUIRED.

Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| relative_gram_difference | 2.356185009416981e-06 | 1.601826657737046e-07 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.relative_gram_difference | ["WORSE", 2.356185009416981e-06, 1.601826657737046e-07, -0.932016091633933] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | WORSE |
| candidate_vs_baseline.worst | -0.932016091633933 |
| metric_directions.relative_gram_difference | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-09b759b2c9f3e9fd"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/qr@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--qr-svd-full--runs/6df9e7692fcc24a89c0c/attempt-0001/receipt.json>)

Status: QUALITY_FAILED. Scope: full_workload. Observed A: 1.1300790419336408 s; B: 0.6861715409904718 s.

Quality assessment: QUALITY_FAILED. Identity: NOT_REQUIRED.

Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| relative_gram_difference | 2.4421552190152415e-06 | 7.375898458611172e-07 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.relative_gram_difference | ["WORSE", 2.4421552190152415e-06, 7.375898458611172e-07, -0.6979758534109318] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | WORSE |
| candidate_vs_baseline.worst | -0.6979758534109318 |
| metric_directions.relative_gram_difference | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-79b79ad3b31ac9c5"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/svd@dataset=istella/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--qr-svd-full--runs/92b2b8e22831f6a2b014/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 4.673548541031778 s; B: 4.680020125000738 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| max_rel_singular_value_error | 2210.0368514055904 | 2210.0368514055904 |
| relative_reconstruction_error_100k_rows | 3.4308121421654494e-05 | 3.435029814175322e-05 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.max_rel_singular_value_error | ["SAME", 2210.0368514055904, 2210.0368514055904, 0.0] |
| candidate_vs_baseline.metrics.relative_reconstruction_error_100k_rows | ["SAME", 3.4308121421654494e-05, 3.435029814175322e-05, 0.0012278414564168415] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.max_rel_singular_value_error | lower |
| metric_directions.relative_reconstruction_error_100k_rows | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>

<a id="attempt-7f7a1c3b05abd44c"></a>
<details>
<summary>AF.X.complete-proposed — apple/apple-fast — algos/svd@dataset=taxi/attempt-0001</summary>

[Recorded toggles](#toggles-b5f3f72f7b33aadb) · [retained receipt](<receipts/apple/apple--captured--qr-svd-full--runs/cd159d14eb49f9042c45/attempt-0001/receipt.json>)

Status: PENDING_ADMISSION. Scope: full_workload. Observed A: 0.732211249996908 s; B: 0.7446086250711232 s.

Quality assessment: PENDING. Identity: NOT_REQUIRED.

Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.

| Saved quality metric | A: candidate | B: incumbent |
|---|---:|---:|
| max_rel_singular_value_error | 9.44616465827964e-07 | 9.44616465827964e-07 |
| relative_reconstruction_error_100k_rows | 1.7764759081724914e-06 | 1.7764759081724914e-06 |

Saved quality gate and opponent comparisons (no reassessment):

| Evidence field | Recorded value |
|---|---|
| candidate_vs_baseline.metrics.max_rel_singular_value_error | ["SAME", 9.44616465827964e-07, 9.44616465827964e-07, 0.0] |
| candidate_vs_baseline.metrics.relative_reconstruction_error_100k_rows | ["SAME", 1.7764759081724914e-06, 1.7764759081724914e-06, 0.0] |
| candidate_vs_baseline.unknown | [] |
| candidate_vs_baseline.verdict | SAME |
| candidate_vs_baseline.worst | 0.0 |
| metric_directions.max_rel_singular_value_error | lower |
| metric_directions.relative_reconstruction_error_100k_rows | lower |

Source: db59bb9557035da8fd11b0a020a84e33b0581c30
Samples (warmup/scored): {"A": {"scored": 1, "warmup": 1}, "B": {"scored": 1, "warmup": 1}}
Worker exits: [0, 0, 0, 0]
[Recorded implementation IDs and 0 source coverage gaps](#coverage-7f7db161711c3a3c)

</details>
