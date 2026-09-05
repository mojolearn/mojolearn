# NVIDIA integrated UMAP and public k-NN qualification

RTX 4090, frozen source 64fa92f4005172b067973cf45705519c98eac0d8,
2026-09-05. All 23 source follow-up jobs passed. Public CSR UMAP fit,
transform and both held-out quality cases passed in all three modes.
IDENTICAL held-out training/query inputs and embeddings match the retained
Apple public-CSR and AMD 142553 captures byte for byte. Native sparse and
transform checks passed. This is source evidence, not Linux wheel evidence.

Public k-NN dispatch legacy/experimental results match all 133,348 selected
pairs and the AMD public capture. The separate whole-native-request price
fixture uses a 100,000-point index, 32 features and K=10; nine rotating rounds
per arm and all retained output bytes passed the strict comparator.

| Queries | Legacy median ms | Experimental median ms | Paired median speedup |
|---|---:|---:|---:|
| 32 | 10.803166 | 3.600512 | 2.986671x |
| 128 | 33.205752 | 5.533382 | 5.994780x |
| 1000 | 325.093991 | 16.979304 | 18.855937x |

Timing includes native public upload/search/download/synchronize, excludes
context creation, fixture preparation and warmup. It is not the earlier
Python/CUDA comparison fixture. Raw samples, quartiles, paired ratios and
log hashes are in knn-public-price-summary.json. The optimization remains
flag-gated and is not enabled in normal wheels. AMD timing remains pending.

Fresh Mamba source Python API checks passed; all five native backward cases
were GREEN. Original same-source cross-vendor certificates remain separate.
One paid GPU, CPU affinity four. Pod s9qrfl3lf5354z was deleted after fetch;
DELETE returned 204 and GET confirmed 404. Teardown completed 15:12:10 UTC.
