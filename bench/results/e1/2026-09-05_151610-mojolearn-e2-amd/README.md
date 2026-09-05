# AMD whole-native-request k-NN selector pricing

AMD MI325X, frozen source 6476a35c1f6937a8f46178989fa6faed026e266d,
2026-09-05. Focused serial job: two builds and 54 timed processes, nine
rotating rounds per arm per shape. All jobs and strict admission passed.
All selected output bytes match across rounds, legacy/experimental arms
and the NVIDIA 103918 capture. Source and payload hashes are retained.

100,000 indexed points, 32 features, K=10, dyadic-v1 fixture:

| Queries | Legacy median ms | Experimental median ms | Paired median speedup |
|---|---:|---:|---:|
| 32 | 3.885478 | 3.667478 | 1.056588x |
| 128 | 10.111951 | 9.852190 | 1.026364x |
| 1000 | 67.020017 | 66.003946 | 1.008623x |

The large AMD fixture is effectively unchanged within observed variation;
this is not the substantial NVIDIA gain. See raw paired ratios and IQRs in
knn-public-price-summary.json. Timing includes native public input upload,
search, output download and synchronization, excluding fixture preparation,
context creation and warmup. No external competitor or Python-host timing
is measured here. The selector remains experimental, disabled in wheels.

Only the main controller ran builds and measurements, CPU affinity four,
BLAS threads one, pricing budget 480 seconds. DigitalOcean droplet 598047881
was deleted after artifact collection: DELETE 204, GET confirmed 404 at
15:17:32 UTC. No paid device remains from this leg.
