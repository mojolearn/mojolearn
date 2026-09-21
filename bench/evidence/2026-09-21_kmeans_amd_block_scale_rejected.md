# AMD KMeans blocked accumulator plus device scale: rejected as a joint default

The default-off `MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC` and
`MOJOLEARN_EXPERIMENTAL_KMEANS_DEVICE_SCALE` pair was measured on one Hot
Aisle MI300X (gfx942) at `5b7af5271`. The candidate preserves every measured
fit and inference bit, but it is not a safe joint AMD default: Taxi fit became
about 4.7% slower and both Taxi fit arms missed the 1.10 stability gate.
Istella-S improved by 7.35x. The AMD defaults remain unchanged.

Both inputs came from Cloudflare R2 through the guarded provider runner. Taxi
uses 4,000,000 rows and 11 numeric features; Istella-S uses its available
2,043,304 rows and 220 features. Each arm uses the same 64 explicit initial
centroids, `n_init=1`, 20 maximum iterations, and IDENTICAL mode. Fit timing is
five alternating calls after one warmup. Transform is seven calls after one
warmup on 100,000 held-out rows; its early allocation/cache transients make its
max/min spreads diagnostic here.

| dataset | off fit median | candidate fit median | off/candidate | off spread | candidate spread | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Taxi | 116.524 ms | 122.314 ms | 0.9527x | 1.1717 | 1.1303 | reject: regression and unstable |
| Istella-S | 3154.510 ms | 429.134 ms | 7.3509x | 1.0061 | 1.0272 | strong win |

The two-dataset fit geomean is 2.646x, but that aggregate does not override a
measured dataset regression. The result suggests a future wide-feature AMD
dispatch experiment; it does not qualify one without a new conditional A/B.

Correctness is stronger than a metric tolerance. Candidate and baseline fit
hashes agree for centers, labels, iteration count, inertia, sum scale, and
weight scale on every timed call. Transform matrices and public `predict`
outputs also agree byte for byte, and both datasets report finite centers,
in-range labels, finite distances, and exact agreement between the predicted
cluster and the transform row minimum.

The broad gate covered six public KMeans configurations across all nine
standard fixtures (`base`, `ties`, `hashed`, `wide`, `denormal`,
`denormal_ftz`, `dupes`, `odd`, and `negative`), twice in one process: 54/54
cells complete per arm, 108/108 infer/model parts identical, 54/54 batch parts
identical, and no refusal. The block-accumulator planted defect made all 108
infer/model and all 54 batch parts divergent. The device-scale planted defect
made 103 infer/model and 49 batch parts divergent, with no refusal. Separate
full-data reach runs also proved that both planted defects move Taxi and
Istella-S, so neither measured candidate silently fell back.

Three non-algorithm failures are retained outside the repository under
`mojolearn-evidence`: the first lease hit a GitHub HTTP 504 while installing
Pixi; the second exposed an argument-order defect in the new identity diff
driver; the third ran three full Istella worker processes concurrently and the
third process exhausted its device partition. The final harness orders diff
arguments correctly and runs full-data reach workers sequentially. Every VM
was force-deleted and verified absent; the final receipt records HTTP 204 then
GET 404.

Compact performance samples, hashes, quality observations, reach records,
identity summaries, build hashes, R2 hashes, provenance, and teardown are in
[`bench/results/kmeans_amd_block_scale_2026-09-21/`](../results/kmeans_amd_block_scale_2026-09-21/).
