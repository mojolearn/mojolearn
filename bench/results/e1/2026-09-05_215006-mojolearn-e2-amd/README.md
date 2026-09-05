# AMD four-arm public kNN layout qualification

DigitalOcean MI325X, source `64e700357c5aa94ba1096f6cb0e9fe79a3e5e099`.
All eight builds, four correctness runs and 108 rotating timing invocations
passed, with CPU affinity restricted to cores 0–3. Full-output admission
matches the retained NVIDIA four-arm campaign. The validated output hashes
also match the earlier Apple campaign: 143,628 correctness pairs per arm,
plus every pricing output at all three query counts and nine rounds.

The kNN Mojo sources, shared fixtures, core/checks Mojo dependencies and
toolchain files did not change between NVIDIA's `9fe07a33` and this revision;
the only change within those directories was experiment documentation.
The commits remain separately recorded.

100,000 indexed points, 32 features, K=10, dyadic-v1 fixture. Median native
request milliseconds, including upload/search/download/synchronization:

| Queries | Baseline | Selector | Transpose | Both |
|---:|---:|---:|---:|---:|
| 32 | 3.853543 | 3.671054 | 2.125052 | 1.934756 |
| 128 | 10.123987 | 9.841770 | 2.504439 | 2.214244 |
| 1000 | 66.968421 | 65.703237 | 6.670732 | 5.522276 |

AMD's main benefit comes from transposition; NVIDIA's earlier campaign got
most of its benefit from the selector. Neither observation justifies a
universal default. Broader distributions, installed bindings, and external
comparators remain open. This is not a Python-API or cross-vendor speed ratio.

Raw logs and hashes are in `diag/knn-layout-price/`. Full samples, IQRs,
paired ratios and cross-vendor checks are retained in
[the comparison](../../resume/2026-09-05-next-certification/amd-nvidia-knn-layout.json)
and [Apple hash comparison](../../resume/2026-09-05-next-certification/amd-apple-hash-comparison.json).
The [controller log](../../resume/2026-09-05-next-certification/amd-layout-controller-2.log)
records archive verification, collection and teardown. Droplet 598100258 was
deleted after collection: DELETE 204, then verified GET 404 at 21:53:34 UTC.
Classification: `PASS` for this named native source campaign.
