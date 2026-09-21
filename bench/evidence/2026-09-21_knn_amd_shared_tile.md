# AMD kNN shared tile on the two medium-large datasets

This trial measured repeated public `kneighbors` calls in IDENTICAL mode on
exactly the two R2-staged datasets used by the classical performance lane:

| dataset | source shape | timed kNN shape |
|---|---:|---:|
| Taxi | 4,000,000 x 11 | 400,000-row resident index x 4,000 queries, k=10 |
| Istella-S | 2,043,304 x 220 | 400,000-row resident index x 4,000 queries, k=10 |

Each cell used six alternating outer passes. Each visit had one untimed
warmup followed by three timed public calls. The reported ratio is the median
of the six paired outer-pass ratios (candidate/baseline); the geomean is over
Taxi and Istella-S. Every timed call hashed the complete returned distance
and index buffers.

## Unconditional shared tile and block top-k: rejected

The first candidate enabled the shared-memory distance tile and block top-k
for all positive feature counts. It improved the two-dataset geomean, but it
made the 11-feature Taxi workload materially slower on both AMD providers.
That repeatable single-dataset regression rejects unconditional AMD routing.

| provider / GPU | Taxi ms (base -> candidate) | Taxi ratio | Istella-S ms (base -> candidate) | Istella-S ratio | geomean |
|---|---:|---:|---:|---:|---:|
| DigitalOcean MI325X VF (`gfx942`) | 24.6410 -> 34.2718 | 1.3914 | 104.1189 -> 55.7047 | 0.5351 | 0.8629 |
| Hot Aisle MI300X VF (`gfx942`) | 26.1097 -> 34.7844 | 1.3302 | 115.5578 -> 57.7079 | 0.4992 | 0.8149 |

## Route only at 32 or more features: accepted

The follow-up candidate set `KNN_SMEM_MIN_FEATURES=32`. This disabled the
shared tile and block top-k for Taxi, while 220-feature Istella-S used both.
The trial build still admitted the experimental exact-chain metadata path on
Taxi because that admission had not yet been given the same runtime width
guard. Taxi therefore did not measure the final intended narrow request path,
and its result cannot by itself admit the promoted routing. The working
candidate now applies `n_features >= KNN_SMEM_MIN_FEATURES` to exact-chain
admission as well. A later broad identity run confirmed that the corrected
narrow default is unchanged from both the CPU oracle and a shared-tile
sabotage build; the promoted tile cannot affect requests below 32 features.

| provider / GPU | Taxi ms (base -> candidate) | Taxi paired ratio | Istella-S ms (base -> candidate) | Istella-S ratio | geomean |
|---|---:|---:|---:|---:|---:|
| DigitalOcean MI325X VF (`gfx942`) | 23.9059 -> 23.6980 | 0.9893 | 103.8297 -> 55.4420 | 0.5340 | 0.7268 |
| Hot Aisle MI300X VF (`gfx942`) | 26.1292 -> 25.1392 | 0.9627* | 115.6362 -> 57.6100 | 0.4984 | 0.6927* |

`*` The Hot Aisle Taxi baseline had one outlier and a 1.452 max/min spread,
so the harness marked that arm `flagged`; its Taxi ratio and derived geomean
are diagnostic, not promotion evidence. The stable DigitalOcean run and both
providers' roughly halved wide-input time support the width-gated candidate.
The broad identity and reach results below close the remaining promotion gate.

IQR below means Q3-Q1 over the six outer-pass medians, using inclusive
quartiles. The paired range is the minimum and maximum of the same six
outer-pass candidate/baseline ratios used to form the paired median.

| provider / dataset | baseline IQR ms | candidate IQR ms | paired ratio min..max |
|---|---:|---:|---:|
| DigitalOcean MI325X / Taxi | 0.056350 | 0.086727 | 0.986123..0.997484 |
| DigitalOcean MI325X / Istella-S | 0.081992 | 0.067299 | 0.533417..0.535732 |
| Hot Aisle MI300X / Taxi | 0.029652 | 0.034910 | 0.953902..0.963556 |
| Hot Aisle MI300X / Istella-S | 0.387202 | 0.073114 | 0.496925..0.499617 |

## Fixed-workload digest and quality result

All four races completed all six outer passes for both arms. Digests were
stable within each arm and bitwise equal across the baseline and candidate:

| output | full-buffer SHA-256 |
|---|---|
| Taxi distances plus indices | `884393f0a3370fd5f447499499d869c342ca4609a2693b140fce5fb11c122804` |
| Istella-S distances plus indices | `7dd293c5a7db136265c14a9f2a1d1ac1a3710ddc475c9659ed3096433a7574da` |

Because every returned neighbor index and distance bit is unchanged, kNN
recall and all downstream quality computed from these outputs are unchanged.
The fixed two-dataset digest gate passed on MI325X and MI300X for both the
unconditional and 32-feature candidates. The separate broad kNN identity gate
below supplies cross-column reference coverage and a shared-tile sabotage
witness before promotion.

## Broad identity and shared-tile reach: passed

The final `gfx942` candidate was built three ways from source `38cd6ce6c`:
the promoted HIP default, the same HIP build with
`MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1`, and the CPU host oracle. The broad
matrix covered 22 neighbor and density lanes, nine fixtures, two repeats, and
198 cells per arm. Every arm was complete with no skipped cells. The default
HIP and CPU outputs were bitwise equal. The narrow default and sabotage
outputs were also equal, proving that the shared tile remains isolated below
the 32-feature threshold.

A separate width probe hashed complete caller-visible output buffers for 36
cases at 32 and 220 features. All default HIP hashes equaled CPU. Sabotage
moved all 14 cases that traverse the promoted brute-force tile: Euclidean and
squared-Euclidean nearest-neighbor queries at `k=1,8,16`, plus uniform and
distance-weighted classifier and regressor calls. Non-L2 metrics, KDE, and
`RadiusNeighbors` remained equal as expected.

The on-box judge initially listed `radius-d32` as a required sabotage target
and therefore exited 1 after every substantive check had passed.
`RadiusNeighbors` uses the separate ball-cover count/fill implementation, so
it cannot reach the brute-force shared tile. The harness now keeps radius as
an identity check without requiring it to react to the shared-tile sabotage.
The corrected judge passes against the original three width-probe outputs;
no GPU rerun was needed and the uncorrected verdict is retained in the compact
receipts.

## Receipts and teardown

Raw receipts are outside the repository at
`/Users/andrewhendel/mojolearn-evidence/2026-09-21_knn_amd_smem/`. The four
subdirectories contain `race/summary.tsv`, per-call JSON, `verdict.json`, GPU
identity, source hashes, and provider lifecycle logs:

- `digitalocean-mi325x`: unconditional candidate, source `68dd2b56c`.
- `hotaisle-mi300x`: unconditional candidate, source `68dd2b56c`.
- `digitalocean-mi325x-wide`: 32-feature candidate, source `1afe6d028`.
- `hotaisle-mi300x-wide`: 32-feature candidate, source `1afe6d028`.

Compact tracked receipts for both conditional runs and the final broad gate are under
`bench/results/knn_amd_shared_tile_2026-09-21/`. They contain the original
summaries and verdicts, a deterministic projection retaining all round
medians, paired ratios, and timed-call digests, source and teardown receipts,
and hardware/runtime/dependency provenance. DigitalOcean used an MI325X VF,
driver 6.12.12 and ROCm 6.4.0-47. Hot Aisle used an MI300X VF in
`rocm/dev-ubuntu-22.04:6.4.1-complete`, driver 6.16.13 and ROCm 6.4.1-83.
Both used `gfx942`, Mojo 1.0.0 (`ed45d567`), NumPy 2.5.3, and PyTorch
2.6.0+rocm6.4.1.

Both DigitalOcean droplets were deleted and verified with HTTP 404. All three
Hot Aisle VMs, including the final broad-gate MI300X, were deleted, returned
HTTP 404, were absent from the provider list, and released their local lease
slots. The broad gate's Cloudflare R2 staging also verified the pinned Taxi
and Istella-S object hashes, although its fixed identity inputs did not need
to read those datasets. No paid resource from these trials was left running.
