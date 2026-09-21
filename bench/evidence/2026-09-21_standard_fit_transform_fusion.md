# StandardScaler fused `fit_transform` candidate

`MOJOLEARN_EXPERIMENTAL_STANDARD_FIT_TRANSFORM=1` keeps the existing
StandardScaler statistics and transform arithmetic but reuses the uploaded
matrix between them. The public Python path remains default-off.

The focused local matrix covered four shapes and all four `with_mean` /
`with_std` combinations. All 16 cells produced byte-identical fitted
statistics and transformed output. A public-API smoke on a 20,000 by 220
matrix was also byte-identical.

The promotion harness uses only the R2 Taxi and Istella inputs. It records the
source and prepared input hashes, shape, dtype, all fitted-statistics bytes,
the complete transformed output, and per-feature transformed means and
variances. It alternates arms for five retained runs in each of three fresh
outer processes and requires every within-process spread to be at most 1.10.
Promotion requires both datasets and a conservative fastest-baseline /
slowest-candidate speedup of at least 1.02.

## Results

Apple Metal rejected the candidate. All three Taxi outer processes retained
the same complete statistics and output hashes and reported median speedups of
1.2351x, 1.2352x, and 1.2417x. Timing was not stable: the baseline/candidate
spreads were 1.433/1.269, 1.298/1.524, and 1.042/1.305. The 16 GiB M4 could
not run the full Istella public-API cell even with sequential arms because the
1.8 GiB input plus the host, device, and output copies exceeded the available
unified memory. This is a resource refusal, not a result.

The guarded DigitalOcean MI325X trial at commit `58f788564` also rejected the
candidate. Taxi (4,000,000 by 11, float32) was byte-identical and semantically
valid. Its baseline and candidate medians were 143.122 ms and 100.243 ms
(1.4277x), but the candidate spread was 1.278, above the 1.10 limit. The
pre-fix body treated this expected timing rejection as fatal, so it did not
collect Istella or the remaining outer processes. That absence cannot support
promotion and no shipping default changed.

The body now distinguishes a complete explicit `REJECT_TIMING_GATE` record
from execution, identity, and quality failures. A timing rejection is retained
and collection continues; the final exactly-Taxi-and-Istella summary remains
the only promotion decision. The failed setup attempt and completed trial are
preserved externally under `mojolearn-evidence/training-amd-2026-09-21_*`.
The completed trial destroyed droplet `602462012` (DELETE 204, then GET 404).

