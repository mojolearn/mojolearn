# GaussianMixture and resampling above 1 MiB — two H100s and two MI300X

The Cholesky column solve diverged on two MI300X only once its buffers passed
1 MiB (`../../2026-09-14/cholesky-mi300x-diag/`), and every earlier
GaussianMixture and resampling receipt used buffers under 1 MiB. This leg
re-asks both merged drivers above it, with the same body (`body.sh`) and
commit on each vendor (`h100/commit.txt`, `mi300x/commit.txt`).

- `MOJOLEARN_GMM_CHECK_LARGE=1`: E-step shapes 70001x4x3, 262145x2x2 and
  131072x8x4 (every stage bit-equal between one and two devices) and a
  90001-row fit with equal identity traces and state.
- `MOJOLEARN_RESAMPLE_CHECK_LARGE=1`: bootstraps of 300001 and 400000
  replicates, a 300001-permutation test and a 70-million-sample Monte Carlo
  (273,438 chunk partials), traces and results equal between one and two
  devices.

Both vendors print every PASS line (`*/gate.txt`), and `h100/digests.txt`
and `mi300x/digests.txt` (the SHA256 of every one- and two-device trace file)
are equal after sorting by name.
