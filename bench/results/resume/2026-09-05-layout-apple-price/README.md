# Apple four-arm public k-NN pricing and NVIDIA comparison

Apple M4, native source `9fe07a33`. All eight builds, four correctness runs
and 108 rotating timing invocations passed. The strict comparison against
NVIDIA at the same commit passes: every selected distance bit and neighbor
index agrees across correctness fixtures, timing fixtures, arms and rounds.
See cross-vendor-summary.json for full hashes, samples, IQRs and paired ratios.
Metadata records native source hashes verified against the pinned commit.

100,000 indexed points, 32 features, K=10, dyadic-v1 fixture. Median native
request milliseconds (upload, search, download and synchronization):

| Queries | Baseline | Selector | Transpose | Both |
|---:|---:|---:|---:|---:|
| 32 | 49.009 | 49.022 | 63.890 | 71.838 |
| 128 | 115.066 | 114.285 | 127.212 | 126.004 |
| 1000 | 667.716 | 651.234 | 674.806 | 652.839 |

Apple does not reproduce the NVIDIA speed gains. The combined arm is slower
at the smallest query count; the large-request medians are close. This
prevents a vendor-independent default-promotion claim. AMD, broader datasets
and installed-wheel qualification remain pending.

The retained run.py mirrors the shell campaign's build flags, fixed fixture,
two excluded warmups per process, nine rounds, four-arm rotating order and
20-minute aggregate deadline. It uses Python process-group timeouts because
this Mac has neither timeout nor gtimeout. Builds use two workers; all work
ran serially under tools/with_build_lock.sh after NVIDIA was terminated.
Timing is measured inside the same native drivers on both platforms; Python
orchestration is outside the measured request. No cross-vendor speed ratio
is calculated, and neither run has an external-library comparator.

Reproduction: use a fresh copy of the retained runner in a new sibling
results directory, then run it under tools/with_build_lock.sh. It refuses to
overwrite an existing status.tsv. Binaries are hashed and removed; raw logs,
build logs, source hashes and completion records remain here.
