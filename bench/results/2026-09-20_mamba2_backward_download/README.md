# Mamba2 public backward synchronization audit

Date: 2026-09-20
Base: `74095a4bf`
Hardware: Apple M4 Metal, shared local machine

## Profile

The production IDENTICAL Mamba2 prefill backward has one explicit compute
fence after its full forward/backward launch graph. That fence is meaningful:
it preserves compute-error precedence before public result copies. No drains
occur between the scan/state-gradient kernels. The remaining synchronization
cost is ten sequential `mamba_download` calls, one for each public gradient
leaf.

The screened candidate kept the compute fence, allocated all ten result
lists, enqueued their independent copies on the same ordered queue, and used
one final copy fence. Arithmetic, recurrence order, and kernel launch order
were unchanged.

## Long-sequence result

`bench/mamba2_backward_download_main.mojo` runs the complete public backward
at B=2, L=770, d_model=64 twelve times per timed sample and hashes all ten
returned leaves. Three separate B/A/A/B-style process rotations produced:

| rotation | baseline median | batched-copy median | change |
| --- | ---: | ---: | ---: |
| 1 | 928.685 ms | 859.189 ms | -7.48% |
| 2 | 1012.307 ms | 1014.396 ms | +0.21% |
| 3 | 1001.624 ms | 984.160 ms | -1.74% |

Across the nine raw samples per arm, the medians were 985.933 ms baseline
and 978.837 ms candidate (-0.72%). The direction did not remain stable enough
to justify changing the production boundary on this machine, so the source
candidate was fully reverted. Raw logs are retained under `apple/`.

All baseline and candidate samples produced the identical full-gradient hash
`10871010813014004120`. Before rejection, `pixi run mamba-grad-m2-public`
also passed all 55 compared tensors against the bitwise public-prefill oracle,
and `python3 tools/mamba_host_gen.py --check` passed after mechanical
regeneration. These correctness results do not override the inconclusive
performance result.

No cloud resource was provisioned. A discrete-GPU re-screen could price copy
fence latency differently, but there is no portable routing evidence here.
