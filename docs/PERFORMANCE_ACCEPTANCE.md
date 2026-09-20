# Performance optimization acceptance

Mojolearn does not use a fixed minimum percentage speedup for accepting an
optimization. In particular, there is no 10% bar. Small improvements compound
and may ship when evidence shows that they are real, general, and worth the
maintenance cost.

Correctness and numerical contracts come first. An optimization must preserve
the affected numeric mode, public API, error behavior, model quality, and
deterministic routing. IDENTICAL changes require bitwise evidence; FAST changes
require the stated quality and reproducibility checks. A speedup does not excuse
a moved bit where bits are promised or a quality regression.

Performance evidence must match the claim. Use interleaved or rotated
before/after measurements on representative production-sized work, record raw
samples and memory pressure, and include multiple seeds, shapes, and data
distributions when behavior can depend on them. Dispatch may depend on
documented hardware, mode, algorithm configuration, and shape. It must not
recognize a benchmark dataset, seed, label pattern, or measured outcome.

Acceptance is a review judgment, not a percentage threshold. Consider:

- reproducibility and the absence of meaningful regressions across the tested
  support region;
- end-to-end speed, memory, launch, or transfer benefit, including gains below
  10%;
- breadth of the affected production path;
- implementation complexity, maintenance burden, and regression risk;
- whether any guard is simple, data-independent, and evidenced on both sides.

A small, simple, consistently positive change can be accepted. A larger but
noisy, fixture-specific, quality-moving, or high-risk change must be rejected
or narrowed. Microbenchmarks can diagnose a kernel, but do not alone justify a
public default when an end-to-end path is available.

Record unsuccessful experiments in `bench/evidence` with tested scope,
timings, correctness result, and rejection reason. Separate true rejections
from promising work awaiting another hardware column. Do not leave rejected
production code enabled or erase evidence that prevents repeated dead ends.
