# Mamba3 yintra tile — September 10, 2026

Initial same-pod H100 comparison, five timed rounds per arm:

| Shape | Baseline ms | Forced tile ms |
| --- | ---: | ---: |
| B2 L4 D32 | 1.219811 | 1.176375 |
| B8 L4096 D512 | 60.212171 | 56.820129 |
| B8 L1024 D2048 | 106.248042 | 102.802685 |

These are the baseline and candidate on this physical H100; earlier sessions' absolute prices came from another card and are not the denominator here. NVIDIA H100 80GB HBM3/driver 580.126.09, activated Pixi Mojo 1.0.0, image Python 3.11.10/NumPy 1.26.3. Both original large public output files match complete SHA256. No opponent was measured. The archived torch reference remains a resident device scan while ours includes public host transfers and full output return.

The tile shares independently rounded causal coefficients across 32 channels and V across 8 token rows. Each output keeps the entire ascending Q-term fold, including future and padded zero operations. Chunk alignment and partial continuation are covered by the direct gate: 100 cases, 1,128,960 cells plus 17-cell guarded suffixes, both Q32/Q64, continuation offsets including 63/64, poisoned excluded coefficients, signed zeros/subnormals/cancellation/large finite consumed operands. That gate passed on Apple and NVIDIA. NVIDIA native default/long traces match baseline; decode-cross, continuation, refusal, 39,087,232-cell public fresh/state comparison and surface checks passed for both arms.

Initial candidate c17a8f0b; guarded default 2c5fb5d3 requires IDENTICAL, NVIDIA, at least 256 threads/10 KiB shared capacity, and B×chunks×heads>=128. Explicit TILED_YINTRA forces the capacity-admitted tile for testing; LEGACY_YINTRA retains the baseline. Tiny and Apple defaults remain unchanged. The workload threshold supplies occupancy and is not a tuned intermediate-range optimum.

Final default was rebuilt and timed before the existing baseline binary to reverse order. Narrow: 59.503218 → 55.910172 ms (6.0% less); wide: 105.693202 → 101.634823 ms (3.8% less). Tiny medians were 1.183115 ms baseline / 1.219134 ms default despite retaining the same small-call kernel; no tiny-call speedup is claimed. All final public hashes, the 39,087,232-cell fresh/report check and surface gates passed. Reused historical resident-torch ratios are about 3.41×/5.24×; the requested 1–1.5× target remains unmet. No opponent was retimed.

[Final reverse-order samples, source/runtime manifests and identity gates](../../identity_continuation_2026-09-10/final-performance/summary.json) are archived by the parent alongside the logs in that directory. Reproduction: tools/mamba3_yintra_tile_leg.sh plus the source manifests, complete output hashes and raw samples under h100/mamba-yintra. See docs/lanes/HANDOFF_mamba3_yintra_tile_2026-09-10.md for the completed Mamba1 public-step wiring audit.
