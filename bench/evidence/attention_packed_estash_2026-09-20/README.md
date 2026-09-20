# Packed causal exponent stash qualification

Commit under test: `60bbe4879`, rebased by fast-forward onto `7f83fee27`.
Provider resource: RunPod `6unpt1svcjgy5t`, NVIDIA L40S 46,068 MiB,
45-minute API watchdog. It was explicitly deleted (HTTP 204) and absence was
verified (GET HTTP 404) at 2026-09-20 11:37 EDT. The lease record is retained
under `bench/results/runpod_leases/`.

Shape: B=1, L=S=2048, H=12, KVH=12, HD=64, causal, identical mode, hashed
nonuniform operands; 3 warmups and 9 measured alternating rounds per binary.

The default and packed logs have exactly the same seven output digests:
ctx `13ccc93eac28789a`, amax `afaf4c3ad51a2071`, denom
`461ef8b28777ac11`, zdot `aba1ca249126f2a9`, dQ `d66200e4f5fecee1`,
dK `cec17a18979026ec`, dV `1082386d9d9af404`.

Default estash median forward+backward was 4.448241 ms (derived backward
3.127946 ms). Packed estash was 4.427081 ms (derived backward 3.095537 ms),
a 0.48% end-to-end improvement. Forward was 1.320295 vs 1.331544 ms; this
small isolated regression is outweighed in the full training pair and the
candidate remains opt-in.

Retained exponent allocation falls from 50,331,648 float32 cells
(201,326,592 bytes) to 25,178,112 cells (100,712,448 bytes), a 49.98%
reduction. The existing exact recompute profile retains zero cells but its
previous same-L40S qualification measured 4.465--4.472 ms backward alone;
packed keeps tuned-estash speed. The attempted redundant recompute run here
is retained as a negative harness log: an estash-specific requested arm is
correctly unavailable in a recompute build.

Apple trial coverage exercised full causal, windowed, and split-cache
training/prefill cases with bit-identical ctx/amax/denom/zdot/dQ/dK/dV.
Existing decode-arm differences were unchanged and are not a packed-stash
claim.
