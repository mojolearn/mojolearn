# Attention-v2 device forward/backward evidence (2026-09-20)

V2 remains opt-in; production v1 remains the default. All timings are raw
wall-clock launch-plus-synchronize milliseconds under IDENTICAL mode.

Apple M-series, B1 H12 L1024 HD64 forward, v2: 202.783 warmup, then 38.429,
39.691, 40.916. Production fused v1 at the identical shape: 41.474 warmup,
then 35.086, 29.591, 35.792. Both allocate 15,925,248 bytes in this harness;
v1 retained no probability stash. Therefore v2 has no production-memory win
and is slower here.

Apple B1 H12 L2048 HD64 backward, v2: 830.299, 856.707, 1216.607; production
fused v1: 290.998, 252.476, 210.494. V2 allocates 38,240,256 bytes and v1
44,531,712 bytes, a 6,291,456-byte reduction, but v2 is several times slower
and is not promotable. V2 launches three kernels: prepare and dQ each own one
of 24,576 query rows; dK/dV has 24,576 `(group,key)` owners in parallel, each
folding its group's 2,048 query rows in ascending order. There are no atomics
or global serial dK/dV fold.

DigitalOcean MI325X (`gfx942`, droplet 602182087, destroyed and GET-confirmed
404) exact gates passed twice for forward and backward at commit 18aca82b8.
Forward B1 H12 L1024 HD64: 31.660, 31.202, 31.091, 31.061. Backward B1 H12
L2048 HD64: 246.882, 244.273, 245.170. Resident bytes were 15,925,248 and
38,240,256 respectively. Raw logs are under ignored
`bench/results/attention-v2-backward/amd-mi325x/remote/attention_v2/`.

RunPod NVIDIA L40S (pod `jfomr6t7qoch8k`, 46,068 MiB, destroyed and
GET-confirmed 404) qualified the pinned-division commit `2c83aee58` twice.
V2 forward was 17.443, 17.260, 17.249, 17.239 ms, while production v1 was
0.649 warmup then 0.411, 0.411, 0.409 ms. V1 retained 12,582,912 exponent
cells (50,331,648 bytes); v2 retained none. V2 backward was 191.551, 191.383,
191.383 ms, while v1 was 3.406, 2.732, 2.720 ms and retained 50,331,648
exponent cells (201,326,592 bytes). This makes the tradeoff explicit: v2
substantially reduces memory on NVIDIA but is not performance-promotable.

The AMD and NVIDIA raw directories (21 compact objects) were uploaded to R2
and every object was verified by readback. Python analytic-quality witnesses,
repeatability, reverse-fold sabotage, and exact device fixtures passed; no
end-to-end language-model quality claim is made by this stage.
