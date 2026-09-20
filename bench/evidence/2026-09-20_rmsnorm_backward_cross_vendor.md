# Exact RMSNorm backward fusion: NVIDIA and AMD qualification

Source under test: `36709ea83`. Both legs built IDENTICAL split and
force-fused binaries from that same source. Each timing cell is the median of
five alternating whole-process medians, with seven synchronized calls per
process. The complete transformer backward gate ran separately for both arms.

## Identity

On both the NVIDIA RTX 4090 and AMD MI300X, both arms passed all 17 backward
fixtures: 37/37 stages and 412,172/412,172 cells bit-identical to the host
reference. Split and fused emitted the same complete trace SHA-256 on both
vendors:

```
a4166b19a0af95ecb9f4978ac8476f61035aa98474c2a449e4ffe97679ec68c8
```

## Timing

| backend and shape | split ms | fused ms | fused speedup | decision |
|---|---:|---:|---:|---|
| RTX 4090, 512x64 | 0.032931 | 0.034311 | 0.960x | split |
| RTX 4090, 2048x512 | 0.063252 | 0.063792 | 0.992x | split |
| RTX 4090, 8192x768 | 0.308781 | 0.357293 | 0.864x | split |
| RTX 4090, 32768x768 | 1.655832 | 1.615780 | **1.0248x** | fuse |
| MI300X, 512x64 | 0.089039 | 0.082626 | 1.078x | split; outside target |
| MI300X, 2048x512 | 0.164991 | 0.163443 | 1.009x | split; outside target |
| MI300X, 8192x768 | 0.246636 | 0.430880 | 0.572x | split |
| MI300X, 32768x768 | 0.784132 | 1.262635 | 0.621x | split |

NVIDIA therefore joins Apple only at `m >= 32768 && d_model >= 768`. The
measured 0.040052 ms saving occurs twice per layer, or 0.961248 ms across the
24 normalization backwards of a 12-layer step, before overlap. NVIDIA shapes
below that boundary and every AMD shape remain on the split path. This is a
backend scheduling choice around one portable exact kernel, not different
arithmetic.

## Rental cleanup

- AMD HotAisle deployment `4c84e855-49bd-4eb6-8c28-9a5c61cdb08d`:
  DELETE HTTP 204, then GET HTTP 404 and absent from the VM list.
- NVIDIA RunPod `g603hr0e7cd9vm`: DELETE HTTP 204, then GET HTTP 404.

Raw logs, source hashes, device witnesses, process medians, guard records, and
teardown receipts are retained in
`bench/results/e1g/2026-09-20-{nvidia,amd}-rmsnorm-backward-cv/`.
