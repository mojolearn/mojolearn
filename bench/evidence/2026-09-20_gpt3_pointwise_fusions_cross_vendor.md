# GPT-3 pointwise backward fusions: NVIDIA and AMD qualification

Source under test: `e3122767a`.  Both legs built IDENTICAL split and
force-fused transformer-backward binaries from the same source.  Timing used
five processes and seven synchronized samples per process; the gated-SiLU
harness alternates call order within each process, while the norm2/residual
arms alternate as whole processes.

## Exactness

On NVIDIA H100 and AMD MI300X, all four forced arms (gated-SiLU split/fused
and norm2/residual split/fused) passed all 17 backward fixtures: 37/37 stages
and 412,172/412,172 cells bit-identical to the host reference.  Every arm on
both vendors emitted the same complete trace SHA-256:

```
a4166b19a0af95ecb9f4978ac8476f61035aa98474c2a449e4ffe97679ec68c8
```

## Timing

Times are medians in milliseconds.  The target shape is 32,768 token rows,
corresponding to GPT batch 16 at sequence length 2,048.

| backend / fusion / shape | split | fused | speedup |
|---|---:|---:|---:|
| H100 gated-SiLU 2048x3072 | 0.095406 | 0.076689 | 1.244x |
| H100 gated-SiLU 8192x3072 | 0.341166 | 0.275962 | 1.236x |
| H100 gated-SiLU 32768x3072 | 1.324914 | 1.074809 | **1.233x** |
| H100 norm2/residual 2048x768 | 0.030988 | 0.026064 | 1.189x |
| H100 norm2/residual 8192x768 | 0.091553 | 0.078195 | 1.171x |
| H100 norm2/residual 32768x768 | 0.333677 | 0.285419 | **1.169x** |
| MI300X gated-SiLU 2048x3072 | 0.070724 | 0.060933 | 1.161x |
| MI300X gated-SiLU 8192x3072 | 0.273326 | 0.197242 | 1.386x |
| MI300X gated-SiLU 32768x3072 | 1.279190 | 0.748531 | **1.709x** |
| MI300X norm2/residual 2048x768 | 0.039299 | 0.030003 | 1.310x |
| MI300X norm2/residual 8192x768 | 0.075185 | 0.055869 | 1.346x |
| MI300X norm2/residual 32768x768 | 0.266357 | 0.206868 | **1.288x** |

At batch 16, gated-SiLU saves 3.001 ms per 12-layer backward on H100 and
6.368 ms on MI300X.  Norm2/residual saves another 0.579 ms and 0.714 ms,
respectively.  Both fusions therefore route on NVIDIA and AMD IDENTICAL as
well as Apple.  FAST and sabotage builds retain their previous behavior.

## Hardware and cleanup

- NVIDIA RunPod `tk6yb46r76epyt`: H100 80GB HBM3, driver 580.126.09.
  DELETE HTTP 204, then GET HTTP 404.  The externally owned attention pod
  `cgp2jehyh5elty` was not touched and remained running after cleanup.
- AMD HotAisle deployment `48e3722c-7db2-41ff-b846-5adebeb318d9`: MI300X
  VF (`gfx942`). DELETE HTTP 204, then GET HTTP 404 and absent from the VM
  list; its local ownership slot was released.

Raw device witnesses, timing logs, check logs, trace digests, watchdog
records, and teardown receipts are retained under
`bench/results/e1g/2026-09-20-{nvidia,amd}-gpt3-pointwise-fusions/`.
Both wrapper records say `extra_exit=1` because the original summarizer glob
also admitted `norm-*-check.log`, which contains no `PRICE` rows, after every
benchmark and exactness check had completed.  The retained raw logs produced
the table above; the glob is corrected in the committed body.
