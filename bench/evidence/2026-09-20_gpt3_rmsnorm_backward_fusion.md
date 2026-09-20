# GPT-3-small exact RMSNorm backward fusion

Commit under test: this commit. Column: Apple M4, IDENTICAL. No cloud device
was touched. The production change is restricted to IDENTICAL; FAST and every
sabotage build retain the old split path.

The old backward launched a cell kernel to materialize
`dh[i] = pinned_mul(dy[i], weight[j])`, then a row kernel reread `dh` and
folded `c = fma(dh[j], x[j], c)` in ascending `j`. The fused kernel retains
one owner per row, computes and stores the same rounded `dh[j]`, then feeds
that stored value's identical local bits into the same ascending fold. It
removes one launch and the row kernel's global reread of `dh`. The remaining
cell kernel and weight-gradient GEMM are unchanged.

## Identity

Both binaries were built from the same source. The control added
`-D MOJOLEARN_BWD_NORM_SPLIT_TRIAL=1`; production omitted it. Both completed
`transformer_backward_check.mojo` clause (a): 17 cases, 37/37 stages, 412,172
cells bit-identical to the host reference. Their complete 37-stage trace
files also had the same SHA-256:

```
a4166b19a0af95ecb9f4978ac8476f61035aa98474c2a449e4ffe97679ec68c8
```

## Repeated-step timing

`bench/samba_rms_price_main.mojo` now includes the GPT-3-small training shape
`m=32768, d_model=768` (batch 16 times sequence 2048). Three alternating
whole-process pairs were run; each process reports seven synchronized device
measurements, and the number below is the median of each process median.

| path | backward milliseconds |
|---|---:|
| retained split control | 24.051 |
| fused IDENTICAL path | 18.129 |

That is 1.327x for one complete RMSNorm backward (26.7% lower latency). A
12-layer GPT step invokes two such backwards per layer, so the isolated-stage
weighted saving is 142.128 ms per step at this Apple shape. This is not a
whole-step claim: GEMM, attention, optimizer, and launch overlap are absent.

The smaller curve is mixed on Apple (the fusion is not a universal latency
win): 512x64 and 2048x512 regress in the noisy local measurement, while
8192x768 improves. Production therefore routes the fusion only at
`m >= 8192 && d_model >= 768` on Apple; smaller shapes and the unmeasured
NVIDIA/AMD columns retain the split path. The fused kernel itself is portable
and compiled everywhere. This commit is evidence for that implementation and
its Apple route, not a cross-vendor speed claim.
