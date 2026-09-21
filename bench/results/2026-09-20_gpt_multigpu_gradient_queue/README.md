# GPT-3-small disjoint-owner gradient queue rejection

Hardware: one guarded RunPod with two NVIDIA L40S GPUs. The target was
IDENTICAL FP32 GPT-3-small: `B=1, L=2048, d_model=768, heads=12, layers=12,
intermediate=2048, vocab=50257` (162,147,840 parameters), six steps, seed
93261.

The scheduling-only candidate queued one logical shard's copy/add on every
disjoint optimizer owner before draining the owner contexts. It retained the
existing rank-by-rank ascending left fold and drained all owners before the
next rank. No arithmetic changed.

The strict target matrix passed complete baseline/candidate and one/two-device
parameter, first-moment, second-moment, flag, and gradient hashes for logical
shards 2, 3, and 5. The strict fault matrix also passed baseline/candidate
state hashes, native `grad_nonfinite`, `opt_refuse`, `after_nonfinite`, and
`after_negative` failures, post-update Python failure, rollback, and replay.

Two-device elapsed seconds from the two rotated target rounds:

| round | shards | baseline | candidate | candidate speedup |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.296963 | 0.294936 | 1.00687x |
| 1 | 3 | 0.487341 | 0.485722 | 1.00333x |
| 1 | 5 | 0.697988 | 0.698847 | 0.99877x |
| 2 | 2 | 0.298761 | 0.358031 | 0.83446x |
| 2 | 3 | 0.489512 | 0.560911 | 0.87271x |
| 2 | 5 | 0.700350 | 0.790085 | 0.88642x |

One-device candidate/baseline ratios remained within 0.9967x to 1.0025x.
The first rotation was effectively neutral, but the reverse rotation showed
an 11.4% to 16.6% two-device regression across all shard counts. The candidate
is therefore rejected as non-repeatable and materially regressing; production
source was restored unchanged.

Pod `uwzao6ig3z1jh1` was terminated at 2026-09-20 23:37:11 EDT. DELETE
returned HTTP 204 and the immediate verification GET returned HTTP 404.
