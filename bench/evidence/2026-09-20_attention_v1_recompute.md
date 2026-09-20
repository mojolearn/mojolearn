# V1-exact attention backward recompute arm (2026-09-20)

The opt-in build flag `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD` prevents the
forward wrapper from retaining its `[B,H,L,S]` exponent buffer. Backward then
uses the pre-existing production v1 recompute kernels. Default routing is
unchanged.

Apple B1 H12 L2048 HD64: the default and forced recompute paths both retained
zero cells on this column. Default backward was 130.160, 132.702, 129.905 ms;
forced recompute was 126.382, 127.510, 133.349 ms. All dQ/dK/dV cells were
bit-identical (`exact_bad 0`). The full fused/eager gate also passed all 15
cases under the flag.

RunPod L40S (pod `9k6bq011nxaz8t`, $0.79/hour, driver 580.159.03, destroyed
and GET-confirmed 404), same B1 H12 L2048 HD64 inputs:

- tuned v1 estash backward: 3.342 warmup, 2.728, 2.732 ms;
- v1 exact recompute: 4.677 warmup, 4.465, 4.472 ms;
- recompute is 1.64x the warm estash time;
- recompute removes 50,331,648 float cells, 201,326,592 bytes;
- dQ/dK/dV comparison: `exact_bad 0`.

This is a memory/throughput tradeoff, not a new default. It is useful when the
201 MB stash prevents the requested batch/sequence from fitting; the tuned
estash path remains appropriate when memory is available.

The supported build surface is:

```sh
MOJOLEARN_ATTENTION_MEMORY_PROFILE=recompute sh bindings/build_byte_lm.sh
MOJOLEARN_ATTENTION_MEMORY_PROFILE=recompute sh bindings/build_transformer.sh
```

Unset or `estash` preserves the default. Other values are refused. The byte-LM
binding reports `byte_lm_attention_memory_profile(shape)` as
`[profile_name, retained_exp_bytes_per_layer]`; checkpoint/run provenance
records both fields under `native_attention_arm`. A locally built recompute
binding reported `['v1-recompute', 0]` for the default byte-LM shape.

No safe speed follow-up survived static review. Recompute already reuses the
forward's exact row maxima and denominators. Reusing per-cell work requires
retaining either scores or exponentials, both the same float32 LxL size.
Losslessly packing only visible causal cells could halve full-causal storage,
but changes every forward/backward address and needs an independent NVIDIA
kernel campaign; it is not smuggled into this usability stage.
