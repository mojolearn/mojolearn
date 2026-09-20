# GPT-3-small K/V gradient-slice fusion

## Scope and structural proof

- Base: `c570c7109`; mode: explicit `MOJOLEARN_NUMERIC_IDENTICAL`, FP32.
- Hardware: local Apple M4 Max; no cloud resource used.
- Production stages: transformer backward stages 25 and 26.
- Shapes: GPT-3-small K/V width 768 at 2,048, 8,192, and 32,768 token
  rows (1,572,864 through 25,165,824 cells per output).

The KV append backward previously launched the same slice kernel twice, once
for K and once for V. Both calls derive the identical source address from
`(batch, token, kv_head, head_dim, key-span offset)` and then perform a plain
bit copy. The fused kernel derives that integer address once and copies the
independent K and V inputs to their original outputs. It performs no floating
point operation, changes no reduction, and introduces no cross-output
dependency.

Launches fall from two to one. Explicit global traffic is unchanged at two
reads and two writes (16 bytes per cell pair), while the duplicated integer
division/modulo address work is halved. Retained and scratch storage are
unchanged.

## Production-shape timing

Each row contains nine alternating baseline/fused samples in nanoseconds,
after warm-up. Complete outputs from both K and V were compared as `uint32`
bits before timing; every shape had zero mismatches.

| rows | baseline samples | fused samples | median fused / baseline |
|---:|---|---|---:|
| 2,048 | 8942000, 9445000, 8042000, 6977000, 10429000, 8784000, 5852000, 5227000, 8818000 | 4764000, 4871000, 4686000, 4767000, 3865000, 4838000, 5358000, 5089000, 4140000 | 0.543 |
| 8,192 | 28987000, 19219000, 17168000, 15106000, 14537000, 13573000, 12798000, 12139000, 11699000 | 10389000, 10582000, 8241000, 7782000, 7167000, 7336000, 6311000, 6841000, 6094000 | 0.505 |
| 32,768 | 59291000, 38110000, 40200000, 40985000, 44992000, 50751000, 53888000, 45379000, 44665000 | 19106000, 19484000, 20008000, 19760000, 26684000, 25314000, 31949000, 38668000, 27477000 | 0.563 |

Every measured round improved. The median stage reduction is 43.7--49.5%.

## Gates

Both commands used `MOJOLEARN_BUILD_JOBS=2`.

```
pixi run mojo run -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  transformer/checks/transformer_backward_check.mojo
```

Passed all 17 fixtures, all 37 backward stages, all 412,172 compared cells,
and all 37 trace-card records against the host oracle. This includes MHA,
GQA, odd-head-dimension, windowed, cache-hot-tail, and signed-zero fixtures.

```
pixi run mojo run -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  training/checks/byte_lm_head_v2_integration_check.mojo
```

Passed `BYTE_LM_HEAD_V2_INTEGRATION_OK`, including repeated ByteTrainer
updates. Optimizer state, loss, and update behavior therefore remain exact on
the integrated training path.

## Adjacent audit

The next adjacent pair is Q/K RoPE backward. It was not fused here: Q and K
can have different head counts under GQA, and duplicating the heavily audited
rotation/sign/sabotage body would create a second arithmetic spelling. The KV
slice pair is preferable because its outputs are pure copies with one shared
integer address calculation and no numerical seam.
