# Transformer reference corpus

This is the independent, tolerance-based reference for the Llama-shaped block
defined by `mojolearn.identical.transformer.fp32.v1`. `gen_corpus.py`
reimplements the block order from Hugging Face Transformers commit
`d56c55bf564ddb176759eb6ec199442682564916` in PyTorch without importing
mojolearn or the `transformers` package.

It complements, but does not replace, the bitwise host/device gate:

- `ref64` can detect a shared wrong algorithm.
- `ref32` calibrates an honest FP32 tolerance and is not a target.
- summation order, cross-vendor identity, and performance remain properties of
  the Mojo gate and its sabotage arms.

## Commands

```sh
pixi run -e skgpu python transformer/corpus/gen_corpus.py --verify
pixi run -e skgpu python transformer/corpus/gen_corpus.py --self-test
python tools/transformer_corpus_check.py transformer/corpus/<case> <dump_dir>
python tools/transformer_corpus_check.py transformer/corpus/<case> --self-test
python tools/transformer_corpus_check.py transformer/corpus/<case> --negative-control
```

`--verify` checks byte determinism, batch composition, decode/prefill
composition and its negative control, rotary anchors, attention scaling,
adversarial reach, finite references, and a 17-perturbation sensitivity
matrix. The checker distinguishes missing and short dumps from numerical
disagreement and separately verifies signed zero.

## Reproducible inputs

Hash specification: `mojolearn.transformer.corpus.hash.v1`.

```c
uint64 splitmix64(uint64 z) {
    z += 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
    return z ^ (z >> 31);
}
uint64 key = splitmix64(seed ^ (tid << 32));
uint64 h = splitmix64(key + (uint64)i);
double f = (double)(h >> 40) * 0x1p-24;
float value = (float)(lo + (hi - lo) * f);
```

Arithmetic wraps modulo `2^64`; `i` is the flat row-major index. All ranges
are dyadic, making the float64 mapping exact before its single float32
rounding. Case `k` uses seed
`0x58666D72436F7270 + 0x1000*k`; composition and decode subcases reuse their
parent seed. A case manifest is authoritative for overrides.

| tensor | id | shape | default range |
|---|---:|---|---|
| `x` | 1 | `[B,L,d_model]` | `[-2,2]` |
| `norm1.weight` | 2 | `[d_model]` | `[0.5,1.5]` |
| `norm2.weight` | 3 | `[d_model]` | `[0.5,1.5]` |
| `q_proj.weight` | 4 | `[n_heads*head_dim,d_model]` | `[-0.5,0.5]` |
| `k_proj.weight` | 5 | `[n_kv_heads*head_dim,d_model]` | `[-0.5,0.5]` |
| `v_proj.weight` | 6 | `[n_kv_heads*head_dim,d_model]` | `[-0.5,0.5]` |
| `o_proj.weight` | 7 | `[d_model,n_heads*head_dim]` | `[-s,s]` |
| `gate_proj.weight` | 8 | `[intermediate_size,d_model]` | `[-0.25,0.25]` |
| `up_proj.weight` | 9 | `[intermediate_size,d_model]` | `[-0.25,0.25]` |
| `down_proj.weight` | 10 | `[d_model,intermediate_size]` | `[-s,s]` |

For output projections, `s = 0.5 / 2^ceil(log2(fan_in)/2)`. There are no
biases.

Exact post-hash plants:

- `adv_signed_zeros_b2_l4_d32_kv2`: rows with `t%4==0` become `+0.0`, rows
  with `t%4==2` become `-0.0`, and remaining flat indices with `i%7==3`
  become `-0.0`.
- `adv_softmax_saturation_b1_l8_d32_kv1`: multiply row `t` by `2^(t-3)`.

## Shapes and dump convention

The 30 stage names, definitions, shapes, and calibrated tolerances live in the
manifests. `M=B*L`; ordinary activations are token-major `[M,F]`. Cache and
attention stages lead with `B`. RMSNorm `sumsq` stages are sums, not means.
Rotary files contain half tables `[n_positions,head_dim/2]`; a checker may
select used absolute positions from a full `[P_max,head_dim/2]` dump.

A driver dump contains raw, row-major, little-endian `<stage>.f32` files. It
should also contain inputs under their manifest names and `positions.i32`;
the checker verifies inputs bitwise before attributing output differences to
arithmetic.

## Cases

| case | B | L | model | heads/kv | head | inter | pos0 | purpose |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base_b1_l1_d32_kv2` | 1 | 1 | 32 | 2/2 | 16 | 64 | 0 | minimum/decode-empty |
| `base_b2_l4_d32_kv2` | 2 | 4 | 32 | 2/2 | 16 | 64 | 0 | causal and batch composition |
| `comp_row0_b1_l4_d32_kv2` | 1 | 4 | 32 | 2/2 | 16 | 64 | 0 | parent row 0 |
| `comp_row1_b1_l4_d32_kv2` | 1 | 4 | 32 | 2/2 | 16 | 64 | 0 | parent row 1 |
| `base_b1_l4_d32_kv1` | 1 | 4 | 32 | 2/1 | 16 | 64 | 0 | GQA repetition |
| `base_b2_l16_d32_kv1_i300` | 2 | 16 | 32 | 2/1 | 16 | 300 | 0 | ragged GEMM partition |
| `base_b1_l4_d48_hd24_kv1` | 1 | 4 | 48 | 2/1 | 24 | 64 | 0 | inexact attention scale |
| `adv_mask_extreme_b1_l4_d32_kv2` | 1 | 4 | 32 | 2/2 | 16 | 64 | 0 | mask addition reach |
| `adv_subnormal_scores_b1_l4_d32_kv2` | 1 | 4 | 32 | 2/2 | 16 | 64 | 0 | subnormal scores |
| `adv_signed_zeros_b2_l4_d32_kv2` | 2 | 4 | 32 | 2/2 | 16 | 64 | 0 | signed-zero plants |
| `adv_rope_far_pos_b1_l4_d32_kv2` | 1 | 4 | 32 | 2/2 | 16 | 64 | 8188 | rotary ceiling |
| `adv_softmax_saturation_b1_l8_d32_kv1` | 1 | 8 | 32 | 2/1 | 16 | 64 | 0 | softmax denominator arms |
| `decode_b1_l4_d32_kv1` + four steps | 1 | 4 | 32 | 2/1 | 16 | 64 | 0 | decode/prefill composition |

Every manifest records `certifies`, `cannot_certify`, and measured
`sensitivity`; the measured table wins over prose.

## Precision and tolerance

Reference arithmetic is float64 except for constants whose FP32 bits are part
of the profile: rotary inverse frequencies and angles, attention scale,
RMSNorm epsilon `1e-6`, and mask fill `-3.4028234663852886e+38`. The checker
requires the FP32 `rope.inv_freq` anchor bitwise before treating cosine/sine
comparisons as unconditional.

Each stage stores `rtol_torch_fp32`, `rtol_correctly_rounded`, and
`rtol_gate`; the gate is exactly one rung looser than the measured PyTorch
FP32 arm. Rungs are `{1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1}` and default
absolute tolerance is `1e-6`. Signed zeros are always compared by sign bit.
An infinity passes only with matching sign when the finite reference exceeds
the FP32 maximum.

## Files

Within `transformer/corpus/<case>/`:

- `<tensor>.f32`: input and weights, raw little-endian float32;
- `positions.i32`: absolute positions, raw little-endian int32;
- `ref64/<stage>.f64`: independent float64 reference;
- `ref32/<stage>.f32`: informative PyTorch FP32 stage;
- `manifest.json`: shape, calibration, sensitivity, and provenance.

The top-level manifest lists cases, seeds, tensor IDs, stage definitions,
perturbations, precision rules, and the upstream commit.

## Scope

This corpus does not certify bits, vendors, summation order, FlashAttention,
SDPA, rope scaling, noncausal masks, projection biases, or performance. Those
belong to the normative contract and executable Mojo gates.
