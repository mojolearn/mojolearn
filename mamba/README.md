# Mamba

This directory implements reference-pinned Mamba 1, Mamba 2, and Mamba 3
blocks. They are low-level forward blocks with caller-owned continuation
state, not trainable estimator or complete language-model APIs.

## Current surface

| family | forward | decode/continuation | Python binding | backward |
|---|---|---|---|---|
| Mamba 1 | implemented | implemented | implemented | complete whole-pass validation for public gradients |
| Mamba 2 | implemented | implemented | implemented | all ten public prefill leaves; explicit incoming-state gradient; decode/window backward unsupported |
| Mamba 3 | implemented | implemented | implemented | all ten public prefill leaves; decode/window backward unsupported |

All forward families expose `fast`, `deterministic`, and `identical` builds.
FAST is allowed to differ from the IDENTICAL oracle; recording that difference
is not a FAST failure. Cross-vendor claims require matching cards from a named
commit and configuration.

The Python binding keeps each `DeviceContext` alive through the final buffer
download. This is required on every backend and is especially important on
AMD, where premature context destruction previously presented as a GPU memory
access fault.

Local identity gates must be wrapped explicitly; the Pixi tasks remain
mode-neutral because the multi-vendor harness supplies FAST and IDENTICAL:

```sh
tools/with_identical_mode.sh pixi run check-mamba-block
tools/with_identical_mode.sh pixi run check-mamba2-block
tools/with_identical_mode.sh pixi run check-mamba3-block
```

## Contracts and evidence

- [Mamba 1 contract](IDENTICAL_MAMBA_CONTRACT.md)
- [Mamba 2 contract](IDENTICAL_MAMBA2_CONTRACT.md)
- [Mamba 3 contract](IDENTICAL_MAMBA3_CONTRACT.md)
- [Corpus format and cross-check](corpus/README.md)

The contracts define the profile; executable checks are the status source.

## Backward implementation order

1. Validate Mamba 2's explicit incoming-state gradient at a multichunk
   boundary.
2. Define cotangents for mutable convolution windows before implementing
   token-at-a-time decode backward.
3. Extend that state-boundary contract to Mamba 1 and Mamba 3.

## Independent gradient oracle

`tools/mamba_gradient_oracle.py` differentiates the cited upstream PyTorch
forward transcription in float64 and independently audits selected gradient
cells with central finite differences. It covers all three families and emits
portable `grad.<tensor>.f64` files plus a manifest:

```sh
pixi run -e skgpu mamba-grad-m1
pixi run -e skgpu mamba-grad-m2
pixi run -e skgpu mamba-grad-m3
```

For an accelerator run, invoke the script through `pixi run -e skgpu python`
and add `--device cuda`. The same spelling selects a ROCm device in a ROCm
PyTorch build; the manifest records CUDA versus HIP. These are tolerance
references. IDENTICAL validation still requires a Mojo device card to match
the pinned Mojo host oracle bit for bit.

Strict public-prefill gates are:

```sh
pixi run -e skgpu mamba-grad-m1-public
pixi run -e skgpu mamba-grad-m2-public
pixi run -e skgpu mamba-grad-m3-public
```

Mamba 2 additionally exposes the cotangent of `initial_states` immediately
before chunk zero. The strict witness uses L257 so the reverse recurrence
crosses a chunk boundary; the block-output objective has an explicit zero
final-state cotangent:

```sh
pixi run -e skgpu mamba-grad-m2-initial-state
```

The incoming-state leaf is compared with float64 autograd through a rebuilt
forward recurrence. Its separate `stage.initial_state` diagnostic uses the
float32 schedule reference. The state gate checks those two tensors; run
`mamba-grad-m2-l257` to check the public parameter/input gradients for the same
multi-chunk fixture. Both strict policies can also be passed together to
`tools/mamba_gradient_oracle.py --compare`.

At L257, `conv1d.bias` and `block_norm.weight` use named compositional
policies: each per-token operand must match both the independent float64
reference and the float32 reference at the default tolerances, then the
public gradient must match the pinned reduction of the native operands bit
for bit. Bias uses the serial batch/token sum; block norm uses GEMM-v1's
128-cell leaves and balanced fold. Direct float64 public-gradient errors
remain reported. This separates accumulated trajectory rounding from an
incorrect reduction without widening the numerical tolerances.

Strict-policy sabotage tests cover missing tensors, altered provenance,
nonfinite values, and corrupted public gradients with both policies enabled:

```sh
pixi run -e skgpu python tools/test_mamba_gradient_oracle.py
```

This does not claim token-at-a-time decode backward: the convolution-window
cotangent and mutable cache update contract remain undefined.

Backward certificates retain native public-gradient bytes for all three short
prefill fixtures and the Mamba 2 L257 public/state gates (five result rows):

```sh
pixi run -e skgpu mamba-backward-cert-apple
# Use mamba-backward-cert-nvidia or mamba-backward-cert-amd on those devices.
python tools/mamba_backward_identity.py compare CERT_A CERT_B
```

Schema v2 stores raw public gradients under each case's `native/` directory
and fingerprints the source, rejecting source changes during a run. The
comparison requires every semantic gate to be GREEN before comparing native
bytes and provenance. Cross-device identity is established only when actual
captures from the named hardware agree.

At source commit `718495cd`, Apple M4 and NVIDIA RTX 4090 (driver
580.159.04) passed all five semantic gates and matched all 54 captured native
gradient tensors bit for bit. The fixtures are Mamba1 `base_b2_l4_d8`,
Mamba2 `m2_base_b2_l4_d32`, Mamba3 `m3_base_b2_l4_d32`, and the Mamba2
`m2_base_b1_l257_d64` public/state gates. See the
[comparison record](../bench/results/resume/2026-09-05/cross-device.json)
and its linked certificates. This evidence is for these source fixtures,
not arbitrary shapes or installed wheels.

AMD backward identity remains unmeasured for this certificate: the available
ROCm 6.1 image was below the compiler's minimum, and the corrected ROCm 6.4
startup request could not acquire an instance. These attempts are recorded
as infrastructure failures, not numerical disagreements.
