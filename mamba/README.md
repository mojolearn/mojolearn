# Mamba

This directory implements reference-pinned Mamba 1, Mamba 2, and Mamba 3
blocks. They are low-level forward blocks with caller-owned continuation
state, not trainable estimator or complete language-model APIs.

## Current surface

| family | forward | decode/continuation | Python binding | backward |
|---|---|---|---|---|
| Mamba 1 | implemented | implemented | implemented | kernels and host oracle exist; no whole-pass composer or public API |
| Mamba 2 | implemented | implemented | implemented | routing/workspace scaffold only |
| Mamba 3 | implemented | implemented | implemented | routing/workspace scaffold only |

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
- [Historical parity ledger](../archive/evidence/mamba/FEATURE_PARITY.md)
- [Historical backward plans](../archive/plans/mamba/)

The contracts define the profile. Historical plans and ledgers explain how it
was reached but are not the current status source.

## Backward implementation order

1. Compose the existing Mamba 1 backward kernels into one end-to-end pass.
2. Add an independent whole-pass oracle, non-finite refusal, and the MB1–MB10
   separating gates.
3. Add a native binding and Python API only after those gates close.
4. Implement Mamba 2 backward from an independent oracle.
5. Implement Mamba 3 backward last.

Do not infer Mamba 2/3 backward capability from their compile probes. Those
probes currently validate routing and workspace topology, not backward
arithmetic.

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

Once a whole-pass Mojo backward runner emits matching files, compare it with:

```sh
python tools/mamba_gradient_oracle.py \
  --compare /tmp/mamba1-grad /tmp/mamba1-mojo-grad
```

Mamba 2 currently has an explicitly partial device tail covering the residual
output projection, gated RMSNorm, the SiLU gate, and the D-skip join. It emits
`d_gnorm`, `d_gate`, `d_skip`, `d_scan`, the z-slice gradient, `dD`, the
D-path partial x gradient, and both tail weight gradients. It names every
remaining seam:

```sh
pixi run -e skgpu mamba-grad-m2
pixi run dump-mamba2-backward-tail
pixi run -e skgpu python tools/mamba_gradient_oracle.py --allow-partial \
  --compare /tmp/mojolearn-mamba2-grad /tmp/mojolearn-mamba2-mojo-grad
```

Passing this comparison certifies only the output projection, gated RMSNorm,
SiLU gate, and D-skip derivatives, not the SSD recurrence or a whole Mamba 2
backward pass.

The first pinned SSD-backward kernel now exists separately in
`impl/mamba_ssm/ops/mamba2_ssd_backward.mojo`: it implements the descending
inter-chunk state recurrence and emits chunk-increment, incoming-state,
initial-state, and per-cell scale-product gradients. The scale products are
now reduced by GEMM-v1 over the exact p-major `P*N=8192` layout, then chained
through `exp(dacs[last])` into a sparse per-chunk `d_dacs` contribution. It is
compile-probed but not yet connected to S18, which must produce its direct
incoming-state gradient before the recurrence can join the partial composer.
