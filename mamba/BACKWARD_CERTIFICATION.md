# Native backward certification

Run the baseline and long profiles separately on an identified GPU:

```bash
MOJOLEARN_MAMBA_CERT_VENDOR=amd pixi run bash tools/mamba_backward_certify.sh
MOJOLEARN_MAMBA_CERT_VENDOR=amd MOJOLEARN_MAMBA_CERT_PROFILE=long-sequence-v1 \
  pixi run bash tools/mamba_backward_certify.sh
```

Use `nvidia` for the NVIDIA leg. The task commands compile native runners
with `-D MOJOLEARN_NUMERIC_IDENTICAL=1`. The comparator requires matching
frozen source, profile, case inventory and raw native tensor bytes; a passing
numerical gate on one GPU is not a cross-vendor certificate.

## Mamba3 long arithmetic contract

The `m3_base_b1_l65_d64` gate retains the independent whole-forward float64
autograd check on **every public gradient**, at `rtol=1e-5, atol=1e-6`.
The generator audits selected parameter/input cells by central finite
differences as well. The public inventory is ten leaves; the native
diagnostic inventory is another 76 tensors. Missing named diagnostics fail.

Thirteen intermediate reductions, copies and joins use the explicit
`mamba3.l65.compositional_operands_plus_exact_dag.v1` policy:

1. Check external backward operands against both independent float64 and
   staged float32 references at the unchanged tolerances.
2. Check the forward normalized B/C values, biases, dt, sigma and angle rate
   independently against the float64 forward. The rate capture reuses the
   exact scalar rate helper used by the native forward kernel.
3. Reconstruct the angle recurrence by bits: zero-state prefill, float32
   FTZ arithmetic, and modulo the pinned float32 `2pi` after **every token**.
   Modulo at chunk/end boundaries is a different arithmetic profile and is
   rejected by the recurrence check and its negative control.
4. Evaluate rotary K independently in float64 at the checked native
   normalized-B, bias and angle inputs. Compare that local forward operation
   at the unchanged tolerances.
5. Reconstruct all thirteen backward outputs from the external operands.
   S14/S15 dots use correctly rounded float32 FMA with the prescribed serial
   order. Subsequent sums, scale copies and shifted-beta joins consume the
   reconstructed predecessors. Every output must match its native dump by
   bits; a consistent downstream error cannot conceal an upstream error.

This is a compositional arithmetic certificate. It does not assert that
every intermediate lies within the direct whole-float64 or PyTorch-float32
comparison threshold. Those direct differences remain printed in the logs.
The earlier 13 direct misses arose along reductions and an accumulated
wrapped-angle path; the contract now distinguishes independently checked
operands from the exact floating-point operations on those operands. It
does not enlarge tolerance constants or remove those thirteen outputs.

`tools/test_mamba3_backward_arithmetic.py` tests one-bit sabotage of every
output, incorrect inputs with self-consistent downstream outputs, a matching
float32 reference that disagrees with float64, rotation/angle corruption,
missing operands, wrong mode, batch boundaries and changed mod placement.
The general oracle policy tests independently prevent a matching staged
backward from hiding a wrong public gradient.

Successful L65 certificates retain `diagnostic-actual` and
`diagnostic-oracle`, including all 86 gradient tensors and nine exact forward
operands. Failed attempts retain `failed-actual` and `failed-oracle`.
`tools/mamba3_join_diagnostics.py` compares those retained bytes across
vendors without re-executing a model. These native prefill/state fixtures do
not imply installed Python backward, decode/cache backward, arbitrary shapes
or corrected Apple evidence beyond the hardware actually rerun.
