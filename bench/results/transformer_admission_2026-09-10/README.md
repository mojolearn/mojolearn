# Transformer numerical-admission investigation

This investigation records numerical comparisons only, with no opponent timings.
The original benchmark tolerance (rtol=0.0005, atol=0.00001) is unchanged.

Apple production RoPE probe exports HD64/HD128 inverse-frequency and sine/cosine
bits at eight positions. The original NumPy FP32 power expression disagrees
with the production portable exp/log power expression in 35/64 HD128 inverse
frequencies, maximum absolute difference 5.96046448e-8. At position 4095,
cosine maximum error is 2.43149698e-4. Aligning only the inverse-frequency
bits reduces that to 5.96046448e-8. This localizes the table discrepancy to
constant generation rather than primarily to the trig approximation.

It does not yet establish that constant alignment fixes full-block admission.
The separate-process original-grid runner checks both original Torch FP32 and
Torch FP32 using production inverse bits. It also reports position-zero error,
where RoPE cannot cause a difference. This avoids mistaking accumulation error
or ill-conditioned outputs for a constant-generation defect.

Runnable sources:
- transformer/checks/transformer_rope_admission_probe.mojo (IDENTICAL build)
- tools/transformer_rope_admission_check.py <native-probe.log>
- tools/transformer_admission_diagnose.py --spec <original-grid-speed_torch_seq.py>
  --shape narrow|wide --arm ours|reference --output <ours.npy>
  [--rope-log <native-probe.log>]

Use separate processes for ours and reference to keep MAX and Torch runtimes
apart. The reference runner uses the original generator, seed, shapes and
weight ranges and records every tensor SHA256. It has no timing loop and
changes no production arithmetic. No comparator patch has been adopted.

## Initial full-output H100 result

Both reconstructed own outputs exactly match the stored baseline SHA256.
Original H100 discrepancies reproduce the September7 raw log exactly.

| Shape | Original max abs | Shared inverse max abs | Original outside gate | Shared inverse outside gate |
|---|---:|---:|---:|---:|
| narrow B8/L4096/D512 |0.0138196945|0.0004343987|1,280,511|46,991|
| wide B8/L1024/D2048 |0.0285353661|0.0043530464|934,854|333,850|

Each output has16,777,216 float32 values. Both shared-inverse comparisons
still fail the original tolerance. Position-zero differences are unchanged
(narrow maximum1.0967e-5, wide1.0872e-4), so some error is independent of RoPE.
The next diagnostic shares complete production sine/cosine tables and compares
both implementations with float64 evaluation using those same FP32 constants.
No default comparator or production arithmetic has been changed.

## Complete-table and float64 isolation

Final numerical-only run: NVIDIA H10080GB HBM3, driver580.126.09,
Torch2.4.1+cu124, CUDA12.4, TF32 disabled, float32 matmul precision highest,
deterministic algorithms disabled. No warmup/timing loop or opponent price.

| Comparison | Narrow max abs | Narrow outside gate | Wide max abs | Wide outside gate |
|---|---:|---:|---:|---:|
| Our FP32 vs Torch FP32, exact shared production tables |0.000406504|46,826|0.004357815|334,709|
| Our FP32 vs FP64, same production FP32 tables |0.000386773|31,165|0.003286085|209,595|
| Torch FP32 vs FP64, same production FP32 tables |0.000266671|29,988|0.003413000|252,164|

Every row fails the unchanged5e-4/1e-5 criterion. Against matched FP64, narrow
RMS absolute errors are1.293e-5 for ours and1.263e-5 for Torch; wide errors
are1.302e-4 and1.486e-4 respectively. The remaining discrepancy therefore is
not a leftover trig-table mismatch. Its magnitude is consistent with FP32
accumulation/nonlinear rounding and fixture conditioning; every individual
residual has not been stage-localized. In particular, Torch FP32 itself does
not satisfy this gate against higher-precision evaluation of the same inputs.

Conclusion: the original comparator has a documented pinned-constant mismatch,
but removing it does not admit the comparison. Production arithmetic and the
default comparator remain unchanged. The explicit shared-inverse/shared-table
options stay in the numerical diagnostic only. No tolerance was loosened and
no qualified performance ratio or new opponent timing row is justified.

`h100/rope_full.log.gz` contains the native HD128 production table for
positions0..4095. Decompress it before passing `--full-rope-log`; table/export
SHA256 values and NumPy output-container hashes are in`h100/evidence.sha256`.
The raw float32 output hashes matching the stored baseline are separately in
`h100/*.ours.log`. The final script verifies these raw hashes before writing
the NumPy containers. Original benchmark inputs/weights are regenerated from
the archived original spec source hash and every tensor SHA is logged.

Reproduction (from a built IDENTICAL checkout with the original companion spec):

```sh
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python tools/transformer_admission_diagnose.py --spec /path/to/original_speed_torch_seq.py --shape narrow --arm ours --output /tmp/narrow.npy
MOJOLEARN_TRANSFORMER_ROPE_FULL=1 /path/to/rope_probe > /tmp/rope_full.log
python tools/transformer_admission_diagnose.py --spec /path/to/original_speed_torch_seq.py --shape narrow --arm reference --output /tmp/narrow.npy --rope-log /path/to/rope.log --full-rope-log /tmp/rope_full.log --reference64
```

Repeat with`--shape wide`. Compile the probe in IDENTICAL mode with the checkout
on the Mojo include path. The original spec is the unchanged companion
`mojolearn-grid/tools/speed_torch_seq.py`; main's older benchmark module does
not expose its public-grid shape/generator helpers. Numerical GPU work is
complete and the shared pod was returned to root/kNN; root owns its cleanup.
