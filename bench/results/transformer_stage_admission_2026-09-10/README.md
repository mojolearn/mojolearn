# Original large Transformer FP32 admission attribution

Numerical diagnosis on NVIDIA H100 80GB HBM3, driver 580.126.09,
Torch 2.4.1+cu124, NumPy 1.26.3. Full original seed-7 shapes are
B8/L4096/D512 and B8/L1024/D2048, each with 16,777,216 final outputs.
Both use the retained production RoPE tables, promoted unchanged for FP64.
TF32 is disabled. No opponent timing was collected and no gate, tolerance,
production arithmetic or qualified performance ratio changed.

The final Torch FP32-versus-FP64 statistics exactly reproduce the earlier
matched-constants diagnostic: 29,988 narrow and 252,164 wide violations of
`atol=1e-5, rtol=5e-4`. The added comparisons separate local rounding from
upstream error by passing exactly the FP32 stage input words into the
original comparator's FP64 methods. These are diagnostics of the Torch
comparator, not an oracle for mojolearn's prescribed FP32 reduction order.

| Comparison (outside tolerance) | Narrow | Wide |
|---|---:|---:|
| norm1 cumulative | 0 | 0 |
| Q projection + RoPE, same norm1 input | 2 | 1,590 |
| K projection + RoPE, same norm1 input | 0 | 390 |
| V projection, same norm1 input | 0 | 220 |
| attention core, same Q/K/V words | 23,209 | 24,535 |
| propagation of Q/K/V error through FP64 attention | 27,288 | 63,627 |
| o_proj, same attention context words | 0 | 0 |
| norm2, same residual words | 0 | 0 |
| MLP, same norm2 words | 0 | 4,848 |
| propagation through FP64 MLP | 28,599 | 259,131 |
| each residual addition, same operands | 0 | 0 |

The remaining attention discrepancy has two sources: rounding within the
attention core, and amplification of Q/K/V projection/RoPE discrepancies.
For wide, the latter has RMS error 2.1821e-4 versus 8.8463e-5 for local
attention rounding. Narrow has comparable RMS errors, 2.9273e-5 and
2.5740e-5. Counts across stages are not additive; widths and tolerance
reference values differ. The o_proj, normalization and residual glue are
not the numerical bottleneck under this diagnostic. Wide has a smaller
additional local MLP discrepancy. Distinguishing Q/K/V GEMM rounding from
RoPE arithmetic and isolating attention logits/softmax remains future work.
This evidence does not establish an implementation bug or justify changing
IDENTICAL arithmetic to follow Torch.

Reproduction (CUDA host; original comparator is external, hash in manifest):

```sh
gzip -dc bench/results/transformer_admission_2026-09-10/h100/rope_full.log.gz > /tmp/transformer-rope-full.log
for shape in narrow wide; do
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 tools/transformer_admission_diagnose.py \
    --spec /path/to/mojolearn-grid/tools/speed_torch_seq.py \
    --shape "$shape" --arm reference --reference-only --reference64 \
    --stage-errors --full-rope-log /tmp/transformer-rope-full.log \
    > "$shape.stage.log" 2>&1
done
```

`--reference-only` avoids requiring a native Mojo rebuild or an own-output
NPY. It cannot report own-versus-opponent admission. The original full-output
comparison remains available with `--output` and without that flag. Each
report now retains the worst tolerance witness separately from the largest
absolute-error witness; the two need not be the same output element.

Validation: both complete CUDA runs exited zero and emitted 20 stage rows
plus the final Torch-versus-FP64 row. All shared final numerical statistics
match the prior retained H100 diagnostics. Six host report/CLI tests pass;
Python compilation and diff whitespace checks pass. No small run was used
to promote a gate. The GPU was released to the Mamba lane after these runs.
