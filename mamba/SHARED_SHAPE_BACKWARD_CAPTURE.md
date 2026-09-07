# Shared B2/L8/D32 backward capture: authored, not executed

`tools/mamba23_shared_shape_backward.py` is a new bounded qualification harness. Its author ran no tests, builds, model code, measurements or provisioning. Root executes it only on the authorized remote NVIDIA/AMD device with explicit opt-in and the corresponding serial guard. No Apple or CPU model execution is admitted.

The shared shape is batch 2, sequence length 8, model width 32. Each family has nine public constructor weight tensors and one input-gradient tensor: all 20 leaves across Mamba2/Mamba3 are checked and retained. The family-specific weight layouts remain distinct: Mamba2 projection `[385,32]`, convolution `[320,1,4]`; Mamba3 projection `[419,32]`, B/C norms `[128]` and biases `[1,128]`. Existing nine weight files are read from each family's `m{2,3}_base_b2_l4_d32` directory because constructor weights are independent of sequence length. Actual L8 inputs and arbitrary cotangents are newly specified exact dyadic arrays, fully retained with hashes. No old L4 input/output/reference result is relabeled as an L8 result, and no full corpus generation is required.

The independent reference follows the existing `mamba/corpus/gen_corpus.py` PyTorch forward transcriptions (`m2_forward`, `m3_forward`) in FP64 on CUDA/HIP, with FP64 autograd for `sum(residual.out*dy)`. All reference-created defaults are scoped to the accelerator. Every public gradient cell must meet the unchanged existing gate `atol=1e-6, rtol=1e-5`. Separately, two public IDENTICAL backward calls must match byte for byte; a zero cotangent must produce numerically zero gradients; and negated captured gradients must fail the same reference gate. A zero-gradient check does not assert a particular signed-zero bit.

The harness retains raw constructor weights, L8 input, cotangent, native gradients, repeated gradients, zero-cotangent gradients and FP64 references. It also records exact tensor names/shapes, loaded native mode/vendor/binary hash, reference/wrapper source hashes, objective, thresholds and all per-leaf verdicts. Output directories must be new. Numerical jobs are sequential, CPU library/thread settings are limited to two, and the external graph/cache is released before Mojo allocates its state.

Root execution command on NVIDIA, from repository root with the current public package/extension selected:

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_MAMBA23_SHARED_SHAPE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 tools/nvidia_serial_guard.py --seconds 600 --rss-gib 12 -- \
  python3 tools/mamba23_shared_shape_backward.py --expected-vendor cuda \
  --output /artifacts/mamba23-b2-l8-d32-new
```

On AMD substitute `tools/amd_serial_guard.py`, `--expected-vendor hip`, and a fresh output path. Dependencies are NumPy, accelerator-enabled PyTorch, einops, the selected Mamba extension, the repository reference source, and the 18 constructor fixture files. `--corpus` can point at another retained copy of those fixture files; their exact hashes are recorded.

A successful summary establishes only this new fixture's external tolerance agreement, same-device repeat bits, and zero-state public VJP scope once root retains the complete successful guard exit/log/command evidence. The harness does not certify cross-vendor bytes itself: root must separately compare retained raw inputs and all corresponding native gradient arrays with consistent provenance. Recurrent-state cotangents, decode/stateful backward, other shapes, FAST backward, training convergence and performance are outside this fixture.
