# afn-optim: optimizer and loss kernels on Apple FAST (lane notes, 2026-10-03)

Branch lane/apple-fast-neural-optim. Code: training/afn_optim.mojo (new, every kernel and launcher),
dispatch hooks in training/checks/optimizer.mojo (`identical_optimizer_step`) and training/estimator.mojo
(`identical_optimizer_step_resident_host`, `identical_ce_loss_resident`). Every hook sits under a
`comptime if` on an `AFN_*` alias that is False unless `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()` and the define is named, so IDENTICAL (and FAST on NVIDIA/AMD) compiles
main's code unchanged.

## Profile of main's path (per call, read from the code)

Optimizer step (`identical_optimizer_step`, reached by mlp-train-step directly, samba-train-step through
T.AdamW -> the resident host entry, lm-train-step through byte_lm.mojo):
- refusal scan `opt_refuse_device_inputs`: 4 `nonfinite_partial_kernel` launches (param, grad, m, v),
  1 host readback of the partials, 1 wait.
- clip (only when max_norm > 0; none of the three board lanes clip): per-tensor squared-norm GEMVs, a sqrt
  launch, a finish launch, a scale launch, host reads of the norm; five waits.
- update: Adam/AdamW is one launch over the flat model; SGD is one launch per tensor (per-tensor momentum
  flag). Then a wait.
- resident host entry: eight small `enqueue_create_buffer` calls per step (denom_out, q_out, sumsq, norms,
  total_cell, out2, ws, sab_partials), each a fresh Metal buffer (~20 us + live-buffer growth).
Cross-entropy (`identical_ce_loss_resident`, mlp and samba; byte_lm calls ce_forward_into/backward_into
itself and is not reached): ~12 launches including 2 vendor matmuls against a host-built ones vector,
5 waits, ~16 scratch buffers per call.

## Candidates (default OFF, one define each)

| define | mechanism | launches/waits after |
|---|---|---|
| MOJOLEARN_AFN_OPT_FUSE_SCAN | 4 scans -> 1 grid-stride scan, 1 fold kernel writes a device gate, the update reads the gate (no partial write on refusal); host reads 32 B of cells after the one wait and raises the oracle's message | scan + fold + update, 1 wait |
| MOJOLEARN_AFN_OPT_CLIP_FUSE | squared-norm block partials (in the scan launch when FUSE_SCAN), fold computes total_norm and coef on device, update scales grad on load and writes the clipped grad back | clip costs 1-2 launches, 0 extra waits |
| MOJOLEARN_AFN_OPT_MULTITENSOR | SGD: one launch over the flat model, a device table of offsets + momentum flags, thread finds its tensor by binary search | SGD j launches -> 1 |
| MOJOLEARN_AFN_OPT_VEC4 | Adam/AdamW 4 consecutive elements per thread, 4-wide loads/stores, scalar tail | same launches, wider memory ops |
| MOJOLEARN_AFN_OPT_RESIDENT_STATE | process-wide pool (`_Global`) of scratch buffers and a pinned host mirror, created once per shape; the resident host entry's eight buffers become sub-buffer views; step scalars stay kernel args | 0 allocations per step |
| MOJOLEARN_AFN_LOSS_FUSED | CE forward+backward in one launch, one block per row (max, sum-exp, smoothing, loss, dlogits, non-finite + target-range flags); a one-block fold writes the scalar loss and gate | 2 launches, 1 wait, no matmuls |
| MOJOLEARN_AFN_OPTIM_ALL | all of the above | |

Brief's candidate 5 (device-side lr schedule) needs nothing: the step scalars (`device_step_scalars(cfg, t)`)
are already kernel arguments, so there is no per-step upload to remove.

Any OPT_* define routes `identical_optimizer_step` into `afn_optimizer_step` (AFN_OPT_ANY), which always
waits exactly once; with only VEC4/MULTITENSOR/RESIDENT_STATE named, main's refusal scan (and main's clip
when clipping) still run first. Sabotage and OPT_RECORD_INTERMEDIATES builds keep main's path.

## Quality

f32 everywhere; fold orders free (norm partials, loss fold); no approximate exp/log/sqrt; refusals keep
the oracle's messages; a refused step writes nothing (gate checked in the update kernel).

## Builds (compile only, M4 laptop, through compile_slot.sh, `-j 1`)

Logs in ~/mojolearn-evidence/afn-optim/build-*.log (not committed). The training build script exports
MOJOLEARN_SKIP_BUILD_GATE=1 itself, so no smoke runs. Results are listed in the lane's final reply and in
the table below.

BUILD_RESULTS_PLACEHOLDER

byte_lm (lm-train-step) is not compiled here: bindings/build_byte_lm.sh refuses FAST until the afn-lm lane
lands; its request lines are written anyway (binding byte_lm, since byte_lm.mojo is what calls
`identical_optimizer_step` for that lane). LOSS_FUSED has no lm line: byte_lm does not call
`identical_ce_loss_resident`.
