# afn-optim A/B requests (docs/apple-fast/ab-neural/optim.txt)

Arm A is FAST with no define, arm B names one define. Binding `training` for mlp-train-step and
samba-train-step, `byte_lm` for lm-train-step (byte_lm.mojo calls `identical_optimizer_step`; those lines
need the afn-lm lane's FAST byte LM build). None of the three board lanes clips gradients and all three use
AdamW, so CLIP_FUSE and MULTITENSOR do not exercise their own mechanism there: their lines measure only the
dispatch into `afn_optimizer_step` (main's scan, the AFN Adam kernel, one wait). Kept so the manager sees
that cost; expect ratio ~1.0.

**MOJOLEARN_AFN_OPT_FUSE_SCAN** (training/afn_optim.mojo, scan/fold kernels and `afn_optimizer_step`).
Main's four non-finite scan launches, a host readback and a wait become one scan launch, one fold launch
that writes a device gate, and the update launch that reads the gate; one wait per step, then a 32-byte
read of the gate cells. Expected: fewer launches and one fewer round trip per step (~0.2-0.4 ms on Apple),
largest share on mlp-train-step where the step is tiny. Risk: a refused step must still raise the same
message; the update is withheld on device. Lanes: mlp, samba, lm train steps.

**MOJOLEARN_AFN_OPT_CLIP_FUSE**. Global-norm clip folded into the scan/fold launches; the coefficient is
applied as the update loads the gradient. Removes main's GEMVs, sqrt/finish/scale launches and five
waits, but only when max_norm > 0. Board lanes do not clip: expect ~1.0 (dispatch cost only). Risk: free
fold order of the norm (f32 reassociation only).

**MOJOLEARN_AFN_OPT_MULTITENSOR**. SGD's one-launch-per-tensor becomes one launch over a device table of
offsets and momentum flags. Board lanes use AdamW (already one launch in main): expect ~1.0 (dispatch
only). Risk: none beyond the table upload (one small copy per step).

**MOJOLEARN_AFN_OPT_VEC4**. Adam/AdamW updates four consecutive elements per thread with 4-wide
loads/stores, same per-element arithmetic, scalar tail. Expected: a modest kernel-time gain at the LM and
Samba parameter counts (memory-bound update), nothing on mlp. Risk: alignment of the flat buffers (offsets
are element offsets into one allocation; the vector path handles the aligned body and the scalar tail).

**MOJOLEARN_AFN_OPT_RESIDENT_STATE**. The step's scratch (scan partials, gate cells, pinned host mirror,
SGD table, the resident host entry's eight small buffers, the loss's row/flag scratch) lives in a
process-wide pool created once per shape; a step allocates no Metal buffer. Expected: removes ~8-14
buffer creations per step (each ~20 us plus live-buffer growth); visible on samba-train-step (resident
host entry) and mlp. Risk: the pool is per process and resized when the shape changes; not thread-safe
across concurrent trainers in one process.

**MOJOLEARN_AFN_LOSS_FUSED** (`afn_ce_loss_resident`). Cross-entropy forward and backward in one launch,
one block per row, plus a one-block fold of the scalar loss and the refusal flags: two launches and one
wait instead of ~12 launches, two vendor matmuls and five waits. Expected: the largest win for
mlp-train-step and samba-train-step (loss is a fixed per-step overhead). Risk: label smoothing and the
target-range refusal are in the fused kernel; the quality judge checks loss and grads against IDENTICAL
and torch. Not on lm-train-step (byte_lm uses its own CE calls).

**MOJOLEARN_AFN_OPTIM_ALL**. Every candidate above together; the composition the manager would keep if
the singles each win.
