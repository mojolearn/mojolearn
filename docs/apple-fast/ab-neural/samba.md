# afn-samba A/B requests (docs/apple-fast/ab-neural/samba.txt)

Every arm A is FAST with no define; arm B is FAST with one define. Board lanes samba-forward and
samba-train-step at shape full; judge with `tools/neural_fast_quality.py samba`. All defines are Apple FAST
only; IDENTICAL and the other vendors compile main's code. Profile and mechanism detail:
docs/apple-fast/notes/neural-samba.md.

**MOJOLEARN_AFN_SAMBA_FUSE (training).** Three fused binding entries replace per-op calls: the final
RMSNorm plus the head GEMM (forward), the whole train tail (norm forward, head GEMM, CE with gradient, both
head backward GEMMs, norm backward reusing the forward's row sums), and the tied embedding gradient
(embedding backward plus the pair add). Saves 4 binding calls per step (4 waits, the upload/readback of
hn, dhn and the logits) and 1 per forward. Expected: a few ms per step from waits and bus traffic; the
forward gains less. Risk: low; same kernels and operands in the same order, so FAST bits should match
the per-op FAST path. Touches samba-forward and samba-train-step.

**MOJOLEARN_AFN_SAMBA_ARENA (training).** Per-op operands and scratch are views of one arena chunk, opened
at the op's start and released after its wait, instead of fresh Metal buffers. Expected: lower per-launch
host cost and no allocation churn per op; small, since each op holds few buffers. Risk: the first call
allocates a 256 MB chunk that stays in the process pool. Touches both lanes.

**MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT (training).** The non-finite input refusals run as a device flag kernel
per operand and are read back with the op's own wait, instead of host loops over every operand (and the
head-loss logits readback, M*V floats, only for that scan). Expected: removes the host scans (the
embedding weight, activations, the logits) from every op; visible on the train step. Risk: a refusal now
raises after the op's kernels ran (outputs are not returned); messages keep their names. Touches both
lanes.

**MOJOLEARN_AFN_SAMBA_EMB_ATOMIC (training).** Embedding backward by f32 atomic scatter (free order)
instead of the run-sorted identical fold; in the tied entry (with FUSE) the head gradient is the scatter's
seed, so the pair add disappears. Expected: fewer launches and no int32 sort scratch; small at vocab 256.
Risk: the gradient's fold order is free (f32 reassociation only); vocab 256 with 1024 tokens means many
collisions per row, so atomic contention could cost more than it saves. Touches samba-train-step.

**MOJOLEARN_AFN_MAMBA3_BWD_CHUNK (mamba).** The Mamba-3 backward angle stage, twice per layer backward:
the theta reverse chain (one thread walking 512 tokens per chain) becomes two launches over 64-token
chunks, and the per-token suffix fold (L^2/2 dependent adds per angle) becomes a segment scan, O(L).
Expected: the largest kernel-side win in this lane on the train step. Risk: fold order changes (exact f32
adds, reassociated); judge quality against IDENTICAL and torch. Touches samba-train-step (and
mamba3 training generally).

**MOJOLEARN_AFN_MAMBA3_BWD_ARENA (mamba).** The Mamba-3 backward's ~75 scratch buffers per pass become
zero-filled views of one arena the pass opens and releases after its wait; the gradient outputs stay real
buffers. Expected: lower per-launch cost across the pass's ~38 launches (fewer live buffers) and no
allocation churn. Risk: as ARENA. Touches samba-train-step.

**MOJOLEARN_AFN_SAMBA_ALL.** Every define above. It is requested twice, once per binding (training and
mamba), because afn_ab.sh builds one binding per line; both A/Bs together give the combined effect.

Note: ARENA, DEVICE_ADMIT and EMB_ATOMIC reroute the training binding's generic `embedding_forward`,
`embedding_backward`, `rms_norm_forward`, `rms_norm_backward` and `samba_head_loss` entries
(training/samba_ops.mojo), so any other model calling those entries in an Apple FAST build with the define
takes the same path.
