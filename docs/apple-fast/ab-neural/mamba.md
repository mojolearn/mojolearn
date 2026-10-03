# afn-mamba A/B requests (docs/apple-fast/ab-neural/mamba.txt)

Every line is arm A = FAST with no afn define, arm B = FAST with the one
define, binding mamba, shape full (batch 1, length 2048, d_model 384). Profile
and code locations: docs/apple-fast/notes/neural-mamba.md.

**MOJOLEARN_AFN_MAMBA1_CHUNKSCAN** (mamba1-forward). The Mamba-1 selective
scan runs one simdgroup of 32 lanes per (batch, channel) with the sequence cut
into 32 chunks: local chunk states, a threadgroup carry fold, and a re-walk
from the carried start. Expected: the scan stops being 768 serial threads over
2048 steps, the largest single win on mamba1-forward. Risk: fold order across
chunk boundaries changes (f32 reassociation, the quality judge must stay
equal); exp(dt A) is computed twice per element.

**MOJOLEARN_AFN_MAMBA1_FUSE_IN** (mamba1-forward). Conv1d + SiLU + window
update in one token-parallel launch, x_proj split + A = -exp(A_log) in one
launch, waits removed. Expected: a few launches and waits fewer and a conv
that uses the whole GPU. Bits equal FAST-off. Risk: small; window update for
l < D_CONV is a special case.

**MOJOLEARN_AFN_MAMBA2_SSD_MMA** (mamba2-forward). The three chunked-SSD
matmuls (C.B^T, (G o L).X_d, X_d^T.(B o decay)) on simdgroup 8x8 f32 tiles,
one launch each, causal tiles skipped. Expected: the SSD matmul stages drop
by several times. Risk: fold order per cell changes (f32 only); applies only
when the chunk is a multiple of 64.

**MOJOLEARN_AFN_MAMBA3_SISO_FUSED** (mamba3-forward). Nine small elementwise
launches around the SISO core become four, per-stage waits dropped. Expected:
launch-bound time on the M3 falls. Bits equal FAST-off. Risk: low; the
serial angle chain is unchanged.

**MOJOLEARN_AFN_MAMBA_ARENA** (mamba1/2/3-forward). Every per-call buffer of a
block (weights, state, stages, x) is a view of one arena, one fill, uploads
with no per-buffer wait, no per-stage waits. Expected: each launch stops
paying for 30 to 45 live buffers and dozens of waits go away. Bits equal
FAST-off. Risk: view lifetime (the arena is kept alive explicitly).

**MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL** (mamba1/2/3-forward). The non-finite
input refusal as device reductions with one readback and one wait per call
instead of a host download (Mamba-1/2, about 7 MB) or one readback per name
(Mamba-3). Expected: removes 13 round trips per Mamba-1/2 call. Bits equal
FAST-off; the refusal text is unchanged.

**MOJOLEARN_AFN_MAMBA_ALL** (mamba1/2/3-forward). All six together; the
number the M3 manager would merge if the singles each hold quality.
