# Generalized LM training capacity and performance — September 10, 2026

The shape generalization and device-to-device inter-layer gradients are on main
(72f79a18, 28699cc7). They establish functionality on the recorded small checks,
not measured large-model throughput. Owned resident Python/native sessions are now implemented; see
[the session handoff](HANDOFF_lm_session_2026-09-10.md). Full host-state
traffic remains. This capacity report adds no kernel speed or opponent measurement.

## Current relative performance

| Area | Last usable comparison | Limitation |
|---|---|---|
| H100 dense GEMM, Llama t512 | 4.07–4.52x cached FP32 cuBLAS; 9.73–11.28 TFLOP/s | Component shapes, not a training step |
| L40S attention forward / forward+backward | 3.91x / 4.32x cached FP32 SDPA | Recorded HD64 fixture; earlier physical pod for opponent |
| NVIDIA kNN, 400k/4k/d32 k10/k15 | 2.61x / 2.88x cached cuML | Complete requests, scoped default |
| Apple kNN | 15.7–25.0% less own request time | Internal improvement, not opponent ratio |
| Mamba3 | No newly qualified opponent ratio | Historical slow regime did not recur in 192 admitted calls |
| Transformer | Opponent ratio unqualified | Large numerical admission still fails |

Sources and exact scope: PERFORMANCE_STATUS_2026-09-10.md and
bench/OPPONENT_REFERENCE.md. No cached opponent rerun; no new opponent row.
kNN and Mamba improvements do not directly accelerate the present decoder LM.

## Allocation blocker identified at 6154aeac (now addressed for fused execution)

At 6154aeac, ByteTrainer allocated LlamaDeviceStages and LlamaBackwardStages
for every layer without their existing `lean=True` argument. The follow-up
HANDOFF_lm_lean_attention_2026-09-10.md wires lazy allocation and records checks. Forward owns four full
`B*H*L*L` FP32 arrays (scores, masked, aexp, weights); backward owns another
four (d_attn_weights, d_attn_masked, d_attn_scores, d_qk_cell). Fused attention
was already integrated at that revision.

ByteBuffers separately allocates five `B*L*V` FP32 arrays: logits, ce_shift,
ce_expo, ce_weights, ce_dlogits. Keeping parameters on device does not remove
these. Disabling trace alone does not change constructor allocations.

For 12 layers, width 768, 12 query/KV heads, head dimension 64, intermediate
2048 and vocabulary 50257, the actual untied model has **162,147,840**
parameters. It is an RMSNorm/RoPE/SwiGLU decoder without final norm, not the
exact 125M GPT-3 Small architecture. The old description that everything else
was plumbing underestimated the remaining memory and numerical work.

| Microbatch | Attention matrices | Five loss/logit matrices | These plus param/grad/Adam m/v |
|---|---|---|---|
| B8, L2048 | 144 GiB | 15.34 GiB | 161.75 GiB |
| B1, L2048 | 18 GiB | 1.92 GiB | 22.33 GiB |
| B1, L1024 | 4.5 GiB | 0.96 GiB | 7.87 GiB |

These are disjoint allocation **subtotals**, not measured peaks or fit
admission. They omit linear activations, duplicate weights/gradients,
workspaces, host snapshots and runtime overhead. B8/L2048 cannot fit an
80-GB device with those materialized allocations. The smaller rows are candidates
for measurement, not proven fits. Do not reduce context/batch and present
that as an improvement on the original workload.

Reproduce without loading a GPU backend:

```sh
python3 tools/lm_training_capacity.py --shape 8 2048 768 12 12 64 2048 12 50257 --materialized-attention
```

Reports for all three rows, with source hashes, live in
bench/results/lm_capacity_2026-09-10/. Verified parameter count, the 144-GiB
attention calculation, invalid GQA rejection, and no GPU-package import.
This is a source-backed allocation inventory; it must be updated when those
constructors change. Corrected ByteBuffers' misleading 'weights are views'
description: the current code allocates and copies them.

## Time estimates and next work

Using the rough `6*parameters*tokens` convention and extrapolating the measured
9.73–11.28 TFLOP/s component rate gives **46–54 hours** for a nominal 125M model
over 2.5B tokens, or **60–69 hours** for this 162M configuration. At 10B tokens
those arithmetic estimates quadruple. These are not full-step measurements,
lower-bound proofs or rental quotes: the convention approximates work, the
GEMM shapes differ, and attention, loss, transfers, optimizer and data handling
are not priced. The prior blanket '2–3x overhead' multiplier is unmeasured.
No current BF16 opponent estimate is justified by these captures.

Priority order:

1. Lean attention allocation is now wired and checked on small native
   training fixtures in both explicit paths; see the follow-up handoff.
   Large peak memory/full-step timing is still required before claiming gains.
2. Own model, optimizer and context across Python calls with explicit close,
   failure poisoning, state export and restore semantics. Provide a lean
   step result; current calls return full gradients and host state snapshots.
3. Bound loss/logit storage for the large vocabulary, then benchmark a complete
   pilot at realistic context. Keep dense GEMM tuning tied to that profile.
4. Add accumulation, schedule/clipping wiring and a large binary checkpoint
   path. The current public JSON checkpoint still has a 2-MiB limit.
5. Measure tokens/s, allocation peak and checkpoint/resume on the intended
   GPU; use those results to price the full token budget. Large Transformer
   opponent admission and cross-vendor generalized-shape evidence remain open.

No performance gate flipped, no rental started, no tree implementation edited.
