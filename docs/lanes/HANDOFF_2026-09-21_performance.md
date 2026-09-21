# Performance handoff — 2026-09-21

This handoff covers the IDENTICAL training and tree-performance lanes. The
integration worktree is `/Users/andrewhendel/mojolearn-wt/handoff-sep18`; use
`origin/main` as the source of truth after fetching. Agent work uses separate
worktrees and normal merges, not cherry-picks. A clean worktree is removable
only after `tools/worktree_prune_check.sh` and a live-owner/evidence check.

## Shipped and qualified

- GPT-3-small repeated training: large embedding backward selects stable sort
  (about 65 ms to 18–19 ms on Apple at B16×L2048, identical gradient cells);
  Apple RMSNorm backward, residual/norm boundaries, norm2/residual backward,
  and gated-SiLU backward have measured IDENTICAL routes. NVIDIA and AMD
  gated-SiLU plus norm2/residual fusions were also qualified: H100 1.23× and
  1.17×, MI300X 1.71× and 1.29× for the respective isolated stages. All
  forced arms matched 37/37 backward stages and 412,172 cells; merged
  `check-train-step` passed. These are isolated-stage timings, not a summed
  whole-step claim.
- Exact NVIDIA QKVO dWeight GEMM uses a narrow fold-storage specialization at
  768×768×2048, measured 2.11–2.15% faster over its 48 repeated calls. Other
  shapes/vendors retain their existing routes. The opt-in broader trial is
  not promoted: physical registers remained capped at 255 and occupancy at
  one block/SM despite smaller local storage.
- Mamba2 production forward defers 13 redundant stage host drains while
  tracing retains its stage boundaries. Twelve-layer Apple forward improved
  20.5–50.2% across rotated runs; output/state hashes, 26 traced stages and
  55 public backward tensors matched.
- IDENTICAL symmetric resident inference groups up to four consecutive trees
  per launch. Public million-row cases improved 1.17×–1.53× and the repeated
  apply kernel about 2×; complete model/prediction/probability hashes match.
  FAST and DETERMINISTIC routes are unchanged. The earlier shared split-
  metadata staging also preserved complete IDENTICAL model/loss hashes and
  gave a small 1–2.7% million-row fit gain (FAST fit gained 21.8%).

## Rejected or pending

- Two-GPU disjoint-owner gradient queueing was exact across 1/2 devices,
  logical shards 2/3/5, full optimizer state, faults, rollback and replay.
  Reverse-order timing regressed 11.4–16.6%, so production source was
  restored. Raw target/fault evidence is in
  `bench/results/2026-09-20_gpt_multigpu_gradient_queue/`.
- AMD attention full-estash `y` aliasing removed 201,326,592 scratch bytes
  per L2048 layer and matched all seven hashes, but whole forward+backward
  was 7.598761 to 7.610369 ms (0.15% slower). AMD routing is unchanged.
- Mamba2 backward download batching matched all ten gradient leaves and the
  55-tensor public oracle, but aggregate speed was only 0.72% and changed
  with run order. It was reverted.
- Attention TT32/TQ32, causal n_rep specialization, host-wait removal and
  zdot→dQ fusion were exact but slower/noisy; see
  `bench/evidence/2026-09-20_optimization_attempt_ledger.md`.
- Exact RF sequential accumulator reuse and Extra Trees RPT64/publication
  trials preserved hashes but did not improve elapsed time; source reverted.

## Cloud and worktree safety

Owned qualification resources were terminated with DELETE 204 and GET 404:
the two-GPU L40S pod `uwzao6ig3z1jh1`, the H100/MI300X fusion legs, and the
AMD attention-alias leg. Other sessions' pods were never modified. A prior
external attention pod `cgp2jehyh5elty` was running at last inventory; its
current state and owner must be checked before any action. Do not infer that
an empty lease file means an external pod is disposable.

The multi-GPU evidence branch is clean and production source is unchanged.
The remaining worktree may be pruned only after the normal eligibility and
ownership checks. Preserve dirty or locked trees and raw evidence. Do not
use commit ancestry alone as a pruning test: prior cherry-picks can make
shipped content appear unmerged; `tools/worktree_prune_check.sh` uses stable
patch equivalence.

## Next measurements

1. Profile a fresh full GPT training step at representative batch/sequence
   shapes on NVIDIA and AMD before prioritizing further isolated kernels;
   keep IDENTICAL hashes and account for 12-layer call counts.
2. For multi-GPU, retain fixed shard-fold and rollback semantics. Try another
   communication/staging design only with strict 1-vs-2 device target/fault
   A/B and reversed timing order; do not revive the rejected owner queue.
3. For attention, focus on measured numerical work (AMD L2048 zdot about
   0.91 ms, dQ about 2.46–2.49 ms, dK+dV about 1.78–1.79 ms in the alias
   screen) rather than allocation-only or host-wait changes. Preserve all
   seven output hashes and pinned folds.
4. Keep vendor routing evidence-bounded: a portable kernel need not be
   enabled everywhere when another vendor or size regresses. Accept small
   repeatable wins; there is no fixed 10% bar. Record failed ideas.
