# Orchestrator handoff, 2026-09-09 (identical fan-out, rounds 1 and 2)

Read this first, then the lane handoffs it names. Andrew's order (13:50
EDT): the round-1 TREES lane winds down after its in-flight H100 A/B; the
four round-2 lanes CONTINUE their full briefs; no further lanes until he says so.

## Policy in force (Andrew, Sep 9)

- The only number is OUR IDENTICAL arm vs THE OPPONENT'S FAST arm on the
  same GPU model and shape. Never time, test, qualify or modify our FAST or
  DETERMINISTIC arms; the existing three-tier bindings stay exactly as they
  are (their fast arms speed up MacBooks).
- Opponents are measured ONCE and stored in `bench/OPPONENT_REFERENCE.md`;
  a round runs ours alone and reads the opponent from that file. A new
  opponent run is owed only for a missing (GPU, driver, version, shape)
  tuple. Published numbers do not exist at our shapes (checked Sep 9).
- Subagents write code and rent NVIDIA boxes; only the orchestrator runs
  anything on the Mac, one thing at a time, nice 19. Every lane leaves a
  RUN OWED list; the orchestrator runs it at merge.
- Fix forward in the same commit; document only opponent-side gaps and
  multi-session optimization targets.

## State of main

- `71d2ba71` PUSHED: round-1 lanes merged (`lane/gemm-identical`,
  `lane/knn-identical`, `lane/samba-attention`, `lane/samba-training`),
  the opponent reference table, one training-test expectation fix, and the
  gate fixes (dbscan_main at -O1 in `tools/check_unsupervised_identity.sh`
  and `tools/e1_unsupervised.sh`; `tools/knn_layout_dispatch_price.sh` runs
  without timeout(1) on macOS).
- All 26 RUN OWED steps of the four merged lanes are GREEN on the Apple M4
  at 71d2ba71: GEMM device check, GEMM card byte-equal to the shipped
  NVIDIA card, tuned probe 20 match / 0 moved, batch invariance, backward,
  OLS both modes, kNN identity + main, unsupervised identity both modes,
  transformer forward and backward cards byte-equal to the shipped cards,
  transformer surface 116/116, training surface 69/69, samba surface green,
  loss card a87615d9 and optimizer card 97d160b0 unchanged, all three
  training builds compile.
- UNCOMMITTED on main at the time of writing: the Apple kNN flip in
  `checks/kernel_matrix.mojo` (`or column == COLUMN_APPLE` in
  `_knn_identical_round_column`, docstring carries the numbers). Its
  post-flip checks were running (`check-knn-identity`, `knn_main`,
  unsupervised identity both modes, the 400k index-tile fingerprint gate,
  the three UMAP checks). If they are green: commit that one file and push.
  If not: revert the one line, record the failure here.
- Apple flip evidence (M4, 100k x 32, k 10, 9 rounds, PRICE_MS medians,
  baseline -> both): 20.5 -> 15.1 ms at 32 queries, 26.9 -> 14.9 at 128,
  182.2 -> 66.6 at 1000; all four arms byte-equal to the NVIDIA baseline
  (143,628 cells). Transpose-only was slightly faster than both at 128 and
  1000 (12.1, 60.2) and slower at 32 (16.3).

## Lanes open at handoff time (trees winding down, the other four continuing)

| lane | branch | worktree (under .claude/worktrees/) | handoff file it must leave |
|---|---|---|---|
| trees (round 1) | lane/trees-identical | agent-abadb3e27843a787c | docs/lanes/HANDOFF_trees.md |
| UMAP optimizer + kNN selector | lane/umap-optimizer | agent-a4b445131c31d9311 | docs/lanes/HANDOFF_umap.md |
| GEMM split-K + H100 table | lane/gemm-splitk | agent-a68e7f0e68dc3bc9f | docs/lanes/HANDOFF_gemm_splitk.md |
| fused attention (bits unchanged) | lane/fused-attention | agent-a94c0adce7c14179f | docs/lanes/HANDOFF_fused_attention.md |
| DBSCAN compiler crash | lane/dbscan-compiler-crash | agent-aa81defd01a6607a1 | docs/lanes/HANDOFF_dbscan_crash.md |

Merge procedure for each: `git merge --no-ff lane/<x>` onto main, run its
RUN OWED list on the M4 one step at a time (nice 19), fix forward, push.
Check RunPod for orphaned pods afterwards (terminate anything not named
samba-*).

## Where the egregious gaps are (ours identical vs their fast, same GPU)

| area | shape | ours | theirs | ratio | owner |
|---|---|---|---|---|---|
| UMAP optimizer (serial host SGD) | 100k x 32, k 15, 200 epochs, H100 | 62.7 s | cuML 0.321 s | 195x | umap lane |
| Attention forward, unfused | L40S, d 1024, seq 4096, batch 4, window 2048 | 685.7 ms | torch SDPA 33.6 | 20x | fused-attention lane |
| Attention fwd+bwd | same | 1672 ms | 106.9 | 16x | fused-attention lane |
| v1 GEMM SPLITK plan | gram.32x32x1M, L40S | 138 ms | cuBLAS fp32 0.378 | 366x | gemm-splitk lane (estimators use the core path, at 0.368 ms = parity) |
| kNN, many queries | 400k, 4000 q, k 10, H100 | 66.5 ms | cuML 10.2 | 6.5x | umap lane task 2 |
| Mamba-2/3 blocks | Sep 7 grid, H100 | | torch reference scan | 72-85x | NOBODY this round |
| Transformer end to end | Sep 7 grid, H100 | | torch eager | 22-41x | partly the two lanes above |
| Symmetric GBDT vs CatBoost | HIGGS 1M-5M, H100 | | CatBoost GPU 846-2459 ms | paper said 4.9-31x; not measured in this tree yet | trees lane |
| Dense GEMM (TUNED plans) | Llama-8B t512, L40S | 1.26-35.6 ms | cuBLAS fp32 0.54-16.9 | 2.1-2.7x | done for now |
| kNN, few queries; RF/ET; OLS/PCA Gram | | | | at or near parity | none needed |

The structural floor: identical means one fixed FP32 FMA order on every
vendor, so tensor cores (TF32, bf16) are unavailable; against the
opponents' FP32 paths ~1.5-2x is the practical floor, against their
TF32/flash paths several x is permanent.

## Not started, in priority order after the open lanes land

1. Mamba-3 SISO block speed (72-85x vs torch reference scan; the byte-LM
   training step is already 2x faster than eager torch, so the gap is in
   the prefill/scan kernels).
2. Symmetric-tree leaf estimation and partition launches (per the trees
   lane's profile once it lands).
3. Missing opponent rows: LightGBM CUDA extra_trees (valid build), cuML
   DBSCAN/PCA at 1M rows, torch byte-LM step time on H100.
4. Round-1 leftovers: estimator-level GEMM swap timing; AMD column for the
   kNN rows and the tuned GEMM plans; the training lane's 64-step runs and
   checkpoint sha (it has NO measured number yet).
