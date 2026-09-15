# LANE STATUS: lane/istella-ranking-bench

Goal (Andrew, Sep 15 2026): benchmark the learning-to-rank GBDT (QueryRMSE,
PairLogit, YetiRank; our IDENTICAL tier) on Istella-S as a ranking dataset
(full train and test, query ids as `group_id`) on one RunPod H100, against
CatBoost GPU (same three losses), XGBoost GPU (`rank:pairwise`, `rank:ndcg`)
and LightGBM `lambdarank` (CUDA if the source build passes its probe, else
CPU, labeled). NDCG@5 and @10, fixed cost against per-tree cost, peak GPU
memory. Results under `bench/results/istella_ranking_2026-09-15/`.

Worktree: /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/wt-istella-rank

## Pieces
- `tools/istella_rank_prep.py`: writes `istella_rank.npz` (train and test
  query ids, the whole 681,250-row test split). Refuses unless the train grades
  and the first 500,000 test rows equal `istella_speed.npz` bit for bit.
- `tools/speed_gbdt_arm.py`: `load_istella_rank` (the binary and regression
  Istella cells are unchanged).
- `tools/speed_gbdt_rank.py`: one process per (library, loss); timing at 1, 10
  and 100 trees, repeated; NDCG computed by one function for every library.
- `tools/istella_rank_leg.sh`: pod body, phases setup, smoke, cells, lgbm_cuda.
- `tools/dataset_store.sh`: catalog key `gbm-bench/istella/istella_rank.npz`.

## Resume
1. If `bench/results/dataset_store/manifest.tsv` has no `istella_rank.npz` row:
   rerun the prep script on the R2 tarball and npz, place the output at
   `~/datasets/gbm-bench/istella/istella_rank.npz`, then run `dataset_store.sh
   manifest` and `push gbm-bench/istella/istella_rank.npz`.
2. Rent (see the header of `tools/istella_rank_leg.sh`). State lives in
   `~/mojolearn-evidence/istella-rank/pod`. Before renting, run
   `sh tools/trees_leg.sh pods` and reap any `mojolearn-rank-*` pod left over.
3. Phases setup, smoke, cells, lgbm_cuda; pull `/root/rank_out/`; reap and verify.

## Log
- 2026-09-15: code written; prep passed; istella_rank.npz pushed to R2 and
  pinned (eda607eda).
- 2026-09-15 19:05 ET: leg 1 pod z0jbxz113hwf3n (H100, driver 580.126.09),
  60-minute watchdog armed (deadline about 20:05 ET), shipped eda607eda, R2
  staged 2 keys in 64 s. If the Mac restarted: `TREES_LEG_STATE=$HOME/mojolearn-evidence/istella-rank/pod sh tools/trees_leg.sh reap`.
- 2026-09-15 19:26 ET: leg 1 DONE. All ten cells recorded, pulled to
  `bench/results/istella_ranking_2026-09-15/`, pod DELETED and verified gone
  (HTTP 404), 21 minutes at $3.49/hr, about $1.22. Two fixes during the leg:
  scikit-learn was missing on the pod (the XGBoost and LightGBM sklearn
  wrappers need it, and the first LightGBM CUDA probe failure was that, not
  CUDA), and the XGBoost arm now relabels qids by order of appearance.
- Result in one line: our fastest cell (QueryRMSE 3,530 ms) against CatBoost's
  fastest (2,355 ms) is 1.50x, against XGBoost's fastest 0.90x, against
  LightGBM CUDA 0.65x; QueryRMSE and PairLogit gaps are fixed cost, YetiRank's
  is per tree (6.57x of CatBoost's slope) and is the cell to work on next.

## Owed
- Nothing for this leg. A YetiRank per-tree investigation is the obvious next
  lane; 1,278 ms of our fixed cost is the Python `_group_sizes` loop.
