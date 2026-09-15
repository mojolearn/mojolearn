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
- 2026-09-15: code written; prep running locally (one core).
