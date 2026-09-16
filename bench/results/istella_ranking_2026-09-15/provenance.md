# Provenance, Istella-S ranking leg (lane/istella-ranking-bench)

- Pod `z0jbxz113hwf3n`, RunPod SECURE, NVIDIA H100 80GB HBM3 (81,559 MiB,
  compute capability 9.0), driver 580.126.09, image
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, host Intel Xeon
  Platinum 8480+, 224 cores, Linux 6.8.0-106-generic. Rented 2026-09-15 19:05
  ET with a 60-minute on-pod watchdog armed BEFORE any work (verified alive,
  `tools/runpod_guard.sh`), one pod for this lane at a time.
- Source: commit `eda607eda` shipped as a git archive; `SHIPPED_COMMIT.txt` on
  the box carries it.
- Data: staged from R2 in 64 s, verified against
  `bench/results/dataset_store/manifest.tsv`:
  `gbm-bench/istella/istella_speed.npz` (2,248,281,826 bytes) and
  `gbm-bench/istella/istella_rank.npz` (624,022,440 bytes, written by
  `tools/istella_rank_prep.py` and pushed to R2 by this lane). Nothing was
  downloaded from an origin on the box and nothing was decoded there.
- Ours: IDENTICAL tier, `MOJOLEARN_NUMERIC_MODE=identical`, bindings built on
  the box (`bindings/build.sh` 56 s, `bindings/build_gbdt.sh` 98 s); the import
  read back `identical cuda`.
- Opponents, pinned: catboost 1.2.10, xgboost 3.2.0, lightgbm 4.7.0 (the
  versions of the existing H100 rows in `bench/OPPONENT_REFERENCE.md`), numpy
  2.4.6, scikit-learn 1.9.1, Python 3.11.
- LightGBM CUDA: built from source on the box with `USE_CUDA=ON` into
  `/root/lgbm_cuda` (268 s) and kept OUT of the CPU wheel's path, so the CPU
  and CUDA arms are the same version, 4.7.0. Its probe passed only after
  scikit-learn was installed; the first probe failure was a missing dependency,
  not a CUDA failure.
- Dataset: Istella-S LETOR. Train 2,043,304 rows in 19,245 queries, test
  681,250 rows in 6,562 queries, 220 dense features, grades 0..4. Largest
  query 182 rows (YetiRank refuses a query over 1,023). The prep refused to
  write unless the train grades and the first 500,000 decoded test rows equal
  `istella_speed.npz` bit for bit; they did.
