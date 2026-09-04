# E1 RESULTS — cross-vendor bit-identity, three GPU vendors

Superseded by `archive/evidence/E2_RESULTS.md` round 2 (`53d56ef`) and `archive/evidence/E3_RESULTS.md`. Every divergence recorded below is closed, and cross-device portability is measured (95/95 in E2 round 2, 111/111 both directions in E3 round 9).

**Claim demonstrated 2026-08-23**: training runs of tree-ensemble models on
three different GPU vendors — Apple M4 (Metal), AMD MI325X (HIP/CDNA3),
NVIDIA H100 (PTX/Hopper) — from ONE source tree at ONE commit, under
`NUMERIC_IDENTICAL` mode, with byte-identical inputs, produce
**bit-identical predictions AND bit-identical per-stage training
certificates** for Extra Trees classification and Random Forest
regression, and bit-identical predictions for symmetric-tree GBDT RMSE.
To our knowledge this is the first demonstration of bit-identical GPU
*training* of tree ensembles across GPU vendors with per-stage
certificates. (Prior art pins same-device determinism, reproducible
BLAS summation, or cross-platform *inference*; not cross-vendor training.)

## Protocol

One commit (`39a0d888e08be50230360db3e5963c61f6598dff`), one driver
(`tools/e1_traced_fit.py`), one bootstrap (`tools/e1_bootstrap.sh`, see
`archive/evidence/E1_RUNBOOK.md`). Inputs are a pure function of a fixed seed with
integer-exact target construction (no platform BLAS anywhere), proven by
hash before any fit is compared:

| input | sha256 (all three machines) |
|---|---|
| X | `5cbddce1deb73cde4aeb6a4be43a3cdb3efd2150cdbfd6e1887eb5ec73b1be8a` |
| y_reg | `58be14758af89c51c76e3dae88bb1986391d5e6afd234a43bcc1e5e9bea1240b` |
| y_clf | `3d78c1ea6e3ba776dd102193f112c835bfd8a8fc48993afba187536f4a5ec75d` |

The device+toolchain stack is the ONLY variable. Every fit writes an
identity-trace card (one hash per training stage — histograms, partition
stats, winners, leaves — 99 to 302 stages per fit); cards are compared by
`tools/identity_trace_diff.py`.

## The three-vendor table (prediction sha256, first 16 hex)

| fit | Apple M4 / Metal | AMD MI325X / HIP | NVIDIA H100 / PTX | card verdict |
|---|---|---|---|---|
| et_clf (ExtraTrees, 10 trees, depth 8) | `f6ecbc1ac97d3cf0` | `f6ecbc1ac97d3cf0` | `f6ecbc1ac97d3cf0` | **IDENTICAL, all stages, all three vendors** |
| rf_reg (RandomForest, 10 trees, depth 8) | `99ba0e31c9548deb` | `99ba0e31c9548deb` | `99ba0e31c9548deb` | **IDENTICAL, all stages, all three vendors** |
| gbdt_rmse (symmetric GBDT, 20 trees, depth 6) | `da34f396f968e546` | `da34f396f968e546` | `da34f396f968e546` | Apple↔AMD **IDENTICAL through all 302 stages**; NVIDIA diverges at `tree001.winners.scores` (see honesty note) |
| gbdt_logloss (same, Logloss) | `6a9045148f9f48e8` | `5ce9266642475b1b` | `0e1fcf6ed140e7d5` | DIVERGENT as the ledger predicted — Apple↔AMD first diverge at `tree000.perm0.leaves.estimated` (row 12, device exp/log), NVIDIA at `tree000.winners.scores` (row 9 score seams) |

## Honesty notes — what the certificates add over output hashes

1. **gbdt_rmse on NVIDIA is an output identity WITHOUT a certificate.**
   The prediction hash matches all three vendors, but the card shows the
   H100's split *scores* differ in last bits from `tree001` on — the
   argmax happened to pick the same winners, so the trees and predictions
   came out identical by margin, not by construction. Apple↔AMD agreeing
   through every stage while NVIDIA differs is the measured proof that
   two backends agreeing closes nothing. The open seams are the
   `compute_scores` multiply-add sites named by ledger row 9 (closure in
   flight as DEVIATION 253).
2. **Every divergence was named by the ledger before hardware existed.**
   Logloss was predicted to diverge at the leaf-estimation exp/log stage
   (row 12) and did, on both vendor pairs that share score bits; the
   RMSE divergence found earlier at `tree000.depth00.pstats` was row 8's
   named site and closed (`c077a22`), which moved the first divergence
   to the cursor-update seam (row 9), closed at `39a0d88` — after which
   Apple↔AMD agree on the entire fit.
3. **Same-device determinism is included in the evidence**: AMD runs 5b,
   6 and 7 (separate boots, fresh rebuilds) reproduced identical hashes;
   the NVIDIA leg ran ONCE and matched with no fix loop.

## Artifacts

Per-machine run directories under `bench/results/e1/` (each holds
`e1_fits.json` with commit + input hashes + prediction hashes, one
`.card` per fit, `bootstrap.log`, environment records):

- `2026-08-22_175315-MacBook-Air-1-terrabyte/` — Apple leg at `39a0d88`
- `2026-08-23_074048-mojolearn-e1-AMD/` — AMD leg at `39a0d88` (run 7)
- `2026-08-23_075219-mojolearn-e1-NV/` — NVIDIA leg at `39a0d88` (run 1)

Earlier dirs (`2026-08-22_*`) are the fix-loop history: the runs that
localized rows 8 and 9 and the driver's own BLAS bug (`ca0e635`).

## What is NOT claimed

- One configuration per family is certified, not the config matrix
  (losses, criteria, policies — the E2 sweep). Exp/log-family losses
  remain divergent until row 12's portable pair is routed through the
  loss kernels (core landed `ed0fe5d`, consumers in flight).
- Nothing is claimed for FAST mode, which is deliberately per-vendor.
- Cross-device model portability (train on A, predict on B) is expected
  but not yet measured. Serialization landed (`ee80324`: save/load on all
  three families, model-file bytes a pure function of the model, so a
  bit-identical fit gives a bit-identical model-file sha256 across
  machines; `tools/e1_cross_infer.py` is the round-trip driver). The
  cross-device round trip runs with the next vendor legs.
