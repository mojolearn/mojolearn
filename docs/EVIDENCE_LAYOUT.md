# Evidence layout: small core in the repo, bulk out

Rule (Andrew, 2026-10-28): everything core is versioned in the repo; bulk stays out. Nothing larger than a few MB
enters the repo, and no binaries (`.so`, `.tgz`, `.npz`) or raw logs. Copies into the repo never delete or move the
original in `~/mojolearn-evidence/`.

Retention (Andrew, 2026-10-28: we retain every set of evidence, not just the latest): every run's small summary is
added under a dated directory and never deleted; bulk stays in `~/mojolearn-evidence/` and R2; `latest/` is only a
pointer (a README naming the current dated directories, no data and no symlinks).

## Where things live

| Kind | Where | Examples |
|---|---|---|
| Ops scripts the rules name | repo `tools/`, `tools/ops/` | `tools/compile_slot.sh` (evidence path is a shim), `tools/ops/lq`, `tools/ops/archive_all_boxes.sh`; origins in `tools/ops/README.md` |
| Compile and lane receipts | repo `experiments/<scope>/` | `compile.json`, `CHECKPOINT.txt`, `HANDOFF*.txt`; lane handoffs in `experiments/handoffs_<YYYYMMDD>/<lane>.md` (dated, accumulate, never deleted) |
| Lane briefs | repo `docs/briefs/<YYYYMMDD>/` (dated, never deleted; mirror of `~/mojolearn-evidence/briefs-<date>/`) | `docs/briefs/README.md` |
| Board and verifier summaries | repo `bench/results/bench_board/main-<box>/` (canonical board), `docs/measurements/<YYYYMMDD>-<tag>/`, `docs/verification/<YYYYMMDD>-<tag>/`, `bench/results/light_verifier/<date>/`; `docs/*/latest/README.md` points at the newest | config, manifest, controller state, `races.tsv`, summary JSON |
| Claim files for a merge | repo `bench/results/<topic>-<date>/` | SUMMARY.md, races.txt, digest JSON (`docs/MEASUREMENT_STORAGE.md`) |
| Full logs, build output, per-recipe boards, race tarballs, binaries, source bundles, box archives | `~/mojolearn-evidence/<lane-or-run>/` on the Mac | `full-measurement-*/`, `box-archive/`, compile logs |
| Retention of bulk | R2 (`measurements/`, `boards/`, `measurements/box-archive/`), indexed in `bench/results/r2-index.tsv` | uploaded from the box by `tools/box_persist_run.sh`, `tools/box_archive.sh`, `tools/box_evidence_r2.sh` |
| Datasets | R2 bucket `mojolearn-data` (`docs/REMOTE_DATA_R2.md`) | never in git |

## Size facts (2026-10-28)

- `~/mojolearn-evidence`: about 75 GB (box-archive 20 GB, full-measurement-20261027-r2 19 GB, other run dirs).
- `bench/results` tracked in git: about 1.5 GB in a fresh checkout; the main checkout's `bench/results` directory is
  about 8.5 GB with untracked and ignored local outputs.
- The 2026-10-28 core copy (scripts, briefs, handoffs, measurement and verifier summaries) added under 1 MB.

## Adding something

1. If a rule, tool or brief names a file that exists only in `~/mojolearn-evidence/`, copy it into the repo and record
   its origin (scripts: `tools/ops/README.md`; summaries: a README in the new dated dir naming source dir, run tag
   and date).
2. A new run gets a new `<YYYYMMDD>-<tag>/` directory (for example `20261027-nvidia-l40s`,
   `20261027-amd-mi325x-r3`); never overwrite or delete an older one. Then update the `latest/README.md` pointer.
3. Keep only what a reader needs to know what ran and what it showed: summaries, manifests, receipts, one-line
   results. Leave the logs and binaries where they are and make sure they reach R2.
4. Never keep the only copy of a script in an evidence folder.
