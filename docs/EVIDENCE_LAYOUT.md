# Evidence layout: small core in the repo, bulk out

Rule (Andrew, 2026-10-28): everything core is versioned in the repo; bulk stays out. Nothing larger than a few MB
enters the repo, and no binaries (`.so`, `.tgz`, `.npz`) or raw logs. Copies into the repo never delete or move the
original in `~/mojolearn-evidence/`.

Retention (Andrew, 2026-10-28): the repo holds only the latest run's small summary (`docs/measurements/latest/`,
`docs/verification/latest/`); it is overwritten on each new run. The full history of every run is retained in
`~/mojolearn-evidence/<run dir>` (for example `full-measurement-nvidia-20261027/`, `full-api-verifier-20261027/`) and in
R2 (`measurements/<date>/<job>.tar.gz`, `boards/<box>/<root>.tar.gz`, `measurements/box-archive/<box>/`, indexed in
`bench/results/r2-index.tsv`), and the per-release boards under `bench/results/bench_board/` (old roots in
`bench/results/board-archive/`). Lane briefs (`docs/briefs/<YYYYMMDD>/`) and lane handoffs
(`experiments/handoffs_<YYYYMMDD>/`) stay dated and accumulate.

## Where things live

| Kind | Where | Examples |
|---|---|---|
| Ops scripts the rules name | repo `tools/`, `tools/ops/` | `tools/compile_slot.sh` (evidence path is a shim), `tools/ops/lq`, `tools/ops/archive_all_boxes.sh`; origins in `tools/ops/README.md` |
| Compile and lane receipts | repo `experiments/<scope>/` | `compile.json`, `CHECKPOINT.txt`, `HANDOFF*.txt`; lane handoffs in `experiments/handoffs_<date>/` |
| Lane briefs | repo `docs/briefs/<date>/` (mirror of `~/mojolearn-evidence/briefs-<date>/`) | `docs/briefs/README.md` |
| Board and verifier summaries | repo `bench/results/bench_board/main-<box>/` (canonical board), `docs/measurements/latest/`, `docs/verification/latest/`, `bench/results/light_verifier/<date>/` | config, manifest, controller state, `races.tsv`, summary JSON |
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
   its origin (scripts: `tools/ops/README.md`; summaries: the README of the target dir). A new run's summary
   replaces the previous one in `latest/`; update its README to name the new source dir, run tag and date.
2. Keep only what a reader needs to know what ran and what it showed: summaries, manifests, receipts, one-line
   results. Leave the logs and binaries where they are and make sure they reach R2.
3. Never keep the only copy of a script in an evidence folder.
