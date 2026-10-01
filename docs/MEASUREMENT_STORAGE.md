# Where measurements run and where their results live

The laptop orchestrates; it does not measure and does not hold the bulk. Every measurement runs on a remote box,
reads its data from R2, and its results go to git (small) or R2 (bulk).

## Boxes

| vendor | box | how to reach it |
|---|---|---|
| NVIDIA | RunPod L40S pods (`nvc2` runs the board; a second pod is rented for PR checks while the board runs) | `tools/nvidia_central.sh status / sync / submit` |
| AMD | DigitalOcean MI325X droplet (held) | `sh tools/do_amd_steward.sh ssh '<cmd>'` |
| Apple | M3 Ultra cloud Mac (`m3ultra-b`), plus the M2 Pro for the board | `ssh ec2-user@<ip>` with the mambik key; PR jobs push the branch to the Mac's bare repo |

PR measurements and the bench board share these boxes. While the board runs, PR jobs go between board chunks on
AMD and Apple, and on a separate NVIDIA pod.

## Data in

Datasets, corpora and token shards come from the R2 bucket `mojolearn-data` (`docs/REMOTE_DATA_R2.md`):
`sh tools/dataset_store.sh stage "<ssh flags+target>" <key>...`, pinned by
`bench/results/dataset_store/manifest.tsv`. A box never downloads from the internet what R2 holds.

## Results out

| what | where | how |
|---|---|---|
| The claim: SUMMARY.md, races.txt (ours ms + digest per arm), digest JSONs, rc.txt | git, `bench/results/<topic>-<date>/` | committed with the merge it supports |
| Bulk: full logs, job directories, raw outputs, board roots | R2 `measurements/<date>/<job>.tar.gz`, `boards/<box>/<root>.tar.gz` | `sh tools/box_evidence_r2.sh "<ssh flags+target>" <remote dir> <key> [exclude...]` uploads from the box and verifies by read-back |
| Index of everything in R2 | git, `bench/results/r2-index.tsv` (key, sha256, bytes, box, source dir) | appended by `box_evidence_r2.sh` |
| Bench board (live) | each box's newest root (`/root/board-0833`, `/Users/ec2-user/board-0833`) | the board controller; one root per wheel |
| Old boards | git `bench/results/board-archive/<box>/<root>/` (BOARD.md, board.json.gz); full roots in R2 `boards/` | never deleted |
| Opponents | git `bench/results/opponent-archive/<box>/` (merged store + every opponent cell) | measured once, reused, never overwritten |

To read a bulk object: `sh tools/dataset_store.sh presign <key>` gives a short-lived URL; fetch it on a box, not the
laptop, unless it is small.

## Current gaps

The live board's BOARD.md per box (pulled into `bench/results/board-latest/<box>/` after each pass) is the source
for "where are we behind"; stage breakdowns of specific rows are in `bench/results/neural-stage-timing-*`.
