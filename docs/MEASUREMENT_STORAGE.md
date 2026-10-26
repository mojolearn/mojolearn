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

## Box jobs persist from the box (2026-10-26)

A result that lives only on a rented box dies with it: a Mac that disconnects stops the watcher that would have
pulled it home, and the box's dead-man deletes it on schedule. So every job on a rented box writes its results
to R2 from the box itself, by default:

| entry point | results dir (on the box) |
|---|---|
| `tools/nvidia_central.sh submit <lane> [--results DIR] <script>` | `DIR` relative to the lane tree; default `box-out/<tag>` |
| `tools/do_amd_steward.sh run <tag> <results-dir> <command>` | `<results-dir>`, relative to `/root/mojolearn` |
| `tools/dev_pod.sh run <lane> [--results DIR] <command>` | `DIR` relative to the box tree; default `box-out/<tag>` |

The Mac mints presigned PUT and GET URLs for the job's keys only (`tools/box_persist.py prepare`; the R2
credentials never leave the Mac; the URLs expire, at most 7 days) and hands them to the box as a 0600 json on ssh
stdin. On the box, `tools/box_persist_run.sh` runs the command (its output also goes to `command.log` in the
results dir, which the command sees as `$MOJOLEARN_RESULTS_DIR`), tars the results dir every 10 minutes into a
ring of slots, and uploads `final.tgz` on any exit (normal, TERM, INT). HUP is ignored: an ssh drop leaves the job
running and uploading. Every self-destruct path (`tools/runpod_guard.sh` watchdog, the `gpu_queue_box.sh` lease
delete, the Hot Aisle watchdog, `do_amd_steward.sh`'s `mojolearn-selfkill` unit) runs
`box_persist_run.sh --flush-all 120` before its delete.

R2 layout: `measurements/box-jobs/<YYYY-MM-DD>/<tag>/{00..N-1}.tgz`, `final.tgz`, each with a `.sha256` sidecar,
and `manifest.json` (every upload: seq, name, sha256, bytes, utc; state running|flushed|final; exit code). The
box logs each upload to `persist.log` in the results dir (`UPLOADED`, `FAILED`, `REJECTED`, `EXPIRED`).
Tarballs leave out venvs, `.pixi`, wheels and caches; one over the size cap (3.8 GB) drops its largest files and
names them in `PERSIST_TRUNCATED.txt`.

Read a job back: `python3 tools/box_persist.py fetch --tag <tag> [--all|--list] [--extract]` downloads `final.tgz`
(else the newest slot) to `~/mojolearn-evidence/box-persist/<tag>/fetched/` and verifies it against its sidecar.
The tag is printed when the job starts and indexed in `~/mojolearn-evidence/box-persist/index.tsv`.
`--no-persist` on an entry point is logged and refused unless `MOJOLEARN_ALLOW_UNPERSISTED=1`.

## Closed-run cold archives (2026-10-21)

Andrew authorized moving old local evidence to S3 Glacier Deep Archive and removing verified local copies.
This is an exception to keeping bulky evidence on the Mac for the selected closed runs. Current measurement
delivery and the automatic box archive still use the Mac and R2. Keep boards, accepted results, recipes and
compact provenance readily available; age alone does not establish that a run is closed or disposable.

The reusable Terraform module and mojolearn's applied configuration live in Mambik:
`infra/modules/cold-archive/` and `infra/cold-archive/`. See that repository's `infra/README.md` for administration.
The bucket is `s3://mojolearn-archive-569378724596-us-east-1`, in `us-east-1`; transfers and restores use
`AWS_PROFILE=mambik-evidence`. The archive has versioning, encryption and public-access blocking. Completed
objects have no automatic expiration. Archive packs explicitly use `DEEP_ARCHIVE`; catalogs, manifests and
restore tools use `STANDARD` so they remain immediately readable.

The initial migration is indexed under `cold-evidence/2026-10-21/`, with the local controller, receipts and
deletion records in `~/mojolearn-evidence/storage-cleanup-20261021/`. Each group has `catalog.json`,
`manifest.jsonl.zst` and `restore_archive.py`. Catalogs identify exact object versions, compressed checksums
and content chunks; a group can reference packs stored under another group's prefix. Preserve every referenced
pack. The manifest records original paths and complete file hashes.

Before local removal, the migration fully decodes and checks each new pack locally, sends its complete SHA-256
with the upload for S3 validation, and verifies the exact version's size, checksum and storage class with HEAD.
It also preserves the complete manifest and restore program. This establishes server-side checksum retention;
it does not claim a completed Glacier restore/download. Open or changed files remain local. Duplicate-container
removals have separate proofs recording the complete content comparisons performed before deletion.

During the initial migration, a separate cleanup subsequently removed `measure-0852`, including the verified
extracted counterparts of three duplicate-pruned race tarballs. This migration did not archive those tarballs
in Glacier, and no complete replacement copy has been verified. The latest `index.json` and `README.txt` record
this exception and link the separate cleanup's retained deletion journal; earlier object versions preserve
the original records. The 27 cataloged Glacier groups remain retained.

To recover a group, fetch its catalog, manifest and restore program, request Glacier restoration of every pack's
exact version, and wait for availability. Download each pack using the catalog's `download_name`, falling back
to its key basename for older catalogs. Run the group's restore program into an empty directory; it checks
compressed packs, decoded chunks and reconstructed whole-file hashes. Restore requests take hours and incur
charges; use the retained instructions rather than treating Glacier as an online dataset store.

## Current gaps

The live board's BOARD.md per box (pulled into `bench/results/board-latest/<box>/` after each pass) is the source
for "where are we behind"; stage breakdowns of specific rows are in `bench/results/neural-stage-timing-*`.

## Measurement receipts (2026-10-13)

`tools/bench_board.py` snapshots the actual repository code and explicit driver files before a new race.
It probes constructed arms (including reused opponents) before timing and refuses missing or mismatched
parameter/version read-back. Functional opponent adapters identify the actual library callable with
`__implementation__` in their declared parameter record; a hand-written or unidentified opponent is refused
before timing. The timed worker records its own package versions, resource settings and loaded
mojolearn source/binary hashes. The retained run archive includes those loaded bytes, full logs, the recipe,
preflight, and unmodified result JSON. These are benchmark controllers, not product execution.

Each cell has a `measurement` object: `software`, `parameters` (raw and canonical), `dataset`, `shape`,
`hardware`, `numeric_mode`, `timing`, `command`, `workers`, `source`, `artifacts`, and `completeness`/`missing`.
Missing information stays explicit. Dataset file hashes and generator source identify the input; actual
shape records the workload that ran, independently of a requested `--rows full`. Third-party implementations
are identified by worker package versions; the source bundle is repository code, not an archive of every
third-party package. Existing artifact-manifest guards continue to check frozen numerical binaries.

Source and run archives go to content-addressed R2 keys under `measurements/records/v1/`. Uploads verify
by GET and SHA-256 before a board records `status: verified`. The measurement worker needs the standard R2
configuration (`MOJOLEARN_R2_CREDS` or `~/.mojolearn_r2`, or the existing R2 environment variables); credentials
and signed URLs never enter a receipt. The default checks R2 before timing. `--evidence-local-only` is restricted
to explicit smoke runs and produces unpublished local evidence.

If a post-run upload fails, the board saves the result and stops before the next race. Retry retention only:

```
python3 tools/bench_board_evidence.py sync /path/to/run/board.json
python3 tools/race_fairness.py /path/to/run/board.json
```

Pull the resulting board JSON when collecting a remote job. Main-board ingestion preserves each original
cell's measurement record and each cached opponent's original record; it never labels old timings with the
current checkout's parameters or versions. Compact receipts remain in board.json; large source/log bytes stay
in R2. Retain local bytes until upload read-back succeeds. Existing model/output retention requirements still
apply; this archive does not replace model-state or unique input retention.

Historical supported audits remain valid without manufacturing new receipts or rerunning measurements.
`--upgrade-legacy-audits` is a one-time explicit migration of existing unchanged audit pairings to full evidence
fingerprints. Ordinary rebuilds cannot approve an old audit solely because median times still match.

Fetch the exact archived bytes named by a board, with checksum verification (no extraction or execution):

```
python3 tools/bench_board_evidence.py fetch /path/to/board.json --race RACE_ID --arm ours --kind artifacts --destination /path/to/run.tar.gz
```

Use `--kind source` to retrieve the source snapshot. Never substitute current source for an older record.
