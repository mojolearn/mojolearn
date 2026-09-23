# Remote runs take their data from R2

**Rule (maintainers): every rented or remote run gets its datasets, corpora
and token files from the project's Cloudflare R2 bucket, never from the
original source on the internet, and never by decoding on the box.** A leg
that needs a file that is not in R2 puts it in R2 first (from the Mac), then
stages it.

Why: a rented box that fetched Istella-S from its origin spent about 21
minutes on setup for 4.5 minutes of measurement (2026-09-12), and origin
servers time out mid-download. R2 charges no egress, so a pod pulls the
pinned, pre-decoded file at datacenter speed.

## How

| what | tool | where it is pinned |
|---|---|---|
| datasets, corpora, token shards | `sh tools/dataset_store.sh stage "<ssh flags and target>" <key>...` | `bench/results/dataset_store/manifest.tsv` (size and sha256; the box refuses a mismatch) |
| compiled bindings (Linux release sets) | `tools/bincache.py`, R2 prefix `bincache/v1/<device>/<image>/` | the cache key and per-file sha256 in each archive; `build-provenance.json` records every hit |

- `stage` mints a short-lived presigned URL on the Mac and hands it to the box
  over stdin. The R2 credentials (`~/.mojolearn_r2`, mode 600, outside the
  repository) never leave the Mac and never appear in argv or logs.
- `sh tools/dataset_store.sh list` shows what is stored; `push` and
  `manifest` add a new file and its pins.
- A leg script that downloads from an origin URL on the box is a defect: route
  it through `dataset_store.sh stage`.

What does not use R2: the local release check (`pixi run -e test
release-check`) runs on the Mac with generated fixtures, and the wheel smoke
installs only the wheel.

See also [RUNPOD_CPU_LEG.md](RUNPOD_CPU_LEG.md) (the staged dataset table) and
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) (section 2c, the binding cache).
