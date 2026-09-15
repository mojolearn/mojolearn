# R2 prebuilt-binding cache, local evidence (2026-09-15)

Lane `lane/r2-binding-cache`. Code: `tools/bincache.py` (the key, the archive
check, the box-side build wrapper, and the Mac-side presign, list and promote
steps), `tools/bincache_leg.sh` (what the runners call) and
`tools/test_bincache.py`. Hooks sit in `tools/gemm_remote_leg.sh`,
`tools/do_extra_leg.sh` and `tools/hotaisle_leg.sh`. The cache is OFF by
default. It needs `MOJOLEARN_BINCACHE=1` on the Mac for the runner to stage
URLs, and a body opts in for each binding with
`python3 tools/bincache.py build bindings/build_X.sh`.

No GPU box was rented. Andrew's Sep 15 rule allows GPU legs only at a PyPI
release. Everything below ran on the M4 on one core, on a shared machine, and
the R2 steps used a few KB in the real `mojolearn-data` bucket, deleted
afterward. The fresh-then-cached pair on NVIDIA and AMD is still OWED; see
`docs/lanes/LANE_STATUS_lane-r2-binding-cache.md`.

## The problem, from legs already in the repo

The build rows of `remote/identity/status.tsv` in three 120-lane legs at
65ae7612f, each building 27 bindings:

| leg | build seconds |
|---|---|
| 2026-09-14_123742-nvidia-h100-identity-120-lanes-clean | 652 (10.9 min) |
| 2026-09-14_130705-amd-mi325x-do-identity-120-lanes-clean | 481 (8.0 min) |
| 2026-09-14_131953-amd-mi300x-hotaisle-identity-120-lanes-clean-b | 779 (13.0 min) |

Builds already reproduce across boxes of one image. Two H100 pods on Sep 13
wrote the same `_mojolearn.so` sha256 (def14add29...), and two MI325X
droplets a day apart wrote 4316fe282c... for it. The MI325X and the MI300X at
one commit and one arch (gfx942) wrote DIFFERENT digests, because their images
differ. That is why the image and the measured OS are in the key.

## What the key covers

Every field is in `key_fields`, and the key is sha256 over its canonical JSON:

- the build script, plus the sha256 of every file in the binding's Mojo import closure, the shell files it execs, `pixi.toml` (task tables and comments removed) and `pixi.lock`
- the modular packages pinned in `pixi.lock` for the platform
- the numeric mode, `MOJOLEARN_GPU_ARCHS`, `MOJOLEARN_TARGET_COLUMN`, and every other `MOJOLEARN_*`, `MOJO_*` and `MODULAR_*` variable except a short list that cannot reach the compiler (commit, run threads, gate skip, this cache). Compile job counts ARE keyed.
- the device arch the box reports, the image the runner declared, and the OS the build ran in (os-release ID and VERSION_ID, glibc, machine, first lines of `ld --version` and `cc --version`)
- the absolute repo path, because the rpath into `.pixi` is baked into each binding

A build whose environment or arguments contain SABOTAGE or FAULT_INJECT is
never looked up and never uploaded. The Mac's promote step refuses such a row
again, from the recorded fields.

If an import cannot be resolved to a file in the tree and does not name a
toolchain package, the key falls back to every `.mojo` file. That costs hits,
never correctness.

## Evidence files

- `unit_tests.txt` has 22 tests, all passing. They check that the key moves with the toolchain, imported sources, the build script, image, mode (fast and deterministic), a define, a trial variable, GPU archs, column, job count, device arch, OS release and glibc, and that it does not move with non-build variables, unrelated files or pixi tasks. An unresolvable import widens to the tree. Sabotage is refused. The host shim resolves through `build_host_family.sh`. The presigner matches the AWS documented SigV4 vector. Archive checks reject a flipped byte, a truncated archive, another key, a forged manifest, an extra member and path traversal. The flow tests run fresh, then cached (the build script does not run and the bytes are identical), then corrupt (rejected and rebuilt), plus mode miss, sabotage, off pass-through, failed-build exit code and promote refusals.
- `sabotage_arms.txt` holds four copies of `bincache.py`, each with one guard removed (sha check, image in key, sabotage refusal, job count in key). Every copy FAILS the suite.
- `closure_compiles.txt` is a ground-truth check of the import closure with real `mojo build -j 1` runs. Every `.mojo` file OUTSIDE the computed closure was replaced by a garbage line (1235 to 1403 files). The preprocessing host, hdbscan host, linalg (Metal), metrics (Metal) and gbdt (Metal) bindings still compiled, exit 0. Breaking one file INSIDE the preprocessing host closure (`checks/numerics.mojo` or `bindings/hostptr.mojo`) fails the build, exit 1. The seconds shown come from a warm compiler cache and are not a timing.
- `r2_end_to_end.txt` is the whole path against the real bucket, with a stand-in `ssh` that runs the piped script on the Mac. Stage wrote a 0600 map with 0 entries. Box A missed, built, and uploaded to an inbox slot through a presigned PUT. Promote copied it to `bincache/v1/none/local-e2e-selftest/<key>.tar.gz` server side. Stage then listed 1 entry. Box B, a fresh tree, took a HIT without running the build script, and the placed file's sha256 equals box A's. No provenance file names a signature, the key id or the secret; the same scan finds the planted value in a control file. The file ends with the `selftest-r2` round trip (put, list, get, copy, 404, delete).
- `closure_stability.txt` shows how often a binding's closure is unchanged between past record commits, so how often a later leg on the same image and arch could hit. 65ae7612f to 4048e1b51: 10 of 27. 4048e1b51 to 426bba511: 13 of 27. 426bba511 to 1edd3054f: 1 of 29, because `checks/kernel_matrix.mojo` (in 27 closures), `core/multi_gpu.mojo` and `build_host_family.sh` changed. Before `pixi.toml` tasks were dropped from the key, every one of those transitions read 0.
- `scripts/` holds the capture scripts exactly as run.

## Cost and expected savings

R2 has no egress fees. A full set of 27 bindings is about 40 MB on the M4
(`python/mojolearn/identical/*.so`), stored gzip level 1. One leg does one
LIST, one PUT, COPY and DELETE per missed binding, and one GET per hit. At
Cloudflare's published R2 prices that is well under a cent per leg, and
storage for a few hundred sets is cents a month. This session spent no box
money.

A hit replaces the compile with a download and a sha256 check of a few MB.
That is seconds, not measured on a box yet. The ceiling is the table above,
8 to 13 minutes a leg, reached only when every closure is unchanged:
- a re-leg or second half at the same commit, image and arch
- a later release record on the same image after a quiet stretch

Against the stability numbers, a leg a few hours after the previous one on the
same image would skip about 40 to 50 percent of its bindings. A day with edits
to `checks/kernel_matrix.mojo` skips almost none. A miss costs one gzip and
one PUT per binding on top of the build. The four boxes of a record are four
different arches or images, so they never share entries with each other.
