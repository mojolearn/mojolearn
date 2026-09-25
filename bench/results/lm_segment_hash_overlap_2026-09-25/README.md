# The per-step hash of `tools/lm_segment.py run`, off the step's path, 2026-09-25

Branch `lane/lm-segment-hash-overlap`. The GPT-3 Small run writes one chain
line per optimizer step: the digests (`sliced-sha256-8.v2`) of the whole state
(parameters, m, v, flags: 1.95 GB at 162,147,840 parameters) and of the
summed gradient (648.6 MB), read back from the device after the step. T3
measured that work at 6.6 s on the H100 hosts, 2.6 s on the DigitalOcean
host and 6.4 s on the Hot Aisle VM, against a 17.2 to 19.5 s step
(`bench/results/lm_t3_2026-09-23/README.md`).

## Where the time goes

On the H100 host (RunPod, Intel Xeon Platinum 8460Y+, 160 CPUs with SHA-NI,
Python 3.11, the published wheel 0.8.18, one device), measured on one
post-step state by `box_split.py` (`h100/box_split.json`):

| part | seconds |
|---|---|
| `export_raw()` (fresh arrays + the binding's device read-back of parameters, m, v, flags) | 5.35 |
| `export_gradients()` (fresh array + read-back of the summed gradient) | 1.22 |
| the digests as the runner took them (`_hash_arrays` + `_hash_gradient`, eight slice threads, one array at a time) | 0.34 |
| **total, the old runner's `hash_seconds`** | **6.9** (the three old-runner lines read 7.04, 6.98, 7.15) |
| of which: allocating the four fresh 648.6 MB arrays (`empty()`, zero-filled pages) | 1.43 |
| the same two export calls into buffers already allocated (`Snapshot.read`, second read) | 3.96 |
| the same digests with every slice in its own thread (`Digests`) | 0.13 |
| one sha256 on one core over the same 2.59 GB (for scale) | 1.53 |

So at this shape on this host the sha256 is about 5 percent of `hash_seconds`.
About 20 percent is the Python side allocating fresh arrays every step, and
about 60 to 75 percent is the binding's read-back (`byte_lm_parallel_export`:
a device copy into a fresh host buffer, an element loop into a Mojo `List`,
then a copy into the caller's memory, once for the state and once for the
gradient). The M4 host (`host_split_mac_m4.txt`, no device) shows the same
host-side shape: 0.52 s to allocate, 0.73 s for the old digests, 0.44 s for
the new, 2.24 s for one core, 1.05 s to memmove 2.59 GB.

## What changed (`tools/lm_segment.py`, commits 9780a0d0f and f8ae397e0)

The digests are unchanged in definition: `_hash_arrays` and `_hash_gradient`
stay as they were and are what the new code is held to.

1. **`Digests`**: all 40 slices (eight per state array, eight of the
   gradient) are hashed at once, one thread and ONE `hashlib` update per
   slice; the slice bounds and the way the hex digests are combined are the
   old ones. hashlib releases the interpreter lock inside a large update.
2. **`Snapshot`**: the post-step state and gradient are read back with the
   binding call `export_raw` and `export_gradients` make, into host buffers
   allocated once and reused (two snapshots, alternating). No `bytes()` of a
   1.95 GB array anywhere; every digest reads the buffers at their address.
3. **`StepPipeline`**: after step N is read back, its digests, its chain line
   (with the `--expect-chain` compare) and a cadence checkpoint are finished
   by one writer thread while the trainer computes step N+1. A snapshot is
   read into only after the job that used it has finished; at most one job
   is pending. Lines are written in step order. When step N disagrees, the
   loop learns it after step N+1 has been computed, stops, never reads back
   or writes N+1, and the chain and segment.json name N as the first
   differing step. Checkpoint uploads go to a second thread in order.
   `--sync-hash` runs the old loop.

The native step (`byte_lm_parallel_step`) keeps the interpreter lock for the
whole step, so each slice's thread must be inside its single update before
the step starts; `Digests` waits for all 40 to start (under 5 ms measured).
The read-back stays between steps: the step is one native call (forward,
backward, fold and AdamW update) and the binding exposes no point between
the fold and the update. Nothing was added to the binding.

What the line's `hash_seconds` means when overlapped: the seconds the step's
line cost the loop outside the step (the read-back, any wait for the
previous line, starting the digests). The digests' own time is in
segment.json `digest_timings`, with the read-back, the wait and the loop's
wall clock from step to step.

## Proofs

**Unit tests** (`tools/tests/test_lm_segment_hash_overlap.py`, 19 tests, CPU):
`Digests` equals `_hash_arrays`/`_hash_gradient` and an independent
sequential definition written in the test, under both schemes, on random
buffers of sizes 0, 1, 3, 7, 8, 9, 15, 17, 1023, 4101, 196615 and 1048589
bytes, on float32/int32 arrays, on all-empty buffers, and on all-zero
buffers with one bit set at either edge of every slice of every array (every
such digest distinct). `Snapshot` makes the export call into reused buffers
and refuses a step mismatch. With a NumPy stand-in trainer, the overlapped
loop writes the same lines (all fields but `seconds`, `hash_seconds`, `prev`)
in the same order as `--sync-hash`, chained; with a disagreement injected at
the first, a middle and the last step both loops stop with that step as the
first differing one, the overlapped loop having computed exactly one step
more (none after the last); checkpoints are the same bytes and manifest
rows, none after a disagreement; job and upload errors reach the main loop.
The existing lm_segment, controls, witness and driver tests pass
(106 passed, 1 skipped; `test_lm_run_driver_hotaisle` has a temp-directory
cleanup race that failed once in eight runs and is unrelated to this change).

**One H100 replay** (RunPod pod `9c37t1ivtbzqi0`, NVIDIA H100 80GB HBM3,
driver 580.178.04, wheel 0.8.18 sha256 `c160fb6d5101...`, Python 3.11, one
device). From route A's `ckpt_00000100.blm` (sha256 `80cd2126a89ba6d8...`)
steps 101 to 103 with `--no-checkpoints --expect-chain` against route A
segment 1's own lines (`h100/expect_chain.jsonl`, from the read-only witness
`A-1.chain.partial.jsonl`), the body rendered by `tools/lm_segment_leg.py
controls --controls "none:none" --wheel 0.8.18 --steps 3` (GETs only) plus
this lane's block. The new runner is this branch's `tools/lm_segment.py`
(sha256 `5f97828ce3d5...`), the old one origin/main's (`7f649b2ddadf...`,
`h100/runners.sha256`), same box, same route/segment/label:

| run | verdict | steps |
|---|---|---|
| new runner (`positive`) | PASS | 101, 102, 103 equal route A's state, gradient, 64 losses, lr bits |
| new runner, `--control none` | PASS | 101 to 103 |
| old runner (`old-positive`) | PASS | 101 to 103 |
| `lm_segment.py compare` new vs old | 15 fields compared, 0 disagreements | |

The two chains' lines are equal byte for byte once `seconds`, `hash_seconds`
and `prev` (the sha256 of the previous line, which covers the seconds) are
left out, with the same key set on every line. Those three fields hold
measured times and differ between any two runs of any runner.

Before and after, steps 102 and 103 (step 101 includes first-call warmup):

| | step seconds | `hash_seconds` | loop wall clock a step |
|---|---|---|---|
| old runner | 38.77, 38.56 | 6.98, 7.15 | 45.75, 45.71 (step + hash) |
| new runner | 38.90, 38.97 | 6.38, 5.64 | 45.28, 44.61 (`wall_since_previous_step`) |

The digests of each step were ready 0.30 to 0.36 s after the writer thread
could look (0.0 s after the next native step returned, in `box_split`:
`overlap_after_step_s`), so the sha256 no longer costs any wall clock. The
steps with the digest threads running read 38.90 and 38.97 s against 38.77 and 38.56 s, 0.1 to 0.4 s longer; three steps a runner cannot tell that from run-to-run spread, so it is not shown to be zero. What remains in `hash_seconds` is the
read-back. In a 3-step run two of the three reads were a snapshot's first
(allocating) read; step 103 was a reuse at 5.63 s, and `box_split` measured
a reuse at 3.96 s, so a long segment should sit between those. Net on this
box: about 1.1 s a step saved of about 45.7 (2.4 percent), not the whole
6.4 to 7 s.

**Checkpoints**: a recipe identical but for `checkpoint_every: 1`, two steps
(101, 102) with each runner, nothing uploaded. Both PASS against route A's
lines; the files are byte-identical: `ckpt_00000101.blm` sha256
`c6e512797785c600...` (1,945,780,167 bytes) and `ckpt_00000102.blm`
`b043a0015be6e0d9...` (1,945,780,168 bytes) from both runners, and the two
manifests' checkpoint rows are identical (`h100/ckpt-old.sha256`,
`h100/ckpt-new.sha256`, `h100/ckpt-*/manifest.tsv`). The overlapped save does
not overlap the next step: `_byte_lm_checkpoint.save` takes the interpreter
lock for every 1 MB chunk, so it runs once the next step returns (step 102
of `ckpt-new` waited 5.6 s for step 101's save). A checkpoint step costs
about what it did before (step 102 with its save: 51.0 s from the previous step against 38.56 + 6.99 + 4.23 = 49.8 s for the old runner); only its upload now runs beside the following steps.

## Not done, and why

- **The read-back itself** (about 4 s steady state here) is the binding's
  `byte_lm_parallel_export`; the runner cannot move it earlier (no hook
  between fold and update) or make it cheaper without a binding change
  (reading straight into the caller's buffer from a pinned staging buffer,
  and reserving the pooled m/v `List`s). That is the remaining cost and the
  next step if it matters; this lane did not touch the binding.
- The two-device and AMD hosts were not measured; the split above is one
  H100 host with SHA-NI. On a host whose sha256 is slower (the plan's AMD VM
  hosts) the digests are a larger share and the overlap saves more.
- The live cross-vendor path (`_run_live`) is unchanged.

## Cost

One RunPod pod, H100 80GB HBM3 at about $3.49 an hour, created 15:11:10 UTC,
deleted at 15:31:22 UTC and confirmed gone (HTTP 404, `h100/teardown.txt`):
about 20 minutes, about $1.20. Two earlier create attempts got "no instances
currently available" and created nothing.

## Files

`host_split.py` and `host_split_mac_m4.txt` (the M4 split), `box_split.py`
(the H100 split, run on the box), and under `h100/`: `status.txt` (the box's
timeline), `box_split.json`, `compare_old_new.txt`, `runners.sha256`,
`positive/`, `none/`, `old-positive/`, `ckpt-new/`, `ckpt-old/` (chain,
segment.json with `digest_timings`, log), the checkpoint digests, the
expected lines, `gpu.txt`, `lscpu.txt`, `wheel.sha256`, `binding.txt`,
`teardown.txt`. The full leg directory stays in
`~/mojolearn-evidence/hash-overlap/leg3/`.
