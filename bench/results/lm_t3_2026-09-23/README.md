# T3: the GPT-3 Small six-segment run, 2026-09-23

The run of `docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md`: a GPT-3 Small shape decoder
(162,147,840 parameters, batch 4, length 2048, vocabulary 50,257, K=64 logical
shards of 8,192 tokens an optimizer step) trained for 5,000 steps over the
pinned FineWeb-Edu id stream in six segments handed between NVIDIA and AMD
boxes, the whole run performed twice by different vendor routes (A and B),
route B held to route A's chain step for step. Driven by
`tools/lm_run_driver.py` from `spec.json` with `recipe.json` (hash scheme
`sliced-sha256-8.v2`). Every box installs the published wheel; nothing is
built on a box. This directory grows as segments land
(`tools/lm_file_evidence.py`); each chain is filed as `chain.summary.tsv` (one
row a step: the learning-rate bits, the mean loss, the seconds, the hash
seconds, the state and gradient digests, a digest of the 64 per-shard losses)
beside `chain.sha256`; the full chains and the checkpoints are in R2 under
`runs/t3/2026-09-22/`.

| segment | box | steps | s per step (median) | hash s | arrival replay | checkpoints |
|---|---|---|---|---|---|---|
| A/1 | NVIDIA H100 80GB HBM3 x2 (RunPod, wheel 0.8.15, commit 30076479a) | 0 to 1000 | 19.5 | 6.7 | none (the seed) | 0, 100, 200, ..., 900, 998, 1000 |
| A/2 | NVIDIA H100 80GB HBM3 x2 (RunPod, another pod, wheel 0.8.15, commit cccf58415) | 1000 to 2000 | 19.5 | 6.6 | PASS (steps 999, 1000 from ckpt 998) | 1100, ..., 1900, 1998, 2000 |
| A/3 (nvidia, coordinator) | NVIDIA H100 80GB HBM3 x1 (RunPod tymq1131yvbhoy, wheel 0.8.17, Python 3.11) | 2000 to 2400 | 101.2 | 4.0 | PASS (steps 1999, 2000 from ckpt 1998) | 2100, 2200, 2300, 2398, 2400 |
| A/3 (amd, worker) | AMD Instinct MI325X x1 (DigitalOcean 603274624, wheel 0.8.17, Python 3.12) | 2000 to 2400 | 103.6 | 1.6 | worker | the coordinator's |

## A/1

Two H100s on one RunPod pod, Python 3.11, the published wheel mojolearn 0.8.15
(`A-1/wheel.sha256`), 1000 optimizer steps from the drawn seed
(`ckpt_00000000.blm`, its sha256 in `ledger.json`). Mean loss over the 64
shards: 10.951 at step 1, 3.872 at step 1000. Median 19.5 s an optimizer step
on two devices and 6.7 s hashing the state and the summed gradient; eleven
1.95 GB checkpoints saved and pushed to R2 (median upload 125 s). Wall clock
17:51 to 01:36 UTC (7 h 44 min); the pod was deleted and the API confirmed it
gone at 01:37 UTC (`A-1/leg.txt`, the leg's teardown record in
`~/mojolearn-evidence/gpt3-run/t3/legs/A-1/`). Verdict PASS, no disagreements
(a first segment is held to nothing; every later segment of route A replays
its last two steps on arrival, and every segment of route B is held to A's
chain).

## A/2

Two H100s on a different RunPod pod (other GPU UUIDs, `A-2/gpu.txt`), the
same wheel. On arrival it fetched `ckpt_00000998.blm`, replayed steps 999
and 1000 on its own hardware and landed on A/1's chain lines bit for bit
(`A-2/arrival/`, verdict PASS), then trained steps 1001 to 2000 from
`ckpt_00001000.blm`. Mean loss 3.928 at step 1001, 3.554 at step 2000.
Median 19.5 s a step and 6.6 s hashing; eleven checkpoints pushed. Wall
clock 01:58 to 09:53 UTC; the pod was confirmed gone at 09:55 UTC. Verdict
PASS, no disagreements. The driver then re-read the spec and started A/3,
the live segment, on the published 0.8.17 wheel (the Python-only release
that fixes the live worker's array reads on Python 3.10 and 3.11; every
binding in it is the 0.8.15 bytes, see
`bench/results/lm_t3_negative_controls_2026-09-23/wheel-0.8.17/`).

## A/3, the live cross-vendor segment

One NVIDIA box and one AMD box trained the same 400 optimizer steps together
through `mojolearn.cross_vendor` over an ssh tunnel, both from the published
0.8.17 wheel (the Python-only release whose bindings are the 0.8.15 bytes),
the H100 as coordinator computing shards 0 to 43 and folding first, the
MI325X as worker computing shards 44 to 63. The H100 first fetched
`ckpt_00001998.blm`, replayed steps 1999 and 2000 on its own hardware and
landed on A/2's chain lines bit for bit (`A-3-nvidia/arrival/`). Then, at
every one of the 400 steps, the two boxes wrote the same state digest, the
same gradient digest, the same 64 losses and the same learning-rate bits
(`A-3-nvidia/segment/chain.summary.tsv` and `A-3-amd/segment/chain.summary.tsv`
agree line for line; the worker's own verdict PASS is recorded in the
ledger). Mean loss 3.552 at step 2001, 3.583 at step 2400. Median 101 s a
step on the H100 side and 104 s on the MI325X side; that pace is the
exchange between the boxes, not either box's arithmetic (T1 measured 40 s
for 64 shards on one H100 and 139 s on one MI325X). Wall clock 10:03 to
22:01 UTC; the pod and the droplet were confirmed gone by 22:04 UTC.
Cost about $88. After A/3 the run paused on Andrew's instruction: no AMD
segment runs before the next PyPI release carries the AMD step-time work
(`bench/results/amd_step_time_2026-09-24/`), so A/4 and every route B
segment that needs AMD are held in `spec.json`.

## Files

`spec.json`, `recipe.json`, `ledger.json`, `driver.log` (every launch of the
driver, including the halts that preceded the relaunches), and per box
`status.txt`, `leg.txt`, `gpu.txt`, `uname.txt`, `binding.txt`,
`commit.txt`, `wheel.sha256`, `checkpoints.sha256`, `uploads.json`,
`segment/{segment.json,manifest.tsv,log.txt,chain.summary.tsv,chain.sha256}`
and `arrival/...` where a replay ran. The full leg directories stay in
`~/mojolearn-evidence/gpt3-run/t3/`.
