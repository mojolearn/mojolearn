# Handoff: the GPT-3 Small six-segment training run (T3)

Written 2026-09-23 for the session that runs T3. Read this, then
`docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md` (the plan; sections 3, 4, 6 and 9 most).
Andrew authorized T2 and T3 on 2026-09-22 ("proceed with T2 when T1c lands and
then do t3 you are authorized"). Rent as needed under the caps in the spec.

## What T3 is, in one paragraph

The real run. A GPT-3 Small shape decoder (162,147,840 parameters, batch 4,
length 2048, vocabulary 50,257) trained for 5,000 optimizer steps of 64
shards (524,288 tokens a step, 2.6B tokens) over the pinned FineWeb-Edu id
stream, in six segments handed between NVIDIA and AMD boxes, one segment
(the third) trained by an NVIDIA box and an AMD box together, the whole run
performed twice by different vendor routes (A and B), route B's segments
starting from route A's checkpoints and held to route A's chain step for
step. Apple is deferred (Andrew: later); route A's segment 6 runs on NVIDIA
(the plan's fallback). The claim is that every optimizer step of both routes
is the same bits on different hardware, evidenced by 10,000 hash-chained
step records, checkpoints in R2 every 100 steps, an arrival replay at every
handoff, and negative controls that were seen to fail.

## State at handoff

- **Main** holds everything; the branch `lane/gpt3-run-dev` is merged. The
  published wheel **mojolearn 0.8.15** (PyPI, manylinux sha256
  2816775838...) carries the four package changes the run needs: `set_lr`,
  `export_raw`, the chained coordinator, the device fold; bindings for
  sm_89, sm_90a and gfx942. The run's boxes install it (`"wheel": "0.8.15"`
  in the spec); no build happens on a box.
- **Evidence so far** (all under `bench/results/`): `lm_segment_t0_2026-09-22`
  (the tooling on the M4), `fineweb_tokens_2026-09-22` (the stream, identical
  on three CPUs), `lm_t1_2026-09-22/{nvidia,amd,live}` (T1: the measured
  steps, the cross-vendor passes at the target shape, the live pair).
- **T2** (the rehearsal, 3 segments of 10 steps on the real stream, driver
  output in `~/mojolearn-evidence/gpt3-run/t2/`): segments 1 (H100) and 2
  (MI325X, arrival replay PASS) landed. Segment 3 (the live pair) failed
  three times, every time on orchestration or infrastructure and never on
  the arithmetic (T1c proved the live arithmetic): an ordering bug, a stalled
  12 GB staging transfer, and a hash-scheme mismatch from code that changed
  under the driver. All three are fixed on main. A fourth attempt was
  running at handoff; check `~/mojolearn-evidence/gpt3-run/t2/driver.log`.
  **If it passed, T2 is done. If it failed, read the cause before T3; the
  plan's fallback for segment 3 is one vendor per route (edit the T3 spec:
  vendor "nvidia" in A, "amd" in B), which keeps every constraint.**
- **Not yet done, both cheap:** the 8-device MI325X check (a one-box segment
  of 3 steps on `gpu-mi325x8-2048gb` with `amd_devices` `0,...,7`, held to the
  T1 NVIDIA chain: `runs/t1/2026-09-22/nvidia/chain.jsonl`, from checkpoint
  `runs/t1/2026-09-22/nvidia/ckpt_00000000.blm` sha 5871ef0f...; about $8),
  which T3's AMD segments assume, and a rehearsal of the T3 spec itself with
  `steps` set to 10 for one segment per vendor if anything above changed.

## How to run it

Everything is driven from a CLEAN worktree at main (a dirty tracked file
refuses every rental by name). Never develop in the driver's worktree; use
another one and merge to main, and never merge into the driver's worktree
mid-run (the recipe pins the hash scheme now, but the runner archives HEAD).

    cd ~/mojolearn-wt/gpt3-tooling            # or a fresh `git worktree add` at main
    git merge --ff-only main
    PYTHONPATH=python ~/CascadeProjects/mojolearn/.pixi/envs/test/bin/python \
      tools/lm_run_driver.py plan --spec ~/mojolearn-evidence/gpt3-run/t3_spec.json
    PYTHONPATH=python ~/CascadeProjects/mojolearn/.pixi/envs/test/bin/python \
      tools/lm_run_driver.py run --spec ~/mojolearn-evidence/gpt3-run/t3_spec.json \
      --out ~/mojolearn-evidence/gpt3-run/t3 --parallel 2 > ~/mojolearn-evidence/gpt3-run/t3/driver.out 2>&1 &

- The spec is `~/mojolearn-evidence/gpt3-run/t3_spec.json`; the recipe
  `~/mojolearn-evidence/gpt3-run/t3_recipe.json` is in R2 at
  `runs/t3/2026-09-22/recipe.json` (hash scheme `sliced-sha256-8.v2`, 5,000
  steps, warmup 250, boundaries 1000/2000/2400/3900/4900/5000). Re-upload it
  if you change it (`sh tools/dataset_store.sh presign-put KEY 3600`, then
  curl -T).
- The driver is resumable: rerun the same command and it skips landed
  segments (`ledger.json`). It halts on a FAIL verdict (the halt rule) and
  on a missing result; read `driver.log`, the leg's `status.txt` and the
  segment's `segment.json` disagreements before rerunning.
- `--parallel 2` runs one NVIDIA and one AMD segment at once; a live segment
  runs alone. AMD is the choke point: one DigitalOcean GPU droplet per
  account (other sessions use it too), RunPod MI300X and Hot Aisle often
  without stock; the driver walks providers and waits (`amd_wait_minutes`).
- Boxes: NVIDIA on RunPod via `tools/gemm_remote_leg.sh` (a GPU-type walk),
  AMD on DigitalOcean via `tools/do_extra_leg.sh` (the 8x MI325X size in the
  spec, $30.40 an hour; the segment lease is priced live against the
  per-segment `dollar_cap`). Every rental has an on-box watchdog and a local
  dead-man; the runners fetch and verify the delete. Never reap another
  session's box.
- Live segments: `tools/lm_live_leg.sh` rents AMD first, then NVIDIA, joins
  them by an ssh tunnel on the AMD box, and stops the survivor if one side
  ends. Bodies fetch the token parts themselves (eight at once, timed).

## What to watch and what to report

- Each landed segment: steps, seconds per step (chain `seconds`), hash
  seconds, checkpoint save and upload seconds, the arrival replay verdict,
  and for route B the boundary checkpoint equal to route A's (the driver
  checks it). Spend per leg is in each runner's log.
- Any DISAGREE: stop, name the column that is alone (the arrival replay and
  a CPU witness of one shard: `python -m mojolearn.cross_vendor` is not it;
  see plan section 6), and tell Andrew before anything is rerun.
- File evidence under `bench/results/lm_t3_<date>/` as the T1 directories
  are filed (small files only; checkpoints stay in R2), commit on a branch,
  merge to main when verified, push (`push-local-main-always`).
- Cost expectation from measured steps: NVIDIA about 45 GPU-hours, AMD about
  160 GPU-hours (20 hours on the 8-GPU droplet at $30.40), live about 24 box
  hours; about $1,000 in all. Wall clock about three days with `--parallel 2`.

## Rules that bit this work

- The recipe names its token stream and its hash scheme; stage and compare
  only what it names (`recipe-names-its-token-stream`).
- A dirty tree refuses a rental; a change that reaches a running run's
  worktree changes the run. Two worktrees, always.
- Every rental is capped, self-terminating, and verified gone; a box that
  cannot be reached is not "gone" until the API says 404.
- Never say we are faster; no price of determinism; report the numbers.
