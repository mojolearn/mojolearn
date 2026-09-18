# Handoff: the 2026-09-17 verification and speed fan-out

Written for a session with NO CONTEXT, at Andrew's request, because the Claude
Code session driving this fan-out may run out of credits. Read this file first.

**Nothing here is blocked on the session surviving.** Rented pods self-delete,
merged work is on main, and each running lane has been told to commit, push and
leave a `docs/lanes/LANE_STATUS_<lane>.md` on its own branch. Read those next.

## 1. What is safe if everything dies right now

- **Rented pods self-terminate.** `tools/runpod_cpu_leg.sh` `write_deadman()`
  composes a script that runs ON THE POD and DELETEs the pod through the RunPod
  API when its timer fires. The orphan case is the laptop going away, so the
  guard does not live on the laptop. No runaway billing.
- **`tools/runpod_guard.sh list` IS NOT A VIEW OF THE FLEET.** On 2026-09-17 it
  printed 14 EXPIRED leases and not one matched a live pod, while the API showed
  14 pods RUNNING at $12.68/hr. The lease files are stale artifacts. The only
  truthful source is:

      export MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key
      export RUNPOD_API_KEY=$(cat "$MOJOLEARN_RUNPOD_KEY_FILE")
      curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods

  Pod `name` carries the owning lane. On 2026-09-17, SEVEN of the fourteen
  belonged to EARLIER sessions' worktrees (python-hotpath, cpu-complete-dep,
  knn-selector-speed, forest-train-speed, kmeans-linear-speed, gbdt-train-speed,
  gbdt-resident), about $4.00/hr. Do not reap those; they are owed to work this
  fan-out knows nothing about.
- **Resuming a killed leg is cheap**, because the expensive inputs are in R2:
  datasets and models pinned in the store, bindings under `bincache/v1`,
  sabotage builds under `bincache/sabotage-v1` (cold 302 s, warm 180 s), and as
  of `744f44945` the opponent wheels too. What is lost is only the compute since
  each pod's last partial upload.

## 2. Merged to main already

| commit | what |
|---|---|
| `744f44945` | r2-opponent-hygiene. Opponent installs now opt-in behind `--opponents` (6 to 8 min back on a forest leg); 71 opponent wheels, 2.47 GB, mirrored into R2 with size+sha256 pins; `cuvs-cu12==26.8.0` corrected to 26.8.1 because that version never existed |
| `712eedd16` | compare-commit-reveal. `verify --commitment` seals a report with a nonce; `verify --compare --commitment-a/-b` verifies it. Closes copy-after-seeing |
| DEVIATION 2900 | attention-speed. `_bswz` causal block swizzle, geomean 0.9592, lean step 0.2065 to **0.1979 s**. Pod terminated, worktree removed |

## 3. Lanes still open, with their branch

Each has been told to leave `docs/lanes/LANE_STATUS_<name>.md` on its branch.

| branch | owns | state at pause |
|---|---|---|
| `lane/reference-regen` | `verify_reference/table.json`, `docs/VERIFY_EXTERNALLY.md` | 3 of 6 legs home, all clean; trial rebuild already moves stale lanes 5 to 0, admission policy off `legacy`, `stepfull` 4 to 1,317 parts, zero-hash entries 28 to 17. Light leg 2 and an NVIDIA par-lane leg still out |
| `lane/sabotage-sweep` | `bench/results/identity_break/` | no report yet. Target: 54 of 229 lanes have a sabotage SEEN to move a build |
| `lane/gemm-next` | `gemm/`, `core/gemm.mojo` | just started. AMD authorized |
| `lane/lm-training-shakedown` | training probes | no report yet |
| `lane/lm-step-memory-build` | `training/byte_lm.mojo`, `training/checks/loss.mojo`, `_byte_lm_impl.py` | just started |
| `lane/tri-vendor-handoff-plan` | one doc, `PLAN_tri_vendor_handoff_training_2026-09-17.md` | complete, unmerged, safe to merge |

## 4. Findings that cost real money to rediscover

- **The step itemization ALREADY EXISTS.** `core/step_phase.mojo` compiles its
  ticks behind a SECOND build define, `-D MOJOLEARN_STEP_PHASE_TIMERS=1`, which
  is why it looks absent. `tools/step_breakdown_leg.sh`,
  `tools/step_breakdown_summary.py` and a committed `breakdown.tsv` under
  `bench/results/e1g/2026-09-11_190725-nvidia-h100-80gb-hbm3-step-breakdown/`
  are the instrument and one result. It needs a RE-RUN, never a rebuild.
- **At a 299.78 ms envelope: GEMM 144.90 ms (48.3%), attention 116.23 ms
  (38.8%), everything else about 13%.** Projected after both flips, the current
  step is roughly GEMM 58%, attention 29%. Halving GEMM buys 29% of the step;
  halving attention buys 15%.
- **GEMM has no hotspot.** Thirteen `gemm.<kind>` leaves with NINE inside 13.0
  to 14.4 ms. Broad and flat, not one peak.
- **The GEMM ceiling is 33.5 TFLOP/s and is PERMANENT.** The identity seam is
  two issued instructions; the three vendors' native FMAs disagree, so no
  single-instruction contract exists. Shipped 11.4 is 34% of the real ceiling.
  A rewrite is a 2x, not a 4x.
- **Nothing transfers between vendors.** The same attention arm read 0.8207 on
  one column and 0.9716 on the other. `kpack_hg` shipped NVIDIA-only; Apple and
  AMD kernel-matrix rows are still 0.
- **Gradient accumulation is NOT missing**, despite an archived doc saying so.
  `accumulate_grads` / `accumulation_is_aligned` at `_training_impl.py:1667,1677`,
  exported, used, tested, and verified as identity_break's `batchgrad` part.
- **The capacity json OVERSTATES.** It says 22.33 GiB for batch 1 where the
  measured device peak was 12.14 GB, so its 161.75 GiB for batch 8 is a loose
  analytic bound, not a measurement.
- **The LM memory fixes are already designed, ranked and sized** in
  `BRIEF_lm_step_memory_2026-09-10.md` lines ~276-315, each with its
  bitwise-identity argument and a named gate. Order: rank 3, then 7, then 2.

## 5. Open, unowned, and NOT to be turned into lanes

Andrew reinstated "no new lanes" on 2026-09-17. These are parked deliberately:

- the challenge-nonce design for `--compare`, the only thing that closes
  precompute-from-our-own-table (a party can synthesize a document from
  `verify_reference/table.json` and commit to it without running anything)
- the `|| true` that SWALLOWS R2 staging failures in `do_extra_leg.sh:1175` and
  `hotaisle_leg.sh:2502`; `MOJOLEARN_STAGE_STRICT=1` exists and should be used
- `release_qualified`, hard-coded `False` in two files with three tests
  asserting it. It is a claim about a four-column release record including
  Apple, so it is Andrew's decision, not a lane
- `test_verify_all.py::test_shipped_verifier_hashes_like_the_harness` FAILS on
  main today on `refuse_routine_apple_column` (24 Apple lanes)
- ten call sites still re-measure opponents that already have rows in
  `bench/OPPONENT_REFERENCE.md`, three of them added 2026-09-17 under headings
  saying they were added so the rows would be looked up
- `MOJOLEARN_KNN_SELECTION_CACHED_OPPONENT_HIGGS` is named in the opponent table
  but NO FILE READS IT; only the unsuffixed, `_TAXI` and `_ISTELLA` forms exist
- the Apple GEMM seam: `BRIEF_gemm_kernel_2026-09-11.md` section 14.3 measured
  the SHIPPED Apple seam as flush-before-round, not the contract's
  round-then-flush, at all 315 boundary triples. Called "a correctness item for
  Andrew" and no commit since addresses it. `lane/gemm-next` was told to confirm
  whether it is still live

## 6. To resume

1. Read each open lane's `LANE_STATUS` file on its branch.
2. Check the live fleet with the API call in section 1, not with the guard.
3. Merge `lane/tri-vendor-handoff-plan`; it is a finished doc.
4. Do not open anything new without asking Andrew.
