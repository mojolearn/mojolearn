# Apple M3 and M2 every-lane sweep (lane/m3-sweep, 2026-09-28)

Most identity lanes that shipped before 2026-09-27 were recorded on one Apple
machine only, an M4. This sweep runs every exposed identity lane once on an
M3 (m3ultra-b, released ~21:20Z today) and on an M2 (m2pro, kept, billed
hourly, no release date yet). No Mac was added, extended or rented.

## What runs

- Lanes: every lane `tools/lane_accounting.py --json` reports as exposed,
  minus the `par-*` two-device lanes. At main 17a8b3c1c that is 485 lanes,
  426 without `par-*`.
- Check: `tools/apple_sweep_run.py` (this branch). It runs the CLEAN stage of
  the one lane check (`tools/algos_lane_check.py`) lane by lane: build what is
  stale (from the steward's build store when the sources match), fit on
  Metal, fit on the CPU host bindings, diff with `identity_break.py --diff`.
  No sabotage and no seam drivers: this sweep checks agreement only. A
  failure never ends a batch. Each lane gets one verdict: AGREE, DISAGREE,
  NOTHING COMPARED, BUILD FAIL, ARM FAIL, ERROR or NOT RUN.
- Submission: `tools/apple_steward.py submit --kind speed --target <mac>`.
  Identity requests cannot be pinned to one Mac. They route one copy per
  generation, plus do-amd, and always carry a sabotage. A speed job can be
  pinned to one Mac and runs alone on its GPU. Batches have 36 lanes. No
  lane starts after 5400 s of its batch, well inside the 3 h cap on a speed
  command. On m3ultra-b, no lane starts after 21:00Z.
- Order: first the lanes with no PASS identity verdict on that generation's
  stewards (m3ultra or m3ultra-b for M3; m2pro for M2), in registration
  order (oldest first). Then the rest. M3: 173 lanes never checked, 253
  checked before. M2: 184 never checked, 242 checked before.
- Priority: the steward has none. Its queue is FIFO by submit time.
- Evidence on each Mac: `~/mojolearn-evidence/apple-sweep/<mac>-<batch>/`
  (`results.jsonl`, and a `lane_check.log` for each lane) and
  `~/mojolearn-evidence/apple-steward/done/<request>/speed.stdout`.

## M3 (m3ultra-b)

### DISAGREE

(none yet)

### Build failures and other non-verdicts

(none yet)

### Finished before the release

(in progress)

## M2 (m2pro)

### DISAGREE

(none yet)

### Build failures and other non-verdicts

(none yet)

### Finished

(in progress)
