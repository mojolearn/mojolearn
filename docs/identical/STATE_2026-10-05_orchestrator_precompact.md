# ORCHESTRATOR STATE before compact — 2026-10-05 ~04:10Z (READ FIRST)

## Branch
- **main** = everything (no separate integration needed now): head a36349707+ (rule commits on top of 7c0a1f81c code).
  Contains all lanes: idn speed, fam/fam2, fix-*, review-fixes, compile-fix, cpu2/cpu3/cpu4 (all 8+5), nr-*, ml-*,
  no-bench-tuning 1+2, no-dim-idn, PTX. integration/identical-all-20261004 kept in sync (push HEAD to both).
- NOT on main yet: `lane/box-run-2` compile fixes (6+ batches, e.g. 5e284c314, 368fc68ba KDE fix). Told box-run-2 to merge
  origin/main into its branch, resolve 3 conflicts (core/msel_device.mojo, spectral/.../dense_graph.mojo, x_decomp/kit_device.mojo),
  then fast-forward main to its head and push all further fixes straight to main.
- Local worktree to operate from: ~/mojolearn-wt/identical-all (always `git fetch && git merge --ff-only origin/main` first).

## Subagents in flight
- **box-run-2** (only one running). Brief ~/mojolearn-evidence/briefs-2026-10-04/box-run-2.md. Owns 2 rented boxes:
  RunPod NVIDIA pod aqxujpku19f133 (name mojolearn-dev-brnv-10050104) and DO MI325X droplet 606153646 (mojolearn-br2-amd).
  Harvester PID in ~/mojolearn-evidence/box-run-2/harvester-status.json (disk-checking, renews leases; free ~70 GiB).
  State at 23:53: AMD compile 124/124 green at 5e284c314; NV last round 49/54 (fixes pushed); identity wave running on both
  (br2-wave-368fc68ba). Next it does: identity ON/OFF both boxes -> timing A/B (ON vs MOJOLEARN_IDN_ALL_OFF, 1 warmup+1 measured)
  for cases proven NV==AMD==host -> KMeans acc256 candidate on NV -> verdicts in docs/identical/optimization-ledger.json AND
  main board -> delete boxes + verify. Projection from pace: ~3-5 h from 23:53.
- No other subagents. Earlier ones all finished (reports in ~/mojolearn-evidence/fam/, cpu3-*/, cpu4-*, no-dim-idn/).

## Next actions, in order
1. When box-run-2's fixes are on main: launch lane **py-runtime** (brief ready: ~/mojolearn-evidence/briefs-2026-10-04/py-runtime.md;
   worktree ~/mojolearn-wt/py-runtime from origin/main; 1,146 py-compute baseline rows + checker --owed 109).
2. Freeze one commit (main after box-run-2). Compile cheap first (owner process). Measure IDENTICAL on Apple M3 for the board
   (queue via ~/mq/opp_to_end.py on M3 behind existing jobs; never release/extend/add Apple machines; M2 = compile checks only,
   queue ~/m2-arms/bq/). Apple never votes on IDENTICAL switches.
3. Apply every finding to the main board (board tools only; Apple board moving to board.json + tools/af_board_render.py by peer).
4. After ALL A/B testing: full IDENTICAL board on NVIDIA, AMD, Apple (one run per race, opponents from store). Then PTX final batch.
5. Remaining named CPU items (owed list on main via `python3 tools/hooks/no_host_routes.py --owed`): GBDT device-resident model,
   multi-output walker, lossguide replay stats, leaf partition bookkeeping, train target borders/CTR tables; ET Apple algo-L
   sampler raises (needs float64-free device sampler) — FIX (standing order); AutoARIMA resident series.
6. Run `bash ~/mojolearn-evidence/apple_watch.sh` periodically (syncs M2/M3 bare repos, flags problems).

## Owner rules added today (all in AGENTS.md/CLAUDE.md on main + memory)
- No dimension targeting in any mode (IDENTICAL too; bits may change across versions).
- Measurement process: freeze per A/B round; compile cheap; IDENTICAL switches by NV+AMD (combined faster, neither materially
  slower); measure IDENTICAL on the 3 and update ALL boards; FAST not rerun; FIX problems when found (standing order); grep logs.
- No Python in the runtime; Python is good for the external glue layer.
- Wait for Mojo/Modular instead of workarounds (AMD portable path parked on lane/amd-portable).
- No worktree pruning without asking; never with untracked files. Merge finished work to main promptly.
- Scope: peer finished AutoARIMA + all Apple FAST A/Bs; I do NV/AMD A/B, M3 IDENTICAL measurement, main board.
- Owner decisions still pending (don't decide): L14/K7/X14/L8 (ML roadmap), and the Apple FAST handoff list
  (~/mojolearn-evidence/HANDOFF_APPLE_FAST_TO_IDENTICAL_2026-10-05.md "Decisions waiting on Andrew").

## Lessons from today
- Harvesters died on a full Mac disk at 20:25Z Oct 4; boxes self-deleted; results lost. Watch harvester PID + disk.
- Main merges can bring in Apple FAST host work without the pre-push guard; re-run `--owed` after each main merge.
