# Stop state, 2026-09-15 about 10:25 ET (14:25 UTC)

Andrew stopped every lane and asked for a restart plan and a different path. This file is the ground truth at the
stop. It supersedes the status lines in `RESTART_PLAN_2026-09-15.md`; the rules there, and in
`FANOUT_RULES_2026-09-15.md`, still hold, especially rule 00: **no GPU boxes between PyPI releases, one AMD, one
NVIDIA and one Apple column at a release.**

## 1. What was stopped and what is left running

- **Agents:** all eleven of this session's subagents were stopped.
- **Remote boxes:** none are running, checked at the stop through the APIs.
  - RunPod pods: 0.
  - DigitalOcean droplets: 0.
  - Hot Aisle VMs: 0, balance $19.32.
- **Local work:** no leg drivers, identity runs or Mojo builds were left from this session.
- **NOT stopped (other sessions on the Mac, not this session's to kill):**
  - Three other `claude` processes, about 13.5 h old.
  - Two `codex` processes, one about 13.5 h old and one about 30 min old.
  - The young Codex owns the worktree `/private/tmp/mojolearn-routine-speed`, branch
    `lane/cpu-training-routine-speed`, which has one uncommitted doc edit. That edit is saved as
    `~/mojolearn-evidence/session-stop-2026-09-15/codex-routine-speed-uncommitted.patch`.
- **CI:** queued CPU identity gate runs were left queued, since a restart can read their results.
- **Saved at the stop:**
  - Every lane branch's work is on origin. Uncommitted evidence was committed with `[skip ci]` or moved to
    `~/mojolearn-evidence/<lane>/`.
  - The throwaway `synthetic` commit and `tools/_plan_tool.py` from a detached test worktree are in
    `~/mojolearn-evidence/session-stop-2026-09-15/plantest/`.

## 2. Every unmerged lane at the stop (main = 1edd3054f)

Each branch carries `docs/lanes/LANE_STATUS_<branch-with-dashes>.md` with its own next commands, except the three
gate-speed branches. Read that file first.

| Lane | Branch, head | Ahead / behind main | Evidence so far | Last CPU gate | Next step |
|---|---|---|---|---|---|
| gbdtord: GBDT ordered-rmse, feature-freq, pointwise, categorical-ctr on CPU | `lane/cpu-training-gbdt-ordered` b5c8461aa | 11 / 3 | Four lanes IDENTICAL x4 locally and on gate 34936445589. The merge gate timed out at 60 min, so 3b3ab79ad raises the job timeout to 120. | 34972329290 in progress at 3b3ab79ad | Merge when green, or after the gate speed fix lands |
| cputransformer: Transformer blocks on CPU | `lane/cpu-training-transformer` 90419f41d | 4 / 5 | M4: IDENTICAL=18 train, 18 infer, 18 batch; sabotage DIVERGENT=18. On gate 34969598898 the covered step passed on all seven runners | 34969598898 cancelled at the 60-min job limit on four runners | Re-gate after the gate speed fix, then merge |
| cpumamba: Mamba 1/2/3 and 2-dtlimit on CPU, plus the Mamba-1 oracle fix | `lane/cpu-training-mamba` 9bf64b494 | 3 / 3 | M4: IDENTICAL=36 train, 36 infer, 36 batch; sabotage DIVERGENT=36. The Mamba-1 host backward oracle was wrong (three rejected plan readings); its new check FAILs 151 before the fix and PASSes after | 34973248141 in progress at ddd875cc8 | Re-gate the head, then merge |
| cpusamba: Samba stack on CPU | `lane/cpu-training-samba` 0a1b30c7d | 11 / 2 | M4: IDENTICAL=18 train, 36 infer/model, 18 batch; sabotage DIVERGENT. Built on the mamba and transformer branches | 34975751193 queued | Merge AFTER cputransformer and cpumamba |
| cpuembed: Embedding, IVF, byte-LM GPU lanes, tokenizer on CPU | `lane/cpu-training-embedding-ivf` a84ff049b | 5 / 3 | Fresh-build local gate at bb62eaeb1 on the M4; details in LANE_STATUS | none | Push for gate, then merge |
| rlparity: `rlpair` harness part (sampler log-probs vs trainer log-probs) | `lane/rl-logprob-parity` 8c27a7771 | 5 / 0 | rlpair IDENTICAL=108 across Apple M4 Metal, one RunPod H100 (rented before the rule arrived; pod verified deleted) and an M4 CPU probe; sabotage RLPAIR_MOVED; 21/21 negative controls. Also adds `SambaStack.allocate_state/forward(state)/step` | 34980556800 queued at 7ae35df48 | Merge when green |
| batch2: `batchgrad`, `batchscale`, `ragged` parts plus the new `lengths=` argument | `lane/batch-invariance-2` d4a3cb79b | 6 / 3 | `lengths=` on the Transformer, Mamba, Samba and byte LM paths, with contract text; test_ragged_lengths 20 pass on Metal. The record dir 2026-09-15_batch2 was committed unreviewed at the stop | 34979837191 queued at 1cc7a2f47 | Review the evidence README, gate, merge |
| bincache: prebuilt bindings in R2 | `lane/r2-binding-cache` cbfed9823 | 2 / 0 | Implemented, default OFF, unit tests and sabotage arms, a live R2 end-to-end run with fake bindings | none | Gate, merge; the box proof is owed to the next release |
| gatehyg, part 1: test isolation plus a time budget | `lane/cpu-training-gate-budget` 36f9c1b27 | 2 / 3 | test_byte_lm_host fixture isolation fixed (3ddf1f3c9 is on this line) | 34971932337 success | Merge |
| gatehyg, part 2: gate speed | `lane/cpu-training-gate-speed` fcd773ea9 | 3 / 3 | WIP only; its commit message is a placeholder | 34981486278 queued | Reconcile with Codex's routine-speed branch below; do not merge the WIP message as is |
| Codex, another session: routine gate speed | `lane/cpu-training-routine-speed` 08b50887a | 3 / 0 | Routine gate cut from seven runners to three, suites sharded across up to four workers, sabotage run once, environment cached | 34978769155 queued | Pick ONE of this or gate-speed |
| record2: the 178-lane record, PARTIAL, stopped | `lane/identity-record-next` 925d1e6a8 | 3 / 5 | Three one-device columns complete and IDENTICAL on 1602 cells; only kmeans-sqrt moved vs the 166-lane record, from the fix. The gate switch is saved as a patch, not applied. Finding: several processes sharing one MI325X gave a MOVED cell and a refusal | none | Nothing, unless a release wants these columns. Do not restart |

## 3. Merge order for whoever restarts

1. **Gate speed first, since everything else waits on the gate.**
   - Merge `lane/cpu-training-gate-budget`.
   - Choose between `lane/cpu-training-routine-speed` (Codex, complete message) and `lane/cpu-training-gate-speed`
     (WIP). Merge one; close the other.
2. `lane/cpu-training-gbdt-ordered`: its 120-minute timeout may be moot after step 1; keep whichever the merged gate needs.
3. `lane/cpu-training-transformer`, then `lane/cpu-training-mamba`, then `lane/cpu-training-samba`.
4. `lane/cpu-training-embedding-ivf`, `lane/rl-logprob-parity`, `lane/batch-invariance-2`. All three edit
   `tools/identity_break.py`: merge main before each.
5. `lane/r2-binding-cache` (default off).
6. Remove each lane's worktree after its merge.

## 4. Why this session got slow, for choosing a different path

- **The CPU identity gate grew from 6 minutes (Sep 14 morning) to 56 minutes per runner job, and 2 to 3.5 hours
  from push to result.**
  - Covered CPU lanes went from about 38 to about 110.
  - Every push ran every lane × 9 fixtures × 2 repeats × the 20-call batch protocol, then all of it again under
    sabotage, on 7 runners, on every branch and again on main.
  - Nothing was scoped, parallel or cached, and 9 concurrent lanes queued gates behind each other.
- **GPU legs became the default evidence:** full re-records, two AMD models, extra NVIDIA architectures,
  two-device columns. The 178-lane record alone took 7.4 GPU hours. That is the reason for rule 00.
- **Too many concurrent lanes.** Nine to eleven lanes each needed a gate slot, and several edited the same files
  (host_surface.py, identity_break.py, the gate workflow), so merges re-gated repeatedly.
- **Stale worktrees**, each 3 to 5 GB, filled the disk to 4.2 GiB free before cleanup.
- **The point of the CPU work was cheap verification:** one GPU record, then free CPU runners check it. The
  session drifted from that.
