# E1G — the cross-vendor GEMM identity run (Apple, NVIDIA, AMD)

**STATUS, 2026-08-23: THIS LEG HAS NOT BEEN RUN.** Not once, not partially.
`tools/gemm_remote_leg.sh` has had a `sh -n` syntax check and nothing more:
not even its dry run has executed, no pod has been created, no remote card
exists, and no line below has been exercised against a real NVIDIA or AMD
device.
It cannot be run from this machine today: the session that wrote it has no
RunPod access, and the orchestrator's standing order for this round is
Apple-M4-only construction with no rented GPU. Every result in `gemm/` is
therefore CONSTRUCTION plus one Apple device's gates; no second vendor has
run this. When the leg does run, replace this paragraph with the result
directory and the verdict -- do not append to it.

DEVIATION 536. The operator's document for `tools/gemm_remote_leg.sh`.
`E1_RUNBOOK.md` is the sibling for the tree-ensemble and unsupervised legs
and this is deliberately the same shape.

**The claim under test.** The same commit, built from one source under
`NUMERIC_IDENTICAL`, running profile `mojolearn.identical.gemm.fp32.v1`
(`gemm/IDENTICAL_FP32_CONTRACT.md`), produces a **byte-identical card** --
the same FNV-1a64 hash at every stage, for every shape -- on an Apple M4
(Metal), on an NVIDIA GPU (CUDA) and on an AMD GPU (HIP/CDNA).

Everything else this lane needs is finished on one desk. The contract, both
oracles, the device kernel, its sabotage gates, the column sweep and the
price harness all run on the Mac. **The three-vendor run is the only thing
left in this lane that a Mac cannot answer.**

---

## Why the Mac cannot answer it, in one paragraph

`tools/gemm_column_invariance.sh` compiles the APPLE, NVIDIA and AMD vendor
**columns** onto **one backend**. It answers "does the choice of vendor
column change the product", which is most of what our own source can get
wrong, and it is worth running before every leg because every defect it
catches is a defect the rented box would have found more slowly and more
expensively.

**It cannot see contraction (IDENTITY_PATHS row 9), denormal policy, or
`sqrt` rounding, because those are properties of the BACKEND and not of the
column.** The standing proof that a backend can be wrong in a way only its
own silicon shows is NVIDIA's `sqrt`, which is not correctly rounded --
180,714 of 2^20 patterns disagree with the correctly-rounded result, 176,577
of them on normals. No amount of green from the column sweep would ever have
found that. That is why this leg exists, and it is why a green column sweep
must never be quoted as a cross-vendor result.

---

## Preconditions

- **A clean tree for every path that can reach the bits.** The local card is
  generated from the WORKING TREE; the remote card is generated from the
  COMMIT. If they differ, the device is not the variable being measured.
  `--rent` refuses on a dirty tree and the dry run reports it as `BLOCK G3`.
  Paths checked: `gemm/mojo_only`, `bench/gemm_card_main.mojo`,
  `bench/gemm_shapes.mojo`, `tools/gemm_card.sh`,
  `tools/with_identical_mode.sh`, `tools/with_build_lock.sh`, `mojo_only`,
  `pixi.toml`, `pixi.lock`.
- **Same commit on both sides**, recorded in the result directory as
  `commit.txt` and, because a rented box has no `.git`, cross-checked by
  `source_sha256.txt` computed with the same recipe at both ends.
- **This lane rents on RunPod. It does not rent on DigitalOcean.**
  `IDENTICAL_GEMM_PLAN.md`'s RENTING section is binding. The identity / E2
  lane's DigitalOcean GPU quota is ONE droplet at a time, and a second lane
  taking it would not error, it would silently queue their leg behind ours.
- `RUNPOD_API_KEY` in the environment, or `MOJOLEARN_RUNPOD_KEY_FILE`
  pointing at a **0600 file outside this repository**. Both are checked; a
  key file inside the checkout is refused by name.
- **Neither lane publishes timing while the other has a box up.** Nothing in
  this leg is a timing number anyway.
- Artifacts land in `bench/results/e1g/<stamp>-<vendor>/`.

---

## Phase 0 — the free work, all of it, before renting anything

Run these on the Mac. Every one of them is a defect found for nothing.

    tools/gemm_card.sh compare                      # the fold topology is reached
    tools/gemm_column_invariance.sh                 # host arms, the harness
    MOJOLEARN_GEMM_CARD_ARM=device tools/gemm_column_invariance.sh
    tools/with_identical_mode.sh pixi run mojo run -I . \
        gemm/mojo_only/gemm_device_check.mojo       # the kernel's own gates

A machine that fails its own gates teaches nothing when diffed against
another machine, so a red here is fixed before a box is created.

---

## Phase 1 — the dry run, which is the default

    tools/gemm_remote_leg.sh nvidia
    tools/gemm_remote_leg.sh amd

**Nothing is rented.** `--rent` plus a key in the environment are both
required, and there is an interlock (`MOJOLEARN_GEMM_LEG_REHEARSAL`) that
aborts any paid call reached from inside a rehearsal.

The dry run generates the Apple reference card for real and then executes
every non-paid path in the leg. **That first step takes as long as it takes**
-- it is a device build behind `tools/with_build_lock.sh`, which four
sessions share, and it has been measured queued for more than ten minutes
behind three other agents' builds. Pass `--local-card <path>` to reuse a card
you already have when you only want to rehearse the plumbing. The leg says
plainly that **nothing checks a supplied card came from this commit** -- that
one is on the operator. If the local generation FAILS, the dry run
substitutes a synthetic card so the rest of the plumbing is still exercised,
announces that it did, and reports `BLOCK L1`. It exists because **a path that has never
run is a path nobody has tested**, and this repository has four separate
scars from guard bugs found only by running the guard. What it checks:

| group | what it exercises |
|---|---|
| A1-A4 | argument validation, refusals by name, the one-hour hard cap, `--rent` without a key |
| B1 | the interlock: a rehearsal child cannot reach the create call |
| C1-C3 | `tools/runpod_guard.sh` `arm`, `check` and `reap` refusal paths, run for real against no pod |
| C4 | the leg turns the guard's refusal into a refusal. This is the check that found the bug where `arm ... | sed` reported SED's exit status, so a refusal read as a successful arm |
| D1-D5 | the contamination guard, including a `[FAST]` banner, a missing banner, a missing log and a missing card, plus an `[IDENTICAL]` banner that must be ACCEPTED so the guard is not just always-red |
| E1-E2 | the differ, both outcomes: an identical synthetic remote card, then one with a planted hash flip that must be caught AND named |
| E3-E4 | a card the differ CANNOT PARSE is reported as unreadable rather than as a divergence, and a block-count mismatch is named rather than read as one |
| F1-F2 | the remote body substitutes cleanly, passes `sh -n` (and `dash -n` where available), carries no bashism, and the leftover-placeholder detector is proven against a planted `@LEFTOVER@` |
| K1-K5 | key hygiene: mode 600, outside the repo, not tracked, present in the curl config and absent from curl's argv, and a `ps` detector proven against a planted key |
| L1 | the Apple reference card is a real measurement and not the synthetic placeholder the dry run falls back to when the local device arm fails |
| G1-G3 | `git archive` extracts and hashes, contains the kernel, and the tree is clean |

**Two kinds of red, and they mean different things.**

- `FAIL` means THIS SCRIPT is broken. Exit 1. Do not rent.
- `BLOCK` means the script is fine and the world is not ready -- a dirty
  tree, a live lease. Exit 3. Fix the world, then rent.

---

## Phase 2 — the paid leg

    export RUNPOD_API_KEY=...        # or MOJOLEARN_RUNPOD_KEY_FILE=<0600 file>
    tools/gemm_remote_leg.sh nvidia --rent --minutes 60

Then, separately, and **not at the same time**:

    tools/gemm_remote_leg.sh amd --rent --minutes 60

One leg is one vendor and two vendors in one invocation are refused. Run
them one after the other so that at most one box is ever up, so the
pre-flight's "this lane already has a pod" check stays meaningful, and so a
teardown failure is unambiguous about which box it is about.

### What the leg does, in the order it enforces

1. **Pre-flight, two free GETs.** No unexpired lease on this machine, no pod
   named `mojolearn-gemm-*` on the account. Double-renting is the cheapest
   orphan to prevent and the easiest to cause by re-running a script that
   failed late.
2. **Create.** THE BILL STARTS HERE and the box is not yet armed.
3. **Wait for ssh, on a timer.** `--ready-timeout` (default 600s) bounds the
   billing-and-unarmed window IN CODE. Running out of it terminates the pod.
4. **Arm the lease BEFORE any work.** `tools/runpod_guard.sh arm` installs an
   on-pod watchdog that DELETEs this pod through the API at the deadline.
   **If arm refuses, the box is not used -- it is terminated.** A box that
   cannot be armed is an orphan that has not happened yet.
5. **Key to the pod on stdin into a 0600 file**, then the pod's own `ps`
   output is dumped and searched with `grep -F -f <keyfile>` so the check
   does not leak what it is looking for.
6. **`git archive` at the pinned SHA**, extracted on the box, and the
   `source_sha256` compared at both ends.
7. **`gemm_device_check.mojo`, then `gemm_card.sh device`, under IDENTICAL.**
8. **Fetch, then read the compiled mode back out of BOTH remote logs.**
9. **Diff against the Apple card and name the FIRST diverging stage.**
10. **Terminate at the end of the WORK, verify it by asking the API, and
    print the lease's remaining minutes.** The lease is the backstop for when
    this session disappears. It is not the plan.

### The first paid run is a bring-up run

Budget it as one. As of DEVIATION 536 nothing below the create call has been
executed against RunPod, and three things in the script are written from the
REST v2 shape the guard's DELETE already uses rather than measured:

- the create request and response shape,
- the GPU type ids (`NVIDIA GeForce RTX 4090`, `AMD Instinct MI300X OAM`),
- the images, and whether they carry `pixi`.

Confirm the first two for nothing before you start, with the account's own
catalog:

    GET https://api.runpod.io/v2/gpu-types
    GET https://api.runpod.io/v2/pods

A wrong GPU id must make the create FAIL rather than quietly hand back a
different GPU. The leg records what it actually got in `remote/gpu.txt`, so a
substitution is visible in the artifacts either way -- read that line before
reading the card.

`pixi install` on a cold box is the main risk to the one-hour cap. If a
bring-up run runs out of lease, that is a finding about the box, not a reason
to extend past an hour by reflex. **Extending is re-arming and each
extension is a decision**:

    tools/runpod_guard.sh extend <pod-id> '<ssh target>' 60

---

## Reading the result

`bench/results/e1g/<stamp>-<vendor>/`

| file | what it means |
|---|---|
| `commit.txt`, `leg.txt` | the commit as `%h parent %p`, the vendor, the GPU requested, the lease, the pod id |
| `local/apple.card`, `local/apple.card.log` | the Apple reference leg and its mode banner |
| `source_sha256_local.txt` | hashed from the EXTRACTED ARCHIVE, so it is a measurement of what was shipped |
| `remote/<vendor>.card` | the card that must be compared |
| `remote/<vendor>.card.log` | **the mode witness.** If this does not say `[IDENTICAL]` the card is void |
| `remote/device_check.log` | the kernel's own gates on that silicon, every verdict reported before it raises |
| `remote/source_sha256.txt` | the box's half of the commit-parity evidence |
| `remote/gpu.txt` | what silicon actually answered |
| `remote/leg.txt` | per-step exit codes from the box |
| `key_in_ps.txt` | `KEY_NOT_IN_PS` or `KEY_VISIBLE_IN_PS`. The key file is removed from the pod after the fetch, before the terminate, so a box whose termination fails is left with no credential on it |
| `diff_apple_vs_<vendor>.txt` | the differ's full report, one section per sequence block |
| `teardown.txt` | whether the box was armed, whether termination was VERIFIED, the exit code |
| `remote_body.sh` | exactly what ran on the box, placeholders already substituted |
| `local/generate.log` | the local card driver's output, including its own mode read-back |
| `create_request.json`, `create_response.json` | what was asked for and what came back. Read these first on a bring-up run |
| `pod_id.txt`, `arm.log`, `lease.txt` | the pod, the arming, and the lease as the guard recorded it |
| `remote_console.log` | the box's stdout, including `REMOTE_BODY_DONE` if the body ran to the end |

**Read `remote/device_check.log` before the diff.** A machine that fails its
own gates teaches nothing when diffed against another one, and the leg
refuses to diff at all when anything above the diff is unsound. A stage name
produced from an unsound pair is worse than no stage name.

---

## The device card is THREE blocks, not one sequence

Measured on the Apple leg 2026-08-23. `tools/gemm_card.sh device` emits a card
with **three sequence blocks** -- the same 60 tags each time, the sequence
number restarting at 0 -- because the emitter runs the device arm at three
launch geometries and appends all three to one trace file.

`tools/identity_trace_diff.py` REFUSES that file. Its format requires the
sequence to start at 0 and increase by exactly 1, so it exits 2 with

    parse error: apple.card:63: seq out of order: expected 60, got 0

**Exit 2 is a parse error, not a divergence, and the two must never be
confused.** The leg splits both cards on the sequence restarts, requires both
sides to have the SAME block count, and diffs block against block, announcing
that it is doing so. Nothing is dropped.

Two consequences worth knowing before reading a report:

- **A block-count mismatch is a bigger finding than any hash.** It means the
  two machines ran different numbers of launch geometries. Resolve it before
  a stage name from either side means anything.
- On this Mac the three blocks are byte-identical to each other, which is the
  launch-invariance property `check_device_is_launch_invariant` asserts,
  arriving for free in the card. A leg where block 1 matches across vendors
  and block 3 does not is a launch-invariance failure on one of them, not an
  ordinary vendor difference.

If the emitter later gives each geometry a distinct tag, the card becomes one
sequence and the splitting in the leg becomes a no-op that reports one block.

---

## Walking a divergence — INTEGERS BEFORE FLOATS

`tools/identity_trace_diff.py` localizes. It does not explain. Its report has
four steps and they are a ladder, not a menu.

1. **STEP 1, PARSE.** A malformed card is a hard error naming the file and
   line. A card whose `.bin` dump does not re-hash to its recorded hash exits
   2 and **voids every other conclusion in the report** -- the writer and the
   reader disagree about the format.
2. **STEP 2, ALIGNMENT, computed before any hash is compared.** If the two
   sides do not emit the same stages, **the two runs took different code
   paths and this is a bigger finding than any hash.** Comparing hashes
   across differing stage sets is not meaningful. Fix the path difference,
   then re-run. On this profile the usual cause is a shape SKIPPED on one
   side; `tools/gemm_card.sh` prints the skipped count and the leg carries it
   home in `remote/card_driver.log`.
3. **STEP 3, THE HASH WALK.** The first diverging record is named. Now the
   ladder:
   - **The `.in.a` and `.in.b` stages come first for every shape and they are
     the FIXTURE.** If an input stage diverges, the two machines did not
     compute over the same bytes and nothing below it means anything about
     arithmetic. That is the one thing to check before all others.
   - **Then any integer stage** (`u32`, `i32`, `u8`, `i64`). An integer
     divergence is a code-path, partition or RNG difference. It is bigger
     than any numeric row. Resolve it before reading one float stage.
   - **Only then the float stages** (`.out`). The first one names the seam.
4. **STEP 4, CELL LEVEL.** Re-run both sides with
   `MOJOLEARN_IDENTITY_TRACE_DUMP=<substring>` so a `.bin` sits beside each
   card. It is a SUBSTRING match over tags, not an exact tag, and the leg
   passes the value through to the remote card run -- so the cell-level step
   costs a SECOND rental and the substring should name one stage, not one
   shape. This profile's largest input stage is 4,000,000 f32; a loose
   substring dumps tens of megabytes per shape and then fetches all of them
   over ssh. The differ then classifies per cell: signed ULP distance, and
   DENORMAL-vs-ZERO called out by name. **A scattered 1-ULP divergence
   (reassociation, FMA contraction, a different reduction order) is a
   completely different investigation from an all-DENORMAL-vs-ZERO one**, and
   the SUMMARY block is there so the two can be told apart at a glance.

Every divergence gets a ledger row before it gets a fix. This lane writes ROW
TEXT and hands it to the identity lane; it does not edit `IDENTITY_PATHS.md`.

---

## What a green card does NOT say

Two things, stated plainly, because a green card is very tempting to
over-read.

1. **A matching hash proves two buffers held the same bits at a checkpoint.
   It does not prove the computation was identical, and anything not hashed
   is invisible.** Two different code paths can land on the same bits; a
   buffer can be right for the wrong reason. Absence of divergence is
   evidence about the buffer, not about the algorithm.
2. **`tools/check_linalg_column_invariance.sh` and
   `tools/gemm_column_invariance.sh` compile three vendor COLUMNS onto ONE
   backend.** Those gates cannot see contraction (IDENTITY_PATHS row 9),
   denormal policy or `sqrt` rounding, because those belong to the BACKEND
   and not to the column, and they cannot fail for any of them no matter how
   wrong they are. **The standing proof is that NVIDIA's `sqrt` is not
   correctly rounded -- 180,714 of 2^20 patterns, 176,577 of them on normals
   -- a defect only its own silicon shows.** That is why this remote leg
   exists and why the column sweep does not replace it.

And a third, smaller one. Nothing in this leg is a timing number. A rented
box under an hour lease, with a cold pixi cache and a shared host, is the
worst timing instrument in this project, and timing belongs to the identity
lane anyway.

---

## Known confounds

- **The mode.** Passing `-D MOJOLEARN_NUMERIC_IDENTICAL=1` proves nothing. A
  build that lands in another session's window compiles the other arm and
  every label inside it agrees with the binary (DEVIATION 514, three
  mislabelled measurements in one day). The leg reads the mode back out of
  both remote banners and fails if either is not `IDENTICAL`. If you are
  reading a card by hand, read its `.log` first.
- **A dirty local tree.** The local card comes from the tree and the remote
  card comes from the commit. Two variables, one diff, no conclusion.
- **`MOJOLEARN_GEMM_CARD_FULL`.** It removes the host shape cap and it must
  be the SAME on both ends. The leg passes one value through to both sides
  and records it in `leg.txt`; do not set it by hand on only one machine.
- **A substituted GPU.** Read `remote/gpu.txt`. A card labelled `nvidia` that
  ran on a different NVIDIA part than the one you asked for is still a valid
  cross-vendor data point, but it is not the data point you filed.
- **A shape skipped on one side only.** It looks exactly like a shape that
  agreed. The differ's STEP 2 catches it as an unmatched tag; the skipped
  count is in `remote/card_driver.log`.
- **The box has no `.git`.** `git rev-parse` on the pod would write
  "unknown", which is a belief about what was shipped rather than a
  comparison. That is why `source_sha256.txt` exists at both ends, and why
  the leg goes red if the two do not match.

---

## Teardown checklist

The leg terminates at the end of the work through an EXIT trap, so every
failure path reaches it, and it **verifies** rather than assuming. Check
these by hand anyway, every time.

1. The run printed `VERIFIED: <pod> is gone (HTTP 404)` or
   `VERIFIED: <pod> reports status TERMINATED`.
2. `teardown.txt` says `terminated=1`.
3. `tools/runpod_guard.sh list` shows no lease for that pod.
4. The lease's remaining minutes at exit are in the run's output. **That
   number is how close this run came to needing the backstop.** A leg that
   habitually finishes with two minutes left is a leg that will one day be
   killed by its own watchdog mid-card.
5. https://console.runpod.io/pods shows nothing named `mojolearn-gemm-*`.

**If step 1 did not print VERIFIED, the box may still be billing.** RunPod's
container exit stops GPU billing but disk keeps accruing, and RunPod restarts
a container that exits, so nothing except a successful DELETE ends it.

## Orphan recovery

    tools/gemm_remote_leg.sh reap                 # terminate every EXPIRED lease
    tools/gemm_remote_leg.sh reap <pod-id>        # terminate one, now, and VERIFY

The second form is the one to reach for. It deletes **that pod, by id**, then
removes its lease file, then **verifies by asking the API** -- because "the
DELETE returned 200" and "the pod is gone" are different claims and only the
second one stops the bill.

It deliberately does not call `tools/runpod_guard.sh reap --force <pod-id>`.
See the last section: that helper ignores the pod argument and reaps every
lease it can see.

If the leg died between create and arm, the pod exists, is billing, and has
**no watchdog at all** -- that is the one window where the lease cannot save
you. The leg bounds that window with `--ready-timeout` and adopts a pod by
name when the create response is unparseable, but if a session was killed
inside it, the console is the ground truth.

---

## Recording the result

Write `E1G_RESULTS.md` in the repository root, modelled on `E1U_RESULTS.md`
(86 stages, 0 divergences, Apple vs a real MI300X -- that is the shape of the
result this leg is for). It needs a table with, per leg: device, column
resolved and lane width, toolchain, commit as `%h parent %p`, **the mojo
source sha256**, the stage count, and the findings count.

Stage the result explicitly. **Never `git add -A` in this checkout** -- four
sessions share it:

    git add gemm/E1G_RUNBOOK.md tools/gemm_remote_leg.sh E1G_RESULTS.md \
            bench/results/e1g/<stamp>-<vendor>
    git show --stat        # READ IT before pushing

---

## Two changes `tools/runpod_guard.sh` needs, and why this lane did not make them

That file belongs to the identity / E2 lane and this lane may not edit it.
Both of these were found by reading it against the leg that calls it.

### 1. `reap --force <pod-id>` ignores the pod id and reaps EVERYTHING

`cmd_reap` shifts `--force` off, which shifts the pod argument off with it,
and the loop that follows terminates every `*.lease` in the lease directory:

    cmd_reap() {
        force=""
        if [ "$1" = "--force" ]; then force="1"; shift; fi
        ...
        for f in "$LEASES"/*.lease; do ... if [ -n "$force" ] || expired; then
            curl -X DELETE .../$pod

So `reap --force <pod-id>` READS like a targeted terminate and is a
reap-everything. It is harmless under a one-box-at-a-time discipline and it
is exactly the kind of gap that stays harmless until the day two boxes are
up. **The change**: accept an optional pod id after `--force` and, when one
is given, terminate only that lease. `tools/gemm_remote_leg.sh` works around
it today by issuing the targeted DELETE itself and removing that pod's lease
file, so `check` and `list` stay honest afterwards.

### 2. The key is in the remote argv for the length of an `arm`

**The defect.** `cmd_arm` passes the key to the pod inside the ssh command
string:

    RUNPOD_API_KEY='$RUNPOD_API_KEY' nohup /tmp/mojolearn-lease.sh ...

The watchdog itself is clean -- the key is in its environment, not its argv,
which is what the surrounding comment claims and it is true of that process.
But sshd runs the whole command string through `sh -c`, so **for the second
or so that `arm` runs, the key is in the argv of that remote shell** and
visible to anything on the box calling `ps`. That is the same class of defect
as the leak the guard's own history records.

**The change.** Send the key on ssh STDIN into a 0600 file and have the
watchdog read it, instead of interpolating it into the command string. In
`cmd_arm`, ship the key with

    printf '%s' "$RUNPOD_API_KEY" | pod_ssh "$target" \
        'umask 077; cat > /tmp/mojolearn-lease.key'

as a separate call before the heredoc, and drop the `RUNPOD_API_KEY='...'`
prefix from the launch line so it becomes a plain
`nohup /tmp/mojolearn-lease.sh`. The `arm` heredoc is already quoted with
`<<'WATCHDOG'`, so nothing else in it moves.

**Do not substitute `-H "Authorization: Bearer $(cat /tmp/mojolearn-lease.key)"`
in the watchdog.** That moves the leak rather than closing it -- the key would
then be in `curl`'s argv on the pod at fire time. Write a 0600 curl config
beside the key and use `curl -K`, which is what `tools/gemm_remote_leg.sh`
does for the same reason on this end:

    header = "Authorization: Bearer <key>"

Until that lands, `tools/gemm_remote_leg.sh` MEASURES the residue rather than
asserting it is absent: after arming, it dumps the pod's process list and
searches it with `grep -F -f <keyfile>`, and `key_in_ps.txt` in every result
directory says `KEY_NOT_IN_PS` or `KEY_VISIBLE_IN_PS`. **Verify the claim, do
not assert it** -- a previous version of the guard leaked a key into `ps`
while carrying a comment in the same edit saying that it did not.
