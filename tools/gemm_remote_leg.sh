#!/bin/sh
# DEVIATION 536 -- ONE GUARDED RUNPOD LEG: an identity payload taken to a
# second and a third vendor, and the box terminated afterwards.
#
#   tools/gemm_remote_leg.sh nvidia                 DRY RUN. Rents nothing.
#   tools/gemm_remote_leg.sh amd                    DRY RUN. Rents nothing.
#   tools/gemm_remote_leg.sh nvidia --rent          RENTS. BILLS. One hour.
#   tools/gemm_remote_leg.sh nvidia --payload phase8 --rent
#                                                   RENTS. Phase 8 of
#                                                   tools/e1_bootstrap.sh:
#                                                   SEVEN lanes, not one.
#   tools/gemm_remote_leg.sh reap [pod-id]          orphan recovery
#
# **THE DEFAULT IS A DRY RUN AND THAT IS DELIBERATE.** Renting needs BOTH
# `--rent` on the command line AND `RUNPOD_API_KEY` in the environment. A
# typo, a stale shell-history line, a copy-paste out of the runbook -- none
# of them can start a bill.
#
# WHAT THIS IS FOR
# ================
# `gemm/IDENTICAL_FP32_CONTRACT.md` profile `mojolearn.identical.gemm.fp32.v1`
# claims that one source, built under `NUMERIC_IDENTICAL`, produces the SAME
# FP32 BITS on any GPU. Everything needed to test that claim except the other
# GPUs is finished on this desk: the contract, both oracles, the device
# kernel, its sabotage gates, the column sweep, the price harness. The one
# thing a Mac cannot answer is whether the claim holds on silicon that is not
# Apple's, and this file is the repeatable, credential-safe, self-expiring
# procedure that answers it.
#
# It is the sibling of `tools/e2_remote_leg.sh` (the identity lane's
# DigitalOcean leg) and it is deliberately NOT a copy of it. See "TWO LANES,
# TWO PROVIDERS" below.
#
# TWO PAYLOADS, ONE SET OF GUARDS
# ===============================
# The safety path -- pre-flight, the detached dead-man armed BEFORE the
# create, arm-before-work, the ready timeout, the key on stdin, `git archive`
# at a pinned sha, the payload run detached and polled, terminate-and-verify,
# one vendor per leg -- is the same for both. Only what runs on the box
# differs.
#
#   --payload gemm    (default)  gemm_device_check.mojo, then
#       `tools/gemm_card.sh device`, under IDENTICAL. The Apple side is
#       generated HERE by this script and the two cards are diffed HERE. One
#       lane, one profile, one answer, and this file is the whole procedure.
#
#   --payload phase8             `tools/e1_bootstrap.sh`, UNMODIFIED, whose
#       phase 8 writes SEVEN lanes' cards in both modes (gemm, cd, kde,
#       linkage, svm, metrics and mamba) into `<run>/lanes/*.card`. That
#       directory is fetched to `bench/results/e1/<stamp>-runpod-<vendor>/`,
#       beside the Mac's own bootstrap directory, and the verdict is
#       `tools/e3_round_judge.sh` section 7 -- THIS LEG DOES NOT DIFF THOSE
#       CARDS. It gets them home, intact, attributed and mode-witnessed.
#
# WHY phase8 RUNS THE BOOTSTRAP RATHER THAN A COPY OF PHASE 8. A copy would
# be a second list of lanes to keep in step with `tools/e1_bootstrap.sh` and
# `tools/e3_round_judge.sh`, and the day they gained `mamba` (2026-08-23) the
# copy would have gone on producing a green six-lane leg with the new lane
# silently absent. The cards the judge reads have to come out of the program
# the judge is named after. The cost of that choice is that phase8 pays for
# phases 0-7 as well: measured on DigitalOcean at commit 144aa5b, the whole
# bootstrap was 27 minutes (H100) and 18 minutes (MI325X) on images that
# ALREADY HAD PIXI -- a cold `pixi install` here is the risk to the hour, and
# it is why the phase8 body bounds the bootstrap with `timeout` and keeps a
# reserve for the fetch. A leg that runs out of hour then comes home with the
# lanes that finished instead of being killed by its own watchdog holding
# every card.
#
# THE mamba LANE, TWICE, BECAUSE BOTH SURPRISE PEOPLE:
#   * ITS FAST ARM HAS NEVER BEEN BUILT ANYWHERE. A
#     `PHASE8-FINDING: mamba [fast]` line is EXPECTED, is information, and is
#     not an abort. `tools/e1_bootstrap.sh` says so at the lane itself.
#   * ITS SHAPE MUST NOT BE WIDENED ON A RENTED BOX. `mamba_check.mojo` reads
#     `MOJOLEARN_MAMBA_CHECK_B` / `_L` / `_DM` and every shape is another
#     compile inside the lease. The remote body UNSETS all three rather than
#     assuming the pod's environment is empty.
#
# WHAT THIS CANNOT TELL YOU
# =========================
#  1. **A matching card is not proof the computation was identical.** Each
#     record is a hash over a buffer at a checkpoint. Two different code
#     paths can land on the same bits, a buffer can be right for the wrong
#     reason, and anything not hashed is invisible. Absence of divergence is
#     evidence about the buffer, not about the algorithm. This is
#     `tools/identity_trace_diff.py`'s own limit and it is inherited whole.
#  2. **It does not replace the local column sweep and the local column
#     sweep does not replace it.** `tools/gemm_column_invariance.sh` compiles
#     three vendor COLUMNS onto ONE backend, so it cannot see FMA contraction
#     (IDENTITY_PATHS row 9), denormal policy, or `sqrt` rounding, which are
#     properties of the BACKEND. The standing proof that a backend can be
#     wrong in a way only its own silicon shows: NVIDIA's `sqrt` is not
#     correctly rounded on 180,714 of 2^20 patterns, 176,577 of them on
#     normals. No amount of green on this Mac would have found that. That is
#     the entire reason this file exists.
#  3. It says nothing about SPEED. Timing belongs to the identity lane and no
#     number produced here is a timing number (IDENTICAL_GEMM_PLAN.md, "LANE
#     BOUNDARY", working rules). A rented box under an hour lease, with a
#     cold pixi cache and a shared host, is the worst timing instrument in
#     this project.
#  4. It does not prove the pod stopped billing. It CHECKS, by asking the API
#     after the terminate, and it prints what it got. Read that line.
#
# TWO LANES, TWO PROVIDERS, AND WHY THAT IS NOT AN ACCIDENT
# =========================================================
# `IDENTICAL_GEMM_PLAN.md`'s RENTING section, amended by Andrew 2026-08-23:
#
#     identity / E2 lane   DigitalOcean droplets, their e2_remote_leg.sh
#     this lane            RunPod, tools/runpod_guard.sh arm FIRST
#
# The two lanes MUST NOT share a provider account. The identity lane's
# DigitalOcean GPU quota is ONE droplet at a time; a second lane taking it
# would not error, it would silently queue their leg behind this one's, and
# the failure would look like slowness rather than like contention.
#
# **THE TWO PROVIDERS FAIL DIFFERENTLY AND NEITHER GUARD TRANSFERS.**
#   DigitalOcean bills until the droplet is DESTROYED. Power-off does not
#   stop it. That is how a 90-second job once billed 10h48m (~$41).
#   RunPod restarts a container that exits. Measured on a real pod on
#   2026-08-23: the credential-free `kill 1` watchdog fired exactly on time,
#   the container exited, **RunPod brought it straight back up**, billing
#   never paused, and the watchdog had been wiped -- leaving the box LESS
#   protected than an unguarded one, because the lease file on this end still
#   claimed it was guarded. Read `tools/runpod_guard.sh`'s header in full
#   before changing one line of the safety path here.
#
# THE SAFETY STORY, IN THE ORDER THE CODE ENFORCES IT
# ===================================================
#  0. Two pre-flight GETs that cost nothing: this lane must not already have
#     a pod up, and no unexpired lease may be sitting in the lease directory.
#     Double-renting is the cheapest orphan to prevent and the easiest to
#     cause by re-running a script that failed late.
#  1. **A DETACHED DEAD-MAN IS ARMED ON THIS MACHINE, BEFORE THE CREATE
#     CALL** (DEVIATION 862). The pod is then created, and from that instant
#     to step 3 the box is BILLING AND ITS OWN WATCHDOG DOES NOT EXIST YET.
#     That window is bounded IN CODE by --ready-timeout (default 600s), and
#     the timeout is a loop IN THIS PROCESS -- so if this process is killed
#     inside the window, the loop dies with it and only the dead-man is left.
#     The dead-man is keyed by pod NAME as well as by id, because the worst
#     case in this file is a create that succeeded and an id nobody parsed.
#  2. SSH comes up.
#  3. **THE LEASE IS ARMED BEFORE ANY WORK.** `tools/runpod_guard.sh arm`
#     installs an on-pod watchdog that DELETEs this pod through the API at
#     the deadline; that survives this Mac going away, which is the whole
#     requirement, and it is why `arm` REFUSES without a key rather than
#     pretending. **If arm refuses, the box is not used -- it is
#     TERMINATED.** A box that cannot be armed is an orphan that has not
#     happened yet.
#  4. Work runs, DETACHED ON THE BOX, polled from here (DEVIATION 863). A
#     dropped ssh no longer kills the payload and no longer makes this leg
#     mistake a cut-off run for a finished one. Terminate happens at the END
#     OF THE WORK, not at the end of the lease. The lease is the backstop for
#     when this session disappears; it is not the plan.
#  5. Teardown is an EXIT trap, so every failure path above reaches it, and
#     it VERIFIES the termination by asking the API instead of assuming the
#     DELETE worked. It prints the lease's remaining minutes at exit, which
#     is the one number that says how close this run came to needing the
#     backstop.
#
#     LAYER ONE is the on-pod watchdog (survives this Mac). LAYER TWO is the
#     `reap` this script performs from here (reclaims the DISK, which keeps
#     accruing on RunPod even after container exit stops GPU billing). LAYER
#     THREE, added by DEVIATION 862, is the detached dead-man of step 1, and
#     it is the only one that covers the window before layer one exists. All
#     three, always, because they cover different failures. The teardown
#     CANCELS the dead-man only when the pod is confirmed gone; if the
#     terminate could not be verified it is deliberately LEFT ARMED and said
#     so, because that is precisely the run it was written for.
#
# ONE HOUR IS A HARD CAP. --minutes above 60 is refused BY NAME. Extending is
# re-arming (`tools/runpod_guard.sh extend`), each extension is a decision a
# human makes, and there is deliberately no `disarm` anywhere in this path.
#
# THE API KEY
# ===========
# The key never appears in an argv, on this machine or on the pod, and never
# in a tracked file.
#   * Locally: curl reads the Authorization header from a 0600 config file
#     via `-K`. Writing that file uses the SHELL BUILTIN `printf`, so no
#     process is spawned and there is no argv to leak into. A previous
#     version of the guard leaked a key into `ps` while carrying a comment in
#     the same edit claiming it did not, so this file does not assert the
#     property -- the dry run CHECKS it (see `leg_rehearse`, checks K1-K5),
#     including a sabotage that proves the checker is not inert.
#   * On the pod: the key arrives on ssh STDIN into a 0600 file, never in a
#     command line. After arming, the pod's own `ps` output is dumped to a
#     file and searched with `grep -F -f <keyfile>` -- the pattern comes from
#     a FILE, so the check itself does not leak what it is looking for.
#   * `tools/runpod_guard.sh arm` used to pass the key to the pod inside the
#     ssh command string, so for the ~1 second that arm ran it was in the
#     argv of the remote `sh -c` that sshd spawns AND in ssh's own argv
#     here. CLOSED 2026-08-24: the guard sends its credential on ssh stdin
#     into a 0600 curl config and the watchdog uses `curl -K`. The dry run's
#     C10 proves it with a stub ssh that records its own argv, and C11 does
#     the same for the guard's `reap`.
#   * BECAUSE THAT IS CLOSED, THE POST-ARM CHECK IS NOW AN ASSERTION. A
#     `KEY_VISIBLE_IN_PS` result used to be an expected residue to record;
#     it is now an unexplained exposure, and this leg exits non-zero for it
#     and asks for the key to be rotated.
#
# HOW THE SOURCE GETS ONTO THE BOX, AND WHY IT IS `git archive`
# =============================================================
# Four candidates, three rejected for reasons this repository has already
# paid for:
#   rsync of the working tree -- REJECTED. Four sessions share this
#     checkout right now. The first E2 legs rsync'd a worktree that carried
#     another session's numerics flip and a `.git` that was a worktree
#     POINTER FILE, and the remote had no repository at all. A leg must ship
#     a COMMIT, not whatever the tree happened to hold at 11:04.
#   git clone from the origin -- REJECTED. The repository is private, so
#     this needs a credential ON THE RENTED BOX, which is the one thing the
#     whole guard design is trying to avoid putting there.
#   git bundle -- REJECTED, narrowly. It is exact, but `git bundle create`
#     needs a REF ("Refusing to create empty bundle" for a bare sha), so
#     `tools/e2_remote_leg.sh` force-creates a branch in the shared
#     checkout. This lane does not mutate refs in a checkout three other
#     sessions are working in.
#   git archive at a pinned SHA -- CHOSEN. Exact by construction, needs no
#     credential on the box, mutates nothing here, and its content is the
#     COMMIT rather than the worktree.
#
# The box therefore has NO `.git`, exactly like the E1U AMD leg, so commit
# attribution cannot come from `git rev-parse` there. It comes from two
# sides instead:
#   * `commit.txt`, written HERE from `git rev-parse`, so the card is filed
#     against the commit that produced it and can never be attributed to a
#     later one;
#   * `source_sha256.txt`, computed over every `.mojo` file with the same
#     recipe on both ends (the recipe `tools/e1_unsupervised.sh` introduced
#     after the AMD leg's `commit.txt` read "unknown"). The Mac's copy is
#     computed from the EXTRACTED ARCHIVE, not from the worktree, so the two
#     hashes agreeing is a measurement of the transport and not a belief
#     about it. They must match or the leg goes red before any card is read.
# And because the LOCAL card is generated from the WORKING TREE while the
# remote runs the COMMIT, `--rent` additionally REFUSES on a dirty tree for
# every path that can reach the bits. One variable is the device. A local
# uncommitted edit is a second variable and it is invisible in the cards.
#
# THE CONTAMINATION GUARD, WHICH IS NOT OPTIONAL
# ==============================================
# The mode is READ BACK from the remote run's own banner and the leg fails if
# it is not IDENTICAL. Passing `-D MOJOLEARN_NUMERIC_IDENTICAL=1` proves
# nothing: a build that lands in another session's window compiles the other
# arm and every label inside it agrees with the binary (DEVIATION 514, three
# mislabelled measurements in one day). `tools/check_linalg_column_invariance.sh`
# has exactly this guard and this is the same shape. It survives the `-D`
# migration for the same reason it was written: a mis-plumbed define is
# exactly as invisible as a lost flip, and this line is the only thing that
# sees either. BOTH remote artifacts are checked -- the card's log and the
# device-check's log -- because they are two separate builds.
#
# WHAT IN THIS FILE HAS NEVER RUN AGAINST A REAL POD
# ==================================================
# Said up front rather than discovered at $2.39/hr. As of DEVIATION 536,
# NOTHING below `leg_create_pod` has been executed against RunPod. The dry
# run exercises argument validation, the guard's refusal paths, the local
# card, the mode read-back including its contaminated case, the differ in
# both outcomes, the key hygiene checks including a sabotage, and a POSIX
# syntax check of the remote body -- all of it, every time. What it CANNOT
# exercise without billing:
#   * the create call's request and response shape. `RP_CREATE_PATH`, the
#     JSON body, and the id/ssh parsing are written from the REST v2 shape
#     the guard's DELETE already uses. Confirm them with the free GETs in
#     the runbook (`GET /v2/gpu-types`, `GET /v2/pods`) BEFORE the first
#     paid run; a wrong `gpuTypeIds` entry must make the create FAIL rather
#     than silently hand back a different GPU, and the leg records what it
#     actually got in `remote/gpu.txt` so a substitution is visible.
#   * `LEG_GPU_*` and `LEG_IMAGE_*` below are DEFAULTS, not measurements.
#   * whether the images have `pixi`, and how long `pixi install` takes on a
#     cold box. That is the main risk to the one-hour cap.
#   * THE DEAD-MAN FIRING (DEVIATION 862) and THE POLL LOOP SEEING A REAL
#     DETACHED RUN (DEVIATION 863). The dry run composes the dead-man,
#     syntax-checks it, proves its key is in no argv, and RUNS IT against
#     stub `curl` and stub `sleep` so the DELETE it would issue is recorded
#     (checks M1-M4). What no dry run can exercise is the API answering it,
#     or an ssh actually dropping mid-poll.
#   * EVERYTHING THE phase8 PAYLOAD DOES ON THE BOX. `tools/e1_bootstrap.sh`
#     has run on DigitalOcean droplets many times and on a RunPod pod never;
#     its five bindings builds, its pixi environments and its phase-8
#     compiles are what fills the hour, and whether `bash`, `timeout` and
#     `git` are on the image is unmeasured here. The fetch of the bootstrap
#     directory, the commit.txt rewrite and the lane mode read-back are
#     rehearsed against planted artifacts on every dry run, but they have
#     never seen a real one.
# The first paid run is therefore a bring-up run. Budget it as one, and read
# the runbook's "first paid run" section before starting it.
#
# ENVIRONMENT
#   RUNPOD_API_KEY                 required by --rent. Never logged.
#   MOJOLEARN_RUNPOD_PODS_PATH     bring-up only: the pods REST path this leg
#                                  lists, creates and deletes through
#                                  (default /v2/pods). It does NOT move the
#                                  on-pod watchdog's endpoints.
#   MOJOLEARN_RUNPOD_KEY_FILE      read the key from this file instead, so it
#                                  need not be in shell history. The file
#                                  must be 0600 and outside this repository;
#                                  both are checked.
#   MOJOLEARN_GEMM_LEG_MINUTES     lease minutes (default 60, hard cap 60)
#   MOJOLEARN_GEMM_LEG_OUT         result directory
#   MOJOLEARN_GEMM_LEG_LOCAL_CARD  reuse an existing Apple card instead of
#                                  generating one. The card must have come
#                                  from the same commit; nothing can check
#                                  that for you, which is why the default is
#                                  to generate.
#   MOJOLEARN_GEMM_CARD_FULL       passed through to BOTH sides identically
#                                  (1 removes the host shape cap). One
#                                  variable is the device.
#   MOJOLEARN_IDENTITY_TRACE_DUMP  passed through to the remote card run so a
#                                  SECOND leg can bring `.bin` sidecars home
#                                  for the differ's cell-level step. It is a
#                                  SUBSTRING match; keep it narrow.
#   MOJOLEARN_GEMM_LEG_PAYLOAD     gemm (default) or phase8; --payload wins.
#   MOJOLEARN_GEMM_LEG_APPLE_DIR   phase8 only: the Apple bootstrap directory
#                                  this box's column will be judged against.
#                                  The default is the newest one under
#                                  bench/results/e1/ that has a lanes/
#                                  directory AND records this commit.
#   MOJOLEARN_GEMM_LEG_WORK_TIMEOUT phase8 only: seconds the bootstrap may
#                                  run on the box. 0 (default) derives it
#                                  from the lease minus a 600s fetch-and-
#                                  terminate reserve.
#   MOJOLEARN_GEMM_LEG_REHEARSAL   internal interlock. When 1, any paid call
#                                  aborts. Set on the children the dry run
#                                  spawns so a rehearsal can never rent.
set -e

# The snapshot below re-execs this file from /tmp, where `dirname $0` is no
# longer the repository. REPO is therefore resolved ONCE, here, from the
# original path, and carried across the exec in the environment.
if [ -n "${MOJOLEARN_LEG_REPO:-}" ]; then
    cd "$MOJOLEARN_LEG_REPO"
else
    cd "$(dirname "$0")/.."
fi
REPO="$(pwd)"
MOJOLEARN_LEG_REPO="$REPO"
export MOJOLEARN_LEG_REPO

# ---------------------------------------------------------------------------
# DEVIATION 1882 -- RUN FROM AN IMMUTABLE SNAPSHOT OF THIS FILE.
#
# `sh` does not read a script into memory. It reads it LAZILY, BY BYTE
# OFFSET, returning to the file for the next command after each one it runs.
# So editing this file while a leg is executing it shifts every offset past
# the edit out from under the running shell.
#
# That is not hypothetical. On 2026-08-25 this file was edited and committed
# at 18:05 while a leg started at 17:58 was still inside it. The shell
# resumed mid-word inside a comment and tried to execute `SERTED,` -- the
# tail of `ASSERTED,` on line 3961 -- and died at exit 127 BEFORE THE FETCH.
# The payload on the box had finished correctly. Four tree lanes had been
# measured. The pod was terminated on schedule with every log still on it,
# and `et` and `iforest` were lost.
#
# A LEG RUNS FOR AN HOUR AND AN HOUR IS LONG ENOUGH TO WANT TO EDIT
# SOMETHING. So the fix is not a rule about not editing; it is this. The
# first thing this script does is copy itself somewhere the working tree
# cannot reach, make the copy READ-ONLY, and re-exec into it. From that
# point the working tree can be edited, committed, rebased or deleted and
# the running leg does not notice.
#
# MOJOLEARN_LEG_SNAPSHOT is how the copy knows it is the copy. It carries
# the path so the banner can print where the leg is actually running from,
# and unsetting it by hand only buys back the bug.
if [ -z "${MOJOLEARN_LEG_SNAPSHOT:-}" ]; then
    _snapdir="${TMPDIR:-/tmp}/mojolearn-leg-snapshot-$$"
    mkdir -p "$_snapdir"
    _snap="$_snapdir/gemm_remote_leg.sh"
    cat "$0" > "$_snap"
    # 555, not 444: NOT WRITABLE is the whole point, but the dry run's
    # rehearsal checks re-invoke "$0" DIRECTLY, and a snapshot without the
    # execute bit turns all seven of them into exit 126.
    chmod 555 "$_snap"
    MOJOLEARN_LEG_SNAPSHOT="$_snap"
    export MOJOLEARN_LEG_SNAPSHOT
    # The snapshot outlives this exec and nothing else will remove it, so it
    # is removed by the copy's own exit trap, not here.
    exec /bin/sh "$_snap" "$@"
fi

# ---------------------------------------------------------------------------
# defaults and the vendor table
# ---------------------------------------------------------------------------

# UNVERIFIED DEFAULTS -- see "WHAT IN THIS FILE HAS NEVER RUN" above.
LEG_GPU_NVIDIA="${MOJOLEARN_GEMM_LEG_GPU_NVIDIA:-NVIDIA GeForce RTX 4090}"
LEG_GPU_AMD="${MOJOLEARN_GEMM_LEG_GPU_AMD:-AMD Instinct MI300X OAM}"
LEG_IMAGE_NVIDIA="${MOJOLEARN_GEMM_LEG_IMAGE_NVIDIA:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
LEG_IMAGE_AMD="${MOJOLEARN_GEMM_LEG_IMAGE_AMD:-rocm/dev-ubuntu-22.04:6.4.1-complete}"

# The API host is overridable ONLY so the termination-verification path can be
# exercised without a pod: point it at a dead port and every query comes back
# 000, which is the "could not confirm" branch -- the single most important
# message in this file and one that would otherwise never have run. It is not
# a production knob. `tools/runpod_guard.sh` has its own hardcoded host and
# this does not move it.
# DEVIATION 1871 -- THE DEFAULT MOVED TO rest.runpod.io/v1 BECAUSE
# api.runpod.io/v2 REFUSES THIS BODY. Measured 2026-08-25, HTTP 422:
#
#   "$: missing property 'image'"
#   "$: additional properties 'allowedCudaVersions', 'cloudType',
#    'containerDiskInGb', 'gpuCount', 'gpuTypeIds', 'imageName',
#    'interruptible', 'supportPublicIp', 'volumeInGb' not allowed"
#
# So v2 is a DIFFERENT SCHEMA, not a different host for the same one. The
# body this file composes is v1's, confirmed field for field against
# rest.runpod.io/v1/openapi.json's PodCreateInput, which lists every one of
# the properties v2 rejected.
#
# DEVIATION 867 predicted exactly this and left the host overridable and the
# path NOT, which is a knob that cannot fix the thing it is for; DEVIATION
# 972 later made the path overridable too. What was still wrong was the
# DEFAULT, so a leg run without either override armed a dead-man, called a
# paid endpoint and got a 422. It cost nothing this time -- the create is the
# first paid call and it failed -- but the next schema drift might not fail
# so cleanly, which is why this is a default change and not a runbook line.
RP_HOST="${MOJOLEARN_RUNPOD_API_HOST:-https://rest.runpod.io}"
# v1 is deprecated and is kept ONLY as a fallback for the DELETE, for the
# reason tools/runpod_guard.sh gives at the same seam: this is the only thing
# standing between an orphan and the bill, and an API deprecation that
# silently 404s would disarm every terminate at once.
RP_HOST_V1="${MOJOLEARN_RUNPOD_API_HOST_V1:-https://rest.runpod.io}"
# DEVIATION 867 -- THE PODS PATH IS OVERRIDABLE, FOR THE BRING-UP RUN ONLY.
# `/v2/pods` under api.runpod.io is written from the shape
# tools/runpod_guard.sh's DELETE already uses and it has NEVER been confirmed
# for a LIST or a CREATE on this account. If the account's pods REST surface
# turns out to be rest.runpod.io/v1 instead, the host was already overridable
# and the path was not, so the leg could be told half of the truth and no
# more -- which is a knob that cannot fix the thing it is for. Confirm both
# with the free GETs in gemm/E1G_RUNBOOK.md BEFORE the first paid run; a
# wrong path makes the pre-flight GET non-2xx and the leg refuses to create
# anything, which is the safe direction and is where this will show up.
# THE GUARD'S OWN ENDPOINTS DO NOT MOVE WITH THIS. The on-pod watchdog walks
# its own hardcoded v2-then-v1 pair, so the box still self-terminates however
# this is set. That separation is deliberate: the layer that survives this
# Mac must not depend on an environment variable set on this Mac.
RP_PODS_PATH="${MOJOLEARN_RUNPOD_PODS_PATH:-/v1/pods}"
# The guard's lease directory, mirrored (not owned) so a terminate can leave
# `check` and `list` telling the truth afterwards.
LEASE_DIR="${MOJOLEARN_LEASE_DIR:-bench/results/runpod_leases}"

# THE PAYLOAD. See "TWO PAYLOADS" above. `gemm` is this lane's original leg;
# `phase8` runs tools/e1_bootstrap.sh so that phase 8's seven lanes come home
# for tools/e3_round_judge.sh section 7.
PAYLOAD="${MOJOLEARN_GEMM_LEG_PAYLOAD:-gemm}"
# speed only. ONE FAMILY PER LEASE; see the refusal above for why that is
# arithmetic rather than taste.
#   gemmseq    gemm + transformer + mamba. torch is already in the image, so
#              this family installs almost nothing and is the cheapest leg.
#   classical  the cuML and cuVS opponents. Pays a RAPIDS install.
#   forest     CatBoost, XGBoost and LightGBM, both devices each. Pays a
#              LightGBM-from-source build for its CUDA arm.
SPEED_FAMILY="${MOJOLEARN_SPEED_FAMILY:-gemmseq}"
# `trees` and `forest` are the SAME family and `trees` is the better name.
# The lane list is gbdt-symmetric, gbdt-depthwise, gbdt-lossguide, rf, et and
# iforest -- BOOSTING IS IN IT. Calling the family `forest` reads as though
# the gradient-boosted learners were somewhere else, and they are not.
#
# THE NORMALIZATION IS NOT HERE. It has to run AFTER the argument loop, or it
# only ever reaches the environment form and never `--family trees`. That is
# where it was, and `--family trees` was refused by a message that offered
# `trees` as the accepted spelling in the same breath.
# THE LOAD LADDER (trees family only).
#
# `year` stops at 463,715 rows and `covtype` at 522,911, and every NVIDIA
# tree ratio this project has is from that size. Half a million rows is not
# where a GPU tree learner is decided. SPEED_DATASET=higgs plus a SPEED_ROWS
# list runs the SAME lane at several row counts in ONE lease, so the answer
# to "does the gap close with load" is a column and not an opinion.
#
# Empty SPEED_ROWS means one run at the dataset's shipped size, which is the
# old behavior exactly.
# The LightGBM CUDA learner is a FIFTEEN-TO-THIRTY MINUTE source build in a
# sixty-minute lease. It is worth that in the lossguide lane, where LightGBM
# is the algorithm's own author. It is not worth it in a load-ladder leg
# whose lanes are symmetric boosting and forests, where LightGBM is a
# secondary opponent and the build would cost half the rungs.
SPEED_LGBM_CUDA="${MOJOLEARN_SPEED_LGBM_CUDA:-1}"
# Concurrent legs on one account. The pre-flight normally REFUSES to rent
# while any mojolearn-gemm-* pod is up, and that refusal is the orphan guard:
# a leg that finds someone else's pod cannot know whether it is a live run or
# a leak. With this flag it warns and names them instead of dying. Safe only
# because every teardown targets its own POD ID and every dead-man is keyed
# by its own pod NAME, so two legs never reap each other.
LEG_ALLOW_CONCURRENT="${MOJOLEARN_LEG_ALLOW_CONCURRENT:-0}"
SPEED_DATASET="${MOJOLEARN_SPEED_DATASET:-}"
SPEED_ROWS="${MOJOLEARN_SPEED_ROWS:-}"
SPEED_ROUNDS="${MOJOLEARN_SPEED_ROUNDS:-5}"
SPEED_SIZE="${MOJOLEARN_SPEED_SIZE:-shipped}"
# PER ARM, in seconds. One lane that hangs must not eat the lease that
# fifteen other lanes are waiting for, and a lane whose driver spins on a
# shape nobody sized for an H100 is the likeliest way this leg loses its
# hour. 600 is deliberately generous for a measurement and deliberately
# far below the lease.
ARM_BUDGET="${MOJOLEARN_SPEED_ARM_BUDGET:-600}"
# PER DRIVER BUILD. Separate from the arm budget because a first-ever CUDA
# compile of a 2,400-line driver is a different order of wait from a
# measurement, and because a compile that hangs must not be given the whole
# lease on the theory that it might be nearly done.
BUILD_BUDGET="${MOJOLEARN_SPEED_BUILD_BUDGET:-900}"
# PER PIP INSTALL. RAPIDS is a multi-gigabyte resolve and is the single step
# that can consume a lease and return nothing.
PIP_BUDGET="${MOJOLEARN_SPEED_PIP_BUDGET:-900}"
# LightGBM's CUDA learner is not in any wheel and is measured at fifteen to
# thirty minutes from source. Bounded so a build that goes wrong cannot take
# the whole forest family with it.
LGBM_BUILD="${MOJOLEARN_SPEED_LGBM_BUILD:-1500}"
# Filled in after the lane list function is defined; see leg_speed_lanes.
SPEED_LANES=""
# phase8 only: the Apple column this box's column will be judged against.
APPLE_DIR="${MOJOLEARN_GEMM_LEG_APPLE_DIR:-}"
APPLE_MISSING=0
# phase8 only, and it is the same idea as --ready-timeout: the work is
# bounded IN CODE. The lease's watchdog is the backstop for a dead session,
# not a schedule, and a box killed by its own watchdog mid-bootstrap dies
# with every card still on it because the fetch never runs. 0 means "the
# lease minus the reserve".
WORK_TIMEOUT="${MOJOLEARN_GEMM_LEG_WORK_TIMEOUT:-0}"
# DEVIATION 972: which e1_bootstrap phases the box runs. EMPTY MEANS ALL, so
# every existing caller is unchanged. Set it to 8 to answer one lane question
# for the price of a compile instead of an hour.
#
# Leg 12 is why. The full bootstrap took fifty minutes on an RTX 4090 and died
# on the work bound at phase 5 with identical_cards=0, and --minutes is capped
# at 60 BY NAME, so on that box a full round does not fit in a lease at all.
# A phase-8-only run measured 74 SECONDS on the M4 and produced all seven lane
# cards BYTE-IDENTICAL to the full run's, which is the evidence that skipping
# 0-7 does not move a lane card.
#
# A PHASE-SUBSET COLUMN IS NOT A ROUND. Judge sections 1-6 will fail on it,
# correctly. Only section 7 is answerable. tools/e1_bootstrap.sh says the same
# thing at more length and it is the file that enforces it.
E1_PHASES="${MOJOLEARN_GEMM_LEG_E1_PHASES:-}"
# DEVIATION 973: which phase-8 lanes the box runs. Empty means all of them.
# Leg 12 proved the need: the lane order puts mamba LAST behind gemm's device
# check, the largest compile in the set, so on a cold box mamba was never
# reached whatever the bound was. Recorded in leg.txt beside e1_phases so a
# narrowed column can never read as a full one.
E1_LANES="${MOJOLEARN_GEMM_LEG_E1_LANES:-}"

# DEVIATION 974: THE HOST'S CUDA VERSION IS A RENTAL CRITERION, NOT LUCK.
#
# MAX refuses to run at all below a floor, and says so precisely:
#
#   Your current NVIDIA GPU driver version is not supported.
#     Required: driver version >= 580 (CUDA >= 13.0)
#     Detected: 570.169 (CUDA 12.8)
#
# RunPod hosts differ. Leg 12 rented the SAME gpu type from the SAME image
# twice: the first box ran gemm and cd to byte-identical cards against the
# M4, and the second refused every kernel launch on a 570.169 host. That is a
# machine lottery, and losing it costs a whole lease and produces a column
# that looks like a failure of ours.
#
# `allowedCudaVersions` is a create-time constraint in the v1 schema
# (/v1/openapi.json, enum 13.0 down to 11.8), so the pod is only placed on a
# host that satisfies it. Defaulting to 13.0 makes an incompatible host a
# CREATE failure, which is free and instant, rather than a launch failure
# fifty minutes in on a paid box.
#
# Set MOJOLEARN_GEMM_LEG_CUDA to widen it if MAX's floor ever drops. Widening
# it below MAX's actual floor just moves the failure back to where it was.
# EVERY DRIVER THAT CAN RUN THE IMAGE, not just the newest. The default was
# `"13.0"` alone, which asks RunPod for hosts carrying one specific driver
# generation. The images this leg uses are CUDA 12.4, and a 12.4 container
# runs on any driver at or above it, so pinning the newest only narrows the
# pool of machines that can satisfy the request -- and on a scarce GPU type
# that is the difference between a box and a wait. Listed newest first
# because the API reads it as an acceptable set, not a preference order.
CUDA_VERSIONS="${MOJOLEARN_GEMM_LEG_CUDA:-\"13.0\",\"12.9\",\"12.8\",\"12.7\",\"12.6\",\"12.5\",\"12.4\"}"
WORK_RESERVE=600
# phase8 only: where the box's bootstrap directory lands on this Mac.
E1_DEST=""
LANES=""

VENDOR=""
MODE="dry"
MINUTES="${MOJOLEARN_GEMM_LEG_MINUTES:-60}"
MINUTES_CAP=60
GPU_ID=""
IMAGE=""
SSH_TARGET=""
LOCAL_CARD="${MOJOLEARN_GEMM_LEG_LOCAL_CARD:-}"
SWEEP=0
READY_TIMEOUT="${MOJOLEARN_GEMM_LEG_READY_TIMEOUT:-600}"
CARD_FULL="${MOJOLEARN_GEMM_CARD_FULL:-}"
# Passed through to the remote card run so a SECOND leg can bring `.bin`
# sidecars home for the cell-level ladder. It is a SUBSTRING match over tags
# (core/identity_trace.mojo), so keep it narrow: this profile's largest input
# stage is 4,000,000 f32 and a loose substring will dump tens of megabytes
# per shape and then fetch all of them over ssh.
LEG_DUMP="${MOJOLEARN_IDENTITY_TRACE_DUMP:-}"

POD_ID=""
POD_NAME=""
POD_TERMINATED=0
DEADMAN_PID=""
DEADMAN_DIR=""
FETCH_RED=0
KEY_RED=0
ARMED=0
TMPD=""
CURLRC=""
RP_CODE=""
RP_BODY=""

DIFFER="tools/identity_trace_diff.py"
PY=/usr/bin/python3
[ -x "$PY" ] || PY=python3

leg_usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "options: --rent --dry-run --minutes N --gpu ID --image REF"
    echo "         --ssh TARGET --local-card PATH --column-sweep"
    echo "         --ready-timeout SECONDS"
    echo "         --payload gemm|phase8|speed  (--phase8 is the short form)"
    echo "         --apple-dir DIR         phase8: the Apple column to be"
    echo "                                 judged against"
    echo "         --work-timeout SECONDS  phase8/speed: bound the work"
    echo "         --family NAME           speed: gemmseq | classical |"
    echo "                                 forest. One family per leg."
    echo "         --rounds N              speed: timed rounds per arm"
}

leg_say() { printf '[%s %s] %s\n' "$(date +%T)" "${VENDOR:-leg}" "$*"; }
leg_die() { printf '\n%s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# argument validation -- refuse everything unknown BY NAME
# ---------------------------------------------------------------------------

while [ $# -gt 0 ]; do
    case "$1" in
        nvidia|amd)
            [ -z "$VENDOR" ] || leg_die "gemm_remote_leg: two vendors given ('$VENDOR' then '$1'). One leg is one vendor: run it twice."
            VENDOR="$1" ;;
        reap)
            shift
            # LAYER TWO, on its own. The orphan-recovery entry point, here
            # because the moment you need it is the moment you least want to
            # be remembering another script's flag order.
            MODE="reap"
            POD_ID="${1:-}"
            break ;;
        --rent)          MODE="rent" ;;
        --dry-run)       MODE="dry" ;;
        --minutes)       shift; MINUTES="${1:-}" ;;
        --gpu)           shift; GPU_ID="${1:-}" ;;
        --image)         shift; IMAGE="${1:-}" ;;
        --ssh)           shift; SSH_TARGET="${1:-}" ;;
        --local-card)    shift; LOCAL_CARD="${1:-}" ;;
        --ready-timeout) shift; READY_TIMEOUT="${1:-}" ;;
        --column-sweep)  SWEEP=1 ;;
        --payload)       shift; PAYLOAD="${1:-}" ;;
        --phase8)        PAYLOAD="phase8" ;;
        --speed)         PAYLOAD="speed" ;;
        --family)        shift; SPEED_FAMILY="${1:-}" ;;
        --rounds)        shift; SPEED_ROUNDS="${1:-}" ;;
        --dataset)       shift; SPEED_DATASET="${1:-}" ;;
        --rows)          shift; SPEED_ROWS="${1:-}" ;;
        --no-lgbm-cuda)  SPEED_LGBM_CUDA=0 ;;
        --allow-concurrent) LEG_ALLOW_CONCURRENT=1 ;;
        --smoke)         SPEED_SIZE="smoke" ;;
        --large)         SPEED_SIZE="large" ;;
        --apple-dir)     shift; APPLE_DIR="${1:-}" ;;
        --work-timeout)  shift; WORK_TIMEOUT="${1:-}" ;;
        -h|--help|help)  leg_usage; exit 0 ;;
        *)
            echo "gemm_remote_leg: unknown argument '$1'." >&2
            echo "  The target vendor must be exactly 'nvidia' or 'amd'." >&2
            echo "  Not 'nv', not 'NVIDIA', not 'apple' (this Mac is the" >&2
            echo "  reference leg and is never rented), not a GPU model" >&2
            echo "  name -- for that use --gpu, which selects hardware" >&2
            echo "  WITHIN a vendor and does not select the vendor." >&2
            echo >&2
            leg_usage >&2
            exit 2 ;;
    esac
    shift
done

if [ "$MODE" != "reap" ]; then
    [ -n "$VENDOR" ] || { echo "gemm_remote_leg: no vendor given." >&2; leg_usage >&2; exit 2; }
fi

# THE HARD CAP IS CHECKED BEFORE ANYTHING ELSE, including before the key, so
# that an over-long lease is refused whether or not renting is even possible.
case "$MINUTES" in
    ''|*[!0-9]*) leg_die "gemm_remote_leg: --minutes must be a whole number of minutes, got '$MINUTES'." ;;
esac
if [ "$MINUTES" -lt 1 ] || [ "$MINUTES" -gt "$MINUTES_CAP" ]; then
    echo "gemm_remote_leg: REFUSING a ${MINUTES}-minute lease." >&2
    echo "  ONE HOUR IS A HARD CAP on this lane (IDENTICAL_GEMM_PLAN.md," >&2
    echo "  RENTING rule 2), not a default to extend past casually. It is" >&2
    echo "  refused here rather than clamped, because a clamp turns a" >&2
    echo "  deliberate over-run into a silent under-run and the operator" >&2
    echo "  never learns which they got." >&2
    echo >&2
    echo "  If the work genuinely needs longer, EXTENDING IS RE-ARMING and" >&2
    echo "  it is a decision a human makes with the box in front of them:" >&2
    echo "    tools/runpod_guard.sh extend <pod-id> '<ssh target>' 60" >&2
    exit 2
fi
case "$READY_TIMEOUT" in
    ''|*[!0-9]*) leg_die "gemm_remote_leg: --ready-timeout must be seconds, got '$READY_TIMEOUT'." ;;
esac

# THE PAYLOAD IS REFUSED BY NAME LIKE THE VENDOR IS, and for the same reason:
# a payload this script does not understand must not fall through to the
# default one and rent a box to run the wrong program.
case "$PAYLOAD" in
    gemm|phase8|speed) : ;;
    *)
        echo "gemm_remote_leg: unknown payload '$PAYLOAD'." >&2
        echo "  gemm   (default) gemm_device_check.mojo + gemm_card.sh" >&2
        echo "                   device, diffed HERE against this Mac's" >&2
        echo "                   Apple card." >&2
        echo "  phase8           tools/e1_bootstrap.sh, whose phase 8 writes" >&2
        echo "                   the seven lanes' cards; judged by" >&2
        echo "                   tools/e3_round_judge.sh section 7, not by" >&2
        echo "                   this leg." >&2
        echo "  speed            THE FAST PATH AGAINST THE VENDOR. Builds" >&2
        echo "                   and runs the bench/speed/ drivers with NO" >&2
        echo "                   -D MOJOLEARN_NUMERIC_IDENTICAL, beside the" >&2
        echo "                   vendor's own library on the same box, and" >&2
        echo "                   brings the FSPEED lines home for" >&2
        echo "                   tools/fast_speed_table.py. One --family" >&2
        echo "                   per leg; it does not fit in one lease." >&2
        exit 2 ;;
esac

if [ "$PAYLOAD" = "speed" ]; then
    # THE ALIAS IS RESOLVED HERE, after the argument loop, so that BOTH the
    # `--family trees` flag and the MOJOLEARN_SPEED_FAMILY environment form
    # reach it. It used to sit up with the defaults, which the flag overwrites
    # afterwards, so the flag form was refused by a message that named `trees`
    # as accepted in the same breath. Nothing was rented; the refusal is the
    # first thing this payload validates.
    if [ "$SPEED_FAMILY" = "trees" ]; then SPEED_FAMILY="forest"; fi

    # THE SPEED PAYLOAD HAS NO APPLE REFERENCE AND NO CARD DIFF, so the two
    # gemm-payload flags that name one are refused rather than ignored. An
    # ignored flag is an operator who believes something ran.
    if [ "$SWEEP" = "1" ]; then
        echo "gemm_remote_leg: --column-sweep belongs to the gemm payload." >&2
        echo "  The speed payload measures WALL CLOCK on one box; a column" >&2
        echo "  sweep is a bit-identity gate and has no timing to give." >&2
        exit 2
    fi
    if [ -n "$LOCAL_CARD" ]; then
        echo "gemm_remote_leg: --local-card belongs to the gemm payload." >&2
        echo "  The speed payload has no Apple reference at all. It is a" >&2
        echo "  measurement of ONE box against the libraries on THAT box," >&2
        echo "  and a card from a different machine cannot enter it." >&2
        exit 2
    fi
    # THE LANE LIST PER FAMILY, COMPOSED HERE AND NOT ON THE BOX, so that
    # what ran is recorded in this leg's own artifacts rather than being
    # whatever the box happened to find. A lane named here whose driver does
    # not know it fails ONE lane and is a line in the log; a family whose
    # DRIVER is missing at this commit is refused earlier, by the archive
    # check, for nothing.
    case "$SPEED_FAMILY" in
        gemmseq)   SPEED_LANES="${MOJOLEARN_SPEED_LANES:-gemm transformer attention mlp rmsnorm mamba selective_scan}" ;;
        classical) SPEED_LANES="${MOJOLEARN_SPEED_LANES:-kmeans dbscan pca ols knn cd kde linkage svm metrics ivf hdbscan cholesky gmm gp krr nystroem rbfsampler resample spectral holtwinters kpss}" ;;
        forest)    SPEED_LANES="${MOJOLEARN_SPEED_LANES:-gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et iforest}" ;;
    esac
    case "$SPEED_FAMILY" in
        gemmseq|classical|forest) : ;;
        *)
            echo "gemm_remote_leg: --family must be one of gemmseq, classical," >&2
            echo "  trees (alias: forest). Got '$SPEED_FAMILY'." >&2
            echo "  trees covers EVERY decision-tree learner, boosting" >&2
            echo "  included: gbdt-symmetric, gbdt-depthwise, gbdt-lossguide," >&2
            echo "  rf, et, iforest." >&2
            echo "  ONE FAMILY PER LEASE, and that is arithmetic rather than" >&2
            echo "  taste: a cold box has to pixi-install, compile drivers" >&2
            echo "  for CUDA for the first time, and pip-install the vendor" >&2
            echo "  libraries. Measured elsewhere in this file: a full" >&2
            echo "  bootstrap took fifty minutes on a 4090 and died on its" >&2
            echo "  work bound, and --minutes is capped at $MINUTES_CAP BY NAME." >&2
            exit 2 ;;
    esac
    case "$SPEED_ROUNDS" in
        ''|*[!0-9]*) leg_die "gemm_remote_leg: --rounds must be a count, got '$SPEED_ROUNDS'." ;;
    esac
fi

if [ "$PAYLOAD" = "phase8" ]; then
    # Two gemm-payload options that phase8 cannot honor. Refused by name
    # rather than ignored: an ignored flag is an operator who believes
    # something ran.
    if [ "$SWEEP" = "1" ]; then
        echo "gemm_remote_leg: --column-sweep belongs to the gemm payload." >&2
        echo "  It runs tools/gemm_column_invariance.sh on the remote" >&2
        echo "  backend; phase 8 does not run it and this leg will not" >&2
        echo "  pretend it did. Drop the flag or drop --payload phase8." >&2
        exit 2
    fi
    if [ -n "$LOCAL_CARD" ]; then
        echo "gemm_remote_leg: --local-card belongs to the gemm payload." >&2
        echo "  Under phase8 the reference is a whole Apple BOOTSTRAP" >&2
        echo "  DIRECTORY rather than one card, because the judge diffs" >&2
        echo "  seven lanes out of it. Name it with --apple-dir <dir>." >&2
        exit 2
    fi
fi

case "$WORK_TIMEOUT" in
    ''|*[!0-9]*) leg_die "gemm_remote_leg: --work-timeout must be seconds, got '$WORK_TIMEOUT'." ;;
esac
if [ "$WORK_TIMEOUT" -eq 0 ]; then
    WORK_TIMEOUT=$(( MINUTES * 60 - WORK_RESERVE ))
    if [ "$WORK_TIMEOUT" -lt 300 ]; then WORK_TIMEOUT=300; fi
fi

if [ "$MODE" != "reap" ]; then
    case "$VENDOR" in
        nvidia) : "${GPU_ID:=$LEG_GPU_NVIDIA}"; : "${IMAGE:=$LEG_IMAGE_NVIDIA}"; SMI_CMD='nvidia-smi --query-gpu=name,driver_version --format=csv,noheader' ;;
        amd)    : "${GPU_ID:=$LEG_GPU_AMD}";    : "${IMAGE:=$LEG_IMAGE_AMD}";    SMI_CMD='rocm-smi --showproductname'
                # AMD hosts advertise no CUDA version, so the CUDA allow-list
                # filters out EVERY host that could hold this pod: the
                # 2026-08-26_162559 create died "no instances currently
                # available" with the list attached. Send it only when the
                # operator asked for one by name.
                CUDA_VERSIONS="${MOJOLEARN_GEMM_LEG_CUDA:-}" ;;
    esac
    VLABEL=$(echo "$VENDOR" | tr '[:lower:]' '[:upper:]')
fi

STAMP=$(date +%Y-%m-%d_%H%M%S)
PSUF=""
if [ "$PAYLOAD" = "phase8" ]; then PSUF="-phase8"; fi
if [ "$PAYLOAD" = "speed" ]; then PSUF="-speed-$SPEED_FAMILY"; fi
if [ "$MODE" = "dry" ]; then
    OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-${VENDOR}${PSUF}-dryrun}"
else
    OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-${VENDOR}${PSUF}}"
fi
if [ "$MODE" != "reap" ] && [ "$PAYLOAD" = "phase8" ]; then
    # WHERE THE JUDGE WILL LOOK. tools/e3_round_judge.sh takes one bootstrap
    # directory per machine out of bench/results/e1/ and labels each column
    # from the BASENAME (its label_of: *-nv* -> NVIDIA, *-amd* -> AMD). The
    # name is composed HERE rather than taken from the pod, because a RunPod
    # container's hostname is a hex id and `<stamp>-<hex>` would land this
    # column in the judge's default branch under a name nobody can read.
    # tools/e2_remote_leg.sh gets the same effect for free: its droplets are
    # NAMED mojolearn-e2-nv / -amd, so `hostname -s` already carries it.
    E1_DEST="bench/results/e1/${STAMP}-runpod-${VENDOR}"
fi

# ---------------------------------------------------------------------------
# teardown -- an EXIT trap, so every failure path above it lands here
# ---------------------------------------------------------------------------

leg_lease_report() {
    [ -n "$POD_ID" ] || return 0
    echo "  lease at exit:"
    if tools/runpod_guard.sh check "$POD_ID" 2>&1 | sed 's/^/    /'; then :; else
        echo "    (no lease recorded -- the box was never armed)"
    fi
}

leg_verify_terminated() {
    # ASK, DO NOT ASSUME. A DELETE that returned 200 and a pod that is gone
    # are different claims, and only the second one stops the bill.
    [ -n "$POD_ID" ] || return 0
    [ -n "$CURLRC" ] || return 0
    _i=1
    while [ "$_i" -le 6 ]; do
        rp_call GET "$RP_PODS_PATH/$POD_ID" || true
        case "$RP_CODE" in
            404) echo "  VERIFIED: $POD_ID is gone (HTTP 404)."; POD_TERMINATED=1; return 0 ;;
            2*)
                _st=$(rp_json "d.get('desiredStatus') or d.get('status') or (d.get('pod') or {}).get('desiredStatus') or ''")
                case "$_st" in
                    TERMINATED|EXITED|terminated|exited)
                        echo "  VERIFIED: $POD_ID reports status $_st."
                        POD_TERMINATED=1; return 0 ;;
                    *) echo "  $POD_ID still reports status '${_st:-?}' (attempt $_i/6)" ;;
                esac ;;
            *) echo "  status query returned HTTP $RP_CODE (attempt $_i/6)" ;;
        esac
        sleep 10
        _i=$((_i + 1))
    done
    echo
    echo "  ############################################################"
    echo "  # THIS POD MAY STILL BE BILLING: $POD_ID"
    echo "  # The API did not confirm it is gone. The on-pod watchdog is"
    echo "  # the remaining layer and it fires at the lease deadline, but"
    echo "  # DO NOT LEAVE THIS TO IT. Terminate by hand now:"
    echo "  #   tools/runpod_guard.sh reap --force $POD_ID"
    echo "  #   https://console.runpod.io/pods"
    echo "  ############################################################"
    return 1
}

# TARGETED, AND STILL NOT `runpod_guard.sh reap --force <pod>` -- but for a
# different reason than when this was written.
#
# It used to be a workaround: that helper shifted `--force` off and shifted
# the pod id off WITH it, so `reap --force <pod-id>` read like a targeted
# terminate and was a reap-everything. CLOSED 2026-08-24 -- the pod id scopes
# it now, `--force` with no id is refused, and the reap-everything form is
# `--force --all`. What is left is a real difference: this leg DELETEs the pod
# it created, by id, then removes that pod's lease file so `check` and `list`
# stay honest, and then ASKS THE API whether the pod is actually gone. The
# guard does not verify, and "the DELETE returned 200" and "the pod is gone"
# are different claims.
leg_terminate() {
    [ -n "$POD_ID" ] || return 0
    [ "$POD_TERMINATED" = "1" ] && return 0
    echo "  terminating $POD_ID (layer two, from this machine)"
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "    NO KEY IN THIS ENVIRONMENT -- this machine cannot terminate"
        echo "    anything. The on-pod watchdog is the only layer left."
        return 0
    fi
    for _u in "$RP_PODS_PATH/$POD_ID" "$RP_HOST_V1/v1/pods/$POD_ID"; do
        rp_call DELETE "$_u"
        echo "    DELETE $_u -> HTTP $RP_CODE"
        case "$RP_CODE" in 2*|404) break ;; esac
    done
    rm -f "$LEASE_DIR/$POD_ID.lease"
    leg_verify_terminated || true
}

# Every line below runs from the EXIT trap, so shellcheck calls it dead.
# shellcheck disable=SC2317
leg_teardown() {
    _rc=$?
    trap - EXIT INT TERM
    if [ -n "$POD_ID" ] && [ "$MODE" = "rent" ]; then
        echo
        echo "== teardown (exit $_rc) =="
        leg_lease_report
        leg_terminate
        {
            echo "pod=$POD_ID"
            echo "armed=$ARMED"
            echo "terminated=$POD_TERMINATED"
            echo "deadman_pid=$DEADMAN_PID"
            echo "exit=$_rc"
            echo "at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } >> "$OUT/teardown.txt" 2>/dev/null || true
    fi
    # THE DEAD-MAN IS CANCELLED ONLY BY A CONFIRMED TERMINATE (DEVIATION 862).
    # Cancelling it after a DELETE that could not be verified would remove the
    # one layer still able to end that pod from this machine, at exactly the
    # moment it is needed. A leg that never created a pod cancels it too --
    # there is nothing for it to guard and it holds a credential on disk.
    if [ -n "$POD_ID" ] && [ "$POD_TERMINATED" != "1" ]; then
        echo "  THE DEAD-MAN IS LEFT ARMED ON PURPOSE: $POD_ID was not"
        echo "  confirmed gone, so the layer that can still end it from here"
        echo "  keeps running (pid $DEADMAN_PID, $DEADMAN_DIR)."
        echo "  Terminate by hand, and only then cancel it:"
        echo "    tools/gemm_remote_leg.sh reap $POD_ID"
        echo "    kill $DEADMAN_PID; rm -rf $DEADMAN_DIR"
    else
        leg_cancel_deadman
    fi
    [ -n "$TMPD" ] && rm -rf "$TMPD"
    exit "$_rc"
}
trap leg_teardown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------
# the API, with the key kept out of every argv
# ---------------------------------------------------------------------------

leg_load_key() {
    if [ -n "${MOJOLEARN_RUNPOD_KEY_FILE:-}" ]; then
        _kf="$MOJOLEARN_RUNPOD_KEY_FILE"
        [ -f "$_kf" ] || leg_die "MOJOLEARN_RUNPOD_KEY_FILE=$_kf does not exist."
        leg_assert_keyfile_hygiene "$_kf"
        RUNPOD_API_KEY=$(cat "$_kf")
        export RUNPOD_API_KEY
    fi
    [ -n "${RUNPOD_API_KEY:-}" ] || return 1
    return 0
}

leg_assert_keyfile_hygiene() {
    _f="$1"
    _perm=$(stat -f '%OLp' "$_f" 2>/dev/null || stat -c '%a' "$_f" 2>/dev/null || echo "?")
    [ "$_perm" = "600" ] || leg_die "key file $_f is mode $_perm, must be 600.  chmod 600 '$_f'"
    case "$_f" in
        "$REPO"/*)
            leg_die "key file $_f is INSIDE this repository. A key in the checkout is a key one 'git add' away from a public commit. Move it out." ;;
    esac
    if git ls-files --error-unmatch "$_f" >/dev/null 2>&1; then
        leg_die "key file $_f is TRACKED BY GIT. Remove it from the index before using it."
    fi
    return 0
}

leg_curlrc() {
    # `printf` is a SHELL BUILTIN here, so no process is spawned and the key
    # never becomes an argv anywhere on this machine. `curl -K` then reads
    # the Authorization header out of the 0600 file. Checks K1-K5 in the dry
    # run verify this rather than trusting it.
    ( umask 077
      printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' \
          "$RUNPOD_API_KEY" > "$CURLRC" )
    leg_assert_keyfile_hygiene "$CURLRC"
}

rp_call() {
    _m="$1"; _p="$2"; _d="${3:-}"
    case "$_p" in
        http://*|https://*) _url="$_p" ;;
        *)                  _url="$RP_HOST$_p" ;;
    esac
    RP_BODY="$TMPD/rp.body"
    : > "$RP_BODY"
    if [ -n "$_d" ]; then
        RP_CODE=$(curl -K "$CURLRC" -o "$RP_BODY" -w '%{http_code}' \
            -X "$_m" -H 'Content-Type: application/json' \
            --data-binary "@$_d" "$_url" 2>>"$TMPD/curl.err") || RP_CODE=000
    else
        RP_CODE=$(curl -K "$CURLRC" -o "$RP_BODY" -w '%{http_code}' \
            -X "$_m" "$_url" 2>>"$TMPD/curl.err") || RP_CODE=000
    fi
    return 0
}

rp_json() {
    # $1 is a python expression over `d`, the parsed body. Prints "" on any
    # failure rather than raising, because every caller is already checking
    # for empty and a traceback here would bury the HTTP code that matters.
    "$PY" - "$RP_BODY" "$1" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if isinstance(d, list):
    d = {"items": d}
try:
    v = eval(sys.argv[2], {"__builtins__": {"str": str}}, {"d": d})
except Exception:
    sys.exit(0)
if v is None:
    v = ""
sys.stdout.write(str(v))
PYEOF
}

# ---------------------------------------------------------------------------
# the local (Apple) reference leg
# ---------------------------------------------------------------------------

LEG_SOURCE_PATHS="gemm/mojo_only bench/gemm_card_main.mojo bench/gemm_shapes.mojo tools/gemm_card.sh tools/with_identical_mode.sh tools/with_build_lock.sh mojo_only pixi.toml pixi.lock"

# DEVIATION 865 -- THE DIRTY-TREE RULE APPLIES TO phase8 TOO, ONE STEP
# REMOVED, AND THE gemm LIST IS THE WRONG LIST FOR IT.
#
# The phase8 payload builds nothing on this Mac, so at first reading the
# rule looks like it does not apply. It does. That payload's reference is an
# APPLE BOOTSTRAP DIRECTORY, and that directory was produced by running
# `bash tools/e1_bootstrap.sh` FROM THIS WORKING TREE and filed under
# whatever `git rev-parse HEAD` said at the time. The box runs `git archive`
# of that same sha. So an uncommitted edit under any lane phase 8 compiles
# makes the two columns two different programs wearing one commit -- and
# leg_apple_column cannot see it, because all it can compare is the sha, and
# the sha is exactly what the dirt is hiding behind. One variable is the
# device; an uncommitted local edit is a second one and it is invisible in
# every card.
#
# This is the list of what phase 8 actually compiles: the bootstrap itself,
# the injector and the build lock, and each lane's source. Four sessions
# share this checkout, so it WILL block sometimes. That is the check working:
# commit or stash, then rent.
LEG_SOURCE_PATHS_PHASE8="tools/e1_bootstrap.sh tools/with_identical_mode.sh tools/with_build_lock.sh tools/gemm_card.sh bench/gemm_card_main.mojo bench/gemm_shapes.mojo gemm/mojo_only solver kde hierarchy svm metrics mamba mojo_only bindings python/mojolearn pixi.toml pixi.lock"

# THE SPEED PAYLOAD'S SOURCE FOOTPRINT. Wider than either of the others,
# because it compiles a driver per family and every lane those drivers call.
# It is deliberately NOT the whole tree: the tree-clean gate exists so the
# thing being measured is the device, and documentation cannot reach a
# millisecond. But a benchmark driver, a lane it imports, or a vendor arm
# script CAN, so all three are in here.
LEG_SOURCE_PATHS_SPEED="bench/speed tools/speed_gemm_arm.py tools/speed_cuml_arm.py tools/speed_torch_seq.py tools/speed_gbdt_arm.py tools/vendor_gemm_price.py tools/fast_speed_table.py bench/gemm_shapes.mojo core gemm mojo_only bindings python/mojolearn pixi.toml pixi.lock"

leg_check_tree_clean() {
    # THE LOCAL CARD COMES FROM THE WORKING TREE AND THE REMOTE CARD COMES
    # FROM THE COMMIT. If those differ for any file that can reach the bits,
    # the diff has two variables in it and the device is not the one being
    # measured. Documentation and results directories are deliberately not
    # in the list: they cannot reach a float.
    # The list is per payload: phase8 compiles seven lanes, the gemm payload
    # compiles one (DEVIATION 865).
    _paths="$LEG_SOURCE_PATHS"
    if [ "$PAYLOAD" = "phase8" ]; then _paths="$LEG_SOURCE_PATHS_PHASE8"; fi
    if [ "$PAYLOAD" = "speed" ]; then _paths="$LEG_SOURCE_PATHS_SPEED"; fi
    # The list is a deliberate word list, so it is unquoted.
    # shellcheck disable=SC2086
    _dirty=$(git status --porcelain -- $_paths 2>/dev/null || true)
    if [ -n "$_dirty" ]; then
        echo "  the working tree is DIRTY for paths the $PAYLOAD payload's"
        echo "  numbers can reach:"
        echo "$_dirty" | sed 's/^/    /'
        # AN UNTRACKED FILE COUNTS TOO, and that is the half people argue
        # with. `git archive` does not ship it, so the BOX never sees it --
        # but the Apple side does: under phase8 that side is a bootstrap
        # directory built from THIS TREE, and under gemm it is a card built
        # from this tree. A new file that any lane imports is therefore in
        # one column and not the other, which is the two-variable case wearing
        # a different hat. Four sessions share this checkout, so this WILL
        # block sometimes; commit or stash, then rent.
        return 1
    fi
    return 0
}

leg_source_sha_recipe() {
    # The same recipe on both ends, byte for byte. Introduced by
    # tools/e1_unsupervised.sh after the AMD leg's commit.txt read "unknown",
    # which is a belief about what was shipped rather than a comparison.
    ( cd "$1" && \
      { find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
          | LC_ALL=C sort | xargs shasum -a 256 2>/dev/null || \
        find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
          | LC_ALL=C sort | xargs sha256sum ; } \
      | { shasum -a 256 2>/dev/null || sha256sum ; } | awk '{print $1}' )
}

leg_local_card() {
    mkdir -p "$OUT/local"
    if [ -n "$LOCAL_CARD" ]; then
        [ -f "$LOCAL_CARD" ] || leg_die "--local-card $LOCAL_CARD does not exist."
        cp "$LOCAL_CARD" "$OUT/local/apple.card"
        [ -f "$LOCAL_CARD.log" ] && cp "$LOCAL_CARD.log" "$OUT/local/apple.card.log"
        LOCAL_CARD="$OUT/local/apple.card"
        leg_say "local card SUPPLIED (not generated): $LOCAL_CARD"
        leg_say "  nothing here can check it came from this commit. That is on the operator."
        return 0
    fi
    leg_say "generating the Apple reference card (device arm, IDENTICAL)"
    LOCAL_CARD="$OUT/local/apple.card"
    # NOT `cmd 2>&1 | sed`. A pipeline's status is its LAST command's, so
    # piping into `sed` for indentation reports SED's success and swallows
    # the driver's failure. That spelling was in the first version of this
    # function AND in leg_arm, where it would have turned "the guard refused
    # to arm" into "armed" -- the exact rule this file exists to enforce,
    # defeated by an indent. Redirect to a file, then indent the file.
    if MOJOLEARN_GEMM_CARD_FULL="$CARD_FULL" \
       tools/gemm_card.sh device "$LOCAL_CARD" > "$OUT/local/generate.log" 2>&1; then
        sed 's/^/    /' "$OUT/local/generate.log"
    else
        sed 's/^/    /' "$OUT/local/generate.log"
        return 1
    fi
    [ -s "$LOCAL_CARD" ] || return 1
    return 0
}

# ---------------------------------------------------------------------------
# the contamination guard and the artifact checks
# ---------------------------------------------------------------------------

leg_witness_mode() {
    # Both banners, one function. bench/gemm_card_main.mojo prints
    #   == bench/gemm_card_main.mojo [IDENTICAL] ==
    # and gemm/mojo_only/gemm_device_check.mojo prints
    #   == gemm/mojo_only/gemm_device_check.mojo [IDENTICAL]  sabotage: ... ==
    [ -f "$1" ] || { printf ''; return 0; }
    sed -n 's/^== [A-Za-z0-9_/.]*\.mojo \[\([A-Z][A-Z]*\)\].*$/\1/p' "$1" | head -1
}

leg_require_file() {
    if [ ! -s "$1" ]; then
        echo "  MISSING: $1"
        echo "    ($2)"
        return 1
    fi
    return 0
}

leg_require_identical() {
    _log="$1"; _what="$2"
    if [ ! -f "$_log" ]; then
        echo "  $_what: NO LOG at $_log -- the mode cannot be read back, so"
        echo "      nothing this run produced can be claimed as IDENTICAL."
        return 1
    fi
    _got=$(leg_witness_mode "$_log")
    if [ "$_got" != "IDENTICAL" ]; then
        echo "  $_what: CONTAMINATED -- the remote binary reports mode"
        echo "      [${_got:-<no banner>}], not [IDENTICAL]. The card is a"
        echo "      claim about NUMERIC_IDENTICAL and this run cannot support"
        echo "      it. Do not read the diff below; it is a comparison"
        echo "      between two different arithmetics."
        echo "      Check that -D MOJOLEARN_NUMERIC_IDENTICAL=1 reached the"
        echo "      remote build:  $_log"
        return 1
    fi
    echo "  $_what: mode read back from the run itself = [$_got]"
    return 0
}

# ---------------------------------------------------------------------------
# the phase8 payload: the reference column, the lane list, the mode witness
# ---------------------------------------------------------------------------

leg_lanes() {
    # THE LANE LIST COMES FROM THE JUDGE, NOT FROM A COPY OF IT. `mamba` was
    # added to tools/e1_bootstrap.sh phase 8 and to tools/e3_round_judge.sh
    # section 7 on 2026-08-23; a hardcoded list in this file would have gone
    # on reporting a green six-lane leg with the new lane silently absent,
    # which is the whole failure mode this leg exists to prevent. The
    # fallback is announced on stderr so a silent extraction failure cannot
    # look like a lane list.
    _l=$(sed -n 's/^for lane in \(.*\); do$/\1/p' tools/e3_round_judge.sh | head -1)
    if [ -z "$_l" ]; then
        _l="gemm cd kde linkage svm metrics mamba"
        echo "  NOTE: the lane list could not be read out of" >&2
        echo "        tools/e3_round_judge.sh; falling back to the literal" >&2
        echo "        list '$_l', which may be stale." >&2
    fi
    printf '%s' "$_l"
}

leg_apple_column() {
    # THE REFERENCE FOR THE phase8 PAYLOAD IS A WHOLE APPLE BOOTSTRAP
    # DIRECTORY, not a card: tools/e3_round_judge.sh section 7 diffs
    # <apple>/lanes/<lane>.identical.card against the box's, and its section
    # 1 refuses the round outright unless every column records the SAME
    # commit. Renting a box to produce the second column when the first does
    # not exist at this commit buys cards with nothing to compare them to,
    # which is exactly what the gemm payload's L1 refuses to do.
    APPLE_LANES=""
    _cand=""
    if [ -n "$APPLE_DIR" ]; then
        _cand="${APPLE_DIR%/}"
        if [ ! -d "$_cand/lanes" ]; then
            echo "  --apple-dir $_cand has no lanes/ subdirectory, so phase 8"
            echo "  never ran in it and it is not a reference column."
            return 1
        fi
    else
        for _d in $(ls -dt bench/results/e1/*/ 2>/dev/null || true); do
            [ -d "${_d}lanes" ] || continue
            _c=$(head -1 "${_d}commit.txt" 2>/dev/null || echo "")
            if [ "$_c" = "$COMMIT" ]; then _cand="${_d%/}"; break; fi
        done
        if [ -z "$_cand" ]; then
            echo "  NO APPLE COLUMN AT THIS COMMIT. Nothing under"
            echo "  bench/results/e1/ has both a lanes/ directory and"
            echo "  commit.txt = $COMMIT."
            echo "  The Apple column is made on this Mac, for nothing, with"
            echo "      bash tools/e1_bootstrap.sh"
            echo "  at this commit; or name an existing one with --apple-dir."
            return 1
        fi
    fi
    _c=$(head -1 "$_cand/commit.txt" 2>/dev/null || echo "")
    if [ "$_c" != "$COMMIT" ]; then
        echo "  THE APPLE COLUMN IS AT A DIFFERENT COMMIT."
        echo "    $_cand records ${_c:-<no commit.txt>}"
        echo "    this leg would ship $COMMIT"
        echo "  The judge's section 1 requires every column to record the"
        echo "  same commit, so this pair cannot be judged whatever the"
        echo "  cards say. Two commits is two variables."
        return 1
    fi
    APPLE_DIR="$_cand"
    for _l in $LANES; do
        if [ -f "$_cand/lanes/$_l.identical.card" ]; then
            APPLE_LANES="$APPLE_LANES $_l"
        else
            echo "  NOTE: the Apple column has no lanes/$_l.identical.card."
            echo "        The judge will call that lane's pair MISSING and"
            echo "        the box's card will have nothing to answer."
        fi
    done
    echo "  Apple column: $_cand (commit $_c)"
    echo "  its IDENTICAL lane cards:$APPLE_LANES"
    return 0
}

leg_witness_lane_mode() {
    # THE LANE DRIVERS DO NOT ALL SPEAK THE SAME BANNER, so this reads the
    # two spellings tools/e1_bootstrap.sh's own read-back accepts -- `[MODE]`
    # and `mode MODE` -- out of the CARD first and then the LOG. Measured on
    # leg 11 (2026-08-23, bench/results/e1/2026-08-23_165142-mojolearn-e2-nv):
    # gemm, cd, kde and svm print `[IDENTICAL]`, metrics prints `mode
    # IDENTICAL`, and LINKAGE PRINTS NEITHER -- its line in that bootstrap
    # log ends in an empty witness. mamba prints none either. An absent
    # witness is reported as UNWITNESSED by the caller; it is never read as a
    # pass, and it is not treated as a contamination either, because a
    # silence is not a claim.
    grep -m1 -ohE '\[(FAST|IDENTICAL)\]|mode (FAST|IDENTICAL)' "$1" "$2" 2>/dev/null \
        | head -1 | sed -e 's/^\[//' -e 's/\]$//' -e 's/^mode //'
}

# ---------------------------------------------------------------------------
# WHAT CAME HOME FROM A SPEED LEG, AND THE ONE THING THAT MAKES IT RED
#
# This payload has no card and no diff, so "red" cannot mean "the bits
# disagree". It means one of three things, and only three:
#
#   1. NOTHING MEASURABLE CAME HOME. No FSPEED round lines at all is a leg
#      that rented a box and produced no measurement.
#   2. AN `ours` ARM REPORTED THE WRONG MODE. Every driver prints the mode it
#      COMPILED in, read from the comptime constant rather than from the flag
#      it was invoked with. This whole payload is the FAST path; one header
#      saying IDENTICAL is a correctly-labelled measurement of the wrong arm,
#      which this repository was bitten by three times on 2026-08-23, and it
#      voids the table rather than one row of it.
#   3. THE SOURCE SHA DID NOT MATCH, which step 8 checks for every payload.
#
# A REFUSED ARM IS NOT RED. cuML declining to install, LightGBM's CUDA
# learner not building inside the lease, a lane whose first-ever CUDA compile
# fails -- every one of those is a RESULT about this box and this image, it
# is counted and printed, and it does not make the leg fail. Treating it as a
# failure would create pressure to quietly drop the arm that refuses, and a
# table whose arms were selected by whether they cooperated is not a table.
# ---------------------------------------------------------------------------
leg_speed_artifacts() {
    _lt="$OUT/remote/leg.txt"
    _ld="$OUT/remote/logs"
    if [ ! -d "$_ld" ]; then
        echo "  NO logs/ DIRECTORY CAME HOME. The box produced no arm logs at"
        echo "  all, so there is nothing to build a table from. Read"
        echo "  $OUT/remote/console.log and $OUT/remote/leg.txt."
        return 1
    fi
    _rounds=$(cat "$_ld"/*.log 2>/dev/null | grep -c '^FSPEED ' || true)
    _refused=$(cat "$_ld"/*.log 2>/dev/null | grep -c '^FSPEED-REFUSED' || true)
    _fast=$(grep -h '^FSPEED-HEADER' "$_ld"/*.log 2>/dev/null | grep -c 'arm=ours mode=FAST' || true)
    _ident=$(grep -h '^FSPEED-HEADER' "$_ld"/*.log 2>/dev/null | grep -c 'arm=ours mode=IDENTICAL' || true)
    echo "  arm logs:        $(ls "$_ld" | wc -l | tr -d ' ')"
    echo "  timed rounds:    ${_rounds:-0}"
    echo "  refused arms:    ${_refused:-0}   (a result about this box, NOT a failure)"
    echo "  ours headers:    ${_fast:-0} FAST, ${_ident:-0} IDENTICAL"

    # PER LANE, so a lane that produced a vendor arm and no arm of ours is
    # visible here rather than as a missing row in the table forty minutes
    # from now. A row with no `ours` arm is the shape a first-ever CUDA
    # compile failure takes.
    for _l in $SPEED_LANES; do
        _o=$(grep -h "^FSPEED .*lane=$_l .*arm=ours " "$_ld"/*.log 2>/dev/null | wc -l | tr -d ' ')
        _v=$(grep -h "^FSPEED .*lane=$_l " "$_ld"/*.log 2>/dev/null | grep -vc 'arm=ours ' || true)
        if [ "${_o:-0}" = "0" ] && [ "${_v:-0}" = "0" ]; then
            echo "    $_l: NOTHING (neither arm produced a round)"
        elif [ "${_o:-0}" = "0" ]; then
            echo "    $_l: vendor only (${_v} rounds) -- OUR ARM DID NOT RUN"
        elif [ "${_v:-0}" = "0" ]; then
            echo "    $_l: ours only (${_o} rounds) -- no opponent ran on this box"
        else
            echo "    $_l: ours ${_o} rounds, opponents ${_v} rounds"
        fi
    done

    _bad=0
    if [ "${_rounds:-0}" -lt 1 ]; then
        echo "  NO TIMED ROUND CAME HOME AT ALL. This leg rented a box and"
        echo "  measured nothing. Read the arm logs before rerunning; the"
        echo "  likeliest cause is that the first-ever CUDA build of these"
        echo "  drivers failed, which is a finding and is in $_ld."
        _bad=1
    fi
    if [ "${_ident:-0}" -gt 0 ]; then
        echo "  MODE WITNESS FAILED: ${_ident} 'ours' header(s) report"
        echo "  IDENTICAL. This payload is the FAST path and builds with no"
        echo "  -D define, so a binary that compiled IDENTICAL means the"
        echo "  environment on that box carried the mode in. Every ratio in"
        echo "  this run is void: it would be the cost of the pin wearing the"
        echo "  label of the fast arm."
        _bad=1
    fi
    if [ -f "$_lt" ]; then sed 's/^/    /' "$_lt"; fi
    return "$_bad"
}

leg_phase8_artifacts() {
    # WHAT CAME HOME, lane by lane. This leg does NOT judge these cards --
    # tools/e3_round_judge.sh section 7 does, against the Apple column. What
    # is checked here is everything that would make judging them meaningless:
    # a card that is not there, and a card whose own run says it was compiled
    # in the other arithmetic.
    _rc=0
    _unwit=""
    if [ ! -d "$E1_DEST/lanes" ]; then
        echo "  NO lanes/ DIRECTORY CAME HOME. Phase 8 wrote no card on this"
        echo "  box. Read $E1_DEST/bootstrap.log, or"
        echo "  $OUT/remote/bootstrap_console.log if even that is missing."
        return 1
    fi
    echo "  lane cards in $E1_DEST/lanes:"
    for _l in $LANES; do
        _card="$E1_DEST/lanes/$_l.identical.card"
        _log="$E1_DEST/lanes/$_l.identical.log"
        if [ ! -s "$_card" ]; then
            echo "    $_l [IDENTICAL]: NO CARD -- the judge will call this"
            echo "        pair MISSING. ($_log)"
            _rc=1
            continue
        fi
        _n=$(grep -vc '^#\|^$' "$_card" || true)
        _w=$(leg_witness_lane_mode "$_card" "$_log")
        case "$_w" in
            IDENTICAL)
                echo "    $_l [IDENTICAL]: $_n records, mode read back from the run = [IDENTICAL]" ;;
            FAST)
                echo "    $_l [IDENTICAL]: $_n records, AND THE RUN REPORTS MODE [FAST]."
                echo "        CONTAMINATED: this card is filed as a"
                echo "        NUMERIC_IDENTICAL claim and the binary that"
                echo "        wrote it was the other arithmetic. Do not judge"
                echo "        it. ($_log)"
                _rc=1 ;;
            *)
                echo "    $_l [IDENTICAL]: $_n records, mode UNWITNESSED"
                _unwit="$_unwit $_l" ;;
        esac
    done
    if [ -n "$_unwit" ]; then
        echo "  UNWITNESSED lanes:$_unwit"
        echo "    Those drivers print no mode line at all, so nothing in"
        echo "    their own output says which arithmetic they were compiled"
        echo "    with. Measured 2026-08-23: linkage and mamba are the two."
        echo "    This is REPORTED rather than asserted in either direction;"
        echo "    the evidence that the pass was IDENTICAL is the other"
        echo "    lanes' banners from the same loop, which is weaker than a"
        echo "    banner of their own and must not be quoted as one."
    fi
    echo "  FAST cards (recorded, never a cross-vendor claim):"
    for _l in $LANES; do
        if [ -s "$E1_DEST/lanes/$_l.fast.card" ]; then
            echo "    $_l [FAST]: present"
        else
            echo "    $_l [FAST]: absent"
        fi
    done
    if [ -f "$E1_DEST/bootstrap.log" ]; then
        _nf=$(grep -c 'PHASE8-FINDING' "$E1_DEST/bootstrap.log" 2>/dev/null || true)
        echo "  phase-8 findings: ${_nf:-0}"
        grep 'PHASE8-FINDING' "$E1_DEST/bootstrap.log" 2>/dev/null | sed 's/^/      /' || true
        echo "    A FINDING IS NOT AN ABORT and it is not this leg's verdict"
        echo "    (tools/e1_bootstrap.sh: 'a lane that fails here is a"
        echo "    FINDING, never an abort'). In particular mamba's FAST arm"
        echo "    HAS NEVER BEEN BUILT ANYWHERE, so a 'mamba [fast]' finding"
        echo "    is EXPECTED here and is information."
    fi
    return $_rc
}

leg_archive_required() {
    # What must be inside the archive for THIS payload to be worth shipping.
    # A word list on purpose, so the caller can iterate it.
    if [ "$PAYLOAD" = "speed" ]; then
        # PER FAMILY, for the same reason phase8's list is per lane: a
        # driver that is not committed at this sha is caught HERE, for
        # nothing, instead of surfacing forty minutes into a rental with no
        # numbers to show for it. The table builder is in every list because
        # a leg that brings home lines nothing can parse has brought home
        # nothing.
        case "$SPEED_FAMILY" in
            gemmseq)
                echo "bench/speed/gemm_speed_main.mojo bench/speed/seq_speed_main.mojo tools/speed_gemm_arm.py tools/speed_torch_seq.py tools/fast_speed_table.py" ;;
            classical)
                echo "bench/speed/classical_speed_main.mojo tools/speed_cuml_arm.py tools/fast_speed_table.py" ;;
            forest)
                echo "bench/speed/forest_speed_arm.py tools/speed_gbdt_arm.py tools/fast_speed_table.py" ;;
        esac
    elif [ "$PAYLOAD" = "phase8" ]; then
        # EVERY LANE PHASE 8 DRIVES, not only the two this file is named
        # after (DEVIATION 866). G2 checks each of these is inside the
        # archive, so a lane whose driver is not committed at this sha is
        # caught HERE, for nothing, instead of surfacing as a PHASE8-FINDING
        # forty minutes into a rental with no card to show for it.
        echo "tools/e1_bootstrap.sh bench/gemm_card_main.mojo gemm/mojo_only/gemm_identical.mojo solver/cd_main.mojo kde/kde_main.mojo hierarchy/linkage_main.mojo svm/svc_main.mojo metrics/metrics_main.mojo mamba/mojo_only/mamba_check.mojo"
    else
        echo "gemm/mojo_only/gemm_identical.mojo"
    fi
}

# ---------------------------------------------------------------------------
# THE DEVICE CARD IS NOT ONE SEQUENCE. Measured 2026-08-23 on the Apple leg:
# `tools/gemm_card.sh device` produces a card with THREE blocks -- the same 60
# tags, the same 60 hashes, and the sequence number restarting at 0 each time,
# because the emitter runs the device arm at three launch geometries and
# appends all three to one trace file.
#
# `tools/identity_trace_diff.py` REFUSES that file. Its format requires the
# sequence to start at 0 and increase by exactly 1, so it exits 2 with
#
#     parse error: apple.card:63: seq out of order: expected 60, got 0
#
# and an exit 2 is a PARSE ERROR, not a divergence. The first version of this
# function read every non-zero status as "the cards diverge", so a card the
# differ could not read at all would have been reported as a cross-vendor
# divergence -- the worst possible misreading, and it is exactly what a leg
# is for if nobody ever runs it against a real card.
#
# So: split both cards on the sequence restarts, require both sides to have
# the SAME number of blocks (a side that emitted a different number of launch
# geometries is itself a finding, and a loud one), and diff block against
# block. NOTHING IS DROPPED and the split is announced, because a quiet
# substitution here would be a fabricated comparison.
# ---------------------------------------------------------------------------
leg_split_card() {
    # $1 card, $2 path prefix. Writes <prefix>.1, <prefix>.2, ... and echoes
    # the block count. Comments are dropped and one fresh header is written
    # per block, so each block is a whole card by the differ's rules.
    awk -F'\t' -v pre="$2" '
        /^#/ { next }
        NF < 5 { next }
        $1 == "0" {
            b++; out = pre "." b
            printf "# format: mojolearn-identity-trace v1\n" > out
        }
        {
            if (b == 0) {
                b = 1; out = pre "." b
                printf "# format: mojolearn-identity-trace v1\n" > out
            }
            print > out
        }
        END { print b + 0 }' "$1"
}

leg_diff_cards() {
    _a="$1"; _b="$2"; _o="$3"
    rm -f "$TMPD"/blk_a.* "$TMPD"/blk_b.*
    _na=$(leg_split_card "$_a" "$TMPD/blk_a")
    _nb=$(leg_split_card "$_b" "$TMPD/blk_b")

    if [ "${_na:-0}" -lt 1 ] || [ "${_nb:-0}" -lt 1 ]; then
        echo "  ONE OF THE CARDS HAS NO RECORDS AT ALL"
        echo "    APPLE: ${_na:-0} block(s)   $VLABEL: ${_nb:-0} block(s)"
        echo "    This is not a divergence. There is nothing to compare."
        return 1
    fi
    if [ "$_na" != "$_nb" ]; then
        echo "  BLOCK COUNT MISMATCH: APPLE emitted $_na, $VLABEL emitted $_nb."
        echo "    The emitter writes one sequence block per launch geometry,"
        echo "    so the two machines ran DIFFERENT NUMBERS OF GEOMETRIES."
        echo "    That is a bigger finding than any hash and it has to be"
        echo "    resolved before a stage name from either side means"
        echo "    anything. It is not a divergence."
        return 1
    fi
    if [ "$_na" -gt 1 ]; then
        echo "  this card carries $_na sequence blocks (one per launch"
        echo "  geometry, sequence numbers restart in each). The differ"
        echo "  refuses a multi-run file, so every block is diffed against"
        echo "  its opposite number. Nothing is dropped."
    fi

    : > "$_o"
    _i=1
    _rc=0
    _first=""
    _firstblk=""
    while [ "$_i" -le "$_na" ]; do
        _brc=0
        {
            echo "############ block $_i of $_na ############"
        } >> "$_o"
        "$PY" "$DIFFER" "$TMPD/blk_a.$_i" "$TMPD/blk_b.$_i" \
            --labels "APPLE,$VLABEL" > "$TMPD/blkdiff.$_i" 2>&1 || _brc=$?
        cat "$TMPD/blkdiff.$_i" >> "$_o"
        if [ "$_brc" = "2" ]; then
            # EXIT 2 IS "I CANNOT READ THIS", NOT "THEY DIFFER".
            echo "  THE DIFFER COULD NOT READ BLOCK $_i (exit 2)."
            sed 's/^/    /' "$TMPD/blkdiff.$_i" | head -10
            echo "    A parse or dump-integrity failure VOIDS every other"
            echo "    conclusion in the report. This is not a divergence and"
            echo "    must not be recorded as one."
            return 2
        fi
        if [ "$_brc" != "0" ] && [ -z "$_first" ]; then
            _first=$(sed -n 's/^  FIRST DIVERGENCE: //p' "$TMPD/blkdiff.$_i" | head -1)
            _firstblk="$_i"
            _rc=1
        fi
        _i=$((_i + 1))
    done

    if [ "$_rc" = "0" ]; then
        sed 's/^/    /' "$TMPD/blkdiff.1" | head -40
        echo
        echo "  NO DIVERGENCE at any matched stage, in any of $_na block(s)."
        echo "  Read that as what it is: the two buffers held the same bits at"
        echo "  every checkpoint the card hashes. It is not proof the two"
        echo "  computations were identical, and anything not hashed is"
        echo "  invisible."
        return 0
    fi

    sed 's/^/    /' "$TMPD/blkdiff.$_firstblk" | head -40
    echo
    if [ -n "$_first" ]; then
        echo "  FIRST DIVERGING STAGE: $_first   (block $_firstblk of $_na)"
    else
        echo "  BLOCK $_firstblk DIVERGES BEFORE ANY HASH COMPARISON."
        echo "  Read STEP 2 above: if the tag sequences do not match, the two"
        echo "  runs took different code paths and no hash comparison between"
        echo "  them means anything. Fix the path difference first."
    fi
    echo "  This is a LOCATION, not a verdict. Walk the ladder INTEGERS"
    echo "  BEFORE FLOATS: an integer stage that diverges is a code-path or"
    echo "  partition difference and is bigger than any numeric row; resolve"
    echo "  it before reading one float stage. The runbook's 'walking a"
    echo "  divergence' section is the procedure."
    return 1
}

# ---------------------------------------------------------------------------
# DEVIATION 862 -- THE UNARMED WINDOW, CLOSED FROM THIS MACHINE
#
# The safety story above says "the pod is created; from this instant to step
# 3 the box is BILLING AND UNARMED", and until this deviation that window had
# exactly one guard: --ready-timeout, which is a `while` loop IN THIS
# PROCESS. Kill this process inside that window -- a SIGKILL, a terminal that
# goes away, a Ctrl-C at the wrong second -- and NOTHING anywhere ends the
# pod. The EXIT trap does not run for SIGKILL. The on-pod watchdog does not
# exist yet. RunPod bills until a pod is TERMINATED, and a container that
# exits is RESTARTED rather than stopped (tools/runpod_guard.sh header,
# measured on a real pod). So the claim that the window was "bounded IN CODE"
# was true only for the paths where this process survives, which are not the
# paths that produce orphans.
#
# tools/e2_remote_leg.sh met this shape on DigitalOcean as leg 10: the create
# succeeded, the response body was unreadable, the script parsed no id,
# printed "create FAILED" and exited through a teardown that had nothing to
# destroy -- an orphan MI325X found by hand nineteen minutes later. Its
# answer was a DETACHED dead-man armed BEFORE the create and keyed by NAME
# rather than by an id the script might never learn. This is that answer,
# ported to RunPod and to this file's key discipline.
#
# THREE LAYERS, FAILING DIFFERENTLY:
#   1. the on-pod watchdog. Survives this Mac entirely -- power, sleep,
#      network. It cannot exist before the pod does, which is the hole.
#   2. THIS. A detached `sh` here, surviving this script's death and the
#      shell that launched it. It does NOT survive the laptop powering off,
#      which is why it is not layer one.
#   3. leg_terminate at the end of the work, which ends a normal leg.
#
# THE CLOCK IS DELIBERATELY THE LATEST OF THE THREE: READY_TIMEOUT + the
# lease + 300s, measured from just before the create. Later than the on-pod
# watchdog's deadline in every timing this leg can reach, so it never
# pre-empts layer 1 and never kills a leg that is still fetching. It exists
# for the run where layer 1 WAS NEVER ARMED. With the defaults it caps an
# abandoned unarmed pod at 75 minutes instead of forever.
#
# THE KEY IS NOT IN ITS ARGV EITHER. tools/e2_remote_leg.sh interpolates its
# DigitalOcean token into a `nohup bash -c "..."` string, so that token sits
# in a process command line for the whole hour. This one writes its own 0600
# curl config beside its script, OUTSIDE TMPD (which the teardown deletes),
# and both go when it fires or when it is cancelled. The dry run's M2 proves
# the key is in no argv and M4 runs the thing against stubs.
#
# `trap "" HUP INT` BEFORE `exec`, AND NOT TERM. Ignored signals survive an
# exec, so this is how a POSIX shell gets what bash spells `disown`: a Ctrl-C
# in this terminal sends SIGINT to the whole foreground process group and
# would otherwise take the dead-man with it -- the one signal most likely to
# arrive at the exact moment the guard is needed. TERM is left deliverable so
# that leg_cancel_deadman's `kill` still works.
# ---------------------------------------------------------------------------

leg_write_deadman() {   # <dir> <seconds> <pod name>; composes, never arms
    _dmd="$1"; _dmsecs="$2"; _dmname="$3"
    ( umask 077; mkdir -p "$_dmd" )
    ( umask 077
      printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' \
          "$RUNPOD_API_KEY" > "$_dmd/curlrc" )
    cat > "$_dmd/deadman.sh" <<'DEADMAN_EOF'
#!/bin/sh
# Written by tools/gemm_remote_leg.sh (DEVIATION 862). DETACHED ON PURPOSE.
# It ends the pod that leg created, if that leg is no longer here to do it.
# Keyed by id when the id is known and by NAME when it is not, because the
# worst case in this file is a create that succeeded and an id nobody parsed.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
sleep @SECS@
_log="$D/deadman.log"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) dead-man firing for @PODNAME@" >> "$_log"
_ids=""
if [ -s "$D/pod_id.txt" ]; then _ids="$(cat "$D/pod_id.txt")"; fi
if [ -z "$_ids" ]; then
    curl -K "$D/curlrc" -o "$D/pods.json" -X GET "@HOST@@PODSPATH@" >> "$_log" 2>&1
    _ids="$(@PY@ "$D/byname.py" "$D/pods.json" "@PODNAME@")"
fi
if [ -z "$_ids" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) no id file and no pod named @PODNAME@: nothing to end" >> "$_log"
else
    for _id in $_ids; do
        for _u in "@HOST@@PODSPATH@/$_id" "@HOSTV1@/v1/pods/$_id"; do
            _c="$(curl -K "$D/curlrc" -o /dev/null -w '%{http_code}' -X DELETE "$_u" 2>>"$_log")"
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE $_u -> $_c" >> "$_log"
            case "$_c" in 2*|404) break ;; esac
        done
    done
fi
# THE CREDENTIAL LEAVES WITH THE JOB. Its only reason to be on disk was that
# DELETE; the log beside it is the evidence and carries nothing secret.
rm -f "$D/curlrc"
DEADMAN_EOF
    # The by-name parser is a FILE rather than a heredoc inside the script
    # above, because a heredoc nested in a heredoc is how a generator quietly
    # ships a body that ends early. One file, one job, and `sh -n` can see it.
    cat > "$_dmd/byname.py" <<'BYNAME_EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if isinstance(d, list):
    d = {"items": d}
pods = d.get("items") or d.get("pods") or d.get("data") or []
print(" ".join(str(p.get("id", "")) for p in pods if p.get("name") == sys.argv[2]))
BYNAME_EOF
    sed -e "s|@SECS@|$_dmsecs|g" \
        -e "s|@PODNAME@|$_dmname|g" \
        -e "s|@HOST@|$RP_HOST|g" \
        -e "s|@HOSTV1@|$RP_HOST_V1|g" \
        -e "s|@PODSPATH@|$RP_PODS_PATH|g" \
        -e "s|@PY@|$PY|g" \
        "$_dmd/deadman.sh" > "$_dmd/deadman.sh.subst"
    mv "$_dmd/deadman.sh.subst" "$_dmd/deadman.sh"
    # SUBSTITUTION, AND THEN A CHECK THAT IT HAPPENED -- the same shape and
    # the same reason as leg_check_remote_body: a `sed` that missed would
    # leave this guard sleeping for the literal string @SECS@, which is to
    # say not sleeping and not guarding.
    if grep -q '@[A-Z][A-Z0-9_]*@' "$_dmd/deadman.sh"; then
        echo "  UNSUBSTITUTED PLACEHOLDER in the dead-man:"
        grep -n '@[A-Z][A-Z0-9_]*@' "$_dmd/deadman.sh" | sed 's/^/    /'
        return 1
    fi
    sh -n "$_dmd/deadman.sh" || { echo "  the dead-man script is not valid sh"; return 1; }
    chmod 700 "$_dmd/deadman.sh"
    return 0
}

leg_arm_deadman() {
    DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-gemm-deadman-$$"
    _secs=$(( READY_TIMEOUT + MINUTES * 60 + 300 ))
    if leg_write_deadman "$DEADMAN_DIR" "$_secs" "$POD_NAME" > "$OUT/deadman_build.log" 2>&1; then
        :
    else
        sed 's/^/    /' "$OUT/deadman_build.log"
        leg_die "THE DEAD-MAN DID NOT BUILD, so nothing is created.
  It is the only thing that ends this pod if this process is killed between
  the create call and the arm, and a leg that rents without it is renting
  without the guarantee this file exists to make."
    fi
    # `trap "" HUP INT` then `exec`: see the block above. TERM stays
    # deliverable so leg_cancel_deadman can still cancel it.
    nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$DEADMAN_DIR/deadman.sh" \
        > /dev/null 2>&1 < /dev/null &
    DEADMAN_PID=$!
    # READ IT BACK. `nohup ... &` succeeding and a process existing are two
    # claims, and every other gate in this file checks the second one. A
    # dead-man that never started, with this script believing it did, is the
    # `arm` failure mode tools/runpod_guard.sh refuses by name.
    sleep 1
    if kill -0 "$DEADMAN_PID" 2>/dev/null; then
        echo "$DEADMAN_DIR" > "$OUT/deadman_dir.txt"
        leg_say "dead-man ARMED before the create: pid $DEADMAN_PID, fires in ${_secs}s, keyed by name $POD_NAME"
        leg_say "  cancel by hand only after the pod is confirmed gone:  kill $DEADMAN_PID; rm -rf $DEADMAN_DIR"
    else
        rm -rf "$DEADMAN_DIR"
        DEADMAN_PID=""; DEADMAN_DIR=""
        leg_die "THE DEAD-MAN DID NOT START. Nothing was created.
  Renting now would put a box up with no guard covering the window before
  the on-pod watchdog exists, which is the one window nothing else covers."
    fi
}

# Every line below runs from the EXIT trap, so shellcheck calls it dead.
# shellcheck disable=SC2317
leg_cancel_deadman() {
    [ -n "$DEADMAN_PID" ] || return 0
    # KILL THE WRAPPER AND ITS `sleep` CHILD. Killing only the wrapper leaves
    # an orphan `sleep` -- harmless, since the DELETE after it dies with the
    # wrapper -- but tools/e2_remote_leg.sh found seven of them after one day
    # and they are indistinguishable at a glance from a live guard.
    pkill -P "$DEADMAN_PID" 2>/dev/null || true
    if kill "$DEADMAN_PID" 2>/dev/null; then
        echo "  dead-man cancelled (pid $DEADMAN_PID)"
    fi
    [ -n "$DEADMAN_DIR" ] && rm -rf "$DEADMAN_DIR"
    DEADMAN_PID=""; DEADMAN_DIR=""
    return 0
}

# ---------------------------------------------------------------------------
# the paid steps
# ---------------------------------------------------------------------------

leg_preflight() {
    # Two GETs, both free, both preventing an orphan rather than cleaning one
    # up. Re-running a leg that failed late is the ordinary way to end up
    # paying for two boxes.
    echo "  pre-flight: existing leases on this machine"
    tools/runpod_guard.sh list 2>&1 | sed 's/^/    /' || true
    _live=$(tools/runpod_guard.sh list 2>/dev/null | grep -c 'min left' || true)
    if [ "${_live:-0}" -gt 0 ]; then
        leg_die "REFUSING to rent: $_live unexpired lease(s) are recorded above.
  Another leg is running, or one ended without terminating its box. Deal
  with that first -- 'tools/gemm_remote_leg.sh reap' terminates and VERIFIES.
  Renting a second box beside an orphan is how a one-hour cap becomes two."
    fi
    echo "  pre-flight: pods already up on this account"
    rp_call GET "$RP_PODS_PATH"
    case "$RP_CODE" in
        2*) : ;;
        *) leg_die "pre-flight pod listing returned HTTP $RP_CODE. If the API
  is not answering, this leg cannot terminate what it creates, so it does
  not create anything." ;;
    esac
    _mine=$(rp_json "','.join([str(p.get('id','')) for p in (d.get('items') or d.get('pods') or d.get('data') or []) if str(p.get('name','')).startswith('mojolearn-gemm-')])")
    if [ -n "$_mine" ]; then
        if [ "$LEG_ALLOW_CONCURRENT" = "1" ]; then
            echo "    CONCURRENT: pod(s) already up and NOT reaped: $_mine"
            echo "    --allow-concurrent was passed. This leg creates its own"
            echo "    pod, terminates BY ITS OWN POD ID, and arms a dead-man"
            echo "    keyed to its own name, so it will not touch those. YOU"
            echo "    are now responsible for confirming ALL of them are gone:"
            echo "      tools/gemm_remote_leg.sh reap <pod-id>"
        else
            leg_die "REFUSING to rent: this lane already has pod(s) up: $_mine
  Terminate them first:  tools/gemm_remote_leg.sh reap <pod-id>
  Or pass --allow-concurrent if those are legs you started on purpose."
        fi
    fi
    echo "    none named mojolearn-gemm-*"
}

leg_create_pod() {
    # THE INTERLOCK. The dry run spawns children of this script to exercise
    # its refusal paths; not one of them may reach a paid call even if the
    # environment around them is fully credentialed.
    if [ "${MOJOLEARN_GEMM_LEG_REHEARSAL:-}" = "1" ]; then
        leg_die "INTERLOCK: a rehearsal child reached leg_create_pod. Nothing was created. This is the interlock working; if you meant to rent, run the leg directly rather than from inside a dry run."
    fi
    POD_NAME="mojolearn-gemm-${VENDOR}-${STAMP}"
    # THE DEAD-MAN IS ARMED HERE: after the interlock (so no rehearsal child
    # can reach it) and BEFORE the POST that starts the bill. Arming it after
    # the create would leave uncovered the one instant it exists for -- the
    # create that succeeds and is never parsed.
    leg_arm_deadman
    # An empty CUDA_VERSIONS (the AMD default) omits the field entirely --
    # "allowedCudaVersions": [] is a constraint no host satisfies, not the
    # absence of one.
    CUDA_LINE=""
    if [ -n "$CUDA_VERSIONS" ]; then
        CUDA_LINE="\"allowedCudaVersions\": [$CUDA_VERSIONS],"
    fi
    cat > "$TMPD/create.json" <<JSONEOF
{
  "name": "$POD_NAME",
  "imageName": "$IMAGE",
  "gpuTypeIds": ["$GPU_ID"],
  "gpuCount": 1,
  $CUDA_LINE
  "cloudType": "SECURE",
  "containerDiskInGb": 60,
  "volumeInGb": 0,
  "ports": ["22/tcp"],
  "supportPublicIp": true,
  "interruptible": false
}
JSONEOF
    cp "$TMPD/create.json" "$OUT/create_request.json"
    leg_say "creating $POD_NAME ($GPU_ID, $IMAGE)"
    leg_say "  THE BILL STARTS HERE and the box is NOT YET ARMED."
    rp_call POST "$RP_PODS_PATH" "$TMPD/create.json"
    cp "$RP_BODY" "$OUT/create_response.json" 2>/dev/null || true
    POD_ID=$(rp_json "d.get('id') or (d.get('pod') or {}).get('id') or ''")
    if [ -z "$POD_ID" ]; then
        # AN UNPARSED CREATE IS THE WORST CASE IN THIS FILE: a pod may exist,
        # be billing, and have no id here to terminate it with. So look for
        # it by name and ADOPT it, which puts it back under the EXIT trap.
        echo "  create returned HTTP $RP_CODE and no id this script could parse."
        echo "  A pod may nonetheless exist. Looking for it by name ..."
        rp_call GET "$RP_PODS_PATH"
        POD_ID=$(rp_json "([str(p.get('id','')) for p in (d.get('items') or d.get('pods') or d.get('data') or []) if p.get('name')=='$POD_NAME'] or [''])[0]")
        if [ -n "$POD_ID" ]; then
            [ -n "$DEADMAN_DIR" ] && printf '%s\n' "$POD_ID" > "$DEADMAN_DIR/pod_id.txt"
            echo "  ADOPTED $POD_ID by name. The teardown trap now owns it."
            leg_die "create response was unparseable but a pod exists; terminating it and stopping. Fix the create parsing before the next paid run: see $OUT/create_response.json"
        fi
        leg_die "create FAILED (HTTP $RP_CODE) and no pod named $POD_NAME exists.
  Body: $OUT/create_response.json
  CHECK THE CONSOLE ANYWAY -- https://console.runpod.io/pods -- because
  'no pod in the listing' and 'no pod' are the same sentence only if the
  listing was complete."
    fi
    leg_say "pod $POD_ID created"
    echo "$POD_ID" > "$OUT/pod_id.txt"
    # HAND THE ID TO THE DEAD-MAN. Without it the dead-man falls back to a
    # by-name listing, which needs the API to answer at the moment it fires;
    # an id it already holds needs nothing but the DELETE.
    [ -n "$DEADMAN_DIR" ] && printf '%s\n' "$POD_ID" > "$DEADMAN_DIR/pod_id.txt"
}

leg_wait_ready() {
    # THE UNARMED BILLING WINDOW, bounded in code. Every second in here is
    # paid for and unguarded, so it ends on a timer rather than on patience.
    if [ -n "$SSH_TARGET" ]; then
        # The operator supplied a target. That is the escape hatch for the
        # day RunPod's port reporting does not match what is parsed below;
        # it skips discovery and goes straight to probing.
        leg_say "using the supplied ssh target: $SSH_TARGET"
    fi
    _deadline=$(( $(date -u +%s) + READY_TIMEOUT ))
    while [ "$(date -u +%s)" -lt "$_deadline" ]; do
        if [ -z "$SSH_TARGET" ]; then
            rp_call GET "$RP_PODS_PATH/$POD_ID"
        fi
        _ip=$(rp_json "d.get('publicIp') or (d.get('pod') or {}).get('publicIp') or ''")
        # DEVIATION 971: portMappings has TWO SHAPES and this read knew one.
        #
        # api.runpod.io/v2 returns a LIST of {privatePort, publicPort} dicts.
        # rest.runpod.io/v1 returns a DICT, {"22": 28885}. Iterating a dict
        # yields its KEYS, which are strings, so `m.get(...)` raised, rp_json
        # swallowed it by design, and _port came back EMPTY. The leg then had
        # no ssh target, waited out its full 600s READY TIMEOUT and terminated
        # a perfectly healthy box. Measured 2026-08-24 on pod qh9dudcqznnuua:
        # ssh -p 28885 root@213.181.111.2 answered SSH-OK and nvidia-smi named
        # the 4090 from a plain shell, while the leg saw nothing.
        #
        # The v1 POST body schema is the one this script already sends (the
        # /v1/openapi.json spec lists gpuTypeIds, imageName, cloudType,
        # containerDiskInGb, volumeInGb, gpuCount and interruptible, and the
        # v2 POST rejects every one of them), so v1 is where this leg belongs
        # and this read has to speak v1's shape.
        # TWO EXPRESSIONS, TRIED IN ORDER, AND NEITHER MAY USE A BUILTIN.
        # rp_json evals in a sandbox whose globals are exactly
        # {"__builtins__": {"str": str}}, so `isinstance` is a NameError, the
        # bare `except` swallows it BY DESIGN, and the caller sees "". That is
        # how the first attempt at this fix failed: it read correctly in a
        # plain python3 and returned empty inside the leg, which is the worst
        # possible combination for diagnosing it.
        #
        # So the shapes are separated instead of branched. The DICT form's
        # `.get` raises on a list and the LIST form's `m.get` raises on a
        # dict's string keys, so each is self-selecting and the wrong one
        # simply yields "".
        _port=$(rp_json "str((d.get('portMappings') or {}).get('22') or (d.get('portMappings') or {}).get(22) or '')")
        [ -n "$_port" ] || _port=$(rp_json "([str(m.get('publicPort','')) for m in (d.get('portMappings') or (d.get('runtime') or {}).get('ports') or []) if str(m.get('privatePort',''))=='22'] or [''])[0]")
        if [ -n "$SSH_TARGET" ] || { [ -n "$_ip" ] && [ -n "$_port" ]; }; then
            [ -n "$SSH_TARGET" ] || SSH_TARGET="-p $_port root@$_ip"
            leg_say "ssh target: $SSH_TARGET"
            _i=1
            while [ "$_i" -le 20 ]; do
                # shellcheck disable=SC2086
                if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
                       -o BatchMode=yes $SSH_TARGET 'echo SSH-OK' 2>/dev/null \
                       | grep -q SSH-OK; then
                    leg_say "ssh up after $_i attempt(s)"
                    return 0
                fi
                sleep 10
                _i=$((_i + 1))
            done
            leg_die "the pod exposed ssh but never answered it. Terminating."
        fi
        sleep 10
    done
    leg_die "READY TIMEOUT: $READY_TIMEOUT seconds without an ssh endpoint.
  The box has been billing and unarmed for all of it, which is exactly the
  window this timeout exists to close. Terminating."
}

leg_ssh() {
    # shellcheck disable=SC2086
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 \
        -o ServerAliveInterval=30 -o BatchMode=yes $SSH_TARGET "$@"
}

leg_arm() {
    leg_say "ARMING THE LEASE BEFORE ANY WORK ($MINUTES minutes)"
    _armlog="$OUT/arm.log"
    # See leg_local_card: the status of `cmd | sed` is SED's. Here that would
    # read a REFUSAL as a successful arm and start work on an unguarded box.
    if tools/runpod_guard.sh arm "$POD_ID" "$SSH_TARGET" "$MINUTES" \
            > "$_armlog" 2>&1; then
        sed 's/^/    /' "$_armlog"
    else
        sed 's/^/    /' "$_armlog"
        leg_die "ARM REFUSED. The box is not used -- it is terminated.
  A box that cannot be armed is an orphan that has not happened yet
  (IDENTICAL_GEMM_PLAN.md, RENTING rule 1)."
    fi
    # READ IT BACK. `arm` succeeding and a lease existing are two claims, and
    # every other gate in this repository checks the second one rather than
    # the first. A lease file the guard did not write is a box `check`,
    # `list` and `reap` cannot see.
    if tools/runpod_guard.sh check "$POD_ID" > "$OUT/lease.txt" 2>&1; then
        sed 's/^/    /' "$OUT/lease.txt"
        ARMED=1
        return 0
    fi
    sed 's/^/    /' "$OUT/lease.txt"
    leg_die "ARM CLAIMED SUCCESS AND LEFT NO READABLE LEASE. Treating that as
  a refusal and terminating: a lease file that cannot be read is a lease
  nobody can check, and the next person reads it and believes it."
}

leg_key_to_pod() {
    # THE KEY REACHES THE POD ON STDIN, INTO A 0600 FILE. Never in an argv on
    # either side. It is there so the operator can self-terminate from the
    # box and so the ps check below has a pattern FILE to search with.
    printf '%s' "$RUNPOD_API_KEY" | leg_ssh 'umask 077; cat > /root/.mojolearn-rp.key; chmod 600 /root/.mojolearn-rp.key; ls -l /root/.mojolearn-rp.key'
}

leg_check_key_not_in_ps() {
    # VERIFY THE CLAIM, DO NOT ASSERT IT. `grep -F -f <keyfile>` takes the
    # pattern from a FILE, so the check does not itself put the key into an
    # argv. A previous version of the guard leaked a key into `ps` while
    # carrying a comment in the same edit claiming it did not.
    leg_ssh 'ps -eo args= > /tmp/mojolearn-ps.txt 2>/dev/null || ps ax > /tmp/mojolearn-ps.txt;
             if grep -q -F -f /root/.mojolearn-rp.key /tmp/mojolearn-ps.txt; then
                 echo KEY_VISIBLE_IN_PS;
                 grep -c -F -f /root/.mojolearn-rp.key /tmp/mojolearn-ps.txt;
             else
                 echo KEY_NOT_IN_PS;
             fi;
             rm -f /tmp/mojolearn-ps.txt' > "$OUT/key_in_ps.txt" 2>&1 || true
    sed 's/^/    /' "$OUT/key_in_ps.txt"
    if grep -q KEY_VISIBLE_IN_PS "$OUT/key_in_ps.txt"; then
        echo "    THE KEY IS VISIBLE IN THE POD'S PROCESS LIST. Anything with"
        echo "    a shell on this box can read it. Rotate the key after this"
        echo "    leg and fix the leak before the next one."
        return 1
    fi
    return 0
}

leg_build_remote_body() {
    _body="$OUT/remote_body.sh"
    if [ "$PAYLOAD" = "speed" ]; then
        leg_body_speed > "$_body"
    elif [ "$PAYLOAD" = "phase8" ]; then
        leg_body_phase8 > "$_body"
    else
        leg_body_gemm > "$_body"
    fi
    leg_check_remote_body
}

leg_body_gemm() {
    cat <<'REMOTE_BODY'
#!/bin/sh
# Generated by tools/gemm_remote_leg.sh (DEVIATION 536). RUNS ON THE POD.
#
# DELIBERATELY `set -u` AND NOT `set -e`. A gate that goes red is a RESULT
# and its log has to come home; dying at the first non-zero would fetch an
# empty directory and lose the finding. Every step records its own exit code
# into leg.txt and the driving host reads them.
#
# POSIX sh only. RunPod's Ubuntu images link /bin/sh to dash, and `exec -a`
# being a bashism has already broken this repository's guard once.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out
mkdir -p "$OUT"
cd "$ROOT" || exit 9

{
  echo "vendor=@VENDOR@"
  echo "commit=@COMMIT@"
  echo "card_full=@CARDFULL@"
  echo "trace_dump=@DUMP@"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/leg.txt"

uname -a > "$OUT/uname.txt" 2>&1
@SMI@ > "$OUT/gpu.txt" 2>&1 || echo "no vendor smi tool answered" >> "$OUT/gpu.txt"

# THE SOURCE HASH, computed BEFORE pixi installs anything, with the same
# recipe the Mac used on the extracted archive. Two legs whose
# source_sha256 agree ran the same program whatever their .git says -- and
# this box has no .git at all, by design.
{ find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
    | LC_ALL=C sort | xargs shasum -a 256 2>/dev/null || \
  find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
    | LC_ALL=C sort | xargs sha256sum ; } \
  | { shasum -a 256 2>/dev/null || sha256sum ; } \
  | awk '{print $1}' > "$OUT/source_sha256.txt"

if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi > /dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_install.log" 2>&1
fi
PATH="$HOME/.pixi/bin:$PATH"
export PATH
command -v pixi > "$OUT/pixi_which.txt" 2>&1 || echo "NO PIXI" >> "$OUT/pixi_which.txt"

pixi install > "$OUT/pixi_env.log" 2>&1
echo "pixi_install_exit=$?" >> "$OUT/leg.txt"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1 || true

# GATE 1: the device kernel's own invariance gates, on this silicon.
# gemm/mojo_only/gemm_device_check.mojo runs every gate and reports every
# verdict before it raises, so a red here names WHICH gate a defect reaches.
tools/with_identical_mode.sh pixi run mojo run -I . \
    gemm/mojo_only/gemm_device_check.mojo > "$OUT/device_check.log" 2>&1
echo "device_check_exit=$?" >> "$OUT/leg.txt"

# GATE 2: the card. Same invocation shape as the Mac's, driven through
# tools/gemm_card.sh so the mode read-back and the skipped-shape accounting
# are the same code on both ends.
MOJOLEARN_GEMM_CARD_FULL="@CARDFULL@" MOJOLEARN_IDENTITY_TRACE_DUMP="@DUMP@" \
    sh tools/gemm_card.sh device "$OUT/@VENDOR@.card" > "$OUT/card_driver.log" 2>&1
echo "card_exit=$?" >> "$OUT/leg.txt"

# GATE 3, optional: the column sweep on this backend. Off by default -- the
# Mac already answers "does the COLUMN change the product" for free, and an
# hour of lease is better spent on the thing only this silicon can answer.
if [ "@SWEEP@" = "1" ]; then
    MOJOLEARN_GEMM_CARD_ARM=device MOJOLEARN_COLUMN_OUT="$OUT/colinv" \
        sh tools/gemm_column_invariance.sh > "$OUT/column_invariance.log" 2>&1
    echo "column_invariance_exit=$?" >> "$OUT/leg.txt"
fi

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
# THE COMPLETION SENTINEL, WRITTEN LAST (DEVIATION 863). The driving host no
# longer holds an ssh session open for the whole run; it starts this body
# detached and polls for this file. Nothing else on the box distinguishes
# "the payload finished" from "the ssh dropped and the payload was killed",
# and leg_check_remote_body refuses a body that stopped writing it.
: > /root/gemm_leg.done
echo REMOTE_BODY_DONE
REMOTE_BODY
}

leg_body_phase8() {
    cat <<'REMOTE_BODY_P8'
#!/bin/sh
# Generated by tools/gemm_remote_leg.sh (DEVIATION 536, --payload phase8).
# RUNS ON THE POD.
#
# IT RUNS tools/e1_bootstrap.sh AND REIMPLEMENTS NOT ONE LINE OF PHASE 8. A
# copy of that loop here would be a second list of lanes to keep in step with
# e1_bootstrap.sh and e3_round_judge.sh, and the cards the judge read would
# not have come from the program the judge is named after.
#
# DELIBERATELY `set -u` AND NOT `set -e`. A lane that goes red is a RESULT and
# its card and log have to come home; dying at the first non-zero would fetch
# an empty directory and lose the finding. e1_bootstrap.sh says the same
# thing in its own words at phase 8: "a lane that fails here is a FINDING,
# never an abort".
#
# POSIX sh, but it invokes `bash` for the bootstrap on purpose:
# tools/e1_bootstrap.sh line 26 uses process substitution and IS a bash
# script. (`sh -n` reports a false syntax error there for the same reason.)
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out
mkdir -p "$OUT"
cd "$ROOT" || exit 9

{
  echo "vendor=@VENDOR@"
  echo "payload=phase8"
  echo "commit=@COMMIT@"
  echo "work_timeout=@WORKTIMEOUT@"
  echo "e1_phases=@E1PHASES@"
  echo "e1_lanes=@E1LANES@"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/leg.txt"

uname -a > "$OUT/uname.txt" 2>&1
@SMI@ > "$OUT/gpu.txt" 2>&1 || echo "no vendor smi tool answered" >> "$OUT/gpu.txt"

# THE MAMBA GATE'S SHAPE STAYS TINY ON A RENTED BOX. mamba_check.mojo reads
# MOJOLEARN_MAMBA_CHECK_B / _L / _DM from the environment (defaults B=1, L=4,
# d_model=8) and every shape is another compile inside the lease. These are
# UNSET rather than assumed absent: a pod template carries an environment of
# its own, and a widened shape would be invisible here until the bill.
unset MOJOLEARN_MAMBA_CHECK_B MOJOLEARN_MAMBA_CHECK_L MOJOLEARN_MAMBA_CHECK_DM

# THE SOURCE HASH, computed BEFORE pixi installs anything, with the same
# recipe the Mac used on the extracted archive. This box has no .git by
# design, so this is the only comparison of what was shipped against what
# ran -- and it is why e1_bootstrap.sh's own provenance step, which is
# `git rev-parse HEAD | tee commit.txt`, writes an EMPTY commit.txt here. The
# driving host rewrites that file from the sha it PINNED after the fetch and
# leaves commit_provenance.txt beside it saying so.
{ find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
    | LC_ALL=C sort | xargs shasum -a 256 2>/dev/null || \
  find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
    | LC_ALL=C sort | xargs sha256sum ; } \
  | { shasum -a 256 2>/dev/null || sha256sum ; } \
  | awk '{print $1}' > "$OUT/source_sha256.txt"

if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi > /dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_install.log" 2>&1
fi
PATH="$HOME/.pixi/bin:$PATH"
export PATH
command -v pixi > "$OUT/pixi_which.txt" 2>&1 || echo "NO PIXI" >> "$OUT/pixi_which.txt"
command -v bash > "$OUT/bash_which.txt" 2>&1 || \
    echo "NO BASH ON THIS IMAGE -- tools/e1_bootstrap.sh cannot run" >> "$OUT/bash_which.txt"

# THE WORK IS BOUNDED IN CODE, exactly like the ready wait. The lease's
# watchdog is the backstop for a dead session, not a schedule: if it fires
# mid-bootstrap the box dies with every card still on it, because the fetch
# has not run yet. @WORKTIMEOUT@ seconds leaves the lease its reserve, so a
# leg that runs out of hour comes home with the lanes that finished.
if command -v timeout > /dev/null 2>&1; then
    MOJOLEARN_E1_PHASES="@E1PHASES@" MOJOLEARN_E1_LANES="@E1LANES@" timeout -k 30 @WORKTIMEOUT@ bash tools/e1_bootstrap.sh > "$OUT/bootstrap_console.log" 2>&1
    echo "bootstrap_exit=$?" >> "$OUT/leg.txt"
    echo "note: a bootstrap_exit of 124 is the work timeout firing" >> "$OUT/leg.txt"
else
    echo "no timeout(1) on this image: THE BOOTSTRAP RAN UNBOUNDED and the" >> "$OUT/leg.txt"
    echo "  lease watchdog is the only thing between it and the hour" >> "$OUT/leg.txt"
    MOJOLEARN_E1_PHASES="@E1PHASES@" MOJOLEARN_E1_LANES="@E1LANES@" bash tools/e1_bootstrap.sh > "$OUT/bootstrap_console.log" 2>&1
    echo "bootstrap_exit=$?" >> "$OUT/leg.txt"
fi
tail -40 "$OUT/bootstrap_console.log"

# WHICH DIRECTORY DID IT WRITE? Read the bootstrap's own last line -- it
# prints `artifacts in <dir>` -- rather than guessing from a listing. The
# newest-directory fallback is for the run that never got that far, and WHICH
# ONE WAS USED is recorded rather than left to be inferred later.
E1DIR=$(sed -n 's|^artifacts in ||p' "$OUT/bootstrap_console.log" | tail -1)
if [ -n "$E1DIR" ] && [ -d "$E1DIR" ]; then
    echo "e1dir_source=the bootstrap's own 'artifacts in' line" >> "$OUT/leg.txt"
else
    E1DIR=$(ls -dt "$ROOT"/bench/results/e1/*/ 2>/dev/null | head -1)
    E1DIR=${E1DIR%/}
    echo "e1dir_source=newest directory under bench/results/e1 (the bootstrap never printed 'artifacts in')" >> "$OUT/leg.txt"
fi
echo "e1dir=$E1DIR" >> "$OUT/leg.txt"

if [ -n "$E1DIR" ] && [ -d "$E1DIR" ]; then
    cp "$OUT/source_sha256.txt" "$E1DIR/source_sha256.txt" 2>/dev/null
    ls "$E1DIR/lanes" > "$OUT/lanes_listing.txt" 2>&1 || \
        echo "NO lanes/ DIRECTORY" > "$OUT/lanes_listing.txt"
    grep 'PHASE8-FINDING' "$E1DIR/bootstrap.log" > "$OUT/phase8_findings.txt" 2>/dev/null
    echo "identical_cards=$(ls "$E1DIR"/lanes/*.identical.card 2>/dev/null | wc -l)" >> "$OUT/leg.txt"
    echo "fast_cards=$(ls "$E1DIR"/lanes/*.fast.card 2>/dev/null | wc -l)" >> "$OUT/leg.txt"
    echo "phase8_findings=$(grep -c 'PHASE8-FINDING' "$E1DIR/bootstrap.log" 2>/dev/null)" >> "$OUT/leg.txt"
else
    echo "NO ARTIFACT DIRECTORY: the bootstrap wrote nothing under bench/results/e1" >> "$OUT/leg.txt"
fi

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
cat "$OUT/leg.txt"
# THE COMPLETION SENTINEL, WRITTEN LAST (DEVIATION 863). The driving host no
# longer holds an ssh session open for the whole run; it starts this body
# detached and polls for this file. Nothing else on the box distinguishes
# "the payload finished" from "the ssh dropped and the payload was killed",
# and leg_check_remote_body refuses a body that stopped writing it.
: > /root/gemm_leg.done
echo REMOTE_BODY_DONE
REMOTE_BODY_P8
}

# ---------------------------------------------------------------------------
# THE SPEED PAYLOAD'S BODY (DEVIATION 1870). RUNS ON THE POD.
#
# It answers a question none of this file's other payloads asks: how fast is
# the arm an ORDINARY USER GETS, against the library an NVIDIA user would
# actually reach for, on NVIDIA silicon. So:
#
#   * NO `-D MOJOLEARN_NUMERIC_IDENTICAL=1` ANYWHERE IN HERE. Every Mojo
#     invocation below is a bare `pixi run mojo run`, which is the FAST path.
#     That is the whole point of the payload and it is the one thing about it
#     that must not be got wrong, which is why every driver prints the mode
#     it COMPILED in and the driving host refuses a leg whose arms reported
#     IDENTICAL.
#   * NO CARD DIFF, NO APPLE REFERENCE, NO IDENTITY CLAIM. A millisecond is
#     not a hash and this payload makes no statement about bits except one:
#     it records whether the FAST output moved between rounds, which is the
#     evidence for what IDENTICAL buys, measured on the same box in the same
#     hour.
#
# THE VENDOR ARMS RUN ON THE IMAGE'S SYSTEM PYTHON AND NOT UNDER PIXI, and
# that is deliberate rather than lazy. `[feature.gbmbench.dependencies]` pins
# python 3.14 because the prebuilt `python/mojolearn/*.so` were built against
# it, and RAPIDS ships for 3.10 through 3.13. Those two cannot share an
# interpreter, so putting cuML into the pixi environment would fail to
# resolve and take the Mojo half of the leg down with it. The RunPod pytorch
# image already carries a 3.11 with torch in it, which is exactly what the
# vendor arms need.
#
# EVERY ARM IS BOUNDED AND EVERY ARM IS ALLOWED TO FAIL. `set -u` and NOT
# `set -e`: an arm that cannot run is a RESULT about this box and this image,
# it is recorded as FSPEED-REFUSED and the leg goes on. Dying at the first
# non-zero would fetch an empty directory and lose every arm that worked,
# which is the failure DEVIATION 863 was written about.
# ---------------------------------------------------------------------------
leg_body_speed() {
    cat <<'REMOTE_BODY_SPEED'
#!/bin/sh
# Generated by tools/gemm_remote_leg.sh (--payload speed). RUNS ON THE POD.
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out
LOGS="$OUT/logs"
mkdir -p "$LOGS"
cd "$ROOT" || exit 9

{
  echo "vendor=@VENDOR@"
  echo "payload=speed"
  echo "family=@FAMILY@"
  echo "lanes=@SPEEDLANES@"
  echo "commit=@COMMIT@"
  echo "rounds=@SPEEDROUNDS@"
  echo "size=@SPEEDSIZE@"
  echo "dataset=@SPEEDDATASET@"
  echo "rows_ladder=@SPEEDROWS@"
  echo "arm_budget=@ARMBUDGET@"
  echo "work_timeout=@WORKTIMEOUT@"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/leg.txt"

uname -a > "$OUT/uname.txt" 2>&1
@SMI@ > "$OUT/gpu.txt" 2>&1 || echo "no vendor smi tool answered" >> "$OUT/gpu.txt"

# THE SOURCE HASH, computed BEFORE anything installs, with the same recipe
# the Mac used on the extracted archive. This box has no .git by design, so
# this is the only comparison of what was shipped against what ran.
{ find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
    | LC_ALL=C sort | xargs shasum -a 256 2>/dev/null || \
  find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
    | LC_ALL=C sort | xargs sha256sum ; } \
  | { shasum -a 256 2>/dev/null || sha256sum ; } \
  | awk '{print $1}' > "$OUT/source_sha256.txt"

if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi > /dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_install.log" 2>&1
fi
PATH="$HOME/.pixi/bin:$PATH"
export PATH
command -v pixi > "$OUT/pixi_which.txt" 2>&1 || echo "NO PIXI" >> "$OUT/pixi_which.txt"
command -v python3 > "$OUT/python_which.txt" 2>&1 || echo "NO SYSTEM PYTHON3" >> "$OUT/python_which.txt"
python3 -c 'import sys; print(sys.version)' >> "$OUT/python_which.txt" 2>&1 || true

export MOJOLEARN_SPEED_ROUNDS="@SPEEDROUNDS@"
export MOJOLEARN_SPEED_SIZE="@SPEEDSIZE@"

# THE DEVICE NAME ON THE HEADER LINE. `bench/speed/seq_speed_main.mojo` takes
# it from the environment rather than from `DeviceContext`, so if this is left
# unset every one of its headers reads `device=unset` and the table cannot say
# which silicon it is about. Taken from the vendor's own tool, first field,
# with the spaces squeezed because the header is space-delimited `k=v`.
MOJOLEARN_SPEED_DEVICE=$(@SMI@ 2>/dev/null | head -1 | sed 's/,.*//' | sed 's/^ *//; s/ *$//; s/  */_/g')
if [ -z "$MOJOLEARN_SPEED_DEVICE" ]; then MOJOLEARN_SPEED_DEVICE="unknown-@VENDOR@"; fi
export MOJOLEARN_SPEED_DEVICE
echo "device=$MOJOLEARN_SPEED_DEVICE" >> "$OUT/leg.txt"

# WHERE THE TWO SIDES MEET. Several lanes have Mojo-only fixture builders and
# several compare outputs, so our arm writes the fixture (and, for the
# sequence lanes, its raw float32 output) here and the vendor arm reads it.
# Without this the vendor arm either refuses the lane by name or runs on data
# our arm never saw, and the second of those is much worse.
MOJOLEARN_SPEED_DUMP="$OUT/dump"
MOJOLEARN_SPEED_DUMP_DIR="$MOJOLEARN_SPEED_DUMP"
export MOJOLEARN_SPEED_DUMP MOJOLEARN_SPEED_DUMP_DIR
mkdir -p "$MOJOLEARN_SPEED_DUMP"

# ONE ARM, BOUNDED, ITS EXIT RECORDED, AND NEVER FATAL.
runarm() {
    _log="$1"; shift
    printf '\n=== %s :: %s ===\n' "$_log" "$*" >> "$OUT/console.log"
    if command -v timeout > /dev/null 2>&1; then
        timeout -k 15 @ARMBUDGET@ "$@" > "$LOGS/$_log" 2>&1
    else
        "$@" > "$LOGS/$_log" 2>&1
    fi
    _rc=$?
    echo "arm_exit ${_log}=$_rc" >> "$OUT/leg.txt"
    if [ "$_rc" = "124" ]; then
        # WRITTEN INTO THE ARM'S OWN LOG so the table builder carries it.
        # An arm killed by the clock that came home looking merely empty is
        # a silent cap, and a silent cap reads as coverage.
        echo "FSPEED-NOTE lane=${_log} arm=- KILLED BY THE PER-ARM BUDGET (@ARMBUDGET@s)" >> "$LOGS/$_log"
    fi
    tail -3 "$LOGS/$_log" >> "$OUT/console.log" 2>&1 || true
    return 0
}

# BUILD ONCE, RUN MANY, AND IT IS NOT A MICRO-OPTIMIZATION.
#
# `mojo run` COMPILES EVERY INVOCATION. One lane per process is a hard
# requirement here -- it is what stops one lane's failure taking the run down
# -- but paying a full compile per lane means the classical family compiles
# the SAME 2,400-line file twenty-two times inside a sixty-minute lease, and
# the sequence family six times. On a cold box that is the difference between
# a table and an empty directory.
#
# So each driver is BUILT once into a binary and the binary is executed per
# lane. The process isolation is unchanged: still one process per lane, still
# one lane's crash contained. What is dropped is only the repeated compile.
#
# A BUILD THAT FAILS IS RECORDED AND THE FAMILY CONTINUES. It is the single
# likeliest failure on this box -- no line of most of these lanes has ever
# been compiled for CUDA -- so it must produce a readable log rather than a
# silent absence, and the vendor arms must still run so the leg comes home
# with half a table instead of none.
BUILT=""
buildone() {
    _name="$1"; _src="$2"
    printf '
=== build %s ===
' "$_name" >> "$OUT/console.log"
    if command -v timeout > /dev/null 2>&1; then
        timeout -k 30 @BUILDBUDGET@ pixi run mojo build -I . "$_src" -o "$OUT/bin_$_name"             > "$LOGS/build.$_name.log" 2>&1
    else
        pixi run mojo build -I . "$_src" -o "$OUT/bin_$_name" > "$LOGS/build.$_name.log" 2>&1
    fi
    _rc=$?
    echo "build_exit ${_name}=$_rc" >> "$OUT/leg.txt"
    if [ "$_rc" = "0" ] && [ -x "$OUT/bin_$_name" ]; then
        BUILT="$BUILT $_name"
        return 0
    fi
    echo "  BUILD FAILED for $_name (exit $_rc); its lanes will have no arm of ours" >> "$OUT/console.log"
    tail -30 "$LOGS/build.$_name.log" >> "$OUT/console.log" 2>&1 || true
    return 1
}

builtok() {
    case " $BUILT " in *" $1 "*) return 0 ;; esac
    return 1
}

pipget() {
    # Best effort, logged, BOUNDED, never fatal. A vendor library that will
    # not install is a finding about this image and is recorded as one.
    #
    # THE BOUND IS THE POINT. RAPIDS is a multi-gigabyte resolve and it is
    # the one step in the classical family that can eat a sixty-minute
    # lease outright, leaving a box that was rented to measure things and
    # measured none. If it does not land inside the budget the vendor arms
    # refuse by name and OUR side is still timed, which is half a table
    # rather than none.
    printf '\n=== pip install %s ===\n' "$*" >> "$OUT/pip.log"
    if command -v timeout > /dev/null 2>&1; then
        timeout -k 30 @PIPBUDGET@ python3 -m pip install --no-input \
            --disable-pip-version-check "$@" >> "$OUT/pip.log" 2>&1
    else
        python3 -m pip install --no-input --disable-pip-version-check "$@" >> "$OUT/pip.log" 2>&1
    fi
    _prc=$?
    echo "pip_exit $*=$_prc" >> "$OUT/pip.log"
    echo "pip_exit $*=$_prc" >> "$OUT/leg.txt"
    if [ "$_prc" = "124" ]; then
        echo "  PIP INSTALL HIT ITS BUDGET (@PIPBUDGET@s): $*" >> "$OUT/console.log"
    fi
    return 0
}

case "@FAMILY@" in
gemmseq)
    # torch is already in the image, so this family installs almost nothing.
    # `einops` is the exception and it is NOT optional: the corpus generator
    # both sequence arms import for their hash spec and their reference scan
    # uses `rearrange` and `repeat` at module scope. Measured 2026-08-25 --
    # without it EVERY torch arm died at import with "No module named
    # 'einops'" and the whole sequence half of the leg came home with our
    # side only and no opponent at all.
    pipget einops
    # The real mamba opponent, attempted ONCE and bounded. It publishes
    # wheels keyed to an exact (torch, CUDA, abi, python) tuple; when the
    # tuple matches this lands in under a minute, and when it does not pip
    # starts running nvcc over several dtype and dstate instantiations, which
    # does not fit in a lease. If it does not arrive, the torch reference
    # scan is the labelled fallback and the write-up has to say which ran.
    #
    # THE PyPI INSTALL WAS THE WRONG INSTALL AND IT COST US THE OPPONENT.
    #
    # `pip install mamba-ssm` has no matching wheel on PyPI for this
    # (torch, CUDA, abi, python) tuple, so it falls back to a SOURCE BUILD
    # that runs nvcc over many dtype/dstate instantiations. It does not
    # finish in 240 seconds and it does not finish in a lease. The 240s
    # bound then expired, `mamba-ssm-cuda` refused, and the mamba lane was
    # measured against `torch-ref-scan-gpu` -- A SEQUENTIAL PyTorch LOOP
    # OVER THE SEQUENCE. Being 2.14x faster than that is not a result about
    # Mamba; it is a result about a reference implementation, and it was
    # reported as though it were the former.
    #
    # The project publishes PREBUILT WHEELS on its GitHub releases, keyed by
    # exactly that tuple. So ASK THE BOX what its tuple is rather than
    # hardcoding one, build the two URLs, and install those. A wheel lands
    # in under a minute.
    # DISCOVER THE ASSET, DO NOT GUESS IT, AND INSTALL THE TWO SEPARATELY.
    #
    # The first attempt hardcoded `causal-conv1d v1.4.0` and `mamba-ssm
    # v2.2.4` and put BOTH in one pip command. The causal-conv1d asset name
    # was wrong, GitHub returned 404, and because they shared a command pip
    # never even tried the mamba_ssm wheel -- whose URL was correct. One
    # guessed version number cost the whole opponent for a second leg.
    #
    # So: ask the GitHub releases API which assets actually exist, match the
    # box's own (cuda, torch, abi, python) tag against them, and install each
    # wheel in ITS OWN command so one 404 cannot take the other down.
    # `mamba_ssm`'s `selective_scan_fn` -- the only thing this leg probes --
    # does not need causal_conv1d, so losing that one is survivable and
    # losing mamba_ssm is not.
    python3 - > /root/mamba_urls.txt 2>> "$OUT/pip.log" <<'MMTAG'
import json, sys, urllib.request

try:
    import torch
    tv = ".".join(torch.__version__.split(".")[:2])
    cu = "cu12" if (torch.version.cuda or "").startswith("12") else "cu11"
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
except Exception as e:
    sys.stderr.write("no torch tuple: %r\n" % (e,))
    raise SystemExit(0)
py = "cp%d%d" % (sys.version_info[0], sys.version_info[1])
want = "%storch%scxx11abi%s" % (cu, tv, abi)
sys.stderr.write("mamba wheel tag wanted: %s / %s\n" % (want, py))

for repo in ("state-spaces/mamba", "Dao-AILab/causal-conv1d"):
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/%s/releases?per_page=30" % repo,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "mojolearn-leg"})
        rels = json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e:
        sys.stderr.write("%s: releases API failed %r\n" % (repo, e))
        continue
    hit = None
    for rel in rels:                      # newest first
        for a in rel.get("assets") or []:
            n = a.get("name", "")
            if want in n and py in n and n.endswith(".whl"):
                hit = a["browser_download_url"]
                break
        if hit:
            break
    if hit:
        print(hit)
    else:
        sys.stderr.write("%s: NO asset matches %s / %s\n" % (repo, want, py))
MMTAG
    echo "mamba_wheel_urls=$(tr '\n' ' ' < /root/mamba_urls.txt)" >> "$OUT/leg.txt"
    if [ -s /root/mamba_urls.txt ] && command -v timeout > /dev/null 2>&1; then
        while read -r _w; do
            [ -n "$_w" ] || continue
            timeout -k 15 420 python3 -m pip install --no-input \
                --no-build-isolation "$_w" >> "$OUT/pip.log" 2>&1
            echo "mamba_wheel_exit $(basename "$_w" | cut -c1-24)=$?" >> "$OUT/leg.txt"
        done < /root/mamba_urls.txt
    else
        echo "mamba_ssm_install=SKIPPED, no matching wheel was discovered" >> "$OUT/leg.txt"
    fi
    # THE PROBE, for the same reason the LightGBM one exists: an install
    # exit code says pip ran, and only a CALL says the fused kernel is
    # there. If this says NO then the mamba lane's only opponent is the
    # reference scan, and THAT HAS TO BE SAID BESIDE THE NUMBER rather than
    # discovered later in a log.
    python3 - >> "$LOGS/mamba_ssm_probe.log" 2>&1 <<'MMPROBE'
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
d, n, L = 4, 8, 16
dev = "cuda"
u = torch.randn(1, d, L, device=dev)
delta = torch.rand(1, d, L, device=dev)
A = -torch.rand(d, n, device=dev)
B = torch.randn(1, n, L, device=dev)
C = torch.randn(1, n, L, device=dev)
out = selective_scan_fn(u, delta, A, B, C)
torch.cuda.synchronize()
print("MAMBA_SSM_PROBE=ok shape=%s" % (tuple(out.shape),))
MMPROBE
    if grep -q MAMBA_SSM_PROBE=ok "$LOGS/mamba_ssm_probe.log" 2>/dev/null; then
        echo "mamba_ssm_works=yes" >> "$OUT/leg.txt"
    else
        echo "mamba_ssm_works=NO" >> "$OUT/leg.txt"
        echo "mamba_ssm_probe_says=$(tail -1 "$LOGS/mamba_ssm_probe.log" 2>/dev/null)" >> "$OUT/leg.txt"
    fi
    # OURS FIRST, EVERY LANE, because the torch arm reads the raw output our
    # arm dumped in order to answer whether the two sides computed the same
    # thing. A vendor arm that ran before its dump existed refuses the
    # agreement line and the row becomes a pair of timings with nothing
    # tying them together.
    buildone gemmspeed bench/speed/gemm_speed_main.mojo
    buildone seqspeed  bench/speed/seq_speed_main.mojo
    for L in @SPEEDLANES@; do
        MOJOLEARN_SPEED_LANE="$L"; export MOJOLEARN_SPEED_LANE
        case "$L" in
            gemm) builtok gemmspeed && runarm "gemm.gemm.ours.log" "$OUT/bin_gemmspeed" ;;
            *)    builtok seqspeed  && runarm "seq.$L.ours.log"    "$OUT/bin_seqspeed" ;;
        esac
    done
    # THE CORRECTNESS GATES RUN BESIDE THE SPEED DRIVERS, on the same box,
    # in the same lease, and they are NOT optional here.
    #
    # DEVIATION 1876 changed which kernel the FAST arm of transformer/ and
    # mamba/ uses. FAST has never been bit-stable and the profile's promise
    # lives entirely on the IDENTICAL side, so that is legitimate -- but
    # those two lanes have FAST-mode gates with tolerances as tight as
    # rtol 1e-7, and a different summation order can cross that. A speed win
    # measured while a correctness gate quietly went red is not a win, and
    # this Mac is not allowed to run them. So they run here, and their exit
    # codes come home in leg.txt beside the numbers they justify.
    runarm "verify.transformer_block.fast.log" \
        pixi run mojo run -I . transformer/mojo_only/transformer_check.mojo
    runarm "verify.mamba_block.fast.log" \
        pixi run mojo run -I . mamba/mojo_only/mamba_check.mojo
    # AND UNDER IDENTICAL, WHICH IS THE HALF THAT IS EASY TO SKIP.
    #
    # DEVIATION 1876 puts the vendor kernel behind a comptime gate, so the
    # claim is that IDENTICAL is untouched -- not slower, not different, not
    # compiled at all. That is REASONING about a `comptime if`, and the
    # profile this whole repository is built on is the thing it would break.
    # Reasoning is not what this lane accepts anywhere else and it is not
    # accepted here. The two gates run again with the define set, on the same
    # box, in the same lease.
    runarm "verify.transformer_block.identical.log" \
        sh tools/with_identical_mode.sh pixi run mojo run -I . \
            transformer/mojo_only/transformer_check.mojo
    runarm "verify.mamba_block.identical.log" \
        sh tools/with_identical_mode.sh pixi run mojo run -I . \
            mamba/mojo_only/mamba_check.mojo
    runarm "gemm.gemm.cublas.log" python3 tools/speed_gemm_arm.py --rounds "@SPEEDROUNDS@"
    for L in @SPEEDLANES@; do
        [ "$L" = "gemm" ] && continue
        runarm "seq.$L.torch.log" python3 tools/speed_torch_seq.py \
            --lane "$L" --rounds "@SPEEDROUNDS@" --dump-dir "$MOJOLEARN_SPEED_DUMP"
    done
    ;;
classical)
    # RAPIDS is the install that can eat the lease. It runs FIRST so that a
    # failure is known before any Mojo time is spent, and it is bounded.
    pipget --extra-index-url=https://pypi.nvidia.com "cuml-cu12" "cuvs-cu12"
    pipget scikit-learn scipy
    buildone classicalspeed bench/speed/classical_speed_main.mojo
    for L in @SPEEDLANES@; do
        MOJOLEARN_SPEED_LANE="$L"; export MOJOLEARN_SPEED_LANE
        builtok classicalspeed && runarm "classical.$L.ours.log" "$OUT/bin_classicalspeed"
        # NO --lane FLAG: tools/speed_cuml_arm.py takes its lane from
        # MOJOLEARN_SPEED_LANE, which is already exported above, and has no
        # argparse at all. Passing a flag it does not know would abort the
        # arm on every lane.
        runarm "classical.$L.vendor.log" \
            python3 tools/speed_cuml_arm.py
    done
    ;;
forest)
    # THE BINDINGS HAVE TO BE BUILT HERE AND THEY NEVER HAVE BEEN.
    #
    # `python/mojolearn/*.so` are NOT committed (`git ls-files` finds zero),
    # and this leg ships `git archive` at a pinned sha, so the box receives
    # no binaries at all. The ones on the Mac would be useless anyway: they
    # are Apple objects. Every one of these is therefore a FIRST-EVER CUDA
    # BUILD of that binding, which is the likeliest thing in this family to
    # fail, so each is bounded and each records its own exit code and none of
    # them is fatal. A family that comes home with the vendor arms timed and
    # ours missing still says something true about this box.
    #
    # WHICH FOUR, AND WHY THOSE: gbdt (all three boosting lanes), rf, trees
    # (extratrees), svm (which carries isolation forest). estimators comes
    # first because build_gbdt.sh links against it.
    #
    # SYSTEM python3 IS THE RIGHT INTERPRETER FOR THESE, and that is checked
    # rather than assumed: `bindings/build_rf.sh` smoke-tests its own output
    # with `python3`, so the extension is meant to import there. That matters
    # because the vendor arms cannot run under pixi at all -- RAPIDS and
    # CatBoost want 3.10 to 3.13 and the gbmbench feature pins 3.14 -- so
    # both sides of this family have to meet on the image's python.
    # THE BASE EXTENSION IS FIRST AND ITS SCRIPT IS NOT NAMED LIKE THE
    # OTHERS. `bindings/build.sh` emits `python/mojolearn/_mojolearn.so`;
    # every other script is `build_<name>.sh`. Leaving it out of the loop is
    # what broke the 2026-08-25 trees leg, and the failure did not look like
    # a missing dependency at all:
    #
    #   ImportError: cannot import name '_mojolearn' from partially
    #   initialized module 'mojolearn' ... (circular import)
    #
    # THREE OF FIVE BUILDS "FAILED" AND NOT ONE OF THEM WAS A COMPILE ERROR.
    # Each script's smoke test copies `python/mojolearn/` into a temp
    # directory and drops in the ONE .so it just built. On this Mac that
    # directory already holds `_mojolearn.so` from some earlier build, so the
    # package imports and the smoke passes. On the box it holds only .py
    # files -- the .so are untracked and the leg ships `git archive` -- so
    # `__init__.py`'s `from .cluster import KMeans` reaches
    # `from . import _mojolearn` and there is nothing there. gbdt, rf and
    # trees each compiled correctly and then failed to import a package that
    # was missing its base.
    #
    # A LOCAL BUILD ARTIFACT WAS LOAD-BEARING AND NOBODY KNEW, which is the
    # same class of thing as a benchmark that only passes because a cache is
    # warm. It is invisible on any machine that has ever built the package.
    # DEVIATION 1880 -- TWO PASSES, AND NO ORDERING CAN REPLACE THEM.
    #
    # Every build_*.sh ends in a smoke test that copies `python/mojolearn/`
    # into a temp directory, drops in the ONE .so it just built, and imports
    # the package. `__init__.py` pulls in several submodules, so that import
    # needs EVERY sibling extension present -- and each script installs its
    # own .so only AFTER its smoke passes. The dependency is circular and it
    # cannot be ordered away: with base first, base's smoke died on a missing
    # `_mojolearn_estimators`; with estimators first, the others died on a
    # missing `_mojolearn`. Both were observed, an hour apart, on two rented
    # boxes.
    #
    # THE REPOSITORY ALREADY KNEW. bindings/build.sh:315 says it in as many
    # words -- "four gates fail on the siblings' not-yet-built .so files. The
    # caller that sets this owns end-to-end verification" -- and provides
    # MOJOLEARN_SKIP_BUILD_GATE, which installs the artifact and skips the
    # smoke. It is invisible on any machine that has ever built the package,
    # because there the siblings are already sitting in the directory.
    #
    # PASS 1 installs all six with the gate skipped. PASS 2 re-runs them with
    # the gate LIVE, against a now-complete package, so nothing is taken on
    # trust: the smoke tests are the only thing that launches kernels through
    # these extensions, and every broken build in this family's history
    # imported fine and died at the first launch. Pass 2's exit codes are the
    # ones recorded as `binding_build_exit`; pass 1's are recorded separately
    # so a failure can be attributed to the right pass.
    # DEVIATION 1881 -- EVERY BINDING, NOT THE SIX THIS FAMILY USES.
    #
    # `mojolearn/__init__.py` imports the WHOLE package surface, so `import
    # mojolearn` needs every extension present regardless of which lane is
    # being timed. With six built, the base gate died on `_mojolearn_solver`
    # (`__init__.py:106` -> `_hierarchy_impl` -> `_mojolearn_solver`) and the
    # ARMS would have died on it too: nothing in this family touches the
    # solver, and it is still required to import the package at all.
    #
    # THE LIST IS DERIVED FROM THE FILESYSTEM rather than written out, so it
    # cannot drift the way the six-name list did. `bindings/build.sh` is
    # named on its own because it is the one script that is not
    # `build_<name>.sh`, which is what hid it in the first place.
    _bscripts="bindings/build.sh $(ls bindings/build_*.sh 2>/dev/null | tr '\n' ' ')"
    for _pass in 1 2; do
        for _bs in $_bscripts; do
            _b=$(basename "$_bs" .sh | sed 's/^build_//; s/^build$/base/')
            printf '\n=== pass %s: %s ===\n' "$_pass" "$_bs" >> "$OUT/console.log"
            if [ "$_pass" = "1" ]; then
                _skip=1
            else
                _skip=""
            fi
            if command -v timeout > /dev/null 2>&1; then
                MOJOLEARN_SKIP_BUILD_GATE="$_skip" timeout -k 30 @BUILDBUDGET@ \
                    bash "$_bs" > "$LOGS/build.binding.$_b.pass$_pass.log" 2>&1
            else
                MOJOLEARN_SKIP_BUILD_GATE="$_skip" \
                    bash "$_bs" > "$LOGS/build.binding.$_b.pass$_pass.log" 2>&1
            fi
            _brc=$?
            if [ "$_pass" = "1" ]; then
                echo "binding_install_exit ${_b}=$_brc" >> "$OUT/leg.txt"
            else
                echo "binding_build_exit ${_b}=$_brc" >> "$OUT/leg.txt"
            fi
            tail -4 "$LOGS/build.binding.$_b.pass$_pass.log" >> "$OUT/console.log" 2>&1 || true
        done
    done
    # THE ONE CHECK THAT ACTUALLY PREDICTS WHETHER THE ARMS CAN RUN. Every
    # gate above is per-extension; this is the thing the arms themselves do.
    # Recorded rather than fatal: if it fails the lanes will refuse by name
    # anyway, and this line says WHY in one place instead of six times.
    # FROM `python/`, WHICH IS WHERE THE PACKAGE LIVES. The first version of
    # this check ran from the repo root and reported
    # `ModuleNotFoundError: No module named 'mojolearn'` while all ten
    # extensions were built, installed and importable -- a red light on a
    # healthy build, which is the worst kind of check to have. The arms get
    # this right on their own (`forest_speed_arm.py` puts `python/` on
    # sys.path), so the check has to do what the arms do rather than
    # something adjacent to it.
    ( cd python && python3 -c "import mojolearn; print('mojolearn imports OK', mojolearn.__file__)" ) \
        > "$LOGS/import_mojolearn.log" 2>&1
    echo "import_mojolearn_exit=$?" >> "$OUT/leg.txt"
    tail -3 "$LOGS/import_mojolearn.log" >> "$OUT/console.log" 2>&1 || true
    ls -la python/mojolearn/*.so > "$OUT/bindings_listing.txt" 2>&1 \
        || echo "NO .so BUILT AT ALL" > "$OUT/bindings_listing.txt"

    pipget catboost xgboost lightgbm scikit-learn
    # THE DATASETS ARE FETCHED AS THEIR OWN NAMED STEP, ONCE, BEFORE ANY ARM.
    #
    # `year` is a 211 MB zip plus a decode to a ~170 MB npz, and `covtype`
    # comes through sklearn's fetcher. Both CACHE -- year into
    # GBM_BENCH_DATA/year_speed.npz, covtype into ~/scikit_learn_data -- so
    # the cost is paid once and every later lane reads the cache.
    #
    # LEFT INSIDE THE ARMS IT WOULD BE PAID INSIDE A 600s PER-ARM BUDGET, and
    # worse than that: an arm killed mid-download leaves no cache, so the NEXT
    # lane starts the same download from zero, and six lanes could spend a
    # whole lease downloading the same file six times and timing nothing. The
    # budget here is generous because this is a network fetch and not a
    # measurement, and its exit code is recorded so a table missing the year
    # rows says WHY.
    if command -v timeout > /dev/null 2>&1; then
        timeout -k 30 1200 python3 tools/speed_gbdt_arm.py --download year \
            > "$LOGS/download.year.log" 2>&1
        echo "download_exit year=$?" >> "$OUT/leg.txt"
        timeout -k 30 600 python3 tools/speed_gbdt_arm.py --download covtype \
            > "$LOGS/download.covtype.log" 2>&1
        echo "download_exit covtype=$?" >> "$OUT/leg.txt"
        # HIGGS IS 2.6 GB AND ONLY FETCHED WHEN THE LADDER ASKS FOR IT.
        # The budget is 2400s because it is a 2.6 GB pull PLUS a gzip csv
        # parse of 11M x 29 that runs several minutes, and both happen here,
        # once, rather than inside a 600s per-arm budget six times over.
        if [ "@SPEEDDATASET@" = "higgs" ]; then
            timeout -k 30 2400 python3 tools/speed_gbdt_arm.py --download higgs \
                > "$LOGS/download.higgs.log" 2>&1
            echo "download_exit higgs=$?" >> "$OUT/leg.txt"
        fi
    fi
    pipget --extra-index-url=https://pypi.nvidia.com "cuml-cu12"
    # LightGBM's CUDA learner is NOT in the wheel and has to be built. It is
    # attempted LAST of the installs and bounded, because it has been
    # measured at fifteen to thirty minutes and this lease is sixty.
    if command -v timeout > /dev/null 2>&1; then
        # `lightgbm_cuda_build_exit=0` WAS A LYING EXIT CODE.
        #
        # On 2026-08-25 this build reported success and produced a CPU-ONLY
        # wheel, and every `lightgbm-cuda` arm in every lane then refused
        # with "CUDA Tree Learner was not enabled in this build". LightGBM
        # is the lossguide lane's OWN ALGORITHM, so the missing opponent is
        # missing exactly where it would be strongest, and its absence
        # flatters us. That is not a gap to leave unexamined.
        #
        # Two changes, and then a PROBE that does not trust either of them:
        #
        #   --no-cache-dir   pip caches a wheel built from an sdist keyed by
        #                    the sdist, NOT by --config-settings. A build
        #                    from an earlier install WITHOUT the CUDA flag
        #                    is therefore a cache hit for this one, and it
        #                    installs, and it reports success.
        #   CMAKE_ARGS       the same define by the second route, so a
        #                    backend that drops --config-settings still
        #                    gets it.
        rm -rf /root/.cache/pip/wheels 2>/dev/null || true
        if [ "@LGBMCUDA@" != "1" ]; then
            echo "lightgbm_cuda_build=SKIPPED by --no-lgbm-cuda; the wheel's" \
                 "CPU learner still runs and every lightgbm-cuda arm will" \
                 "refuse by name" >> "$OUT/leg.txt"
        else
        # CUDACXX: pip's isolated build env resolves the CUDA compiler
        # through CMake's enable_language(CUDA), which walks CUDACXX and
        # PATH -- and the leg's non-interactive ssh shell has neither.
        # Leg 2026-08-27_031904: "No CMAKE_CUDA_COMPILER could be found",
        # build exit 1, wheel came out CPU-only and every lightgbm-cuda
        # arm refused. The devel images keep nvcc at /usr/local/cuda/bin.
        CMAKE_ARGS="-DUSE_CUDA=ON" \
        CUDACXX=/usr/local/cuda/bin/nvcc \
        PATH="/usr/local/cuda/bin:$PATH" \
        timeout -k 30 @LGBMBUILD@ python3 -m pip install --no-input \
            --no-cache-dir --force-reinstall \
            --no-binary lightgbm \
            --config-settings=cmake.define.USE_CUDA=ON lightgbm \
            >> "$OUT/pip.log" 2>&1
        echo "lightgbm_cuda_build_exit=$?" >> "$OUT/leg.txt"
        # THE PROBE. An exit code says the build ran; only a fit on the
        # device says the learner is in it. Sixteen rows, two leaves, one
        # iteration -- it costs nothing and it is the difference between
        # "we built it" and "it works". Recorded as its own line so the
        # table's reader can see WHY a lightgbm-cuda row is missing without
        # reading a 40 MB pip log.
        python3 - >> "$LOGS/lightgbm_cuda_probe.log" 2>&1 <<'LGBMPROBE'
import numpy as np, lightgbm as lgb
x = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
y = (x[:, 0] > 0).astype(np.float32)
m = lgb.LGBMRegressor(device="cuda", n_estimators=1, num_leaves=2,
                      min_child_samples=1, verbose=-1)
m.fit(x, y)
print("LIGHTGBM_CUDA_PROBE=ok")
LGBMPROBE
        if grep -q LIGHTGBM_CUDA_PROBE=ok "$LOGS/lightgbm_cuda_probe.log" 2>/dev/null; then
            echo "lightgbm_cuda_works=yes" >> "$OUT/leg.txt"
        else
            echo "lightgbm_cuda_works=NO" >> "$OUT/leg.txt"
            echo "lightgbm_cuda_probe_says=$(tail -1 "$LOGS/lightgbm_cuda_probe.log" 2>/dev/null)" >> "$OUT/leg.txt"
        fi
        fi
    else
        echo "lightgbm_cuda_build=SKIPPED, no timeout(1) to bound it" >> "$OUT/leg.txt"
    fi
    # ONE PROCESS PER LANE, BOTH SIDES IN IT, ALTERNATING ROUND BY ROUND.
    #
    # This used to be two processes per lane: `--ours-only` here and
    # tools/speed_gbdt_arm.py separately. That splits the arms across two
    # processes, and a ratio built from two processes is exposed to
    # everything that can change between them. This box is shared and
    # throttled and this repository has measured one drifting 1.7x inside
    # twenty minutes; the whole reason the interleaved format exists is that
    # a box which throttles mid-run throttles BOTH arms and the ratio
    # survives what an absolute number does not.
    #
    # Without `--ours-only`, forest_speed_arm.py appends the opponents to its
    # own rotation -- ours first, then theirs, no arm twice in a row -- which
    # is the format every other comparison in this repository quotes.
    #
    # `--ours-only` stays in that file for the case it was written for: two
    # CUDA runtimes that will not coexist in one process. If a lane crashes
    # on import, splitting it is the fallback, and then the split has to be
    # said out loud beside the number.
    #
    # THE LADDER IS THE INNER LOOP, ON PURPOSE. Rungs of one lane run
    # back to back, so the thing that varies between two rungs is the row
    # count and not forty minutes of a shared box's thermal history. Each
    # rung is a separate process because each rung is a different fit; what
    # is interleaved WITHIN a rung is ours against theirs, which is the
    # comparison the ratio is made of.
    for L in @SPEEDLANES@; do
        # THE iforest LANE KEEPS ITS OWN DATASET AND THE LINE SAYS SO.
        # An isolation forest scored on data with no planted anomalies has
        # no accuracy column, and a timing with no accuracy column is the
        # thing this whole slice refuses to print. So the ladder's dataset
        # is NOT forced onto it; it runs its `anomaly` fixture, whose row
        # count --rows still climbs.
        _dsflag=""
        if [ -n "@SPEEDDATASET@" ] && [ "$L" != "iforest" ]; then
            _dsflag="--dataset @SPEEDDATASET@"
        fi
        if [ -z "@SPEEDROWS@" ]; then
            runarm "forest.$L.log" \
                python3 bench/speed/forest_speed_arm.py --lane "$L" $_dsflag
        else
            for R in @SPEEDROWS@; do
                runarm "forest.$L.r$R.log" \
                    python3 bench/speed/forest_speed_arm.py --lane "$L" \
                        $_dsflag --rows "$R"
            done
        fi
    done
    ;;
esac

# WHAT ACTUALLY CAME OUT, counted on the box, so a fetch that loses files is
# distinguishable from a run that produced none.
echo "fspeed_lines=$(cat "$LOGS"/*.log 2>/dev/null | grep -c '^FSPEED')" >> "$OUT/leg.txt"
echo "fspeed_rounds=$(cat "$LOGS"/*.log 2>/dev/null | grep -c '^FSPEED ')" >> "$OUT/leg.txt"
echo "fspeed_refused=$(cat "$LOGS"/*.log 2>/dev/null | grep -c '^FSPEED-REFUSED')" >> "$OUT/leg.txt"
# THE MODE WITNESS, READ BACK ON THE BOX. Every `ours` header must say FAST.
# One that says IDENTICAL means a correctly-labelled measurement of the WRONG
# ARM, which is the failure this repository has already been bitten by three
# times in one day.
echo "ours_headers_fast=$(grep -h '^FSPEED-HEADER' "$LOGS"/*.log 2>/dev/null | grep -c 'arm=ours mode=FAST')" >> "$OUT/leg.txt"
echo "ours_headers_identical=$(grep -h '^FSPEED-HEADER' "$LOGS"/*.log 2>/dev/null | grep -c 'arm=ours mode=IDENTICAL')" >> "$OUT/leg.txt"

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
cat "$OUT/leg.txt"
: > /root/gemm_leg.done
echo REMOTE_BODY_DONE
REMOTE_BODY_SPEED
}

leg_check_remote_body() {
    _body="$OUT/remote_body.sh"
    # SUBSTITUTION, AND THEN A CHECK THAT IT HAPPENED. A `sed` without `/g`
    # has already left one of this repository's guards holding a literal
    # placeholder, so the trailing grep is the part that matters: it does not
    # trust /g, it verifies that no placeholder survived.
    sed -e "s|@VENDOR@|$VENDOR|g" \
        -e "s|@COMMIT@|$COMMIT|g" \
        -e "s|@FAMILY@|$SPEED_FAMILY|g" \
        -e "s|@SPEEDLANES@|$SPEED_LANES|g" \
        -e "s|@SPEEDROUNDS@|$SPEED_ROUNDS|g" \
        -e "s|@SPEEDDATASET@|$SPEED_DATASET|g" \
        -e "s|@SPEEDROWS@|$SPEED_ROWS|g" \
        -e "s|@LGBMCUDA@|$SPEED_LGBM_CUDA|g" \
        -e "s|@SPEEDSIZE@|$SPEED_SIZE|g" \
        -e "s|@ARMBUDGET@|$ARM_BUDGET|g" \
        -e "s|@BUILDBUDGET@|$BUILD_BUDGET|g" \
        -e "s|@PIPBUDGET@|$PIP_BUDGET|g" \
        -e "s|@LGBMBUILD@|$LGBM_BUILD|g" \
        -e "s|@CARDFULL@|$CARD_FULL|g" \
        -e "s|@SWEEP@|$SWEEP|g" \
        -e "s|@DUMP@|$LEG_DUMP|g" \
        -e "s|@WORKTIMEOUT@|$WORK_TIMEOUT|g" \
        -e "s|@E1PHASES@|$E1_PHASES|g" \
        -e "s|@E1LANES@|$E1_LANES|g" \
        -e "s|@SMI@|$SMI_CMD|g" \
        "$_body" > "$_body.subst"
    mv "$_body.subst" "$_body"
    if grep -q '@[A-Z][A-Z_]*@' "$_body"; then
        echo "  UNSUBSTITUTED PLACEHOLDER in the remote body:"
        grep -n '@[A-Z][A-Z_]*@' "$_body" | sed 's/^/    /'
        return 1
    fi
    # POSIX SYNTAX CHECK BEFORE IT IS EVER SHIPPED. The pod's /bin/sh is
    # dash, this Mac's is not, so `dash -n` runs when it is available and
    # `sh -n` always does.
    sh -n "$_body" || { echo "  the remote body is not valid sh"; return 1; }
    if command -v dash > /dev/null 2>&1; then
        dash -n "$_body" || { echo "  the remote body is not valid DASH"; return 1; }
    fi
    # COMMENTS ARE NOT CODE. The first version of this grep matched the
    # remote body's own header, which explains that `exec -a` is a bashism --
    # a check that fails on its own documentation is a check that will be
    # deleted by the third person who hits it. Found by running it.
    # `local` is matched after ANY of start-of-line, space, tab, `;` or `{`,
    # because `f() { local x=1; }` is the spelling that actually appears and
    # an anchored `^ *local ` walks straight past it.
    # AWK ENDS A RULE AT A NEWLINE unless the line ends in `&&`, `||` or `{`.
    # A version of this split before the `{`, which awk read as TWO rules --
    # a bare pattern (default action: print the line) plus an unconditional
    # print -- so it "found" a bashism on every line including the comments.
    # Caught by running it against a file with three planted bashisms.
    awk '!/^[[:space:]]*#/ &&
         (/exec -a/ || /\[\[/ || /(^|[;{ \t])local / || /<\(/ ||
          /(^|[;{ \t])function / || /(^|[;{ \t])source / || /echo -e/) {
             print FNR ": " $0
         }' "$_body" > "$TMPD/bashisms"
    if [ -s "$TMPD/bashisms" ]; then
        echo "  BASHISM in the remote body (the pod runs dash):"
        sed 's/^/    /' "$TMPD/bashisms"
        return 1
    fi
    # THE SENTINEL IS WHAT THE POLL LOOP WAITS FOR (DEVIATION 863). A body
    # that stopped writing it would make every leg poll all the way to its
    # outer deadline and then report a complete run as a cut-off one, which
    # is a false finding about the box printed with total confidence.
    if ! grep -q 'gemm_leg.done' "$_body"; then
        echo "  the remote body never writes /root/gemm_leg.done."
        echo "  That file is the poll loop's only 'the payload finished'"
        echo "  signal; without it every leg waits out its outer deadline and"
        echo "  then calls a finished run partial."
        return 1
    fi
    if [ "$PAYLOAD" = "phase8" ]; then
        # THE TWO THINGS THIS PAYLOAD IS ABOUT, checked in the shipped text
        # rather than believed from the heredoc above it.
        if ! grep -q 'bash tools/e1_bootstrap.sh' "$_body"; then
            echo "  the phase8 body does not run tools/e1_bootstrap.sh."
            echo "  A body that reimplements phase 8 is a second lane list"
            echo "  and the judge would be reading cards from a different"
            echo "  program than the one it is named after."
            return 1
        fi
        if ! grep -q 'unset MOJOLEARN_MAMBA_CHECK_B' "$_body"; then
            echo "  the phase8 body does not unset the mamba shape variables."
            echo "  MOJOLEARN_MAMBA_CHECK_B/_L/_DM widen the mamba gate and"
            echo "  every shape is another compile inside the lease."
            return 1
        fi
    fi
    return 0
}

leg_ship_and_run() {
    leg_say "shipping the COMMIT ($COMMIT), not the working tree"
    git archive --format=tar "$COMMIT" | gzip > "$TMPD/src.tgz"
    # The Mac's half of the two-sided source hash, computed from the ARCHIVE
    # so that what is compared is what was SHIPPED.
    mkdir -p "$TMPD/archive"
    gzip -dc "$TMPD/src.tgz" | ( cd "$TMPD/archive" && tar xf - )
    # A PIPELINE'S STATUS IS ITS LAST COMMAND'S, so neither `git archive` nor
    # the untar above can report a failure through `set -e`. Check the
    # RESULT instead: the file the leg is entirely about has to be in there.
    for _need in $(leg_archive_required); do
        [ -f "$TMPD/archive/$_need" ] || leg_die \
"THE ARCHIVE DOES NOT CONTAIN $_need.
  Either 'git archive' of $COMMIT failed, or that file is not committed at
  this commit. Shipping it would rent a box to build nothing."
    done
    leg_source_sha_recipe "$TMPD/archive" > "$OUT/source_sha256_local.txt"
    leg_say "  archive source sha256: $(cut -c1-32 < "$OUT/source_sha256_local.txt")"

    leg_ssh 'rm -rf /root/mojolearn /root/gemm_leg_out && mkdir -p /root/mojolearn' \
        > /dev/null
    leg_ssh 'cd /root/mojolearn && tar xzf -' < "$TMPD/src.tgz"
    leg_ssh 'umask 022; cat > /root/gemm_leg.sh' < "$OUT/remote_body.sh"
    leg_run_payload
}

# ---------------------------------------------------------------------------
# DEVIATION 863 -- THE PAYLOAD RUNS DETACHED AND THIS LEG POLLS IT
#
# It used to be one foreground `leg_ssh 'sh /root/gemm_leg.sh'` with `|| true`
# after it. Under --payload phase8 that holds an ssh session open for the
# length of a whole bootstrap: measured 27 minutes on an H100 and 18 on an
# MI325X ON IMAGES THAT ALREADY HAD PIXI, and this lane's images do not. One
# dropped ssh -- a wifi blip, a closed lid, ServerAliveInterval giving up
# after ninety seconds -- did two things, and the second is much worse than
# the first. The remote `sh` took SIGHUP and the bootstrap died mid-lane; and
# the `|| true` swallowed it, so this script walked on to the fetch, brought
# home whichever half of phase 8 had finished, terminated the box and
# reported a leg. NOTHING IN THE ARTIFACTS SAID THE RUN WAS CUT OFF. That is
# a fabricated column, and one whole rental to discover.
#
# So the body is started with `nohup`, in the background, writing to a file
# on the box, and polled with short ssh calls that are each allowed to fail
# and be retried. The sentinel is a file the body writes as its last act. The
# liveness test is `kill -0` on the remote pid, so THREE states are told
# apart: finished (sentinel), still running (pid alive), and DIED WITHOUT
# FINISHING (no sentinel, no pid) -- which is a finding and is recorded as
# one rather than passed off as a complete run.
#
# The body still bounds itself with `timeout @WORKTIMEOUT@`. The deadline
# here is that plus a margin, and it is the bound for the case where the
# body is gone, the sentinel never appeared, and the image had no timeout(1).
# ---------------------------------------------------------------------------
leg_run_payload() {
    leg_say "starting the payload DETACHED on the box, then polling for it"
    leg_ssh 'rm -f /root/gemm_leg.done /root/gemm_leg_console.log;
             nohup sh /root/gemm_leg.sh > /root/gemm_leg_console.log 2>&1 < /dev/null &
             echo "REMOTE_PID=$!"' > "$OUT/remote_start.log" 2>&1 || true
    sed 's/^/    /' "$OUT/remote_start.log"
    _rpid=$(sed -n 's/^REMOTE_PID=//p' "$OUT/remote_start.log" | tr -d "\r" | tail -1)
    if [ -z "$_rpid" ]; then
        leg_die "THE PAYLOAD DID NOT START. The box answered without a pid, so
  there is nothing running there and polling would poll to the deadline for
  no reason. Read $OUT/remote_start.log. Terminating."
    fi
    # THE OUTER DEADLINE IS CAPPED BY THE LEASE, not only by the work bound.
    # Two reasons, and the second one only shows on a bad day. The body binds
    # itself with timeout(1) at WORK_TIMEOUT, so a little past that is all a
    # healthy run needs. But when the POD dies -- the on-pod watchdog fires,
    # RunPod evicts it, the network goes -- every poll from then on is simply
    # unreachable, and an uncapped deadline would sit here retrying a box that
    # no longer exists, for as long as the work bound allowed, with the fetch
    # and the terminate still waiting behind it. The cap leaves the lease its
    # last five minutes so the fetch below still has somewhere to run.
    _polllimit=$(( WORK_TIMEOUT + 240 ))
    _leasecap=$(( MINUTES * 60 - 300 ))
    if [ "$_polllimit" -gt "$_leasecap" ]; then _polllimit=$_leasecap; fi
    if [ "$_polllimit" -lt 120 ]; then _polllimit=120; fi
    leg_say "  remote pid $_rpid; polling every 30s, giving up after ${_polllimit}s"
    _pdeadline=$(( $(date -u +%s) + _polllimit ))
    _unreach=0
    while : ; do
        if [ "$(date -u +%s)" -ge "$_pdeadline" ]; then
            echo "    OUTER POLL DEADLINE reached (${_polllimit}s)."
            echo "    The body bounds itself with timeout(1) at ${WORK_TIMEOUT}s,"
            echo "    so getting here means either that bound did not hold (no"
            echo "    timeout(1) on the image, or no sentinel written) or the"
            echo "    box stopped answering and never came back. Fetching what"
            echo "    exists, then terminating."
            FETCH_RED=1
            break
        fi
        _st=$(leg_ssh "if [ -f /root/gemm_leg.done ]; then echo LEG_DONE;
                       elif kill -0 $_rpid 2>/dev/null; then echo LEG_RUNNING;
                       else echo LEG_GONE; fi" 2>/dev/null) || _st=""
        case "$_st" in
            *LEG_DONE*)
                leg_say "  the payload finished (its sentinel is on the box)"
                break ;;
            *LEG_RUNNING*)
                _unreach=0 ;;
            *LEG_GONE*)
                echo "    THE PAYLOAD PROCESS IS GONE AND WROTE NO SENTINEL."
                echo "    It was killed, or it died before its last line. What"
                echo "    is on the box is a PARTIAL run. The fetch below still"
                echo "    happens so the finding comes home, and this leg goes"
                echo "    red rather than reporting a clean column."
                FETCH_RED=1
                break ;;
            *)
                # AN UNANSWERED POLL IS NOT A DEAD RUN. The work is detached
                # from the ssh session on purpose, so a poll that fails is a
                # statement about the network and about nothing else. Say so
                # occasionally and keep going until the deadline.
                _unreach=$((_unreach + 1))
                if [ "$_unreach" = "1" ] || [ "$_unreach" = "10" ]; then
                    echo "    poll $_unreach: the box did not answer. The payload is"
                    echo "    DETACHED, so this says nothing about the run itself."
                    echo "    Retrying until the deadline."
                fi ;;
        esac
        sleep 30
    done
    leg_ssh 'cat /root/gemm_leg_console.log' > "$OUT/remote_console.log" 2>/dev/null \
        || echo "    could not read the remote console log"
    tail -5 "$OUT/remote_console.log" | sed 's/^/    /'
}

leg_fetch_e1dir() {
    # THE CARDS THE JUDGE READS. tools/e3_round_judge.sh wants one bootstrap
    # directory per machine under bench/results/e1/, so the box's lands
    # beside the Mac's with a name the judge's own label_of resolves to
    # $VLABEL. This is the phase8 payload's whole deliverable.
    _rd=$(sed -n 's/^e1dir=//p' "$OUT/remote/leg.txt" 2>/dev/null | tail -1)
    if [ -z "$_rd" ]; then
        # DEVIATION 864 -- ASK THE BOX BEFORE GIVING UP. leg.txt is written by
        # the remote body and arrives through the tar in leg_fetch, and either
        # of those can fail on its own while the bootstrap directory is
        # sitting on the pod, intact, one ssh away. This is the LAST MOMENT
        # that directory exists: the terminate is two steps below. Giving up
        # here threw away the entire deliverable of a paid leg because a small
        # text file did not come home.
        # The fallback is the one the remote body itself uses -- the newest
        # directory under bench/results/e1 -- and it is ANNOUNCED, because a
        # directory picked by mtime is a weaker claim than one the bootstrap
        # named in its own output.
        _rd=$(leg_ssh 'ls -dt /root/mojolearn/bench/results/e1/*/ 2>/dev/null | head -1' 2>/dev/null | tr -d "\r" | sed 's|/$||')
        if [ -n "$_rd" ]; then
            echo "  NO e1dir RECORDED in the box's leg.txt. Asked the box"
            echo "  directly and took the NEWEST directory under"
            echo "  bench/results/e1:"
            echo "    $_rd"
            echo "  That is an mtime guess and not the bootstrap's own answer."
            echo "  Read $E1_DEST/bootstrap.log to confirm it is this run's."
        else
            echo "  NO e1dir RECORDED, and the box has no directory under"
            echo "  bench/results/e1 either, so there is nothing to fetch for"
            echo "  the judge. Read $OUT/remote/bootstrap_console.log."
            return 1
        fi
    fi
    mkdir -p "$E1_DEST"
    # A PIPELINE'S STATUS IS ITS LAST COMMAND'S, so neither the remote tar
    # nor ssh can report a failure here. Check the RESULT instead.
    leg_ssh "cd '$_rd' && tar czf - ." | ( cd "$E1_DEST" && tar xzf - ) || true
    if [ -d "$E1_DEST/lanes" ] || [ -f "$E1_DEST/bootstrap.log" ]; then
        echo "  fetched $_rd"
        echo "       -> $E1_DEST"
    else
        echo "  FETCH FAILED for $_rd -- neither lanes/ nor bootstrap.log"
        echo "  arrived. It is still on the box, which is about to be"
        echo "  terminated, so read $OUT/remote/ for what did come home."
        return 1
    fi
    # COMMIT ATTRIBUTION ON A BOX WITH NO .git. e1_bootstrap.sh's provenance
    # step is `git rev-parse HEAD | tee commit.txt`, and this leg ships a
    # `git archive`, so on the pod that command writes an EMPTY commit.txt --
    # the same failure as the AMD leg whose commit.txt read "unknown" -- and
    # the judge's section 1 would read this column as MISSING. The sha is
    # written here from the sha this leg PINNED, and commit_provenance.txt
    # says so IN THE DIRECTORY, because a file that looks like git's output
    # and is not must say which it is. What it rests on is not a belief:
    # source_sha256 is compared at both ends in step 8.
    printf '%s\n' "$COMMIT" > "$E1_DEST/commit.txt"
    {
        echo "commit.txt in this directory was written by"
        echo "tools/gemm_remote_leg.sh --payload phase8, NOT by git on the box."
        echo
        echo "commit=$COMMIT_LINE"
        echo "vendor=$VENDOR"
        echo "remote_dir=$_rd"
        echo "pod=$POD_ID"
        echo
        echo "The box has no .git: the source arrives as 'git archive' at a"
        echo "pinned sha (see this leg's header for why a clone and a bundle"
        echo "were both rejected), so the bootstrap's own 'git rev-parse HEAD'"
        echo "wrote an empty file. The evidence that the box ran this commit"
        echo "is source_sha256.txt, computed with one recipe at both ends:"
        echo "  here:  $(cat "$OUT/source_sha256_local.txt" 2>/dev/null)"
        echo "  there: $(cat "$OUT/remote/source_sha256.txt" 2>/dev/null)"
    } > "$E1_DEST/commit_provenance.txt"
    return 0
}

leg_fetch() {
    mkdir -p "$OUT/remote"
    leg_ssh 'cd /root/gemm_leg_out && tar czf - .' | ( cd "$OUT/remote" && tar xzf - ) \
        || echo "  FETCH FAILED -- the remote log is /root/gemm_leg_out"
    if [ "$PAYLOAD" = "phase8" ]; then
        leg_fetch_e1dir || FETCH_RED=1
    fi
    # THE KEY LEAVES THE BOX BEFORE THE BOX DOES. Its only jobs -- letting an
    # operator self-terminate from the pod, and giving the ps check a pattern
    # file -- are both finished here. If the terminate below then fails, what
    # is left billing is a box with no credential on it rather than a box
    # with a live API key in a file.
    leg_ssh 'rm -f /root/.mojolearn-rp.key' > /dev/null 2>&1 \
        || echo "  could not remove the key file from the pod (it dies with the pod)"
    find "$OUT/remote" -mindepth 1 -maxdepth 1 | sed 's|.*/|    |'
}

# ---------------------------------------------------------------------------
# the dry run
# ---------------------------------------------------------------------------

R_PASS=0
R_FAIL=0
R_BLOCK=0
rok()  { R_PASS=$((R_PASS + 1)); printf '  ok    %s\n' "$1"; }
rbad() { R_FAIL=$((R_FAIL + 1)); printf '  FAIL  %s\n' "$1"; [ -n "${2:-}" ] && printf '        %s\n' "$2"; return 0; }
# TWO KINDS OF RED, KEPT APART ON PURPOSE. `rbad` means THIS SCRIPT is
# broken. `rblock` means the script is fine and the WORLD is not ready --
# a dirty tree, a live lease. Collapsing them into one number is how an
# operator learns to read "DRY RUN: RED" as "the tree is dirty again" and
# then walks past a real plumbing failure wearing the same word.
rblock() { R_BLOCK=$((R_BLOCK + 1)); printf '  BLOCK %s\n' "$1"; [ -n "${2:-}" ] && printf '        %s\n' "$2"; return 0; }

leg_rehearse() {
    echo
    echo "== dry run: every path except the paid ones =="
    echo "   A PATH THAT HAS NEVER RUN IS A PATH NOBODY HAS TESTED. This"
    echo "   repository has four separate scars from guard bugs that were"
    echo "   found only by running the guard, so these are executed on every"
    echo "   dry run rather than described in a comment."
    echo

    # -- A. argument validation ---------------------------------------------
    _out=$(MOJOLEARN_GEMM_LEG_REHEARSAL=1 "$0" walrus 2>&1) && _rc=0 || _rc=$?
    if [ "$_rc" = "2" ] && echo "$_out" | grep -q "exactly 'nvidia' or 'amd'"; then
        rok "A1 an unknown vendor is refused by name (exit 2)"
    else
        rbad "A1 an unknown vendor is refused by name" "got exit $_rc"
    fi

    _out=$(MOJOLEARN_GEMM_LEG_REHEARSAL=1 "$0" nvidia amd 2>&1) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "two vendors given"; then
        rok "A2 two vendors in one leg are refused"
    else
        rbad "A2 two vendors in one leg are refused" "got exit $_rc"
    fi

    _out=$(MOJOLEARN_GEMM_LEG_REHEARSAL=1 "$0" nvidia --rent --minutes 90 2>&1) && _rc=0 || _rc=$?
    if [ "$_rc" = "2" ] && echo "$_out" | grep -q "HARD CAP"; then
        rok "A3 --minutes 90 is refused before anything else is checked"
    else
        rbad "A3 --minutes 90 is refused" "got exit $_rc: $_out"
    fi

    _out=$( ( unset RUNPOD_API_KEY MOJOLEARN_RUNPOD_KEY_FILE; MOJOLEARN_GEMM_LEG_REHEARSAL=1 "$0" nvidia --rent 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "RUNPOD_API_KEY"; then
        rok "A4 --rent without a key refuses and rents nothing"
    else
        rbad "A4 --rent without a key refuses" "got exit $_rc: $_out"
    fi

    _out=$(MOJOLEARN_GEMM_LEG_REHEARSAL=1 "$0" nvidia --payload lanes 2>&1) && _rc=0 || _rc=$?
    if [ "$_rc" = "2" ] && echo "$_out" | grep -q "unknown payload"; then
        rok "A5 an unknown payload is refused by name (exit 2)"
    else
        rbad "A5 an unknown payload is refused by name" "got exit $_rc: $_out"
    fi

    _out=$(MOJOLEARN_GEMM_LEG_REHEARSAL=1 "$0" nvidia --payload phase8 --column-sweep 2>&1) && _rc=0 || _rc=$?
    if [ "$_rc" = "2" ] && echo "$_out" | grep -q "column-sweep belongs to the gemm payload"; then
        rok "A6 --column-sweep is refused under phase8 rather than ignored"
    else
        rbad "A6 --column-sweep is refused under phase8" "got exit $_rc: $_out"
    fi

    _out=$(MOJOLEARN_GEMM_LEG_REHEARSAL=1 "$0" nvidia --payload phase8 --local-card /tmp/nope.card 2>&1) && _rc=0 || _rc=$?
    if [ "$_rc" = "2" ] && echo "$_out" | grep -q "local-card belongs to the gemm payload"; then
        rok "A7 --local-card is refused under phase8 rather than ignored"
    else
        rbad "A7 --local-card is refused under phase8" "got exit $_rc: $_out"
    fi

    # -- B. the interlock ---------------------------------------------------
    # `trap - EXIT` first: leg_create_pod refuses by calling leg_die, and a
    # shell that runs EXIT traps inside ( ) would otherwise delete this dry
    # run's temp directory out from under the checks that follow.
    _out=$( ( trap - EXIT; MOJOLEARN_GEMM_LEG_REHEARSAL=1
              export MOJOLEARN_GEMM_LEG_REHEARSAL
              leg_create_pod 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "INTERLOCK"; then
        rok "B1 the paid call is fenced: a rehearsal child cannot create a pod"
    else
        rbad "B1 the paid call is fenced" "got exit $_rc: $_out"
    fi

    # -- C. the guard's refusal paths (the real guard, no pod) --------------
    _out=$( ( unset RUNPOD_API_KEY MOJOLEARN_RUNPOD_KEY_FILE; tools/runpod_guard.sh arm gemm-dryrun-pod "dry target" 60 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "REFUSING to arm"; then
        rok "C1 runpod_guard.sh arm REFUSES without a key"
    else
        rbad "C1 runpod_guard.sh arm REFUSES without a key" "got exit $_rc: $_out"
    fi

    _out=$(tools/runpod_guard.sh check gemm-dryrun-no-such-pod 2>&1) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "NO LEASE"; then
        rok "C2 runpod_guard.sh check calls an unknown pod UNGUARDED"
    else
        rbad "C2 runpod_guard.sh check on an unknown pod" "got exit $_rc: $_out"
    fi

    _out=$( ( unset RUNPOD_API_KEY MOJOLEARN_RUNPOD_KEY_FILE; tools/runpod_guard.sh reap --force gemm-dryrun-no-such-pod 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "reap needs RUNPOD_API_KEY"; then
        rok "C3 runpod_guard.sh reap refuses without a key"
    else
        rbad "C3 runpod_guard.sh reap refuses without a key" "got exit $_rc: $_out"
    fi

    # C4 IS THE CHECK THAT FOUND THE PIPELINE BUG. leg_arm used to indent the
    # guard's output with `| sed`, so its status was SED's and a refusal read
    # as a successful arm -- rule 1 defeated by an indent, invisible until
    # something ran it.
    # POD_ID and SSH_TARGET are set INSIDE the subshell on purpose: the
    # rehearsal must not leave a fake pod id where the teardown can see it.
    # shellcheck disable=SC2030
    _out=$( ( trap - EXIT; unset RUNPOD_API_KEY MOJOLEARN_RUNPOD_KEY_FILE
              POD_ID=gemm-dryrun-pod; SSH_TARGET="dry target"
              leg_arm 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "ARM REFUSED"; then
        rok "C4 leg_arm turns the guard's refusal into a refusal, not a pass"
    else
        rbad "C4 leg_arm propagates the guard's refusal" "IT DID NOT. Work would start on an unguarded box. got exit $_rc"
    fi

    # C5-C11 ARE THE GUARD'S TWO CLOSED DEFECTS, PROVED. Both lived on the
    # paid path, so both are exercised here against no pod and no API: a
    # fabricated lease directory (MOJOLEARN_LEASE_DIR), a `--dry-run` reap
    # that calls nothing, and stub `ssh` / `curl` that record their own argv
    # instead of connecting. Every one of them has an ACCEPT case beside the
    # refusal, because a guard that refuses everything passes a test that
    # only feeds it the bad case.
    _cfake="rpa_FAKE_KEY_FOR_REHEARSAL_0000"
    ( umask 077; printf '%s' "$_cfake" > "$TMPD/cfake.key" )

    _out=$( ( unset RUNPOD_API_KEY MOJOLEARN_RUNPOD_KEY_FILE; tools/runpod_guard.sh reap --force 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" = "2" ] && echo "$_out" | grep -q "would terminate EVERY"; then
        rok "C5 'reap --force' with no pod id is REFUSED (it used to reap every lease)"
    else
        rbad "C5 'reap --force' with no pod id is refused" "got exit $_rc: $_out"
    fi

    # THE ACCEPT CASE FOR C5: the same command WITH --all must get PAST the
    # form check and die at the key gate instead. Without this, a reap that
    # refused every invocation would pass C5.
    _out=$( ( unset RUNPOD_API_KEY MOJOLEARN_RUNPOD_KEY_FILE; tools/runpod_guard.sh reap --force --all 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" = "2" ] && echo "$_out" | grep -q "reap needs RUNPOD_API_KEY"; then
        rok "C5b 'reap --force --all' is ACCEPTED by the form check (it stops at the key)"
    else
        rbad "C5b the reap-everything form is still reachable by name" "got exit $_rc: $_out"
    fi

    _out=$( ( unset RUNPOD_API_KEY MOJOLEARN_RUNPOD_KEY_FILE; tools/runpod_guard.sh reap --wat 2>&1 ) ) && _rc=0 || _rc=$?
    _out2=$( ( unset RUNPOD_API_KEY MOJOLEARN_RUNPOD_KEY_FILE; tools/runpod_guard.sh reap podA podB 2>&1 ) ) && _rc2=0 || _rc2=$?
    if [ "$_rc" = "2" ] && echo "$_out" | grep -q "unknown option" \
       && [ "$_rc2" = "2" ] && echo "$_out2" | grep -q "two pod ids"; then
        rok "C6 reap refuses an unknown option and two pod ids, both by name"
    else
        rbad "C6 reap's argument validation" "unknown-option: exit $_rc; two-ids: exit $_rc2"
    fi

    # The fabricated lease directory. `gemm-dry-zbroken` sorts LAST on
    # purpose: the inheritance bug it tests can only show when a READABLE
    # lease was sourced before it.
    _ld="$TMPD/leases"
    rm -rf "$_ld"; mkdir -p "$_ld"
    _nowe=$(date -u +%s)
    printf "pod='gemm-dry-expired'\ntarget='-p 1 root@nowhere'\narmed_utc='x'\ndeadline_epoch=%s\nminutes=60\n" \
        "$((_nowe - 600))" > "$_ld/gemm-dry-expired.lease"
    printf "pod='gemm-dry-live'\ntarget='-p 2 root@nowhere'\narmed_utc='x'\ndeadline_epoch=%s\nminutes=60\n" \
        "$((_nowe + 3600))" > "$_ld/gemm-dry-live.lease"
    printf "# truncated lease: it names no pod and no deadline\nminutes=60\n" \
        > "$_ld/gemm-dry-zbroken.lease"

    _out=$(MOJOLEARN_LEASE_DIR="$_ld" tools/runpod_guard.sh reap --force gemm-dry-live --dry-run 2>&1) && _rc=0 || _rc=$?
    if echo "$_out" | grep -q "WOULD TERMINATE gemm-dry-live" \
       && ! echo "$_out" | grep -q "gemm-dry-expired"; then
        rok "C7 'reap --force <pod-id>' terminates THAT POD AND NOTHING ELSE"
    else
        rbad "C7 the pod id scopes the reap" "IT DID NOT. Other lanes' boxes are in range of this command. got: $_out"
    fi

    _out=$(MOJOLEARN_LEASE_DIR="$_ld" tools/runpod_guard.sh reap --force --all --dry-run 2>&1) && _rc=0 || _rc=$?
    if echo "$_out" | grep -q "WOULD TERMINATE gemm-dry-live" \
       && echo "$_out" | grep -q "WOULD TERMINATE gemm-dry-expired"; then
        rok "C7b '--force --all' still selects every lease (the scoping is not just always-one)"
    else
        rbad "C7b --force --all selects every lease" "got: $_out"
    fi

    _out=$(MOJOLEARN_LEASE_DIR="$_ld" tools/runpod_guard.sh reap --dry-run 2>&1) && _rc=0 || _rc=$?
    if echo "$_out" | grep -q "WOULD TERMINATE gemm-dry-expired" \
       && echo "$_out" | grep -q "leaving gemm-dry-live alone"; then
        rok "C8 a plain reap takes the EXPIRED lease and leaves the live one alone"
    else
        rbad "C8 a plain reap is expiry-scoped" "got: $_out"
    fi

    _n=$(MOJOLEARN_LEASE_DIR="$_ld" tools/runpod_guard.sh reap --force --all --dry-run 2>&1 \
         | grep -c 'WOULD TERMINATE gemm-dry-live' || true)
    _out=$(MOJOLEARN_LEASE_DIR="$_ld" tools/runpod_guard.sh reap --force --all --dry-run 2>&1) && _rc=0 || _rc=$?
    if [ "${_n:-0}" = "1" ] && echo "$_out" | grep -q "SKIPPING unreadable lease"; then
        rok "C9 a truncated lease file is SKIPPED and does not inherit the previous pod"
    else
        rbad "C9 a truncated lease cannot make the reap terminate the wrong pod" "gemm-dry-live was selected ${_n:-0} time(s) and it must be exactly 1. The lease file is '.'-sourced into the loop's own shell, so an unreadable one inherits the previous iteration's pod and this loop DELETES what \$pod names."
    fi

    # C10: THE KEY DOES NOT REACH THE POD IN AN ARGV. A stub `ssh` records
    # its own argv and its stdin and answers what the guard greps for. The
    # guard is invoked with stdin on /dev/null so the stub's second `cat`
    # cannot block; the credential push makes its own pipe.
    _sd="$TMPD/stub"; rm -rf "$_sd"; mkdir -p "$_sd"
    _rec="$TMPD/rec"; rm -rf "$_rec"; mkdir -p "$_rec"
    {
        echo '#!/bin/sh'
        echo '# stub ssh for the dry run: records argv and stdin, connects to nothing.'
        echo "printf '%s\n' \"\$@\" >> $_rec/argv.txt"
        echo "cat >> $_rec/stdin.bin"
        echo 'echo CREDENTIAL_ON_POD'
        echo 'echo "WATCHDOG_ARMED pid=4242 secs=3600"'
    } > "$_sd/ssh"
    chmod 700 "$_sd/ssh"
    _ld2="$TMPD/leases2"; rm -rf "$_ld2"; mkdir -p "$_ld2"
    ( trap - EXIT
      PATH="$_sd:$PATH"; export PATH
      RUNPOD_API_KEY="$_cfake"; export RUNPOD_API_KEY
      MOJOLEARN_LEASE_DIR="$_ld2"; export MOJOLEARN_LEASE_DIR
      tools/runpod_guard.sh arm gemm-dry-armpod "-p 9 root@nowhere" 60
    ) > "$TMPD/c10.out" 2>&1 < /dev/null || true
    if [ ! -s "$_rec/argv.txt" ]; then
        rbad "C10 the key never enters an ssh argv" "THE STUB RECORDED NOTHING. The check proved nothing: $(head -3 "$TMPD/c10.out")"
    elif ! grep -q 'gemm-dry-armpod' "$_rec/argv.txt"; then
        rbad "C10 the key never enters an ssh argv" "the recorded argv does not even name the pod, so it is not the arm's argv"
    elif ! grep -q -F -f "$TMPD/cfake.key" "$_rec/stdin.bin" 2>/dev/null; then
        rbad "C10 the key reaches the pod on STDIN" "IT DID NOT. Either the credential push did not run or it sent something else."
    elif grep -q -F -f "$TMPD/cfake.key" "$_rec/argv.txt"; then
        rbad "C10 the key never enters an ssh argv" "THE KEY IS IN THE ARM'S ARGV. Anything running ps on the pod can read it, and so can anything on this Mac."
    else
        rok "C10 arm: the key travels on ssh STDIN and appears in NO argv, on either machine"
    fi

    # C11: the same property for the guard's own DELETE. A stub `curl`
    # records its argv and copies whatever `-K` pointed at, so the test can
    # see both halves: the config carries the key, the command line does not.
    {
        echo '#!/bin/sh'
        echo '# stub curl for the dry run: records argv and the -K config, calls nothing.'
        echo "printf '%s\n' \"\$@\" >> $_rec/curl_argv.txt"
        echo 'prev=""'
        echo 'for a in "$@"; do'
        echo "  if [ \"\$prev\" = \"-K\" ]; then cat \"\$a\" >> $_rec/curl_config.txt; fi"
        echo '  prev="$a"'
        echo 'done'
        echo 'exit 0'
    } > "$_sd/curl"
    chmod 700 "$_sd/curl"
    _ld3="$TMPD/leases3"; rm -rf "$_ld3"; mkdir -p "$_ld3"
    printf "pod='gemm-dry-reappod'\ntarget='-p 3 root@nowhere'\narmed_utc='x'\ndeadline_epoch=%s\nminutes=60\n" \
        "$((_nowe - 60))" > "$_ld3/gemm-dry-reappod.lease"
    ( trap - EXIT
      PATH="$_sd:$PATH"; export PATH
      RUNPOD_API_KEY="$_cfake"; export RUNPOD_API_KEY
      MOJOLEARN_LEASE_DIR="$_ld3"; export MOJOLEARN_LEASE_DIR
      tools/runpod_guard.sh reap --force gemm-dry-reappod
    ) > "$TMPD/c11.out" 2>&1 < /dev/null || true
    if [ ! -s "$_rec/curl_argv.txt" ]; then
        rbad "C11 the key never enters curl's argv" "THE STUB RECORDED NOTHING, so the check proved nothing: $(head -3 "$TMPD/c11.out")"
    elif ! grep -q -F -f "$TMPD/cfake.key" "$_rec/curl_config.txt" 2>/dev/null; then
        rbad "C11 the reap's curl config carries the key" "IT DOES NOT, so -K would send no auth and every terminate would 401"
    elif grep -q -F -f "$TMPD/cfake.key" "$_rec/curl_argv.txt"; then
        rbad "C11 the key never enters curl's argv" "THE KEY IS IN curl's COMMAND LINE on this machine"
    else
        rok "C11 reap: the key is in a 0600 curl config and in NO argv"
    fi

    # -- D and E belong to the gemm payload: leg_require_identical and
    # leg_diff_cards are on ITS path and not on phase8's, where the mode
    # read-back is per lane (group P) and the judge does the diffing, and
    # not on speed's, which has no card and asserts the OPPOSITE mode --
    # `leg_require_identical` would refuse every log a speed leg produces,
    # correctly, because a FAST banner is exactly what that payload wants.
    if [ "$PAYLOAD" = "speed" ]; then
        rok "D/E skipped: the speed payload has no card and requires FAST, not IDENTICAL"
    fi
    if [ "$PAYLOAD" != "phase8" ] && [ "$PAYLOAD" != "speed" ]; then

    # -- D. the contamination guard -----------------------------------------
    printf '== bench/gemm_card_main.mojo [FAST] ==\nstages: 60 over 20 shapes; 0 skipped\n' > "$TMPD/fast.log"
    if leg_require_identical "$TMPD/fast.log" "rehearsal" > "$TMPD/d1.out" 2>&1; then
        rbad "D1 a [FAST] banner is rejected" "it was ACCEPTED, which means every card this leg produces could be the wrong arithmetic"
    else
        if grep -q CONTAMINATED "$TMPD/d1.out"; then
            rok "D1 a [FAST] banner is rejected as CONTAMINATED"
        else
            rbad "D1 a [FAST] banner is rejected" "rejected, but not with the contamination message"
        fi
    fi

    printf 'warning: something\nstages: 60\n' > "$TMPD/nobanner.log"
    if leg_require_identical "$TMPD/nobanner.log" "rehearsal" > "$TMPD/d2.out" 2>&1; then
        rbad "D2 a log with no banner is rejected" "it was ACCEPTED"
    else
        if grep -q '<no banner>' "$TMPD/d2.out"; then
            rok "D2 a log with no banner is rejected and says so"
        else
            rbad "D2 a log with no banner" "rejected without naming the missing banner"
        fi
    fi

    printf '== bench/gemm_card_main.mojo [IDENTICAL] ==\n' > "$TMPD/ident.log"
    if leg_require_identical "$TMPD/ident.log" "rehearsal" > /dev/null 2>&1; then
        rok "D3 an [IDENTICAL] banner is accepted (the guard is not just always-red)"
    else
        rbad "D3 an [IDENTICAL] banner is accepted" "it was rejected"
    fi

    if leg_require_identical "$TMPD/does-not-exist.log" "rehearsal" > "$TMPD/d4.out" 2>&1; then
        rbad "D4 a missing log is rejected" "it was ACCEPTED"
    else
        rok "D4 a missing log is rejected"
    fi

    if leg_require_file "$TMPD/no-such.card" "rehearsal" > /dev/null 2>&1; then
        rbad "D5 a missing card is rejected" "it was ACCEPTED"
    else
        rok "D5 a missing card is rejected"
    fi

    # -- E. the differ plumbing, both outcomes -------------------------------
    if [ -s "$LOCAL_CARD" ]; then
        cp "$LOCAL_CARD" "$TMPD/synth_same.card"
        if leg_diff_cards "$LOCAL_CARD" "$TMPD/synth_same.card" "$TMPD/diff_same.txt" > "$TMPD/e1.out" 2>&1; then
            rok "E1 a synthetic remote card equal to the local one reports NO DIVERGENCE"
        else
            rbad "E1 equal cards report no divergence" "$(tail -3 "$TMPD/e1.out")"
        fi

        _tag=$(awk -F'\t' '!/^#/ && NF==5 { c++; if (c==3) { print $2; exit } }' "$LOCAL_CARD")
        awk -F'\t' -v OFS='\t' '
            !/^#/ && NF==5 { c++; if (c==3) { $5 = ($5 ~ /^0/ ? "1" : "0") substr($5,2) } }
            { print }' "$LOCAL_CARD" > "$TMPD/synth_diff.card"
        if leg_diff_cards "$LOCAL_CARD" "$TMPD/synth_diff.card" "$TMPD/diff_planted.txt" > "$TMPD/e2.out" 2>&1; then
            rbad "E2 a planted divergence is caught" "THE DIFF CAME BACK GREEN. The differ plumbing is inert and no red this leg can produce means anything."
        else
            if grep -q "FIRST DIVERGING STAGE: $_tag" "$TMPD/e2.out"; then
                rok "E2 a planted divergence at '$_tag' is caught and NAMED"
            else
                rbad "E2 a planted divergence is named" "caught, but the reported stage was not $_tag"
            fi
        fi
        # E3: A CARD THE DIFFER CANNOT READ IS NOT A DIVERGENCE. Exit 2 is
        # a parse or dump-integrity failure and it VOIDS the report. The
        # first version of leg_diff_cards read every non-zero status as
        # "they diverge", which would have filed an unreadable card as a
        # cross-vendor finding.
        awk -F'\t' -v OFS='\t' '
            !/^#/ && NF==5 { c++; if (c==2) { $5 = "not-a-hash" } }
            { print }' "$LOCAL_CARD" > "$TMPD/synth_unreadable.card"
        leg_diff_cards "$LOCAL_CARD" "$TMPD/synth_unreadable.card" \
            "$TMPD/diff_unreadable.txt" > "$TMPD/e3.out" 2>&1 && _rc=0 || _rc=$?
        if [ "$_rc" = "2" ] && grep -q "COULD NOT READ" "$TMPD/e3.out"; then
            rok "E3 a card the differ cannot parse is called UNREADABLE, not divergent"
        else
            rbad "E3 an unparseable card is not reported as a divergence" "got exit $_rc: $(head -3 "$TMPD/e3.out")"
        fi

        # E4: a side that emitted a different number of launch-geometry
        # blocks is a bigger finding than any hash, and it must not be
        # reported as a stage divergence.
        { cat "$LOCAL_CARD"; grep -v '^#' "$LOCAL_CARD"; } > "$TMPD/synth_blocks.card"
        leg_diff_cards "$LOCAL_CARD" "$TMPD/synth_blocks.card" \
            "$TMPD/diff_blocks.txt" > "$TMPD/e4.out" 2>&1 && _rc=0 || _rc=$?
        if [ "$_rc" != "0" ] && grep -q "BLOCK COUNT MISMATCH" "$TMPD/e4.out"; then
            rok "E4 a block-count mismatch is named, not read as a divergence"
        else
            rbad "E4 a block-count mismatch is named" "got exit $_rc: $(head -3 "$TMPD/e4.out")"
        fi
    else
        rbad "E1/E2/E3/E4 the differ plumbing" "no local card to rehearse against"
    fi

    fi

    # -- F. the remote body --------------------------------------------------
    COMMIT="${COMMIT:-0000000}"
    if leg_build_remote_body > "$TMPD/f1.out" 2>&1; then
        rok "F1 the remote body substitutes cleanly and passes a POSIX sh check"
    else
        rbad "F1 the remote body is built and syntax-checked" "$(cat "$TMPD/f1.out")"
    fi
    # SABOTAGE: prove the placeholder check is not inert. A guard that has
    # never fired is a guard nobody has any reason to believe.
    printf '\n# @LEFTOVER@\n' >> "$OUT/remote_body.sh"
    if grep -q '@[A-Z][A-Z_]*@' "$OUT/remote_body.sh"; then
        rok "F2 the leftover-placeholder detector sees a planted @LEFTOVER@"
    else
        rbad "F2 the leftover-placeholder detector" "it did not see a planted placeholder"
    fi
    sed '$d' "$OUT/remote_body.sh" > "$TMPD/rb" && mv "$TMPD/rb" "$OUT/remote_body.sh"

    # F3: the poll loop's sentinel, proved against a body that stopped
    # writing it. Without this the leg would wait out its outer deadline and
    # then call a complete run partial -- a false finding about the box,
    # printed with the same confidence as a true one.
    cp "$OUT/remote_body.sh" "$TMPD/body.keep3"
    grep -v 'gemm_leg.done' "$TMPD/body.keep3" > "$OUT/remote_body.sh"
    if ( trap - EXIT; leg_check_remote_body ) > "$TMPD/f3.out" 2>&1; then
        rbad "F3 a body that writes no completion sentinel is refused" "IT WAS ACCEPTED. Every leg would poll to its outer deadline and then report a finished run as cut off."
    elif grep -q 'never writes /root/gemm_leg.done' "$TMPD/f3.out"; then
        rok "F3 a body that stopped writing the poll sentinel is refused, by name"
    else
        rbad "F3 a body with no sentinel is refused BY NAME" "it was refused for another reason: $(head -2 "$TMPD/f3.out")"
    fi
    cp "$TMPD/body.keep3" "$OUT/remote_body.sh"

    # -- M. the dead-man (DEVIATION 862) -------------------------------------
    # It is armed BEFORE the create call and it is the ONLY guard covering the
    # window before the on-pod watchdog exists, so it gets what the guard
    # itself gets: composed for real, syntax-checked, proved to keep the key
    # out of every argv, and then RUN against stub `curl` and stub `sleep` so
    # the DELETE it would issue is RECORDED rather than assumed. A guard that
    # has never fired is a guard nobody has any reason to believe.
    _dm="$TMPD/deadman"; rm -rf "$_dm"
    if ( trap - EXIT; RUNPOD_API_KEY="$_cfake"
         leg_write_deadman "$_dm" 0 mojolearn-gemm-dry-podname ) > "$TMPD/m1.out" 2>&1; then
        rok "M1 the dead-man composes, substitutes cleanly and passes sh -n"
    else
        rbad "M1 the dead-man is built and syntax-checked" "$(cat "$TMPD/m1.out")"
    fi

    if [ ! -f "$_dm/deadman.sh" ]; then
        rbad "M2 the dead-man's key is in a 0600 config and in no script" "no dead-man script was composed at all"
    elif ! grep -q -F -f "$TMPD/cfake.key" "$_dm/curlrc" 2>/dev/null; then
        rbad "M2 the dead-man's curl config carries the key" "IT DOES NOT, so every DELETE it issues would 401. It would fire and fail, which is the worst kind of guard."
    elif grep -q -F -f "$TMPD/cfake.key" "$_dm/deadman.sh"; then
        rbad "M2 the key is NOT in the dead-man's script" "THE KEY IS IN THE SCRIPT ITSELF -- a file that outlives this process by design, sitting next to the pod name it is for."
    else
        rok "M2 the dead-man's key is in a 0600 curl config, and in no script and no argv"
    fi

    if [ -f "$_dm/deadman.sh" ]; then
        printf '# @LEFTOVER@\n' >> "$_dm/deadman.sh"
        if grep -q '@[A-Z][A-Z0-9_]*@' "$_dm/deadman.sh"; then
            rok "M3 the dead-man's leftover-placeholder detector sees a planted @LEFTOVER@"
        else
            rbad "M3 the dead-man's placeholder detector is not inert" "it did not see a planted placeholder. A missed @SECS@ would leave the guard sleeping on a literal string, which is to say not guarding."
        fi
        sed '$d' "$_dm/deadman.sh" > "$TMPD/dmrb" && mv "$TMPD/dmrb" "$_dm/deadman.sh"
        chmod 700 "$_dm/deadman.sh"
    fi

    # M4: RUN IT. `sleep` and `curl` are stubs, so it waits for nothing and
    # calls nothing, but every other line is the real one -- the id file it
    # reads, the two DELETE urls it walks, and the credential it removes on
    # the way out.
    {
        echo '#!/bin/sh'
        echo '# stub sleep for the dry run: the dead-man must not actually wait.'
        echo 'exit 0'
    } > "$_sd/sleep"
    chmod 700 "$_sd/sleep"
    : > "$_rec/curl_argv.txt"; : > "$_rec/curl_config.txt"
    printf '%s\n' "gemm-dry-deadpod" > "$_dm/pod_id.txt"
    ( trap - EXIT
      PATH="$_sd:$PATH"; export PATH
      sh "$_dm/deadman.sh" ) > "$TMPD/m4.out" 2>&1 || true
    if ! grep -q 'gemm-dry-deadpod' "$_rec/curl_argv.txt" 2>/dev/null; then
        rbad "M4 the dead-man DELETEs the pod it was armed for" "IT CALLED NOTHING NAMING THAT POD. An armed dead-man that issues no DELETE is a lease file with nothing behind it: $(head -3 "$TMPD/m4.out")"
    elif ! grep -qx 'DELETE' "$_rec/curl_argv.txt"; then
        rbad "M4 the dead-man issues a DELETE" "it called curl for that pod, but not with -X DELETE"
    elif grep -q -F -f "$TMPD/cfake.key" "$_rec/curl_argv.txt"; then
        rbad "M4 the key never enters the dead-man's curl argv" "THE KEY IS IN A COMMAND LINE that, unlike this leg, lives for the whole lease."
    elif ! grep -q -F -f "$TMPD/cfake.key" "$_rec/curl_config.txt" 2>/dev/null; then
        rbad "M4 the dead-man's -K config carries the key at the moment it fires" "it does not, so the DELETE would 401"
    elif [ -f "$_dm/curlrc" ]; then
        rbad "M4 the dead-man removes its credential when it fires" "the 0600 curl config is still on disk after the only job it existed for"
    else
        rok "M4 the dead-man fires: DELETEs its pod by id on both endpoints, keeps the key out of every argv, and removes the credential afterwards"
    fi
    rm -f "$_sd/sleep"

    # M5: THE BY-NAME PATH, WHICH M4 DOES NOT REACH. M4 plants an id file, so
    # it walks the DELETE straight away; the branch that matters most is the
    # other one -- no id, because the create response was unreadable. That is
    # the leg-10 shape exactly (tools/e2_remote_leg.sh: a droplet created, a
    # body nobody could parse, an orphan found by hand nineteen minutes
    # later). Its parser is a separate file for this reason, so it can be fed
    # a planted listing here without an API and without a pod.
    printf '%s' '{"items":[{"id":"other-lane-pod","name":"mojolearn-e2-nv"},{"id":"gemm-dry-bynamepod","name":"mojolearn-gemm-dry-podname"}]}' \
        > "$TMPD/m5_pods.json"
    _m5=$("$PY" "$_dm/byname.py" "$TMPD/m5_pods.json" mojolearn-gemm-dry-podname 2>&1)
    if [ "$_m5" = "gemm-dry-bynamepod" ]; then
        rok "M5 the dead-man's by-name lookup finds ITS pod and no other lane's"
    else
        rbad "M5 the by-name lookup selects this leg's pod only" "it returned '$_m5' and it must be exactly 'gemm-dry-bynamepod'. This is the branch that runs when the create response was unparseable, which is the one case an id-keyed guard cannot cover -- and a lookup that over-matches would terminate another lane's box."
    fi

    # -- P. the phase8 payload -----------------------------------------------
    if [ "$PAYLOAD" = "phase8" ]; then
        # P1 IS THE ONE THAT KEEPS THE REST HONEST: every check below reads
        # $LANES, and a lane list that quietly fell back to a literal would
        # make all of them pass while the newest lane went unchecked.
        _raw=$(sed -n 's/^for lane in \(.*\); do$/\1/p' tools/e3_round_judge.sh | head -1)
        if [ -n "$_raw" ] && echo " $_raw " | grep -q ' mamba '; then
            rok "P1 the lane list is READ OUT OF tools/e3_round_judge.sh ($_raw)"
        else
            rbad "P1 the lane list comes from the judge, not a copy" "the extraction returned '$_raw', so the leg would use a literal list that goes stale the next time a lane is added"
        fi

        # P2: the fetched directory's NAME is what the judge labels the
        # column from, and this asks the judge's OWN label_of rather than a
        # copy of its patterns.
        _lo=$(sed -n '/^label_of() {/,/^}/p' tools/e3_round_judge.sh)
        if [ -n "$_lo" ]; then
            _got=$( eval "$_lo"; label_of "$E1_DEST" )
            if [ "$_got" = "$VLABEL" ]; then
                rok "P2 the judge's own label_of reads $(basename "$E1_DEST") as $_got"
            else
                rbad "P2 the fetched directory is labelled $VLABEL by the judge" "label_of said '$_got'. The column would be judged under the wrong name."
            fi
        else
            rbad "P2 label_of could not be read out of tools/e3_round_judge.sh" "the naming of $E1_DEST is then unverified"
        fi

        _p8="$TMPD/p8"
        rm -rf "$_p8"; mkdir -p "$_p8/lanes"
        printf '# format: mojolearn-identity-trace v1\n0\tp8.in.a\tf32\t4\t0123456789abcdef\n' \
            > "$_p8/lanes/gemm.identical.card"

        printf '== bench/gemm_card_main.mojo [IDENTICAL] ==\n' > "$_p8/lanes/gemm.identical.log"
        if ( trap - EXIT; E1_DEST="$_p8"; LANES="gemm"; leg_phase8_artifacts ) > "$TMPD/p3.out" 2>&1; then
            if grep -q 'mode read back from the run = \[IDENTICAL\]' "$TMPD/p3.out"; then
                rok "P3 a lane whose run reports [IDENTICAL] is accepted (the guard is not just always-red)"
            else
                rbad "P3 an [IDENTICAL] lane is accepted and says so" "$(head -3 "$TMPD/p3.out")"
            fi
        else
            rbad "P3 an [IDENTICAL] lane is accepted" "it was rejected: $(head -3 "$TMPD/p3.out")"
        fi

        printf '== bench/gemm_card_main.mojo [FAST] ==\n' > "$_p8/lanes/gemm.identical.log"
        if ( trap - EXIT; E1_DEST="$_p8"; LANES="gemm"; leg_phase8_artifacts ) > "$TMPD/p4.out" 2>&1; then
            rbad "P4 an IDENTICAL card whose run reports [FAST] is caught" "IT WAS ACCEPTED. Every card this payload brings home could be the wrong arithmetic."
        else
            if grep -q CONTAMINATED "$TMPD/p4.out"; then
                rok "P4 an IDENTICAL card whose run reports [FAST] is CONTAMINATED"
            else
                rbad "P4 a [FAST] lane is rejected" "rejected, but not with the contamination message"
            fi
        fi

        printf 'warning: something, and no mode line anywhere\n' > "$_p8/lanes/gemm.identical.log"
        if ( trap - EXIT; E1_DEST="$_p8"; LANES="gemm"; leg_phase8_artifacts ) > "$TMPD/p5.out" 2>&1; then
            if grep -q UNWITNESSED "$TMPD/p5.out"; then
                rok "P5 a lane with no banner is called UNWITNESSED, not passed and not failed (linkage and mamba are both like this)"
            else
                rbad "P5 an unwitnessed lane is named" "accepted without saying the mode was never witnessed"
            fi
        else
            rbad "P5 an unwitnessed lane is reported, not failed" "it was treated as red; linkage and mamba would make every leg red"
        fi

        if ( trap - EXIT; E1_DEST="$_p8"; LANES="cd"; leg_phase8_artifacts ) > "$TMPD/p6.out" 2>&1; then
            rbad "P6 a lane with no card at all is caught" "IT WAS ACCEPTED"
        else
            if grep -q 'NO CARD' "$TMPD/p6.out"; then
                rok "P6 a lane with no IDENTICAL card is named, and the judge's MISSING is predicted"
            else
                rbad "P6 a missing lane card is named" "rejected for the wrong reason"
            fi
        fi

        if ( trap - EXIT; E1_DEST="$TMPD/no-such-column"; LANES="gemm"; leg_phase8_artifacts ) > "$TMPD/p7.out" 2>&1; then
            rbad "P7 a column with no lanes/ directory is caught" "IT WAS ACCEPTED"
        else
            rok "P7 a column with no lanes/ directory at all is caught"
        fi

        # P8: the two content checks in leg_check_remote_body, proven against
        # a planted body. A body that had stopped running e1_bootstrap.sh, or
        # had stopped unsetting the mamba shape, must not ship.
        cp "$OUT/remote_body.sh" "$TMPD/body.keep"
        grep -v 'bash tools/e1_bootstrap.sh' "$TMPD/body.keep" > "$OUT/remote_body.sh"
        if ( trap - EXIT; leg_check_remote_body ) > "$TMPD/p8.out" 2>&1; then
            rbad "P8 a body that does not run e1_bootstrap.sh is refused" "IT WAS ACCEPTED. The leg could ship a copy of phase 8 and the judge would never know."
        elif grep -q 'does not run tools/e1_bootstrap.sh' "$TMPD/p8.out"; then
            rok "P8 a body that stopped running tools/e1_bootstrap.sh is refused, by name"
        else
            rbad "P8 a body that stopped running e1_bootstrap.sh is refused BY NAME" "it was refused for another reason: $(head -2 "$TMPD/p8.out")"
        fi
        grep -v 'unset MOJOLEARN_MAMBA_CHECK_B' "$TMPD/body.keep" > "$OUT/remote_body.sh"
        if ( trap - EXIT; leg_check_remote_body ) > "$TMPD/p9.out" 2>&1; then
            rbad "P9 a body that does not pin the mamba shape is refused" "IT WAS ACCEPTED. A widened mamba shape is another compile inside the lease."
        elif grep -q 'does not unset the mamba shape' "$TMPD/p9.out"; then
            rok "P9 a body that stopped unsetting the mamba shape is refused, by name"
        else
            rbad "P9 a body that stopped unsetting the mamba shape is refused BY NAME" "it was refused for another reason: $(head -2 "$TMPD/p9.out")"
        fi
        cp "$TMPD/body.keep" "$OUT/remote_body.sh"
    fi

    # -- K. key hygiene ------------------------------------------------------
    # A LOW-ENTROPY FAKE, never a real key: this string is written to disk and
    # quoted in output.
    _fake="rpa_FAKE_KEY_FOR_REHEARSAL_0000"
    ( umask 077; printf '%s' "$_fake" > "$TMPD/fake.key" )
    if leg_assert_keyfile_hygiene "$TMPD/fake.key" > /dev/null 2>&1; then
        rok "K1 a 0600 key file outside the repo passes the hygiene check"
    else
        rbad "K1 key-file hygiene accepts a good file" "it was rejected"
    fi

    chmod 644 "$TMPD/fake.key"
    if ( trap - EXIT; leg_assert_keyfile_hygiene "$TMPD/fake.key" ) > "$TMPD/k2.out" 2>&1; then
        rbad "K2 a world-readable key file is refused" "mode 644 was ACCEPTED"
    else
        if grep -q "must be 600" "$TMPD/k2.out"; then
            rok "K2 a world-readable key file is refused by mode"
        else
            rbad "K2 a world-readable key file is refused" "refused for the wrong reason"
        fi
    fi
    chmod 600 "$TMPD/fake.key"

    # THE PROBE HAS TO BE INSIDE THE CHECKOUT or the check it is testing has
    # nothing to fire on. The first version put it under $OUT, which the
    # operator can point anywhere -- with MOJOLEARN_GEMM_LEG_OUT set outside
    # the repo the probe passed the hygiene check and K3 reported a failure
    # that was really the probe's. Under bench/results/ rather than the repo
    # root so a stray `git add` of a source path cannot pick it up, and
    # removed immediately either way.
    _probe="$REPO/bench/results/.gemm_leg_keyprobe_$$.key"
    mkdir -p "$REPO/bench/results"
    ( umask 077; printf '%s' "$_fake" > "$_probe" )
    if ( trap - EXIT; leg_assert_keyfile_hygiene "$_probe" ) > "$TMPD/k3.out" 2>&1; then
        rbad "K3 a key file inside the checkout is refused" "it was ACCEPTED"
    else
        if grep -q "INSIDE this repository" "$TMPD/k3.out"; then
            rok "K3 a key file inside the checkout is refused"
        else
            rbad "K3 a key file inside the checkout is refused" "refused for the wrong reason"
        fi
    fi
    rm -f "$_probe"

    # K4: the curl config carries the key and the curl ARGV does not.
    _saved_rc="$CURLRC"
    CURLRC="$TMPD/rehearsal.curlrc"
    ( trap - EXIT; RUNPOD_API_KEY="$_fake"; leg_curlrc )
    if grep -q -F -f "$TMPD/fake.key" "$CURLRC"; then
        _argv="curl -K $CURLRC -o $TMPD/rp.body -w %{http_code} -X GET $RP_HOST$RP_PODS_PATH"
        if printf '%s' "$_argv" | grep -q -F -f "$TMPD/fake.key"; then
            rbad "K4 the key is in the config file and NOT in the argv" "the key appears in the command line"
        else
            rok "K4 the key is in the 0600 config file and not in curl's argv"
        fi
    else
        rbad "K4 the config file carries the key" "the key is not in the config file, so -K would send no auth"
    fi
    CURLRC="$_saved_rc"

    # K5: the ps detector, and a SABOTAGE proving it is not inert. The
    # negative case (nothing on this machine has the key in an argv) cannot
    # distinguish "clean" from "broken grep", so a line containing the fake
    # key is appended to a copy of the dump and the detector must find it.
    # WHAT THIS PROVES: that `grep -F -f keyfile dump` finds a key in a ps
    # dump. WHAT IT DOES NOT PROVE: that ps on the POD renders argv the way
    # this Mac's does. That is checked for real by leg_check_key_not_in_ps
    # on the first paid run.
    ps -eo args= > "$TMPD/ps.txt" 2>/dev/null || ps ax > "$TMPD/ps.txt"
    if grep -q -F -f "$TMPD/fake.key" "$TMPD/ps.txt"; then
        rbad "K5 no process on this machine has the key in its argv" "the fake key is in a live command line"
    else
        cp "$TMPD/ps.txt" "$TMPD/ps_sab.txt"
        printf 'sh -c something --token %s\n' "$_fake" >> "$TMPD/ps_sab.txt"
        if grep -q -F -f "$TMPD/fake.key" "$TMPD/ps_sab.txt"; then
            rok "K5 the ps detector finds a planted key and finds none in the real dump"
        else
            rbad "K5 the ps detector is not inert" "IT DID NOT FIND A PLANTED KEY. Every 'KEY_NOT_IN_PS' this leg prints would be meaningless."
        fi
    fi

    # -- L. the reference leg ------------------------------------------------
    if [ "$PAYLOAD" = "phase8" ]; then
        if [ "$APPLE_MISSING" = "1" ]; then
            rblock "L1 an Apple column exists at this commit" "THERE IS NONE. The box would produce seven cards with nothing to compare them to, and the judge's section 1 refuses a round whose columns record different commits. Make it here first: bash tools/e1_bootstrap.sh -- or name one with --apple-dir."
        else
            rok "L1 the Apple column exists at this commit ($APPLE_DIR)"
        fi
    elif [ "$PAYLOAD" = "speed" ]; then
        # NOT `rok`, and the distinction matters. There is no Apple
        # reference here at all and there is not supposed to be one: a
        # millisecond from an M4 cannot enter a table of H100 milliseconds.
        # Printing "ok, the reference is a real measurement" would be a
        # green check on a thing that does not exist.
        rok "L1 no Apple reference is required: a speed leg compares two libraries on ONE box"
    else
        rok "L1 the Apple reference card is a real measurement"
    fi

    # -- G. source shipping --------------------------------------------------
    if git archive --format=tar HEAD > "$TMPD/dry.tar" 2>"$TMPD/g1.err"; then
        mkdir -p "$TMPD/dryarch"
        ( cd "$TMPD/dryarch" && tar xf "$TMPD/dry.tar" )
        _h=$(leg_source_sha_recipe "$TMPD/dryarch")
        if [ ${#_h} -eq 64 ]; then
            rok "G1 git archive extracts and hashes to a 64-hex source sha ($(echo "$_h" | cut -c1-16)...)"
        else
            rbad "G1 the source hash recipe" "got '$_h'"
        fi
        _miss=""
        for _need in $(leg_archive_required); do
            [ -f "$TMPD/dryarch/$_need" ] || _miss="$_miss $_need"
        done
        if [ -z "$_miss" ]; then
            rok "G2 the archive contains everything the $PAYLOAD payload runs ($(leg_archive_required))"
        else
            rbad "G2 the archive contains what the $PAYLOAD payload runs" "MISSING:$_miss -- the leg would rent a box to build nothing"
        fi
    else
        rbad "G1 git archive" "$(cat "$TMPD/g1.err")"
    fi

    if leg_check_tree_clean > "$TMPD/g3.out" 2>&1; then
        rok "G3 the working tree is clean for every path that reaches the bits"
    else
        rblock "G3 the working tree is clean for the leg's source paths" "--rent WOULD REFUSE right now. The local card comes from the tree and the remote card comes from the commit, so this is two variables:
$(sed 's/^/        /' "$TMPD/g3.out")"
    fi

    echo
    echo "  $R_PASS passed, $R_FAIL failed, $R_BLOCK blocking this box from being rented"
    [ "$R_FAIL" -eq 0 ] || return 1
    [ "$R_BLOCK" -eq 0 ] || return 3
    return 0
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

TMPD=$(mktemp -d "${TMPDIR:-/tmp}/gemm-leg.XXXXXX")
chmod 700 "$TMPD"
CURLRC="$TMPD/curlrc"

if [ "$MODE" = "reap" ]; then
    # LAYER TWO ON ITS OWN. The guard reaps; this adds the verification the
    # guard does not do, because "the DELETE returned 200" and "the pod is
    # gone" are different claims and only the second one stops the bill.
    leg_load_key || leg_die "reap needs RUNPOD_API_KEY (or MOJOLEARN_RUNPOD_KEY_FILE)."
    leg_curlrc
    # shellcheck disable=SC2031
    if [ -n "$POD_ID" ]; then
        echo "reaping $POD_ID"
        MODE=reap leg_terminate
        [ "$POD_TERMINATED" = "1" ] || exit 1
    else
        echo "reaping every EXPIRED lease (unexpired ones are left alone):"
        tools/runpod_guard.sh list 2>&1 | sed 's/^/  /' || true
        tools/runpod_guard.sh reap 2>&1 | sed 's/^/  /' || true
        echo
        echo "Unexpired leases are NOT reaped by this. To end one early:"
        echo "  tools/gemm_remote_leg.sh reap <pod-id>"
    fi
    POD_ID=""
    exit 0
fi

# THE KEY IS CHECKED BEFORE ANY WORK AND BEFORE ANY DIRECTORY IS CREATED.
# Renting needs BOTH --rent and a key. Checking it here rather than after the
# local card means a keyless --rent costs nothing, leaves nothing behind, and
# -- the part that matters -- can never reach the create call by falling
# through a later branch.
if [ "$MODE" = "rent" ]; then
    leg_load_key || leg_die "REFUSING to rent: RUNPOD_API_KEY is not set.
  Renting needs BOTH --rent and a key, deliberately. Without the key the
  guard cannot arm an on-pod watchdog either, and an unarmed box is an
  orphan that has not happened yet:
    export RUNPOD_API_KEY=...      (or MOJOLEARN_RUNPOD_KEY_FILE=<0600 file>)"
fi

mkdir -p "$OUT"
echo "== gemm.fp32.v1 remote leg (DEVIATION 536) =="
echo "   profile:  mojolearn.identical.gemm.fp32.v1"
echo "   vendor:   $VENDOR    label in the diff: $VLABEL"
echo "   payload:  $PAYLOAD$( [ "$PAYLOAD" = phase8 ] && echo '   (tools/e1_bootstrap.sh; phase 8 is seven lanes)' )"
if [ "$PAYLOAD" = "speed" ]; then
echo "   family:   $SPEED_FAMILY"
echo "   lanes:    $SPEED_LANES"
echo "   rounds:   $SPEED_ROUNDS timed per arm, size=$SPEED_SIZE, per-arm budget ${ARM_BUDGET}s"
if [ -n "$SPEED_DATASET" ] || [ -n "$SPEED_ROWS" ]; then
    echo "   dataset:  ${SPEED_DATASET:-<lane default>}"
    echo "   LADDER:   rows = ${SPEED_ROWS:-<shipped>}"
    echo "             Each rung is a separate fit of the SAME lane at a"
    echo "             different row count, scored on the SAME held-out"
    echo "             tail, so the rungs are comparable to each other."
fi
echo "   MODE:     FAST. No -D MOJOLEARN_NUMERIC_IDENTICAL anywhere in this"
echo "             payload. This leg measures the arm a user gets and makes"
echo "             NO identity claim about it."
fi
echo "   mode:     $MODE$( [ "$MODE" = dry ] && echo '  (nothing is rented; --rent opts in)' )"
echo "   gpu:      $GPU_ID"
echo "   image:    $IMAGE"
echo "   lease:    $MINUTES minutes (hard cap $MINUTES_CAP)"
echo "   out:      $OUT"
echo

COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
COMMIT_LINE=$(git log -1 --format='%h parent %p' 2>/dev/null || echo unknown)
{
    echo "$COMMIT"
} > "$OUT/commit.txt"
{
    echo "commit=$COMMIT_LINE"
    echo "vendor=$VENDOR"
    echo "gpu_requested=$GPU_ID"
    echo "image=$IMAGE"
    echo "minutes=$MINUTES"
    echo "mode=$MODE"
    echo "card_full=${CARD_FULL:-<unset>}"
    echo "trace_dump=${LEG_DUMP:-<unset>}"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/leg.txt"
echo "   commit:   $COMMIT_LINE"
echo

if [ "$PAYLOAD" = "speed" ]; then

echo "== step 1: THERE IS NO APPLE REFERENCE, AND THAT IS THE POINT =="
echo "   This payload builds NOTHING on this Mac and compares nothing to it."
echo "   A wall-clock number is a property of ONE box, and the comparison"
echo "   that matters is made ON the rented box, in one hour, between our"
echo "   FAST arm and the vendor's own library running on the same silicon"
echo "   under the same thermal conditions. Carrying an M4 millisecond into"
echo "   that table would be comparing two machines and calling it a"
echo "   comparison of two libraries."
echo
echo "   What comes home is FSPEED lines. The verdict is"
echo "   tools/fast_speed_table.py, run HERE, over the fetched logs."

elif [ "$PAYLOAD" = "phase8" ]; then

echo "== step 1: the Apple reference column =="
echo "   The phase8 payload builds NOTHING on this Mac. Its reference is an"
echo "   Apple bootstrap directory that already exists, and the comparison"
echo "   is made by tools/e3_round_judge.sh once the box's directory lands"
echo "   beside it -- this leg does not diff phase-8 cards."
LANES=$(leg_lanes)
echo "   lanes (read out of tools/e3_round_judge.sh section 7): $LANES"
if leg_apple_column; then
    :
else
    APPLE_MISSING=1
    if [ "$MODE" = "rent" ]; then
        leg_die "NOTHING IS RENTED WITHOUT AN APPLE COLUMN AT THIS COMMIT.
  The box would produce seven cards with nothing to compare them to, and
  the Mac's column cannot be made later at this commit once the shared
  checkout has moved on. Make it first (it costs nothing but the build
  lock):
    bash tools/e1_bootstrap.sh
  or name an existing one:  --apple-dir bench/results/e1/<stamp>-...
  This is the phase8 payload's L1: the gemm payload refuses to rent
  against a reference card it does not have, for the same reason."
    fi
    echo "  (dry run: continuing so the rest of the plumbing is exercised)"
fi

else

echo "== step 1: the Apple reference card =="
if leg_local_card; then
    if [ -f "$OUT/local/apple.card.log" ]; then
        leg_require_identical "$OUT/local/apple.card.log" "local card" || {
            [ "$MODE" = "rent" ] && leg_die "the LOCAL card is not IDENTICAL. Nothing is rented against a reference this leg cannot trust."
            echo "  (dry run: continuing so the rest of the plumbing is still exercised)"
        }
    fi
    echo "  local card: $LOCAL_CARD ($(grep -c . "$LOCAL_CARD") lines)"
else
    if [ "$MODE" = "rent" ]; then
        leg_die "THE LOCAL CARD FAILED. There is nothing to diff a remote card
  against, so nothing is rented. Fix the local device arm first:
    tools/gemm_card.sh device /tmp/apple.card"
    fi
    LOCAL_CARD_SYNTHETIC=1
    echo "  LOCAL CARD FAILED. In --rent this aborts before any billing."
    echo "  The dry run continues with a synthetic card so the plumbing below"
    echo "  is still exercised, and says so rather than quietly substituting."
    LOCAL_CARD="$OUT/local/apple.card"
    {
        echo "# format: mojolearn-identity-trace v1"
        echo "# SYNTHETIC. Not a measurement. The local device arm failed."
        printf '0\tsynthetic.in.a\tf32\t16\t0123456789abcdef\n'
        printf '1\tsynthetic.in.b\tf32\t16\t123456789abcdef0\n'
        printf '2\tsynthetic.out\tf32\t16\tfedcba9876543210\n'
        printf '3\tsynthetic.out2\tf32\t16\t0f1e2d3c4b5a6978\n'
    } > "$LOCAL_CARD"
fi

fi

if [ "$MODE" = "dry" ]; then
    leg_rehearse || DRY_RC=$?
    echo
    echo "== what --rent WOULD do, in order =="
    echo "   1. list leases here and pods there; refuse if either is occupied"
    echo "   1b. ARM A DETACHED DEAD-MAN HERE, before the create: it ends a"
    echo "      pod named mojolearn-gemm-$VENDOR-<stamp> after"
    echo "      $((READY_TIMEOUT + MINUTES * 60 + 300))s even if this process is killed"
    echo "   2. POST $RP_HOST$RP_PODS_PATH  ->  a $GPU_ID pod named"
    echo "      mojolearn-gemm-$VENDOR-<stamp>   [THE BILL STARTS HERE]"
    echo "   3. wait up to ${READY_TIMEOUT}s for ssh; time out -> TERMINATE"
    echo "   4. tools/runpod_guard.sh arm <pod> '<ssh>' $MINUTES"
    echo "      refusal -> TERMINATE, no work, no card"
    echo "   5. key to the pod on stdin (0600), then read the pod's ps back"
    echo "   6. git archive $COMMIT -> the box; compare source sha both ways"
    if [ "$PAYLOAD" = "phase8" ]; then
    echo "   7. bash tools/e1_bootstrap.sh, DETACHED and polled, bounded at"
    echo "      ${WORK_TIMEOUT}s so the fetch keeps its reserve; phase 8 writes"
    echo "      lanes/*.card"
    echo "      READ THIS BEFORE RENTING: phase 8 loops 'for mode in fast"
    echo "      identical', so the ENTIRE FAST PASS -- seven lanes, arm and"
    echo "      check, roughly twenty compiles -- runs BEFORE the first"
    echo "      IDENTICAL card is written. The whole bootstrap was 27 min"
    echo "      (H100) and 18 min (MI325X) on images that ALREADY HAD PIXI;"
    echo "      a cold pixi install here is unmeasured. If the ${WORK_TIMEOUT}s"
    echo "      bound fires inside the fast pass the box comes home with NO"
    echo "      identical cards at all, and that is a clock finding, not a"
    echo "      silicon one. bootstrap_exit=124 in remote/leg.txt says so."
    echo "   8. fetch the bootstrap directory to"
    echo "      $E1_DEST; rewrite its commit.txt"
    echo "      from the pinned sha; read each lane's mode back"
    echo "   9. NO DIFF HERE. tools/e3_round_judge.sh section 7 judges it"
    else
    if [ "$PAYLOAD" = "speed" ]; then
    echo "   7. the bench/speed/ drivers, FAST (NO -D define), and the"
    echo "      vendor arms, on the box, one arm per process, each bounded"
    echo "      at ${ARM_BUDGET}s: family=$SPEED_FAMILY"
    echo "      lanes: $SPEED_LANES"
    echo "   8. fetch the FSPEED logs; REFUSE any 'ours' header that says"
    echo "      IDENTICAL, because this payload is the FAST arm"
    echo "   9. no diff. tools/fast_speed_table.py builds the ratio table"
    echo "      HERE, from the logs, after the box is gone"
    elif [ "$PAYLOAD" = "phase8" ]; then
    echo "   7. tools/e1_bootstrap.sh, phases ${E1_PHASES:-all}, bounded"
    echo "   8. fetch the bootstrap directory; read the mode back per lane"
    echo "   9. no diff. tools/e3_round_judge.sh section 7 is the verdict"
    else
    echo "   7. gemm_device_check.mojo, then gemm_card.sh device, IDENTICAL"
    echo "   8. fetch; read the mode back out of BOTH remote logs"
    echo "   9. diff against the Apple card; name the FIRST diverging stage"
    fi
    fi
    echo "  10. reap, verify by asking the API, print the lease remaining"
    echo
    echo "   the command:"
    echo "     export RUNPOD_API_KEY=...   # or MOJOLEARN_RUNPOD_KEY_FILE"
    if [ "$PAYLOAD" = "phase8" ]; then
    echo "     tools/gemm_remote_leg.sh $VENDOR --payload phase8 --rent --minutes $MINUTES"
    else
    echo "     tools/gemm_remote_leg.sh $VENDOR --rent --minutes $MINUTES"
    fi
    echo
    if [ "${DRY_RC:-0}" = "1" ]; then
        echo "DRY RUN: RED -- THIS SCRIPT is broken (a FAIL above). Fix it"
        echo "before renting anything: every one of those checks stands"
        echo "between a paid box and an orphan."
        exit 1
    fi
    if [ "${DRY_RC:-0}" = "3" ]; then
        echo "DRY RUN: the plumbing is GREEN and the box CANNOT BE RENTED YET"
        echo "(a BLOCK above). Nothing here is broken; the world is not ready."
        exit 3
    fi
    echo "DRY RUN: GREEN, and nothing was rented. That is a statement about"
    echo "THIS SCRIPT'S plumbing, not about cross-vendor identity, which no"
    echo "dry run can say anything about at all."
    exit 0
fi

# ---- from here on it costs money ------------------------------------------

echo
echo "== step 2: pre-flight =="
leg_curlrc

leg_check_tree_clean || leg_die "REFUSING to rent against a dirty tree.
  The local card came from the WORKING TREE and the remote card will come
  from the COMMIT. If they differ, the device is not the variable being
  measured and the diff is uninterpretable. Commit, stash, or pass
  --local-card pointing at a card generated at this exact commit."
leg_build_remote_body || leg_die "the remote body did not build. Nothing rented."
leg_preflight

echo
echo "== step 3: the box =="
leg_create_pod
leg_wait_ready

echo
echo "== step 4: the lease, BEFORE any work =="
leg_arm

echo
echo "== step 5: the key on the pod, and where it is visible =="
leg_key_to_pod 2>&1 | sed 's/^/    /' || true
# ASSERTED, NOT RECORDED. Until 2026-08-24 `tools/runpod_guard.sh arm` put
# the key in the argv of the remote `sh -c` that sshd spawns, so this check
# was a MEASUREMENT of a known residue and could not be a gate. That leak is
# closed -- the guard sends its credential on stdin into a 0600 curl config
# and the dry run's C10 proves it -- so KEY_VISIBLE_IN_PS now means something
# is wrong that nobody knows about, and the leg goes red for it.
leg_check_key_not_in_ps || KEY_RED=1

echo
echo "== step 6: the source =="
leg_ship_and_run

echo
echo "== step 7: fetch =="
leg_fetch

echo
echo "== step 8: what came home =="
RED=$FETCH_RED
if [ "$PAYLOAD" = "speed" ]; then
    leg_speed_artifacts || RED=1
elif [ "$PAYLOAD" = "phase8" ]; then
    leg_phase8_artifacts || RED=1
else
    REMOTE_CARD="$OUT/remote/$VENDOR.card"
    leg_require_file "$REMOTE_CARD" "the remote never produced a card; read $OUT/remote/card_driver.log" || RED=1
    leg_require_identical "$OUT/remote/$VENDOR.card.log" "remote card" || RED=1
    leg_require_identical "$OUT/remote/device_check.log" "remote device check" || RED=1
fi

if [ -f "$OUT/remote/source_sha256.txt" ]; then
    _rs=$(cat "$OUT/remote/source_sha256.txt")
    _ls=$(cat "$OUT/source_sha256_local.txt")
    if [ "$_rs" = "$_ls" ]; then
        echo "  source sha256 MATCHES on both ends ($(echo "$_ls" | cut -c1-16)...)"
    else
        echo "  SOURCE SHA MISMATCH -- the box did not run what was shipped."
        echo "    here:   $_ls"
        echo "    there:  $_rs"
        echo "    Every comparison below is void: this is not one variable."
        RED=1
    fi
else
    echo "  no source_sha256.txt came home; commit parity is UNVERIFIED"
    RED=1
fi

if [ "$KEY_RED" = "1" ]; then
    echo "  CREDENTIAL EXPOSURE: the key was visible in the pod's process"
    echo "    list. That is not a claim about the cards -- they are as sound"
    echo "    as everything else above says -- but this leg exits non-zero"
    echo "    for it and the key must be ROTATED after this run. Read"
    echo "    $OUT/key_in_ps.txt and find what put it there."
fi
if [ -f "$OUT/remote/gpu.txt" ]; then
    echo "  gpu on the box:"
    sed 's/^/    /' "$OUT/remote/gpu.txt" | head -4
fi
if [ -f "$OUT/remote/leg.txt" ]; then
    sed 's/^/    /' "$OUT/remote/leg.txt"
fi
if [ "$PAYLOAD" = "speed" ]; then
    :
elif [ "$PAYLOAD" = "phase8" ]; then
    if grep -q 'bootstrap_exit=0' "$OUT/remote/leg.txt" 2>/dev/null; then
        echo "  the bootstrap ran to the end on this box"
    elif grep -q 'bootstrap_exit=124' "$OUT/remote/leg.txt" 2>/dev/null; then
        echo "  THE WORK TIMEOUT FIRED (${WORK_TIMEOUT}s). The bootstrap was"
        echo "    stopped so the fetch would still happen, which is why any"
        echo "    card above came home at all. Whatever is missing above is"
        echo "    missing because of the clock, not because of the silicon."
    else
        echo "  the bootstrap exited non-zero. That is not by itself a"
        echo "    finding about this silicon -- e1_bootstrap.sh runs every"
        echo "    phase and reports rather than aborts -- but read"
        echo "    $E1_DEST/bootstrap.log before reading any card."
    fi
else
    if grep -q 'device_check_exit=0' "$OUT/remote/leg.txt" 2>/dev/null; then
        echo "  the device kernel's own gates: GREEN on this silicon"
    else
        echo "  the device kernel's own gates: RED or unrun on this silicon."
        echo "    A machine that fails its own gates teaches nothing when diffed"
        echo "    against another. Read $OUT/remote/device_check.log FIRST; the"
        echo "    card diff below is secondary until that is resolved."
        RED=1
    fi
fi

echo
if [ "$PAYLOAD" = "speed" ]; then
echo "== step 9: the table, which is built HERE =="
echo "  Nothing is diffed. The verdict is a ratio table over the FSPEED"
echo "  lines the box printed:"
echo
echo "    python3 tools/fast_speed_table.py $OUT/remote/logs \\"
echo "        --out bench/results/fast_speed/${STAMP}-${VENDOR}-${SPEED_FAMILY}.md"
echo
echo "  Read its ratio column with the arm name in view. Above 1.0 means WE"
echo "  ARE SLOWER, and for the gemm lane the fair opponent is cublas-tf32"
echo "  rather than cublas-fp32, because our FAST arm took the same"
echo "  precision cut. That is written at length in"
echo "  bench/speed/gemm_speed_main.mojo's docstring."
elif [ "$PAYLOAD" = "phase8" ]; then
echo "== step 9: the judge, which is not this leg =="
echo "  This leg does not diff phase-8 cards and must not:"
echo "  tools/e3_round_judge.sh is what the round is recorded from, it"
echo "  reads the whole column (commits, the tree matrix, E2U, the E1U"
echo "  cards, the gate lines, cross-infer) and section 7 is the lanes."
echo
echo "    bash tools/e3_round_judge.sh \\"
echo "        ${APPLE_DIR:-<apple bootstrap dir>} \\"
echo "        $E1_DEST --write"
echo
echo "  Run BOTH vendors' legs first and judge all three columns in one"
echo "  command; the judge takes <mac> <nv> [<amd>] and one invocation is"
echo "  one round."
else
echo "== step 9: Apple vs $VLABEL =="
if [ "$RED" = "0" ]; then
    leg_diff_cards "$LOCAL_CARD" "$REMOTE_CARD" "$OUT/diff_apple_vs_$VENDOR.txt" || RED=1
else
    echo "  NOT DIFFING. Something above is unsound and a stage name produced"
    echo "  from an unsound pair is worse than no stage name at all."
fi
fi

{
    echo "commit=$COMMIT_LINE"
    echo "vendor=$VENDOR"
    echo "payload=$PAYLOAD"
    echo "pod=$POD_ID"
    echo "gpu_requested=$GPU_ID"
    echo "red=$RED"
    if [ "$PAYLOAD" = "phase8" ]; then
        echo "e1_dir=$E1_DEST"
        echo "apple_dir=${APPLE_DIR:-<none>}"
    fi
    if [ "$PAYLOAD" = "speed" ]; then
        echo "family=$SPEED_FAMILY"
        echo "lanes=$SPEED_LANES"
        echo "rounds=$SPEED_ROUNDS"
        echo "size=$SPEED_SIZE"
        echo "dataset=$SPEED_DATASET"
        echo "rows_ladder=$SPEED_ROWS"
    fi
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$OUT/leg.txt"

echo
echo "== step 10: terminate at the end of the WORK =="
leg_lease_report
leg_terminate

echo
echo "artifacts: $OUT"
if [ "$PAYLOAD" = "speed" ]; then
    echo "the arms:    $OUT/remote/logs"
    echo "next: build the table and write the row before this scrollback is"
    echo "the only place the pod's details exist:"
    echo "  python3 tools/fast_speed_table.py $OUT/remote/logs \\"
    echo "      --out bench/results/fast_speed/${STAMP}-${VENDOR}-${SPEED_FAMILY}.md"
elif [ "$PAYLOAD" = "phase8" ]; then
    echo "the column:  $E1_DEST"
    echo "$E1_DEST" > "$OUT/e1_dir.txt"
    echo "next: gemm/E1G_RUNBOOK.md, 'the phase8 payload'. The verdict is"
    echo "tools/e3_round_judge.sh and it is recorded in E3_RESULTS.md, not"
    echo "in E1G_RESULTS.md -- one round per entry, never appended to."
else
    echo "next: gemm/E1G_RUNBOOK.md, 'reading the result'. Write the row into"
    echo "E1G_RESULTS.md before the pod's details are only in this scrollback."
fi
# The two reds are separate on purpose. RED is about whether the cards can
# be believed; KEY_RED is about a credential. Neither one gets to hide the
# other, and either one makes this leg exit non-zero.
[ "$RED" = "0" ] || exit 1
[ "$KEY_RED" = "0" ] || exit 1
exit 0
