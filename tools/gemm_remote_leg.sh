#!/bin/sh
# DEVIATION 536 -- ONE GUARDED RUNPOD LEG: the `gemm.fp32.v1` identity card
# taken to a second and a third vendor, and the box terminated afterwards.
#
#   tools/gemm_remote_leg.sh nvidia                 DRY RUN. Rents nothing.
#   tools/gemm_remote_leg.sh amd                    DRY RUN. Rents nothing.
#   tools/gemm_remote_leg.sh nvidia --rent          RENTS. BILLS. One hour.
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
#  1. The pod is created. **From this instant to step 3 the box is BILLING
#     AND UNARMED.** That window is bounded IN CODE by --ready-timeout
#     (default 600s), not by whoever is watching the terminal, and running
#     out of it terminates the pod.
#  2. SSH comes up.
#  3. **THE LEASE IS ARMED BEFORE ANY WORK.** `tools/runpod_guard.sh arm`
#     installs an on-pod watchdog that DELETEs this pod through the API at
#     the deadline; that survives this Mac going away, which is the whole
#     requirement, and it is why `arm` REFUSES without a key rather than
#     pretending. **If arm refuses, the box is not used -- it is
#     TERMINATED.** A box that cannot be armed is an orphan that has not
#     happened yet.
#  4. Work runs. Terminate happens at the END OF THE WORK, not at the end of
#     the lease. The lease is the backstop for when this session disappears;
#     it is not the plan.
#  5. Teardown is an EXIT trap, so every failure path above reaches it, and
#     it VERIFIES the termination by asking the API instead of assuming the
#     DELETE worked. It prints the lease's remaining minutes at exit, which
#     is the one number that says how close this run came to needing the
#     backstop.
#
#     LAYER ONE is the on-pod watchdog (survives this Mac). LAYER TWO is the
#     `reap` this script performs from here (reclaims the DISK, which keeps
#     accruing on RunPod even after container exit stops GPU billing). Both
#     layers, always, because they cover different failures.
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
#   * KNOWN AND UNFIXED HERE: `tools/runpod_guard.sh arm` passes the key to
#     the pod inside the ssh command string, so for the ~1 second that arm
#     runs the key is in the argv of the remote `sh -c` that sshd spawns.
#     That file belongs to the identity lane and this lane may not edit it.
#     The exact change is written up in this leg's report and in
#     `gemm/E1G_RUNBOOK.md`; the post-arm check below is what makes the
#     residue visible rather than assumed.
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
# The first paid run is therefore a bring-up run. Budget it as one, and read
# the runbook's "first paid run" section before starting it.
#
# ENVIRONMENT
#   RUNPOD_API_KEY                 required by --rent. Never logged.
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
#   MOJOLEARN_GEMM_LEG_REHEARSAL   internal interlock. When 1, any paid call
#                                  aborts. Set on the children the dry run
#                                  spawns so a rehearsal can never rent.
set -e

cd "$(dirname "$0")/.."
REPO="$(pwd)"

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
RP_HOST="${MOJOLEARN_RUNPOD_API_HOST:-https://api.runpod.io}"
# v1 is deprecated and is kept ONLY as a fallback for the DELETE, for the
# reason tools/runpod_guard.sh gives at the same seam: this is the only thing
# standing between an orphan and the bill, and an API deprecation that
# silently 404s would disarm every terminate at once.
RP_HOST_V1="${MOJOLEARN_RUNPOD_API_HOST_V1:-https://rest.runpod.io}"
RP_PODS_PATH="/v2/pods"
# The guard's lease directory, mirrored (not owned) so a terminate can leave
# `check` and `list` telling the truth afterwards.
LEASE_DIR="${MOJOLEARN_LEASE_DIR:-bench/results/runpod_leases}"

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
ARMED=0
TMPD=""
CURLRC=""
RP_CODE=""
RP_BODY=""

DIFFER="tools/identity_trace_diff.py"
PY=/usr/bin/python3
[ -x "$PY" ] || PY=python3

leg_usage() {
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "options: --rent --dry-run --minutes N --gpu ID --image REF"
    echo "         --ssh TARGET --local-card PATH --column-sweep"
    echo "         --ready-timeout SECONDS"
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

if [ "$MODE" != "reap" ]; then
    case "$VENDOR" in
        nvidia) : "${GPU_ID:=$LEG_GPU_NVIDIA}"; : "${IMAGE:=$LEG_IMAGE_NVIDIA}"; SMI_CMD='nvidia-smi --query-gpu=name,driver_version --format=csv,noheader' ;;
        amd)    : "${GPU_ID:=$LEG_GPU_AMD}";    : "${IMAGE:=$LEG_IMAGE_AMD}";    SMI_CMD='rocm-smi --showproductname' ;;
    esac
    VLABEL=$(echo "$VENDOR" | tr '[:lower:]' '[:upper:]')
fi

STAMP=$(date +%Y-%m-%d_%H%M%S)
if [ "$MODE" = "dry" ]; then
    OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-${VENDOR}-dryrun}"
else
    OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-${VENDOR}}"
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

# TARGETED, AND DELIBERATELY NOT `runpod_guard.sh reap --force <pod>`.
#
# THAT HELPER IGNORES ITS POD ARGUMENT. `cmd_reap` shifts `--force` off and
# then shifts the pod id off with it, and the loop that follows terminates
# EVERY lease file in the lease directory. So `reap --force <pod-id>` READS
# like a targeted terminate and is a reap-everything -- harmless under this
# lane's one-box-at-a-time discipline, and exactly the kind of gap that is
# only harmless until the day two boxes are up. This leg deletes the pod it
# created, by id, and then removes that pod's lease file so `check` and
# `list` are still telling the truth afterwards. The exact change the guard
# needs is in gemm/E1G_RUNBOOK.md; that file is the identity lane's.
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
            echo "exit=$_rc"
            echo "at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } >> "$OUT/teardown.txt" 2>/dev/null || true
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

leg_check_tree_clean() {
    # THE LOCAL CARD COMES FROM THE WORKING TREE AND THE REMOTE CARD COMES
    # FROM THE COMMIT. If those differ for any file that can reach the bits,
    # the diff has two variables in it and the device is not the one being
    # measured. Documentation and results directories are deliberately not
    # in the list: they cannot reach a float.
    # LEG_SOURCE_PATHS is a deliberate word list, so it is unquoted.
    # shellcheck disable=SC2086
    _dirty=$(git status --porcelain -- $LEG_SOURCE_PATHS 2>/dev/null || true)
    if [ -n "$_dirty" ]; then
        echo "  the working tree is DIRTY for paths that reach the bits:"
        echo "$_dirty" | sed 's/^/    /'
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
        leg_die "REFUSING to rent: this lane already has pod(s) up: $_mine
  Terminate them first:  tools/gemm_remote_leg.sh reap <pod-id>"
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
    cat > "$TMPD/create.json" <<JSONEOF
{
  "name": "$POD_NAME",
  "imageName": "$IMAGE",
  "gpuTypeIds": ["$GPU_ID"],
  "gpuCount": 1,
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
        _port=$(rp_json "([str(m.get('publicPort','')) for m in (d.get('portMappings') or (d.get('runtime') or {}).get('ports') or []) if str(m.get('privatePort',''))=='22'] or [''])[0]")
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
    cat > "$_body" <<'REMOTE_BODY'
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
echo REMOTE_BODY_DONE
REMOTE_BODY

    # SUBSTITUTION, AND THEN A CHECK THAT IT HAPPENED. A `sed` without `/g`
    # has already left one of this repository's guards holding a literal
    # placeholder, so the trailing grep is the part that matters: it does not
    # trust /g, it verifies that no placeholder survived.
    sed -e "s|@VENDOR@|$VENDOR|g" \
        -e "s|@COMMIT@|$COMMIT|g" \
        -e "s|@CARDFULL@|$CARD_FULL|g" \
        -e "s|@SWEEP@|$SWEEP|g" \
        -e "s|@DUMP@|$LEG_DUMP|g" \
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
    [ -f "$TMPD/archive/gemm/mojo_only/gemm_identical.mojo" ] || leg_die \
"THE ARCHIVE DOES NOT CONTAIN gemm/mojo_only/gemm_identical.mojo.
  Either 'git archive' of $COMMIT failed, or the kernel is not committed at
  this commit. Shipping it would rent a box to build nothing."
    leg_source_sha_recipe "$TMPD/archive" > "$OUT/source_sha256_local.txt"
    leg_say "  archive source sha256: $(cut -c1-32 < "$OUT/source_sha256_local.txt")"

    leg_ssh 'rm -rf /root/mojolearn /root/gemm_leg_out && mkdir -p /root/mojolearn' \
        > /dev/null
    leg_ssh 'cd /root/mojolearn && tar xzf -' < "$TMPD/src.tgz"
    leg_ssh 'umask 022; cat > /root/gemm_leg.sh' < "$OUT/remote_body.sh"
    leg_say "running the leg on the box (this is the long step)"
    leg_ssh 'sh /root/gemm_leg.sh' > "$OUT/remote_console.log" 2>&1 || true
    tail -5 "$OUT/remote_console.log" | sed 's/^/    /'
}

leg_fetch() {
    mkdir -p "$OUT/remote"
    leg_ssh 'cd /root/gemm_leg_out && tar czf - .' | ( cd "$OUT/remote" && tar xzf - ) \
        || echo "  FETCH FAILED -- the remote log is /root/gemm_leg_out"
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

    _out=$( ( unset RUNPOD_API_KEY; MOJOLEARN_GEMM_LEG_REHEARSAL=1 "$0" nvidia --rent 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "RUNPOD_API_KEY"; then
        rok "A4 --rent without a key refuses and rents nothing"
    else
        rbad "A4 --rent without a key refuses" "got exit $_rc: $_out"
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
    _out=$( ( unset RUNPOD_API_KEY; tools/runpod_guard.sh arm gemm-dryrun-pod "dry target" 60 2>&1 ) ) && _rc=0 || _rc=$?
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

    _out=$( ( unset RUNPOD_API_KEY; tools/runpod_guard.sh reap --force gemm-dryrun-no-such-pod 2>&1 ) ) && _rc=0 || _rc=$?
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
    _out=$( ( trap - EXIT; unset RUNPOD_API_KEY
              POD_ID=gemm-dryrun-pod; SSH_TARGET="dry target"
              leg_arm 2>&1 ) ) && _rc=0 || _rc=$?
    if [ "$_rc" != "0" ] && echo "$_out" | grep -q "ARM REFUSED"; then
        rok "C4 leg_arm turns the guard's refusal into a refusal, not a pass"
    else
        rbad "C4 leg_arm propagates the guard's refusal" "IT DID NOT. Work would start on an unguarded box. got exit $_rc"
    fi

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
    if [ "${LOCAL_CARD_SYNTHETIC:-0}" = "1" ]; then
        rblock "L1 the Apple reference card is a real measurement" "IT IS SYNTHETIC. The local device arm failed, so the checks above tested the plumbing against a card that measures nothing. Renting now would buy a remote card with nothing to compare it to."
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
        if [ -f "$TMPD/dryarch/gemm/mojo_only/gemm_identical.mojo" ]; then
            rok "G2 the archive contains the kernel the leg is about"
        else
            rbad "G2 the archive contains gemm/mojo_only/gemm_identical.mojo" "it does not -- the leg would build nothing"
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

if [ "$MODE" = "dry" ]; then
    leg_rehearse || DRY_RC=$?
    echo
    echo "== what --rent WOULD do, in order =="
    echo "   1. list leases here and pods there; refuse if either is occupied"
    echo "   2. POST $RP_HOST$RP_PODS_PATH  ->  a $GPU_ID pod named"
    echo "      mojolearn-gemm-$VENDOR-<stamp>   [THE BILL STARTS HERE]"
    echo "   3. wait up to ${READY_TIMEOUT}s for ssh; time out -> TERMINATE"
    echo "   4. tools/runpod_guard.sh arm <pod> '<ssh>' $MINUTES"
    echo "      refusal -> TERMINATE, no work, no card"
    echo "   5. key to the pod on stdin (0600), then read the pod's ps back"
    echo "   6. git archive $COMMIT -> the box; compare source sha both ways"
    echo "   7. gemm_device_check.mojo, then gemm_card.sh device, IDENTICAL"
    echo "   8. fetch; read the mode back out of BOTH remote logs"
    echo "   9. diff against the Apple card; name the FIRST diverging stage"
    echo "  10. reap, verify by asking the API, print the lease remaining"
    echo
    echo "   the command:"
    echo "     export RUNPOD_API_KEY=...   # or MOJOLEARN_RUNPOD_KEY_FILE"
    echo "     tools/gemm_remote_leg.sh $VENDOR --rent --minutes $MINUTES"
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
leg_check_key_not_in_ps || true

echo
echo "== step 6: the source =="
leg_ship_and_run

echo
echo "== step 7: fetch =="
leg_fetch

echo
echo "== step 8: what came home =="
RED=0
REMOTE_CARD="$OUT/remote/$VENDOR.card"
leg_require_file "$REMOTE_CARD" "the remote never produced a card; read $OUT/remote/card_driver.log" || RED=1
leg_require_identical "$OUT/remote/$VENDOR.card.log" "remote card" || RED=1
leg_require_identical "$OUT/remote/device_check.log" "remote device check" || RED=1

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

if [ -f "$OUT/remote/gpu.txt" ]; then
    echo "  gpu on the box:"
    sed 's/^/    /' "$OUT/remote/gpu.txt" | head -4
fi
if [ -f "$OUT/remote/leg.txt" ]; then
    sed 's/^/    /' "$OUT/remote/leg.txt"
fi
if grep -q 'device_check_exit=0' "$OUT/remote/leg.txt" 2>/dev/null; then
    echo "  the device kernel's own gates: GREEN on this silicon"
else
    echo "  the device kernel's own gates: RED or unrun on this silicon."
    echo "    A machine that fails its own gates teaches nothing when diffed"
    echo "    against another. Read $OUT/remote/device_check.log FIRST; the"
    echo "    card diff below is secondary until that is resolved."
    RED=1
fi

echo
echo "== step 9: Apple vs $VLABEL =="
if [ "$RED" = "0" ]; then
    leg_diff_cards "$LOCAL_CARD" "$REMOTE_CARD" "$OUT/diff_apple_vs_$VENDOR.txt" || RED=1
else
    echo "  NOT DIFFING. Something above is unsound and a stage name produced"
    echo "  from an unsound pair is worse than no stage name at all."
fi

{
    echo "commit=$COMMIT_LINE"
    echo "vendor=$VENDOR"
    echo "pod=$POD_ID"
    echo "gpu_requested=$GPU_ID"
    echo "red=$RED"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$OUT/leg.txt"

echo
echo "== step 10: terminate at the end of the WORK =="
leg_lease_report
leg_terminate

echo
echo "artifacts: $OUT"
echo "next: gemm/E1G_RUNBOOK.md, 'reading the result'. Write the row into"
echo "E1G_RESULTS.md before the pod's details are only in this scrollback."
[ "$RED" = "0" ] || exit 1
exit 0
