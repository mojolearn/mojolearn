#!/bin/sh
# EVERY RENTED GPU EXPIRES ON ITS OWN. This is the code that makes that true.
#
#   tools/runpod_guard.sh arm   <pod-id> <ssh-target> [minutes]   # default 60
#   tools/runpod_guard.sh check <pod-id>
#   tools/runpod_guard.sh list
#   tools/runpod_guard.sh reap                                    # needs a key
#   tools/runpod_guard.sh extend <pod-id> <ssh-target> [minutes]
#
# WHY THIS EXISTS, AND WHY IT IS NOT A CHECKLIST ITEM
# ---------------------------------------------------
# An MI300X is $2.39/hr. The failure being guarded is an ORPHAN: a session
# ends, crashes, or is closed, and the pod keeps billing with nobody watching
# it. Andrew's rule (2026-08-23): the expiry must be BAKED IN, not remembered.
#
# **THE ORPHAN CASE IS THE LAPTOP GOING AWAY**, so a reaper that runs on the
# laptop is the wrong primary defence -- it is exactly the thing that is gone
# when it is needed. The primary defence has to live ON THE POD.
#
# LAYER 1 (primary, credential-free, survives everything on this end)
# -------------------------------------------------------------------
# A RunPod pod runs `sleep infinity` as PID 1. Killing PID 1 exits the
# container, the pod goes to EXITED, and GPU billing stops. So `arm` puts
#
#     nohup sh -c 'sleep <N>; kill 1' &
#
# on the box, detached, before any work starts. No API key is needed on the
# pod, nothing here has to stay running, and the deadline holds if this
# machine is closed, crashes, or loses its network the second after arming.
#
# It is deliberately `kill 1` and not `shutdown`/`poweroff`: those need init
# and privileges the container does not have, and a guard that silently fails
# to arm is worse than no guard, which is why `arm` VERIFIES the watchdog is
# running before it returns and refuses loudly if it is not.
#
# LAYER 2 (secondary, needs a key, reclaims the disk too)
# -------------------------------------------------------
# An EXITED pod still bills its disk (~$0.02/hr at 120 GB). `reap` reads the
# lease files and terminates anything past its deadline through the REST API,
# using RUNPOD_API_KEY from the environment. Run it whenever you are at the
# machine; it is a cleanup, not the safety net.
#
# EXTENDING A LEASE IS RE-ARMING, NEVER REMOVING. `extend` kills the old
# watchdog and starts a new one. There is no `disarm`, deliberately: the only
# way to keep a box alive is to keep saying so.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
LEASES="${MOJOLEARN_LEASE_DIR:-$HERE/../bench/results/runpod_leases}"
mkdir -p "$LEASES"

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes"

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

# `ssh-target` is passed through verbatim so both RunPod forms work:
#   "-p 23432 root@213.173.96.54"          direct tcp
#   "podid-xxxxxxxx@ssh.runpod.io"         the proxy (needs -tt)
pod_ssh() {
    target="$1"; shift
    # shellcheck disable=SC2086
    ssh $SSH_OPTS $target "$@"
}

cmd_arm() {
    pod="$1"; target="$2"; mins="${3:-60}"
    [ -n "$pod" ] && [ -n "$target" ] || usage
    secs=$((mins * 60))

    echo "arming a ${mins}-minute self-kill on pod $pod ..."

    # `setsid` where available so the watchdog does not die with the ssh
    # session; `nohup` alone is enough on these images but both is cheap.
    pod_ssh "$target" "
        pkill -f 'mojolearn-lease-watchdog' >/dev/null 2>&1 || true
        nohup sh -c 'exec -a mojolearn-lease-watchdog sh -c \"sleep $secs; kill 1\"' \
            >/tmp/mojolearn-lease.log 2>&1 &
        sleep 1
        pgrep -f mojolearn-lease-watchdog >/dev/null && echo WATCHDOG_ARMED
    " > /tmp/mojolearn_arm_out 2>&1 || true

    if ! grep -q WATCHDOG_ARMED /tmp/mojolearn_arm_out; then
        echo "REFUSING: the watchdog did not come up on $pod." >&2
        echo "  A guard that silently fails to arm is worse than no guard," >&2
        echo "  because the next person reads the lease file and believes it." >&2
        echo "  ssh output:" >&2
        sed 's/^/    /' /tmp/mojolearn_arm_out >&2
        echo "  Do NOT start work on this pod until the watchdog is up," >&2
        echo "  or terminate it now:  tools/runpod_guard.sh reap --force $pod" >&2
        exit 1
    fi

    now=$(date -u +%s)
    deadline=$((now + secs))
    {
        echo "pod=$pod"
        echo "target=$target"
        echo "armed_utc=$(date -u -r "$now" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "deadline_epoch=$deadline"
        echo "minutes=$mins"
    } > "$LEASES/$pod.lease"

    echo "ARMED. Pod $pod exits its container in $mins minutes."
    echo "  lease: $LEASES/$pod.lease"
    echo "  extend with: tools/runpod_guard.sh extend $pod '$target' <minutes>"
}

cmd_extend() {
    pod="$1"; target="$2"; mins="${3:-60}"
    [ -n "$pod" ] && [ -n "$target" ] || usage
    echo "extending: the old watchdog is killed and a new one armed."
    cmd_arm "$pod" "$target" "$mins"
}

cmd_check() {
    pod="$1"; [ -n "$pod" ] || usage
    f="$LEASES/$pod.lease"
    [ -f "$f" ] || { echo "NO LEASE for $pod -- it is unguarded."; exit 1; }
    # shellcheck disable=SC1090
    . "$f"
    now=$(date -u +%s)
    left=$(( (deadline_epoch - now) / 60 ))
    if [ "$left" -le 0 ]; then
        echo "$pod: EXPIRED ${left#-} minutes ago (watchdog should have fired)"
    else
        echo "$pod: $left minutes left of a $minutes-minute lease"
    fi
}

cmd_list() {
    found=0
    for f in "$LEASES"/*.lease; do
        [ -e "$f" ] || continue
        found=1
        # shellcheck disable=SC1090
        ( . "$f"
          now=$(date -u +%s)
          left=$(( (deadline_epoch - now) / 60 ))
          if [ "$left" -le 0 ]; then
              printf '%-16s EXPIRED %s min ago\n' "$pod" "${left#-}"
          else
              printf '%-16s %s min left\n' "$pod" "$left"
          fi )
    done
    [ "$found" -eq 1 ] || echo "no leases recorded in $LEASES"
}

cmd_reap() {
    force=""
    if [ "$1" = "--force" ]; then force="1"; shift; fi
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "reap needs RUNPOD_API_KEY in the environment." >&2
        echo "  This is the SECONDARY layer -- it reclaims the DISK of a pod" >&2
        echo "  whose container the on-pod watchdog already exited. GPU" >&2
        echo "  billing has already stopped without it." >&2
        exit 2
    fi
    for f in "$LEASES"/*.lease; do
        [ -e "$f" ] || continue
        # shellcheck disable=SC1090
        . "$f"
        now=$(date -u +%s)
        if [ -n "$force" ] || [ "$now" -ge "$deadline_epoch" ]; then
            echo "terminating $pod ..."
            curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$pod" \
                -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null \
                && { echo "  terminated"; rm -f "$f"; } \
                || echo "  FAILED -- terminate $pod by hand in the console"
        fi
    done
}

case "${1:-}" in
    arm)    shift; cmd_arm "$@" ;;
    extend) shift; cmd_extend "$@" ;;
    check)  shift; cmd_check "$@" ;;
    list)   shift; cmd_list "$@" ;;
    reap)   shift; cmd_reap "$@" ;;
    *)      usage ;;
esac
