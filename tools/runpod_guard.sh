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
# WHAT WAS TRIED FIRST, AND WHY IT DOES NOT WORK ON RUNPOD
# ---------------------------------------------------------
# The first design was credential-free and looked right: a RunPod pod runs a
# `sleep`-shaped PID 1, so a detached `sleep N; kill 1` would exit the
# container at the deadline and stop GPU billing, with no API key needed and
# nothing on this end required to stay alive.
#
# **IT WAS TESTED ON A REAL POD ON 2026-08-23 AND IT DOES NOT STOP BILLING.**
# Measured on an RTX 4090 (83fcj1xbnychde, runpod/pytorch, $0.34/hr):
#
#   1. the watchdog armed and fired exactly on time -- ssh dropped at the
#      deadline with "Connection closed by remote host", so `kill 1` did
#      reach PID 1 and the container did exit;
#   2. **RunPod RESTARTED the container.** 30 seconds later the pod was
#      reachable again, PID 1 was a fresh `docker-init`, `status` was still
#      RUNNING and `uptime` had reset. Billing never paused.
#   3. and the watchdog was GONE, because the restart wiped the process --
#      so the box was then MORE exposed than an unguarded one, since the
#      lease file on this end still said it was protected.
#
# That last point is why this is recorded at length rather than quietly
# fixed. A guard that fires, gets undone, and leaves a lease file claiming
# protection is worse than no guard at all: it converts a checkable property
# into a belief, which is the same failure the numerics ledger exists to
# prevent. Two other bugs surfaced in the same test and are noted at their
# sites: `exec -a` is a bash extension and RunPod images link /bin/sh to
# dash, and the lease file's ssh target must be quoted because the file is
# `.`-sourced.
#
# PID 1 IS ALSO NOT WHAT THE FIRST DESIGN ASSUMED. On this image it is
# `docker-init` running an entrypoint, not `sleep infinity`. The earlier
# `sleep infinity` observation came from a pod created with an explicit
# `args`, so the premise held for that pod and not in general.
#
# THE ONLY THING THAT STOPS RUNPOD BILLING IS TERMINATING THE POD through
# the API. And because the orphan case is THIS MACHINE GOING AWAY, the call
# has to be made FROM THE POD. That needs an API key on the box, which is a
# real cost and is why `arm` now REFUSES rather than pretending:
#
#   - with RUNPOD_API_KEY in this environment, `arm` installs a watchdog
#     that sleeps to the deadline and then DELETEs the pod through the REST
#     API. That survives this machine disappearing, which is the whole
#     requirement.
#   - without it, `arm` REFUSES and says the box is unguarded. Do not use a
#     box that could not be armed; terminate it.
#
# LAYER 2, unchanged: `reap` terminates expired leases from this machine.
# It is a cleanup for when you are here, never the safety net.
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
    # PORTABLE `sh`, NOT BASH. The first version of this used
    # `exec -a mojolearn-lease-watchdog ...` to give the watchdog a
    # greppable name. `exec -a` is a BASH extension and RunPod's Ubuntu
    # images link /bin/sh to dash, which does not have it -- so the
    # watchdog never started, `pgrep` found nothing, and `arm` refused.
    # It refused CORRECTLY, which is the only reason this was a caught bug
    # rather than a silent one, but a guard that always refuses is a guard
    # that is never armed. Found by preparing to test it on a real pod;
    # it had never been run against one.
    #
    # So: no `exec -a`, no name matching. The watchdog is an ordinary
    # backgrounded `sh -c` and its PID is written to a file, which is both
    # portable and unambiguous -- `pgrep -f` on a `sleep` pattern would
    # also match this very ssh command line and any other lease on the box.
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "REFUSING to arm $pod: RUNPOD_API_KEY is not set." >&2
        echo "  The credential-free watchdog was TESTED and does not stop" >&2
        echo "  billing on RunPod -- kill 1 exits the container and RunPod" >&2
        echo "  RESTARTS it, wiping the watchdog and leaving the box less" >&2
        echo "  protected than before. See this file's header." >&2
        echo "  Self-termination through the API is the only mechanism that" >&2
        echo "  survives this machine going away, and it needs a key ON THE" >&2
        echo "  POD. Set RUNPOD_API_KEY and re-arm, or terminate the box:" >&2
        echo "    tools/runpod_guard.sh reap --force $pod" >&2
        exit 1
    fi

    # The key goes to the pod through the ssh COMMAND'S ENVIRONMENT, not
    # through a file and not through the argv of the watchdog, so it does
    # not sit in `ps` output for anything else on the box to read.
    pod_ssh "$target" "
        if [ -f /tmp/mojolearn-lease.pid ]; then
            kill \"\$(cat /tmp/mojolearn-lease.pid)\" 2>/dev/null || true
        fi
        umask 077
        cat > /tmp/mojolearn-lease.sh <<'WATCHDOG'
#!/bin/sh
# Installed by tools/runpod_guard.sh. Terminates THIS POD at the deadline.
sleep SECONDS_PLACEHOLDER
# v2 first (v1 is deprecated), v1 as a FALLBACK. Two endpoints because this
# is the only thing standing between an orphan and the bill, and an API
# deprecation that silently 404s would disarm every lease at once. Both were
# confirmed answering 200 with this account on 2026-08-23.
for u in \"https://api.runpod.io/v2/pods/POD_PLACEHOLDER\" \
         \"https://rest.runpod.io/v1/pods/POD_PLACEHOLDER\"; do
    code=\$(curl -s -o /tmp/mojolearn-lease.body -w '%{http_code}' \
            -X DELETE \"\$u\" -H \"Authorization: Bearer \$RUNPOD_API_KEY\")
    echo \"\$(date -u +%FT%TZ) DELETE \$u -> \$code\" >> /tmp/mojolearn-lease.out
    case \"\$code\" in 2*) exit 0 ;; esac
done
echo \"\$(date -u +%FT%TZ) ALL DELETE ENDPOINTS FAILED\" >> /tmp/mojolearn-lease.out
WATCHDOG
        sed -i \"s/SECONDS_PLACEHOLDER/$secs/g; s/POD_PLACEHOLDER/$pod/g\" \
            /tmp/mojolearn-lease.sh
        chmod 700 /tmp/mojolearn-lease.sh
        RUNPOD_API_KEY='$RUNPOD_API_KEY' nohup /tmp/mojolearn-lease.sh \
            >/tmp/mojolearn-lease.log 2>&1 &
        echo \$! > /tmp/mojolearn-lease.pid
        sleep 1
        if kill -0 \"\$(cat /tmp/mojolearn-lease.pid)\" 2>/dev/null; then
            echo \"WATCHDOG_ARMED pid=\$(cat /tmp/mojolearn-lease.pid) secs=$secs\"
        fi
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
        # EVERY VALUE IS QUOTED, because this file is `.`-sourced by `check`
        # and `list`. The ssh target contains spaces ("-p 24054 root@host"),
        # so an unquoted `target=` line makes the shell try to RUN `-p` as a
        # command -- measured: `line 2: 24054: command not found`. A lease
        # file that cannot be read is a lease nobody can check.
        echo "pod='$pod'"
        echo "target='$target'"
        echo "armed_utc='$(date -u -r "$now" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)'"
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
            curl -fsS -X DELETE "https://api.runpod.io/v2/pods/$pod" \
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
