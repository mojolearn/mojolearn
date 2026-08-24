#!/bin/sh
# EVERY RENTED GPU EXPIRES ON ITS OWN. This is the code that makes that true.
#
#   tools/runpod_guard.sh arm   <pod-id> <ssh-target> [minutes]   # default 60
#   tools/runpod_guard.sh check <pod-id>
#   tools/runpod_guard.sh list
#   tools/runpod_guard.sh extend <pod-id> <ssh-target> [minutes]
#
#   tools/runpod_guard.sh reap                    every EXPIRED lease
#   tools/runpod_guard.sh reap <pod-id>           that one, if EXPIRED
#   tools/runpod_guard.sh reap --force <pod-id>   that one, NOW
#   tools/runpod_guard.sh reap --force --all      every lease, NOW
#   tools/runpod_guard.sh reap --dry-run ...      print the selection only
#                                                 (reap needs a key; --dry-run
#                                                  does not, it calls nothing)
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
#     THE CREDENTIAL REACHES THE POD ON SSH STDIN, into a 0600 curl config
#     at /tmp/mojolearn-lease.curlrc, and the watchdog authenticates with
#     `curl -K`. It is in no argv on either machine at any point. That file
#     stays on the pod for the life of the lease because the watchdog needs
#     it; it dies with the pod.
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

GUARD_CURLRC=""

guard_curlrc() {
    # THE KEY GOES INTO A 0600 FILE AND NEVER INTO AN ARGV. `printf` is a
    # SHELL BUILTIN here, so composing the line spawns no process and there
    # is no command line for it to leak into; `curl -K` then reads the
    # Authorization header out of the file. `-H "Authorization: Bearer $KEY"`
    # is the spelling this replaces and it put the key in curl's own argv,
    # readable by anything that could run `ps` while the call was in flight.
    # tools/gemm_remote_leg.sh does exactly this at the same seam.
    GUARD_CURLRC="${TMPDIR:-/tmp}/mojolearn-guard-$$.curlrc"
    ( umask 077
      printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' \
          "$RUNPOD_API_KEY" > "$GUARD_CURLRC" )
    trap 'rm -f "$GUARD_CURLRC"' EXIT INT TERM
}

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
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

    # THE CREDENTIAL GOES OVER STDIN, INTO A 0600 CURL CONFIG, BEFORE ANY
    # COMMAND STRING IS BUILT. It is never in an argv on either machine.
    #
    # What this replaces, and why it was wrong: the arm used to interpolate
    # `RUNPOD_API_KEY='$RUNPOD_API_KEY'` into the ssh command string. The
    # WATCHDOG was clean -- the key was in its environment, not its argv --
    # and the comment that sat here said exactly that and was true of that
    # process. But sshd runs the whole command string through `sh -c`, so for
    # the second or so that arm ran THE KEY WAS IN THE ARGV OF THAT REMOTE
    # SHELL, readable by anything on the box that could run `ps`; and `ssh`
    # on THIS machine had the same string in its own argv. Two process lists,
    # one interpolation. Closed 2026-08-24.
    #
    # `printf` is a shell BUILTIN, so composing the config spawns no process
    # here either. The watchdog reads the header with `curl -K`. Do NOT
    # "simplify" that to `-H "Authorization: Bearer $(cat ...)"`: that moves
    # the leak rather than closing it, into curl's argv on the pod at the
    # moment the lease fires.
    printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' \
        "$RUNPOD_API_KEY" \
        | pod_ssh "$target" '
            umask 077
            cat > /tmp/mojolearn-lease.curlrc
            chmod 600 /tmp/mojolearn-lease.curlrc
            if [ -s /tmp/mojolearn-lease.curlrc ]; then echo CREDENTIAL_ON_POD; fi
        ' > /tmp/mojolearn_cred_out 2>&1 || true
    if ! grep -q CREDENTIAL_ON_POD /tmp/mojolearn_cred_out; then
        echo "REFUSING: the credential did not land on $pod." >&2
        echo "  The watchdog terminates this pod through the API and cannot" >&2
        echo "  do it without one, so arming now would install a watchdog" >&2
        echo "  that fires and fails -- the worst kind, because the lease" >&2
        echo "  file would say the box is guarded." >&2
        echo "  ssh output:" >&2
        sed 's/^/    /' /tmp/mojolearn_cred_out >&2
        exit 1
    fi

    pod_ssh "$target" "
        if [ ! -s /tmp/mojolearn-lease.curlrc ]; then
            echo NO_CREDENTIAL_ON_POD
            exit 3
        fi
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
            -X DELETE \"\$u\" -K /tmp/mojolearn-lease.curlrc)
    echo \"\$(date -u +%FT%TZ) DELETE \$u -> \$code\" >> /tmp/mojolearn-lease.out
    case \"\$code\" in 2*) exit 0 ;; esac
done
echo \"\$(date -u +%FT%TZ) ALL DELETE ENDPOINTS FAILED\" >> /tmp/mojolearn-lease.out
WATCHDOG
        sed -i \"s/SECONDS_PLACEHOLDER/$secs/g; s/POD_PLACEHOLDER/$pod/g\" \
            /tmp/mojolearn-lease.sh
        chmod 700 /tmp/mojolearn-lease.sh
        nohup /tmp/mojolearn-lease.sh >/tmp/mojolearn-lease.log 2>&1 &
        echo \$! > /tmp/mojolearn-lease.pid
        sleep 1
        if kill -0 \"\$(cat /tmp/mojolearn-lease.pid)\" 2>/dev/null; then
            echo \"WATCHDOG_ARMED pid=\$(cat /tmp/mojolearn-lease.pid) secs=$secs\"
        fi
    " > /tmp/mojolearn_arm_out 2>&1 || true

    if grep -q NO_CREDENTIAL_ON_POD /tmp/mojolearn_arm_out; then
        echo "REFUSING: the credential was gone from $pod by the time the" >&2
        echo "  watchdog was installed. Something removed" >&2
        echo "  /tmp/mojolearn-lease.curlrc between the two ssh calls." >&2
        exit 1
    fi
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

# THE POD ID SCOPES THE REAP. Until 2026-08-24 it did not: `cmd_reap` shifted
# `--force` off, which shifted the pod argument off WITH it, and the loop that
# followed terminated EVERY lease in the directory. So
# `reap --force <pod-id>` READ like a targeted terminate and was a
# terminate-everything -- harmless under a one-box-at-a-time discipline, and a
# way to destroy another lane's box on the first day two are up. Found by
# reading this file against tools/gemm_remote_leg.sh, which is why that leg
# issues its own targeted DELETE.
#
# The reap-everything form still exists, because ending every lease at once is
# a real thing to want after a bad night. It has to be asked for BY NAME:
# `reap --force --all`. `reap --force` alone is REFUSED.
reap_one() {
    _pod="$1"; _f="$2"
    if [ -n "$dry" ]; then
        if [ -n "$_f" ]; then
            echo "WOULD TERMINATE $_pod (lease $_f)   [--dry-run: nothing was called]"
        else
            echo "WOULD TERMINATE $_pod (no lease file)   [--dry-run: nothing was called]"
        fi
        return 0
    fi
    echo "terminating $_pod ..."
    # NOT `... && { echo; rm -f "$_f"; } || echo FAILED`. When $_f is empty
    # the `rm` branch of that spelling returns non-zero and the `||` fires,
    # so a SUCCESSFUL terminate reports FAILED. A pipeline's status is its
    # last command's and an && chain's is the last thing that ran.
    if curl -fsS -X DELETE "https://api.runpod.io/v2/pods/$_pod" \
            -K "$GUARD_CURLRC" >/dev/null; then
        echo "  terminated"
        if [ -n "$_f" ]; then rm -f "$_f"; fi
    else
        echo "  FAILED -- terminate $_pod by hand in the console"
    fi
}

cmd_reap() {
    force=""; all=""; only=""; dry=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --force)   force="1" ;;
            --all)     all="1" ;;
            --dry-run) dry="1" ;;
            -*)
                echo "reap: unknown option '$1'." >&2
                echo "  reap                    every EXPIRED lease" >&2
                echo "  reap <pod-id>           that one, if EXPIRED" >&2
                echo "  reap --force <pod-id>   that one, NOW" >&2
                echo "  reap --force --all      every lease, NOW" >&2
                echo "  reap --dry-run ...      print the selection, call nothing" >&2
                exit 2 ;;
            *)
                if [ -n "$only" ]; then
                    echo "reap: two pod ids given ('$only' then '$1')." >&2
                    echo "  One reap is one pod. Run it twice, or say --all." >&2
                    exit 2
                fi
                only="$1" ;;
        esac
        shift
    done

    if [ -n "$all" ] && [ -n "$only" ]; then
        echo "reap: --all and a pod id ('$only') contradict each other." >&2
        echo "  --all is every lease; a pod id is one. Pick one of them." >&2
        exit 2
    fi
    if [ -n "$force" ] && [ -z "$only" ] && [ -z "$all" ]; then
        echo "REFUSING: 'reap --force' with no pod id would terminate EVERY" >&2
        echo "  lease in $LEASES, expired or not -- including another lane's" >&2
        echo "  box, which is not yours to end. It reads like a targeted" >&2
        echo "  terminate, so it is refused rather than obeyed." >&2
        echo "    tools/runpod_guard.sh reap --force <pod-id>   one, now" >&2
        echo "    tools/runpod_guard.sh reap --force --all      all, now" >&2
        echo "    tools/runpod_guard.sh reap --force --all --dry-run" >&2
        echo "                                                  see it first" >&2
        exit 2
    fi

    # --dry-run calls nothing, so it needs no credential. Every other form
    # does, and it is checked here rather than at the curl so that a keyless
    # reap costs nothing and touches no lease file.
    if [ -z "$dry" ] && [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "reap needs RUNPOD_API_KEY in the environment." >&2
        echo "  This is the SECONDARY layer -- it reclaims the DISK of a pod" >&2
        echo "  whose container the on-pod watchdog already exited. GPU" >&2
        echo "  billing has already stopped without it." >&2
        exit 2
    fi
    if [ -z "$dry" ]; then guard_curlrc; fi

    matched=0
    for f in "$LEASES"/*.lease; do
        [ -e "$f" ] || continue
        # EVERY VARIABLE IS CLEARED BEFORE THE SOURCE. This file is
        # `.`-sourced into THIS shell (the loop needs $pod afterwards), so a
        # lease that fails to set `pod` would otherwise inherit the PREVIOUS
        # iteration's value -- and this loop DELETES what $pod names. A
        # truncated lease file would have terminated the pod before it.
        pod=""; deadline_epoch=""; target=""; minutes=""
        # shellcheck disable=SC1090
        . "$f"
        if [ -z "$pod" ] || [ -z "$deadline_epoch" ]; then
            echo "SKIPPING unreadable lease $f (it names no pod or no deadline)"
            continue
        fi
        if [ -n "$only" ] && [ "$pod" != "$only" ]; then
            continue
        fi
        if [ -n "$only" ]; then matched=1; fi
        now=$(date -u +%s)
        if [ -n "$force" ] || [ "$now" -ge "$deadline_epoch" ]; then
            reap_one "$pod" "$f"
        else
            echo "leaving $pod alone: $(( (deadline_epoch - now) / 60 )) minute(s) left"
            echo "  (--force ends an unexpired lease early; it is a decision)"
        fi
    done

    if [ -n "$only" ] && [ "$matched" = "0" ]; then
        # A NAMED POD WITH NO LEASE FILE IS THE ORPHAN CASE ITSELF: a box
        # that was created and never armed has no lease here and is exactly
        # the one that is still billing. Silence would be the worst answer,
        # so say so, and terminate it when it was named with --force.
        if [ -n "$force" ]; then
            echo "no lease file for $only -- terminating it BY ID anyway,"
            echo "  because you named it. A pod with no lease is the orphan"
            echo "  case: created, never armed, still billing."
            reap_one "$only" ""
        else
            echo "no lease file for $only. It may never have been armed." >&2
            echo "  If it exists and is billing, end it now:" >&2
            echo "    tools/runpod_guard.sh reap --force $only" >&2
            exit 1
        fi
    fi
}

case "${1:-}" in
    arm)    shift; cmd_arm "$@" ;;
    extend) shift; cmd_extend "$@" ;;
    check)  shift; cmd_check "$@" ;;
    list)   shift; cmd_list "$@" ;;
    reap)   shift; cmd_reap "$@" ;;
    *)      usage ;;
esac
