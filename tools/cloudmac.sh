#!/bin/bash
# tools/cloudmac.sh -- the AWS EC2 Macs that act as Apple stewards for the
# algorithm expansion (docs/lanes/ALGORITHM_EXPANSION_PLAN.md). Subagents
# never use the MacBook's CPU or GPU; every Apple build and Metal check runs
# on these hosts.
#
#   tools/cloudmac.sh list                 name, instance, ip
#   tools/cloudmac.sh ssh <name> [cmd]     a shell (or one command) on that Mac
#   tools/cloudmac.sh bootstrap <name>     disk, Xcode/Metal toolchain check, pixi,
#                                          bare repo + push of origin/main, pixi install
#   tools/cloudmac.sh push <name|all> <ref>...   laptop -> the Mac's bare repo
#   tools/cloudmac.sh steward <name> install|restart|status
#                                          the Apple steward as a launchd daemon
#                                          (Label mojolearn.steward): install/restart
#                                          drain the steward (its queue never moves),
#                                          push origin/main, move ~/mojolearn to it,
#                                          write the plist, bootstrap or kickstart -k
#   tools/cloudmac.sh stop-all             TERMINATE instances and RELEASE hosts
#                                          (hosts can only be released after 24 h)
#
# Local settings (AWS profile, SSH key) come from ~/.config/mojolearn/cloudmac.env,
# which is never committed; MOJOLEARN_CLOUDMAC_* in the environment override it.
# The hosts are in us-east-1d. This tool never allocates, launches or extends
# a Mac; it only manages hosts that already exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$HOME/.config/mojolearn/cloudmac.env" ] && . "$HOME/.config/mojolearn/cloudmac.env"
REG="${MOJOLEARN_CLOUDMAC_REG:-$HOME/mojolearn-evidence/cloudmacs.tsv}"   # name  instance  host  ip
KEY="${MOJOLEARN_CLOUDMAC_KEY:?set MOJOLEARN_CLOUDMAC_KEY (see ~/.config/mojolearn/cloudmac.env)}"
AWSP=(--profile "${MOJOLEARN_CLOUDMAC_PROFILE:?set MOJOLEARN_CLOUDMAC_PROFILE}" --region us-east-1)
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

die() { echo "cloudmac: $*" >&2; exit 1; }
row() { awk -v n="$1" '$1 == n' "$REG"; }
ip_of() { local r; r=$(row "$1"); [ -n "$r" ] || die "no cloud Mac named $1 in $REG"; awk '{print $4}' <<< "$r"; }
cm() { local ip; ip=$(ip_of "$1"); shift; ssh "${SSH_OPTS[@]}" "ec2-user@$ip" "$@"; }

case "${1:-}" in
list) column -t "$REG" ;;
ssh) n="${2:?name}"; shift 2; cm "$n" "$@" ;;
bootstrap)
    n="${2:?name}"
    echo "== $n: grow the APFS container to the full EBS volume"
    cm "$n" 'PDISK=$(diskutil list physical external | head -n1 | cut -d" " -f1);
             APFSCONT=$(diskutil list physical external | grep Apple_APFS | tr -s " " | cut -d" " -f8);
             yes | sudo diskutil repairDisk "$PDISK" >/dev/null 2>&1; sudo diskutil apfs resizeContainer "$APFSCONT" 0 >/dev/null 2>&1; df -h / | tail -1'
    echo "== $n: toolchain (must match the laptop: Xcode 26.6, Metal toolchain 17.6.109)"
    cm "$n" 'sw_vers -productVersion; sysctl -n machdep.cpu.brand_string; xcode-select -p; xcodebuild -version 2>&1 | head -1; xcrun -f metal 2>&1 | tail -1'
    echo "== $n: pixi"
    cm "$n" 'command -v pixi >/dev/null || [ -x ~/.pixi/bin/pixi ] || curl -fsSL https://pixi.sh/install.sh | bash >/dev/null 2>&1; ~/.pixi/bin/pixi --version'
    echo "== $n: bare repo (the laptop PUSHES here; the org disables deploy keys, so no GitHub credential lives on this Mac)"
    cm "$n" '[ -d ~/mojolearn.git ] || git init -q --bare ~/mojolearn.git; echo BARE_OK'
    "$0" push "$n" origin/main
    echo "== $n: worktree + pixi install"
    cm "$n" "[ -d ~/mojolearn/.git ] || git clone -q --no-hardlinks ~/mojolearn.git ~/mojolearn; cd ~/mojolearn && git fetch -q origin && git checkout -q --detach origin/main && ~/.pixi/bin/pixi install -e default >/tmp/pixi_install.log 2>&1 && echo PIXI_OK && git log -1 --format=%h"
    ;;
push)
    # push <name|all> <commit-ish>...: every ref lands on the Mac as refs/heads/main
    # (for origin/main) or refs/heads/<branch>; a bare sha lands as refs/steward/<sha>.
    n="${2:?name|all}"; shift 2; [ $# -gt 0 ] || die "push needs refs"
    names=$n; [ "$n" = all ] && names=$(awk '{print $1}' "$REG")
    for m in $names; do
        ip=$(ip_of "$m"); specs=()
        for r in "$@"; do
            sha=$(git -C "$ROOT" rev-parse --verify "$r^{commit}") || die "no commit $r"
            case "$r" in origin/*) specs+=("$sha:refs/heads/${r#origin/}") ;;
                         [0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) specs+=("$sha:refs/steward/$sha") ;;
                         *) specs+=("$sha:refs/heads/$r") ;; esac
        done
        GIT_SSH_COMMAND="ssh ${SSH_OPTS[*]}" git -C "$ROOT" push -q -f "ec2-user@$ip:mojolearn.git" "${specs[@]}" && echo "$m: pushed $*"
    done
    ;;
steward)
    # A steward started by nohup or screen dies when macOS sshd closes the
    # session, so it runs as a system launchd daemon as ec2-user, KeepAlive,
    # under caffeinate, logging to ~/mojolearn-evidence/steward-<name>.log.
    # A Mac listed in MOJOLEARN_STEWARD_DEFERRED (default: none. m3ultra was deferred while busy with a
    # GPT-3 segment) is refused: it must not be contacted until it is free.
    n="${2:?name}"; act="${3:?install|restart|status}"
    case ",${MOJOLEARN_STEWARD_DEFERRED-}," in *",$n,"*)
        die "$n is deferred (MOJOLEARN_STEWARD_DEFERRED); clear it once the Mac is free" ;; esac
    case "$act" in
    status)
        cm "$n" 'sudo launchctl print system/mojolearn.steward 2>/dev/null | grep -E "state =|pid =|last exit" || echo "mojolearn.steward: not loaded";
                 cd ~/mojolearn && echo "clone at $(git log -1 --format=%h)"; [ -f ~/mojolearn-evidence/apple-steward/drain ] && echo DRAINING; tail -5 ~/mojolearn-evidence/steward-'"$n"'.log 2>/dev/null' ;;
    install|restart)
        # Never kill a steward mid-request (its request would be stranded in
        # working/) and never move its queue (requests outside queue/ vanish
        # from `apple_steward.py status` and from coalescing, and lanes
        # resubmit them as duplicates). A steward that knows the drain file
        # DRAINS: it claims nothing new and writes `drained` once its request
        # is done. An older one is booted out the first moment nothing runs.
        # The wait is a laptop-side poll; an interrupted restart removes the drain.
        # (Under zsh nullglob an empty `ls glob` lists the cwd and succeeds:
        # count the matches instead.)
        SQ='~/mojolearn-evidence/apple-steward'
        if cm "$n" "sudo launchctl print system/mojolearn.steward >/dev/null 2>&1" 2>/dev/null; then
            if cm "$n" "grep -q DRAIN_FILE ~/mojolearn/tools/apple_steward.py"; then
                trap 'cm "$n" "rm -f $SQ/drain" || true' EXIT
                cm "$n" "mkdir -p $SQ; rm -f $SQ/drained; touch $SQ/drain"
                echo "$n: draining (the queue stays visible); waiting for the running request"
                until cm "$n" "test -f $SQ/drained" 2>/dev/null; do sleep 20; done
            else
                echo "$n: the steward predates the drain file: booting it out the first moment nothing runs"
                until cm "$n" "setopt nullglob 2>/dev/null || true; set -- $SQ/working/[0-9]*.json; [ \$# -gt 0 ] || {
                        sudo launchctl bootout system/mojolearn.steward 2>/dev/null
                        for f in $SQ/working/[0-9]*.json; do [ -f \"\$f\" ] || continue; b=\$(basename \"\$f\"); mv \"\$f\" $SQ/queue/\${b%%.*}.json; echo \"requeued \$b\"; done
                        echo STOPPED; }" 2>/dev/null | tee /dev/stderr | grep -qx STOPPED; do sleep 5; done
            fi
        fi
        git -C "$ROOT" fetch -q origin main
        "$0" push "$n" origin/main
        cm "$n" 'set -e; cd ~/mojolearn && git fetch -q origin && git checkout -q --detach origin/main && echo "clone at $(git log -1 --format=%h)"'
        plist=$(cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>mojolearn.steward</string>
<key>UserName</key><string>ec2-user</string>
<key>WorkingDirectory</key><string>/Users/ec2-user/mojolearn</string>
<key>EnvironmentVariables</key><dict><key>HOME</key><string>/Users/ec2-user</string><key>PATH</key><string>/Users/ec2-user/.pixi/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
<key>ProgramArguments</key><array><string>/usr/bin/caffeinate</string><string>-i</string><string>/usr/bin/python3</string><string>tools/apple_steward.py</string><string>work</string><string>--steward</string><string>$n</string></array>
<key>KeepAlive</key><true/><key>RunAtLoad</key><true/>
<key>StandardOutPath</key><string>/Users/ec2-user/mojolearn-evidence/steward-$n.log</string>
<key>StandardErrorPath</key><string>/Users/ec2-user/mojolearn-evidence/steward-$n.log</string>
</dict></plist>
PLIST
)
        printf '%s\n' "$plist" | cm "$n" 'set -e; mkdir -p ~/mojolearn-evidence; cat > /tmp/mojolearn.steward.plist; plutil -lint /tmp/mojolearn.steward.plist >/dev/null;
            P=/Library/LaunchDaemons/mojolearn.steward.plist
            if [ -f $P ] && cmp -s /tmp/mojolearn.steward.plist $P && sudo launchctl print system/mojolearn.steward >/dev/null 2>&1; then
                sudo launchctl kickstart -k system/mojolearn.steward && echo "kickstarted (plist unchanged)"
            else
                sudo launchctl bootout system/mojolearn.steward 2>/dev/null || true
                sudo install -m 644 -o root -g wheel /tmp/mojolearn.steward.plist $P
                sudo launchctl bootstrap system $P && echo "bootstrapped $P"
            fi
            rm -f ~/mojolearn-evidence/apple-steward/drain
            sleep 3; sudo launchctl print system/mojolearn.steward | grep -E "state =|pid =" '
        trap - EXIT ;;
    *) die "steward <name> install|restart|status" ;;
    esac
    ;;
stop-all)
    while read -r n inst host ip; do
        [ -n "$n" ] || continue
        aws ec2 terminate-instances "${AWSP[@]}" --instance-ids "$inst" --query 'TerminatingInstances[0].CurrentState.Name' --output text
        echo "$n: release host $host after termination completes and 24 h have passed:"
        echo "  aws ec2 release-hosts --profile $MOJOLEARN_CLOUDMAC_PROFILE --region us-east-1 --host-ids $host"
    done < "$REG"
    ;;
*) sed -n 2,19p "$0"; exit 2 ;;
esac
