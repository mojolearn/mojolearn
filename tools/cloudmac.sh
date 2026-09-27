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
#   tools/cloudmac.sh stop-all             TERMINATE instances and RELEASE hosts
#                                          (hosts can only be released after 24 h)
#
# The hosts live in the mambik AWS account (profile `mambik`), us-east-1d.
# Its org policy refuses RunInstances without `lane` and `owner` tags.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REG="${MOJOLEARN_CLOUDMAC_REG:-$HOME/mojolearn-evidence/cloudmacs.tsv}"   # name  instance  host  ip
KEY="${MOJOLEARN_CLOUDMAC_KEY:-$HOME/.ssh/mambik-l8.pem}"
AWSP=(--profile "${MOJOLEARN_CLOUDMAC_PROFILE:-mambik}" --region us-east-1)
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
    cm "$n" "[ -d ~/mojolearn/.git ] || git clone -q ~/mojolearn.git ~/mojolearn; cd ~/mojolearn && git fetch -q origin && git checkout -q --detach origin/main && ~/.pixi/bin/pixi install -e default >/tmp/pixi_install.log 2>&1 && echo PIXI_OK && git log -1 --format=%h"
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
stop-all)
    while read -r n inst host ip; do
        [ -n "$n" ] || continue
        aws ec2 terminate-instances "${AWSP[@]}" --instance-ids "$inst" --query 'TerminatingInstances[0].CurrentState.Name' --output text
        echo "$n: release host $host after termination completes and 24 h have passed:"
        echo "  aws ec2 release-hosts --profile mambik --region us-east-1 --host-ids $host"
    done < "$REG"
    ;;
*) sed -n 2,14p "$0"; exit 2 ;;
esac
