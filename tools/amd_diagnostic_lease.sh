#!/usr/bin/env bash
# Installed-wheel diagnostic only. Provider operations stay in the shared lease library.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[[ $# == 3 ]] || { echo 'usage: amd_diagnostic_lease.sh BUNDLE OUT rent|dry-run' >&2; exit 2; }
BUNDLE=$1 OUT=$2 MODE=$3
[[ "$MODE" == rent || "$MODE" == dry-run ]] || exit 2
[[ -f "$BUNDLE" && ! -e "$OUT" ]] || exit 2
mkdir -p "$OUT"
TMPD=$(mktemp -d)
say() { printf '[amd-diagnostic] %s\n' "$*"; }
die() { say "$*" >&2; exit 1; }
. "$ROOT/tools/runpod_pod_lib.sh"
. "$ROOT/tools/hotaisle_vm_lib.sh"
HA_SPEC_WANT=1gpu
REMOTE=/root/mojolearn-amd-diagnostic
FETCH_READY=0
finish() {
    rc=$?
    trap - EXIT INT TERM HUP
    if [[ "$FETCH_READY" == 1 ]]; then
        # Fetch even after a failed setup, timeout or diagnostic refusal.
        if ha_ssh 120 "tar czf - -C $REMOTE results" > "$OUT/results.tgz" 2> "$OUT/fetch.stderr"; then
            tar xzf "$OUT/results.tgz" -C "$OUT" || rc=1
        else
            rc=1
        fi
    fi
    if ! ha_teardown; then rc=1; fi
    ha_spend >> "$OUT/provider.txt"
    printf 'exit=%s\ndestroy_confirmed=%s\n' "$rc" "$HA_GONE" > "$OUT/teardown.txt"
    # The independent dead-man retains its own credentials on uncertainty.
    rm -rf "$TMPD"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if [[ "$MODE" == dry-run ]]; then
    ha_write_deadman "$TMPD/check" "$(( $(date +%s) + 3000 ))" "$OUT/provider.txt"
    ha_write_watchdog "$TMPD/watchdog.sh" "$REMOTE-guard" 1800 DRYRUN_REF
    say 'DRY RUN: 1 MI300X; 30-minute lease, 50-minute priced horizon, $3 cap; 1200-second diagnostic, 2 CPU cores, 12 GiB RSS; no resource created'
    exit 0
fi
ha_rent amd-oob-diagnostic 30 300 "$OUT/provider.txt" "$REMOTE-guard" "$OUT" || die "Lease refused: $HA_REFUSED"
ha_ssh 60 "mkdir -p $REMOTE/results" </dev/null
FETCH_READY=1
sha=$(shasum -a 256 "$BUNDLE" | awk '{print $1}')
ha_ssh 300 "cat > $REMOTE/bundle.tgz" < "$BUNDLE"
ha_ssh 60 "cd $REMOTE && echo '$sha  bundle.tgz' | sha256sum -c - && tar xzf bundle.tgz && sha256sum -c SHA256SUMS" </dev/null > "$OUT/staging.log" 2>&1
ha_ssh 1500 "cd $REMOTE && timeout -k 20 1450 bash run.sh > results/run.log 2>&1" </dev/null
