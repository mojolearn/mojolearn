#!/bin/sh
# tools/bincache_leg.sh -- the Mac half of tools/bincache.py, called by the
# three leg runners (tools/gemm_remote_leg.sh, tools/do_extra_leg.sh,
# tools/hotaisle_leg.sh) only when MOJOLEARN_BINCACHE=1. DEFAULT OFF.
#
#   sh tools/bincache_leg.sh stage "<ssh flags+target>" "<image>"
#        after the source is unpacked on the box: detect the device arch
#        there, list bincache/v1/<arch>/<image slug>/ in R2, presign a GET for
#        every entry and a PUT for 64 inbox slots, and pipe the map INSIDE a
#        script over ssh stdin into /root/.mojolearn_bincache/urls.tsv (0600).
#   sh tools/bincache_leg.sh promote "<leg out>/remote/bincache"
#        after the fetch: server-side copy each inbox object the box uploaded
#        to its content address (refusing any row whose recorded fields do not
#        hash to its key or name a sabotage define), then delete the inbox copy.
#   sh tools/bincache_leg.sh selftest
#        a live R2 round trip under bincache-selftest/ (no box).
#
# CREDENTIALS: exactly tools/dataset_store.sh's rules. ~/.mojolearn_r2 is read
# here; its values reach python3 as a shell env prefix (never argv, never
# exported into this shell's wider environment); the box receives presigned
# URLs only, inside a piped script, so they appear in no process list, no leg
# archive and no log. The last line printed is a one-line summary for a
# runner's `tail -1`.
#
#   MOJOLEARN_BINCACHE_REMOTE_SH   the box shell (default `sh -s`; Hot Aisle
#                                  needs `sudo -n -H sh -s`)
# POSIX sh.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CREDS="${MOJOLEARN_R2_CREDS:-$HOME/.mojolearn_r2}"
REMOTE_SH="${MOJOLEARN_BINCACHE_REMOTE_SH:-sh -s}"

with_creds() {
    [ -f "$CREDS" ] || { echo "BINCACHE OFF: no $CREDS on this machine"; return 1; }
    (
        # shellcheck disable=SC1090
        . "$CREDS"
        R2_ACCOUNT_ID="${R2_ACCOUNT_ID:-}" R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-}" \
        R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-}" R2_BUCKET="${R2_BUCKET:-}" \
            python3 "$ROOT/tools/bincache.py" "$@"
    )
}

detect_script() {
    cat <<'EOF'
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$cc" in 9.0) echo sm_90a ;; 12.0) echo sm_120a ;; *) echo "sm_$(echo "$cc" | tr -d .)" ;; esac
elif command -v rocminfo > /dev/null 2>&1 && rocminfo > /dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+' || echo amd-unknown
else
    echo none
fi
EOF
}

cmd_stage() {
    target="${1:?usage: stage \"<ssh flags+target>\" <image>}"
    image="${2:?usage: stage \"<ssh flags+target>\" <image>}"
    # shellcheck disable=SC2086
    arch=$(detect_script | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        $target "$REMOTE_SH" 2>/dev/null | tail -1 | tr -cd 'A-Za-z0-9._-')
    [ -n "$arch" ] || arch=unknown
    slug=$(printf '%s' "$image" | tr -c 'A-Za-z0-9._-' '-' | cut -c1-100)
    leg="$(date -u +%Y%m%dT%H%M%SZ)-$(od -An -N4 -tx4 /dev/urandom | tr -d ' \n')"
    map=$(mktemp "${TMPDIR:-/tmp}/bincache-map.XXXXXX") || return 1
    chmod 600 "$map"
    # MOJOLEARN_BINCACHE_NEGATIVE=1 (tools/runpod_cpu_leg.sh) also lists the
    # sabotage namespace, as sget rows only a negative-control build reads.
    neg=""
    [ "${MOJOLEARN_BINCACHE_NEGATIVE:-0}" = 1 ] && neg="--negative"
    # shellcheck disable=SC2086
    if ! with_creds plan --partition "$arch/$slug" --image "$image" --leg-id "$leg" --slots "${MOJOLEARN_BINCACHE_SLOTS:-64}" $neg > "$map"; then
        rm -f "$map"
        echo "BINCACHE STAGING FAILED (plan); the body builds from source"
        return 1
    fi
    n=$(grep -c '^get' "$map")
    # shellcheck disable=SC2086
    if ! { printf 'umask 077\nrm -rf /root/.mojolearn_bincache\nmkdir -p /root/.mojolearn_bincache\n'
           printf "cat > /root/.mojolearn_bincache/urls.tsv <<'BINCACHE_MAP_EOF'\n"
           cat "$map"
           printf 'BINCACHE_MAP_EOF\necho BINCACHE_MAP_OK\n'
         } | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
             $target "$REMOTE_SH" | grep -q '^BINCACHE_MAP_OK'; then
        rm -f "$map"
        echo "BINCACHE STAGING FAILED (ssh); the body builds from source"
        return 1
    fi
    rm -f "$map"
    echo "BINCACHE STAGED partition=$arch/$slug leg=$leg entries=$n"
}

case "${1:-}" in
    stage)    shift; cmd_stage "$@" ;;
    promote)  shift; with_creds promote "$@" ;;
    selftest) shift; with_creds selftest-r2 ;;
    *) sed -n '2,30p' "$0"; exit 2 ;;
esac
