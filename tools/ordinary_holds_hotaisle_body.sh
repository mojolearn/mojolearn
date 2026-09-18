#!/bin/sh
# No provisioning: entry point executed by hotaisle_leg.sh's guarded VM.
set -eu
cd /root/mojolearn
if [ ! -s commit.txt ]; then
    commit=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt | head -1)
    case "$commit" in
        ''|*[!0-9a-f]*) echo 'Missing archived source provenance' >&2; exit 2 ;;
    esac
    [ "${#commit}" = 40 ] || { echo 'Invalid commit witness length' >&2; exit 2; }
    printf '%s\n' "$commit" > commit.txt
fi
exec bash tools/ordinary_holds_gpu_leg.sh hip amd-mi300x-gfx942 \
    /root/gemm_leg_out/ordinary-holds 2400
