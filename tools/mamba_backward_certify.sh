#!/usr/bin/env bash
# Lightweight GPU certificate for strict public-prefill and state backward
# gates. This performs no training or timing and never provisions hardware.
set -u
set -o pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAMP=$(date -u +%Y-%m-%d_%H%M%S)
VENDOR=${MOJOLEARN_MAMBA_CERT_VENDOR:-nvidia}
case "$VENDOR" in
    apple|nvidia|amd) ;;
    *)
        echo "MAMBA-BACKWARD-CERT REFUSED: unsupported vendor '$VENDOR'" >&2
        exit 2 ;;
esac
OUT=${MOJOLEARN_MAMBA_CERT_OUT:-$ROOT/bench/results/mamba_backward_cert/${STAMP}-${VENDOR}}
mkdir -p "$OUT"
if [ -e "$OUT/results.tsv" ]; then
    echo "MAMBA-BACKWARD-CERT REFUSED: output already contains a certificate" >&2
    exit 2
fi

if [ "$VENDOR" = nvidia ]; then
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "MAMBA-BACKWARD-CERT REFUSED: nvidia-smi is unavailable" >&2
        exit 2
    }
    nvidia-smi --query-gpu=index,name,uuid,driver_version \
        --format=csv,noheader > "$OUT/device.csv" 2>/dev/null || {
        echo "MAMBA-BACKWARD-CERT REFUSED: NVIDIA device query failed" >&2
        exit 2
    }
elif [ "$VENDOR" = amd ]; then
    command -v rocm-smi >/dev/null 2>&1 || {
        echo "MAMBA-BACKWARD-CERT REFUSED: rocm-smi is unavailable" >&2
        exit 2
    }
    rocm-smi --showproductname > "$OUT/device.csv" 2>/dev/null || {
        echo "MAMBA-BACKWARD-CERT REFUSED: AMD device query failed" >&2
        exit 2
    }
else
    system_profiler SPDisplaysDataType > "$OUT/device.csv" 2>/dev/null || exit 2
    if ! grep -q 'Apple M' "$OUT/device.csv"; then
        echo "MAMBA-BACKWARD-CERT REFUSED: no Apple silicon GPU" >&2
        exit 2
    fi
fi
if [ ! -s "$OUT/device.csv" ]; then
    echo "MAMBA-BACKWARD-CERT REFUSED: no $VENDOR device was reported" >&2
    exit 2
fi
if [ "$VENDOR" = amd ] &&
   ! grep -Eqi 'card series|card model' "$OUT/device.csv"; then
    echo "MAMBA-BACKWARD-CERT REFUSED: rocm-smi reported no AMD card" >&2
    exit 2
fi

if [ -n "${MOJOLEARN_COMMIT:-}" ]; then
    COMMIT=$MOJOLEARN_COMMIT
elif [ -s "$ROOT/commit.txt" ]; then
    COMMIT=$(sed -n '1p' "$ROOT/commit.txt")
else
    COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
fi
printf '%s\n' "$COMMIT" > "$OUT/commit.txt"
SOURCE_SHA=$(python "$ROOT/tools/mamba_backward_identity.py" source "$ROOT") || exit 2
{
    echo "schema=mojolearn.mamba.backward-certificate.${VENDOR}.v2"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "kernel=$(uname -srm)"
    echo "mode=IDENTICAL"
    echo "vendor=$VENDOR"
    echo "mode_injector=tools/with_identical_mode.sh"
    echo "commit=$COMMIT"
    echo "source_sha256=$SOURCE_SHA"
} > "$OUT/environment.txt"

: > "$OUT/results.tsv"
printf 'family\tverdict\trc\toracle_manifest_sha256\tdump_manifest_sha256\n' \
    >> "$OUT/results.tsv"
RC=0

run_family() {
    family=$1
    task=$2
    oracle=$3
    actual=$4
    family_out="$OUT/$family"
    mkdir -p "$family_out"
    (cd "$ROOT" && pixi run -e skgpu "$task") \
        > >(tee "$family_out/gate.log") 2>&1
    gate_rc=$?
    if [ -f "$oracle/manifest.json" ]; then
        cp "$oracle/manifest.json" "$family_out/oracle_manifest.json"
        oracle_sha=$(sha256sum "$family_out/oracle_manifest.json" | awk '{print $1}')
    else
        oracle_sha=missing
    fi
    if [ -f "$actual/dump_manifest.json" ]; then
        cp "$actual/dump_manifest.json" "$family_out/dump_manifest.json"
        dump_sha=$(sha256sum "$family_out/dump_manifest.json" | awk '{print $1}')
    else
        dump_sha=missing
    fi
    # Preserve the native bytes even on a red semantic gate for localization.
    # The identity comparator refuses every certificate with a red row.
    python "$ROOT/tools/mamba_backward_identity.py" capture \
        "$actual" "$oracle" "$family_out/native" "$SOURCE_SHA" \
        > "$family_out/capture.log" 2>&1
    capture_rc=$?
    verdict=GREEN
    if [ "$gate_rc" -ne 0 ] || [ "$capture_rc" -ne 0 ]; then
        verdict=RED
        RC=1
        if [ "$gate_rc" -eq 0 ]; then gate_rc=$capture_rc; fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$family" "$verdict" "$gate_rc" "$oracle_sha" "$dump_sha" \
        >> "$OUT/results.tsv"
}

run_family mamba1 mamba-grad-m1-public \
    /tmp/mojolearn-mamba1-grad /tmp/mojolearn-mamba1-actual
run_family mamba2 mamba-grad-m2-public \
    /tmp/mojolearn-mamba2-grad /tmp/mojolearn-mamba2-actual
run_family mamba3 mamba-grad-m3-public \
    /tmp/mojolearn-mamba3-grad /tmp/mojolearn-mamba3-actual
run_family mamba2-l257 mamba-grad-m2-l257 \
    /tmp/mojolearn-mamba2-l257-grad /tmp/mojolearn-mamba2-l257-actual
run_family mamba2-state mamba-grad-m2-initial-state \
    /tmp/mojolearn-mamba2-state-grad /tmp/mojolearn-mamba2-state-actual

FINAL_SOURCE_SHA=$(python "$ROOT/tools/mamba_backward_identity.py" source "$ROOT") || RC=1
if [ "$FINAL_SOURCE_SHA" != "$SOURCE_SHA" ]; then
    echo "MAMBA-BACKWARD-CERT FAILED: source changed during certification" >&2
    echo "source_changed=1" >> "$OUT/environment.txt"
    RC=1
fi

(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS) || RC=1
cat "$OUT/environment.txt"
cat "$OUT/device.csv"
cat "$OUT/results.tsv"
echo "MAMBA-BACKWARD-CERT artifacts=$OUT"
exit "$RC"
