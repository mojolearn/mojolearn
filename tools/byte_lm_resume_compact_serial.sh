#!/usr/bin/env bash
# ROOT ONLY, remote Linux; compact references are diagnostic, not admission.
# head64 BASELINE_BUNDLE NEW_OUT; resume128 BASELINE_BUNDLE NEW_OUT FOREIGN_HEAD_BUNDLE.
# No builds, installs, transport, rental or full-comparator changes.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ $(uname -s) == Linux && ( $# == 3 || $# == 4 ) ]] || exit 2
action=$1 baseline=$2 OUT=$3 foreign=${4:-}
case "$action:$#" in head64:3|resume128:4) ;; *) exit 2 ;; esac
vendor=${MOJOLEARN_BYTE_LM_EXPECT_VENDOR:?set cuda or hip}
arch=${MOJOLEARN_GPU_ARCHS:?set actual single GPU architecture}
case "$vendor:$arch" in cuda:sm_[0-9]*) guard=tools/nvidia_serial_guard.py ;; hip:gfx[0-9]*) guard=tools/amd_serial_guard.py ;; *) exit 2 ;; esac
[[ "$arch" != *[!A-Za-z0-9_]* ]] || exit 2
PY=${MOJOLEARN_PYTHON:?existing absolute Python executable required}
baseline_sha=${MOJOLEARN_BYTE_LM_BASELINE_HANDOFF_SHA256:?root-pinned handoff SHA required}
foreign_sha=${MOJOLEARN_BYTE_LM_FOREIGN_HANDOFF_SHA256:-}
checkpoint_sha=${MOJOLEARN_BYTE_LM_FOREIGN_SHA256:-}
[[ "$PY" = /* && -x "$PY" && "$baseline_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$baseline" = /* && -d "$baseline" && "$OUT" = /* && ! -e "$OUT" && ! -L "$OUT" ]] || exit 2
if [[ "$action" == resume128 ]]; then
    [[ "$foreign" = /* && -d "$foreign" && "$foreign_sha" =~ ^[0-9a-f]{64}$ && "$checkpoint_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
else
    [[ -z "$foreign_sha" && -z "$checkpoint_sha" ]] || exit 2
fi
seconds=${MOJOLEARN_BYTE_LM_RESUME_SECONDS:-2100}
[[ "$seconds" =~ ^[0-9]+$ ]] && ((seconds >= 120 && seconds <= 2400)) || exit 2
export MOJOLEARN_NUMERIC_MODE=identical CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$ROOT/python:$ROOT" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MAX_JOBS=2 MOJOLEARN_CPU_THREADS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2 MOJOLEARN_COMPILE_JOBS=2
umask 077
mkdir "$OUT"
deadline=$(($(date +%s) + seconds - 30))
active_guard=
cleanup() {
    local status=$?
    trap - EXIT
    trap '' TERM HUP INT
    if [[ -n "$active_guard" ]]; then
        kill -TERM "$active_guard" 2>/dev/null || true
        wait "$active_guard" 2>/dev/null || true
    fi
    (set -o noclobber; printf '%s\n' "$status" > "$OUT/exit_code")
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap 'exit 130' INT
: > "$OUT/results.tsv"
cp "$ROOT/tools/byte_lm_resume_compact_serial.sh" "$OUT/campaign-source.sh"
run() {
    local name=$1 cap=$2 remaining status
    shift 2
    remaining=$((deadline - $(date +%s)))
    if ((remaining < 15)); then
        printf '%s\t124\tSKIPPED_DEADLINE\n' "$name" >> "$OUT/results.tsv"
        return 124
    fi
    ((cap <= remaining)) || cap=$remaining
    printf '%q ' "$PY" "$guard" --seconds "$cap" --rss-gib 12 -- "$@" > "$OUT/$name.command.txt"
    printf '\n' >> "$OUT/$name.command.txt"
    "$PY" "$guard" --seconds "$cap" --rss-gib 12 -- "$@" > "$OUT/$name.log" 2>&1 &
    active_guard=$!
    status=0
    wait "$active_guard" || status=$?
    active_guard=
    printf '%s\t%s\n' "$name" "$status" >> "$OUT/results.tsv"
    printf '%s exit=%s\n' "$name" "$status"
    return "$status"
}
capture_receipt() {
    "$PY" tools/root_job_receipt.py --vendor "$vendor" --exit-code 0 --job-kind capture \
        --command-file "$OUT/$1.command.txt" --guard-log "$OUT/$1.log" \
        --result "$OUT/$1/summary.json" --output "$OUT/$1.receipt.json"
}
cat > "$OUT/compact_checks.py" <<'PY'
import os
from pathlib import Path
import sys

phase, repo_arg, base_arg, base_sha, out_arg, vendor, action, foreign_arg, foreign_sha, cp_sha = sys.argv[1:]
repo, out = Path(repo_arg), Path(out_arg)
sys.path.insert(0, str(repo / 'tools'))
from byte_lm_resume_handoff import (load_handoff, compatible_handoffs, compare_capture_hashes,
    compact_control, exclusive)
from byte_lm_state_compare import canonical, load_capture, parse, read, receipt, require, sha
from byte_lm_real_text_capture import source_inventory

base = load_handoff(Path(base_arg), base_sha, kind='baseline128', vendor=vendor)
reference = base['data']
require(sha(read(repo / 'tools/byte_lm_resume_handoff.py')) == reference['author_source_sha256'],
        'local handoff helper differs from root authoring source')
require(source_inventory() == reference['source'], 'numerical source differs from compact baseline')
binary = read(repo / 'python/mojolearn/identical/_mojolearn_byte_lm.so')
require(binary == read(base['directory'] / reference['binding_file']) and
        sha(binary) == reference['binding_sha256'], 'installed binding differs from retained baseline bytes')
head = None
if action == 'resume128':
    head = load_handoff(Path(foreign_arg), foreign_sha, kind='head64')
    require(head['data']['vendor'] != vendor and sha(head['checkpoint']) == cp_sha,
            'foreign vendor/checkpoint root pin differs')
    compatible_handoffs(base, head)

if phase == 'preflight':
    # Preserve offered compact witnesses and their original receipt-relative
    # files so fetched outputs carry the complete offered transfer chain.
    for label, bundle in (('baseline', base), ('foreign', head)):
        if bundle is None:
            continue
        for parent, _, files in os.walk(bundle['directory']):
            for name in files:
                path = Path(parent) / name
                exclusive(out / 'handoffs' / label / path.relative_to(bundle['directory']), read(path))
    if head is not None:
        exclusive(out / 'transferred-head64.checkpoint.json', head['checkpoint'])
    exclusive(out / 'preflight.json', canonical(dict(
        schema='mojolearn.byte-lm.compact-resume-preflight.v1', vendor=vendor, action=action,
        baseline_handoff_sha256=base_sha, foreign_handoff_sha256=foreign_sha or None,
        foreign_checkpoint_sha256=cp_sha or None, binding_sha256=sha(binary),
        baseline_summary_sha256=reference['summary_sha256'],
        foreign_summary_sha256=head['data']['summary_sha256'] if head is not None else None,
        campaign_source_sha256=sha(read(out / 'campaign-source.sh')),
        checks_source_sha256=sha(read(out / 'compact_checks.py')),
        identity_admitted=False, learning_admitted=False,
        boundary='Compact root handoff verified; remote hashes cannot admit a full baseline. '
                 'Fetch all new raw outputs and run final all-raw comparator locally.')))
else:
    before = parse(read(out / 'preflight.json'))
    require(before['baseline_handoff_sha256'] == base_sha and before['binding_sha256'] == sha(binary),
            'compact baseline changed during campaign')
    result = dict(schema='mojolearn.byte-lm.compact-resume-diagnostic.v1', phase=phase,
                  identity_admitted=False, learning_admitted=False,
                  boundary='Remote diagnostic only; final full raw baseline comparison remains mandatory')
    if head is not None:
        require(read(out / 'transferred-head64.checkpoint.json') == head['checkpoint'], 'transferred bytes changed')
    if phase == 'verify-head':
        leg = load_capture(out / 'head64', 'head64', vendor)
        compare_capture_hashes(leg, base)
        receipt(out / 'head64.receipt.json', leg['summary_sha256'], vendor)
        result['checkpoint_sha256'] = sha(leg['checkpoint'])
    elif phase in ('verify-resume', 'verify-control'):
        leg = load_capture(out / 'resume128', 'resume128', vendor)
        compare_capture_hashes(leg, base)
        require(leg['incoming'] == head['checkpoint'], 'resume did not consume actual foreign checkpoint')
        receipt(out / 'resume128.receipt.json', leg['summary_sha256'], vendor)
        result['terminal_checkpoint_sha256'] = sha(leg['checkpoint'])
        if phase == 'verify-control':
            control = load_capture(out / 'zero-moments65', 'zero-moments65', vendor)
            receipt(out / 'zero-moments65.receipt.json', control['summary_sha256'], vendor)
            result['control'] = compact_control(head, leg, control, base)
    else:
        raise ValueError('unknown compact check phase')
    exclusive(out / (phase + '.json'), canonical(result))
print('Compact checks complete; not a full-raw admission.')
PY
check_args=("$ROOT" "$baseline" "$baseline_sha" "$OUT" "$vendor" "$action" "$foreign" "$foreign_sha" "$checkpoint_sha")
run preflight 120 "$PY" "$OUT/compact_checks.py" preflight "${check_args[@]}"
if [[ "$action" == head64 ]]; then
    run head64 1200 "$PY" tools/byte_lm_real_text_capture.py --output "$OUT/head64" \
        --expected-vendor "$vendor" --steps 128 --action head64
    capture_receipt head64
    run verify-head 180 "$PY" "$OUT/compact_checks.py" verify-head "${check_args[@]}"
else
    run resume128 1200 "$PY" tools/byte_lm_real_text_capture.py --output "$OUT/resume128" \
        --expected-vendor "$vendor" --steps 128 --action resume128 \
        --resume-checkpoint "$OUT/transferred-head64.checkpoint.json"
    capture_receipt resume128
    run verify-resume 180 "$PY" "$OUT/compact_checks.py" verify-resume "${check_args[@]}"
    run zero-moments65 240 "$PY" tools/byte_lm_real_text_capture.py --output "$OUT/zero-moments65" \
        --expected-vendor "$vendor" --steps 128 --action zero-moments65 \
        --resume-checkpoint "$OUT/transferred-head64.checkpoint.json"
    capture_receipt zero-moments65
    run verify-control 240 "$PY" "$OUT/compact_checks.py" verify-control "${check_args[@]}"
fi
echo 'COMPACT DIAGNOSTIC COMPLETE. Fetch raw outputs; final local all-raw comparator is mandatory.'
