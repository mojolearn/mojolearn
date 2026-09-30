#!/usr/bin/env bash
# ROOT ONLY, on an already available remote Linux CUDA/HIP host. This helper
# never installs, builds, rents, transports, or selects a GPU architecture.
# Usage: bash tools/byte_lm_resume_serial.sh head64 BASELINE_CAMPAIGN NEW_OUT
#        bash tools/byte_lm_resume_serial.sh resume128 BASELINE_CAMPAIGN NEW_OUT FOREIGN_HEAD_CAMPAIGN
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ $(uname -s) == Linux ]] || { echo 'Remote Linux only' >&2; exit 2; }
[[ $# == 3 || $# == 4 ]] || { echo 'Expected action, baseline campaign, new output, optional foreign head campaign' >&2; exit 2; }
action=$1
baseline=$2
OUT=$3
foreign=${4:-}
case "$action:$#" in head64:3|resume128:4) ;; *) echo 'head64 requires3 arguments; resume128 requires4' >&2; exit 2 ;; esac
vendor=${MOJOLEARN_BYTE_LM_EXPECT_VENDOR:?set cuda or hip}
arch=${MOJOLEARN_GPU_ARCHS:?set the actual single rented GPU architecture}
case "$vendor:$arch" in
    cuda:sm_[0-9]*) guard=tools/nvidia_serial_guard.py ;;
    hip:gfx[0-9]*) guard=tools/amd_serial_guard.py ;;
    *) echo 'Expected explicit cuda/sm_NN or hip/gfxNNN' >&2; exit 2 ;;
esac
[[ "$arch" != *[!A-Za-z0-9_]* ]] || exit 2
PY=${MOJOLEARN_PYTHON:?set the existing baseline Python executable; no environment is created}
[[ "$PY" = /* && -x "$PY" ]] || { echo 'Existing absolute Python executable required' >&2; exit 2; }
[[ "$OUT" = /* && ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'New absolute output directory required' >&2; exit 2; }
[[ "$baseline" = /* && -d "$baseline" && ! -L "$baseline" ]] || exit 2
expected_foreign_sha=${MOJOLEARN_BYTE_LM_FOREIGN_SHA256:-}
if [[ "$action" == resume128 ]]; then
    [[ "$foreign" = /* && -d "$foreign" && ! -L "$foreign" ]] || exit 2
    [[ "$expected_foreign_sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'Root-pinned MOJOLEARN_BYTE_LM_FOREIGN_SHA256 required' >&2; exit 2; }
else
    [[ -z "$expected_foreign_sha" ]] || { echo 'Foreign checkpoint SHA applies only to resume128' >&2; exit 2; }
fi
seconds=${MOJOLEARN_BYTE_LM_RESUME_SECONDS:-2100}
[[ "$seconds" =~ ^[0-9]+$ ]] && ((seconds >= 120 && seconds <= 2400)) || exit 2
export MOJOLEARN_NUMERIC_MODE=identical CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_GPU_ARCHS="$arch"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2 MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_CPU_THREADS=2
umask 077
mkdir "$OUT"
deadline=$(($(date +%s) + seconds - 30))
active_guard=
cleanup() {
    local status=$?
    trap - EXIT
    trap '' TERM HUP INT
    if [[ -n "$active_guard" ]]; then
        # The guard owns the model's separate process group. Ask the guard to
        # tear it down, then wait; do not kill the supervisor and orphan it.
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
printf '%s\n' "action=$action" "vendor=$vendor" "gpu_arch=$arch" "baseline=$baseline" \
    "foreign_head=$foreign" "expected_foreign_checkpoint_sha256=$expected_foreign_sha" \
    'scope=retained fixed byte-LM continuation; qualification requires final file comparator' \
    > "$OUT/provenance.txt"
cp "$ROOT/tools/byte_lm_resume_serial.sh" "$OUT/campaign-source.sh"

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

write_receipt() {
    local name=$1
    # Called only after the actual guard returned zero, including teardown.
    "$PY" tools/root_job_receipt.py --vendor "$vendor" --exit-code 0 --job-kind capture \
        --command-file "$OUT/$name.command.txt" --guard-log "$OUT/$name.log" \
        --result "$OUT/$name/summary.json" --output "$OUT/$name.receipt.json"
}

# Retain the file-only pre/postcheck program in the campaign itself. Imports
# below are stdlib-only at module load; none calls a model or loads a binding.
cat > "$OUT/artifact_checks.py" <<'PY'
import json
import os
from pathlib import Path
import stat
import sys

phase, repo_arg, baseline_arg, out_arg, vendor, action, foreign_arg, expected_sha = sys.argv[1:]
repo, baseline, out = map(Path, (repo_arg, baseline_arg, out_arg))
sys.path.insert(0, str(repo / 'tools'))
from byte_lm_state_compare import (canonical, compatible, load_capture, parse, read,
    receipt, require, resume_control, same_steps, sha)
from byte_lm_real_text_capture import source_inventory
from byte_lm_validation_admit import admit


def exclusive(path, raw, readonly=False):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if readonly:
            os.fchmod(fd, 0o444)
    finally:
        os.close(fd)


base = load_capture(baseline / 'full128', 'continuous', vendor)
require(source_inventory() == base['source'], 'current numerical source differs from continuous baseline')
installed = repo / 'python/mojolearn/identical/_mojolearn_byte_lm.so'
installed_raw = read(installed)
retained_raw = read(baseline / 'bindings/_mojolearn_byte_lm.so')
require(installed_raw == retained_raw and sha(installed_raw) == base['runtime']['binding_sha256'],
        'installed/retained binding differs from continuous baseline; rebuilding is not allowed here')
head = None
if action == 'resume128':
    foreign = Path(foreign_arg)
    require(read(foreign / 'exit_code').strip() == b'0', 'foreign head campaign did not exit successfully')
    head = load_capture(foreign / 'head64', 'head64')
    require(head['runtime']['native_vendor'] != vendor, 'checkpoint must originate on the other vendor')
    receipt(foreign / 'head64.receipt.json', head['summary_sha256'], head['runtime']['native_vendor'])
    compatible(base, head)
    require(same_steps(base, head) == 64, 'foreign first64 steps differ from local continuous baseline')
    require(sha(head['checkpoint']) == expected_sha, 'foreign checkpoint differs from root-pinned transfer SHA')

if phase == 'preflight':
    admitted = admit(baseline)
    require(admitted['passed'] is True and admitted['vendor'] == vendor,
            'continuous baseline lacks single-vendor admission')
    exclusive(out / 'baseline-admission.json', canonical(admitted))
    (out / 'bindings').mkdir()
    exclusive(out / 'bindings/_mojolearn_byte_lm.so', installed_raw, readonly=True)
    witness = dict(schema='mojolearn.byte-lm.resume-preflight.v1', action=action, vendor=vendor,
        baseline_summary_sha256=base['summary_sha256'], binding_sha256=sha(installed_raw),
        source=base['source'], schedule=base['summary']['schedule'], config=base['metadata']['config'],
        python_executable=sys.executable, numeric_mode=os.environ.get('MOJOLEARN_NUMERIC_MODE'),
        campaign_source_sha256=sha(read(out / 'campaign-source.sh')),
        artifact_checks_sha256=sha(read(out / 'artifact_checks.py')),
        qualification='preflight only; model work and guard exits are still required')
    if head is not None:
        # Exclusive read-only copy of actual transferred bytes. Every model
        # call also makes its own sealed memfd and retains incoming bytes.
        exclusive(out / 'transferred-head64.checkpoint.json', head['checkpoint'], readonly=True)
        witness['transfer'] = dict(file='transferred-head64.checkpoint.json', bytes=len(head['checkpoint']),
            sha256=expected_sha, foreign_vendor=head['runtime']['native_vendor'],
            foreign_summary_sha256=head['summary_sha256'],
            foreign_receipt_sha256=sha(read(Path(foreign_arg) / 'head64.receipt.json')),
            copied_actual_checkpoint=True, publication='exclusive file, mode0444; per-model load is sealed')
    exclusive(out / 'preflight.json', canonical(witness))
elif phase in ('verify-head', 'verify-resume', 'verify-control'):
    before = parse(read(out / 'preflight.json'))
    require(before['baseline_summary_sha256'] == base['summary_sha256'] and
            before['binding_sha256'] == sha(installed_raw) and before['source'] == base['source'] and
            read(out / 'bindings/_mojolearn_byte_lm.so') == installed_raw,
            'baseline/source/binding changed during continuation')
    if head is not None:
        transferred = out / 'transferred-head64.checkpoint.json'
        require(read(transferred, limit=2 * 1024 * 1024) == head['checkpoint'] and
                stat.S_IMODE(transferred.stat().st_mode) == 0o444 and
                before['transfer']['sha256'] == expected_sha,
                'read-only transferred checkpoint changed')
    if phase == 'verify-head':
        leg = load_capture(out / 'head64', 'head64', vendor)
        compatible(base, leg)
        require(same_steps(base, leg) == 64, 'head trajectory differs from continuous')
        receipt(out / 'head64.receipt.json', leg['summary_sha256'], vendor)
        result = dict(head_checkpoint_sha256=sha(leg['checkpoint']), compared_steps=64)
    else:
        leg = load_capture(out / 'resume128', 'resume128', vendor)
        compatible(base, leg)
        require(same_steps(base, leg) == 64 and leg['checkpoint'] == base['checkpoint'] and
                leg['incoming'] == head['checkpoint'], 'resume trajectory/checkpoint chain differs')
        receipt(out / 'resume128.receipt.json', leg['summary_sha256'], vendor)
        result = dict(transferred_checkpoint_sha256=expected_sha, compared_steps=64,
                      terminal_checkpoint_sha256=sha(leg['checkpoint']))
        if phase == 'verify-control':
            control = load_capture(out / 'zero-moments65', 'zero-moments65', vendor)
            compatible(base, control)
            receipt(out / 'zero-moments65.receipt.json', control['summary_sha256'], vendor)
            result['control'] = resume_control(head, leg, control, base)
    result.update(schema='mojolearn.byte-lm.resume-local-check.v1', phase=phase,
                  claim='checked retained local continuation bytes; full cross-vendor admission remains separate')
    exclusive(out / (phase + '.json'), canonical(result))
else:
    raise ValueError('unknown artifact check phase')
print(json.dumps(dict(phase=phase, complete=True), sort_keys=True))
PY

check_args=("$ROOT" "$baseline" "$OUT" "$vendor" "$action" "$foreign" "$expected_foreign_sha")
run preflight 180 "$PY" "$OUT/artifact_checks.py" preflight "${check_args[@]}"
if [[ "$action" == head64 ]]; then
    run head64 1200 "$PY" tools/byte_lm_real_text_capture.py --output "$OUT/head64" \
        --expected-vendor "$vendor" --steps 128 --action head64
    write_receipt head64
    run verify-head 180 "$PY" "$OUT/artifact_checks.py" verify-head "${check_args[@]}"
else
    run resume128 1200 "$PY" tools/byte_lm_real_text_capture.py --output "$OUT/resume128" \
        --expected-vendor "$vendor" --steps 128 --action resume128 \
        --resume-checkpoint "$OUT/transferred-head64.checkpoint.json"
    write_receipt resume128
    run verify-resume 180 "$PY" "$OUT/artifact_checks.py" verify-resume "${check_args[@]}"
    run zero-moments65 240 "$PY" tools/byte_lm_real_text_capture.py --output "$OUT/zero-moments65" \
        --expected-vendor "$vendor" --steps 128 --action zero-moments65 \
        --resume-checkpoint "$OUT/transferred-head64.checkpoint.json"
    write_receipt zero-moments65
    run verify-control 240 "$PY" "$OUT/artifact_checks.py" verify-control "${check_args[@]}"
fi
echo 'Continuation artifacts retained; use the separate full comparator with both continuous campaigns and all root receipts.'
