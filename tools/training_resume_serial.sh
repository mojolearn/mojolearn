#!/usr/bin/env bash
# Root-only remote orchestration. AUTHORED WITHOUT EXECUTION.
# No provisioning or transfers. Never invoke from a subagent or Apple host.
# Usage: --vendor cuda|hip --gpu-arch sm_XX|gfxXXX --out /new/absolute/path
# Optional: --incoming /actual/foreign-head8 --build-from /retained/base-campaign
# Default builds once then runs continuous16 and head8; incoming mode reuses
# the exact retained binary and runs resume16 only in a new output directory.
set -euo pipefail

resume_script=$(realpath -- "${BASH_SOURCE[0]}")
resume_root=$(dirname -- "$(dirname -- "$resume_script")")
[[ $(uname -s) == Linux ]] || { echo 'Resume campaign requires remote Linux; no Apple execution' >&2; exit 2; }
command -v timeout >/dev/null
command -v taskset >/dev/null
# GNU timeout supervises the whole process group. TERM reaches an active vendor
# guard, whose handler drains/kills its separately grouped compiler/model child.
# The final 15 seconds are reserved for that cleanup, never additional work.
if [[ ${MOJOLEARN_RESUME_DEADLINE_ACTIVE:-0} != 1 ]]; then
    resume_cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
    export MOJOLEARN_RESUME_DEADLINE_ACTIVE=1
    exec timeout --signal=TERM --kill-after=15s 1785s \
        taskset -c "$resume_cores" bash "$resume_script" "$@"
fi
cd "$resume_root"

vendor='' arch='' out='' incoming='' build_from=''
while (($#)); do
    case "$1" in
        --vendor) vendor=${2:?missing vendor}; shift 2 ;;
        --gpu-arch) arch=${2:?missing GPU architecture}; shift 2 ;;
        --out) out=${2:?missing fresh output}; shift 2 ;;
        --incoming) incoming=${2:?missing incoming checkpoint}; shift 2 ;;
        --build-from) build_from=${2:?missing retained build}; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ $out == /* ]] || { echo 'Require fresh absolute --out path' >&2; exit 2; }
case "$vendor:$arch" in
    cuda:sm_[0-9]* ) guard=tools/nvidia_serial_guard.py ;;
    hip:gfx[0-9]* ) guard=tools/amd_serial_guard.py ;;
    *) echo 'Require explicit matching vendor and one GPU architecture' >&2; exit 2 ;;
esac
[[ $arch != *[!A-Za-z0-9_]* ]] || { echo 'Malformed GPU architecture' >&2; exit 2; }
if [[ -n $incoming || -n $build_from ]]; then
    [[ $incoming == /* && $build_from == /* ]] || { echo 'Incoming mode requires absolute --incoming AND --build-from' >&2; exit 2; }
fi
mkdir -- "$out"  # refuses existing directories, files and symlinks
cp -- "$resume_script" "$out/orchestrator.sh"
cp -- "$guard" "$out/vendor-guard.py"
sha256sum "$out/orchestrator.sh" "$out/vendor-guard.py" > "$out/orchestration.sha256"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_CPU_THREADS=2 MOJOLEARN_COMPILE_JOBS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export MOJOLEARN_NUMERIC_MODE=identical
unset MOJOLEARN_TRAIN_SEED
helper=tools/training_cross_vendor_resume.py
started=$SECONDS

# Retain the exact shell-safe argv spelling which is subsequently executed.
recipe() {
    local path=$1
    shift
    printf '%q ' "$@" > "$path"
    printf '\n' >> "$path"
}
run_recipe() {
    local command_file=$1 log=$2 status=$3 rc=0
    if bash "$command_file" > "$log" 2>&1; then rc=0; else rc=$?; fi
    printf '%s\n' "$rc" > "$status"
    return "$rc"
}
remaining_budget() {
    local requested=$1 remaining=$((1740 - (SECONDS - started)))
    ((remaining > 5)) || { echo 'Campaign deadline budget exhausted' >&2; exit 124; }
    if ((requested < remaining)); then printf '%s' "$requested"; else printf '%s' "$remaining"; fi
}

# Runtime witness is read-only; each command is bounded by the global deadline
# and a short individual timeout, under the two-core outer affinity.
{
    printf 'vendor=%s\narchitecture=%s\nmode=identical\ncompiler_threads=2\n' "$vendor" "$arch"
    uname -a
    timeout 30s pixi run mojo --version
    timeout 15s pixi --version
    if [[ $vendor == cuda ]]; then
        timeout 15s nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
    elif command -v rocm-smi >/dev/null; then
        timeout 20s rocm-smi --showproductname --showdriverversion
    else
        timeout 20s rocminfo
    fi
} > "$out/runtime.txt" 2>&1

if [[ -z $incoming ]]; then
    cpu_flags=()
    case "$(uname -m)" in
        x86_64) cpu_flags=(--target-cpu x86-64-v3) ;;
        aarch64) ;;
        *) echo 'Unsupported Linux CPU architecture' >&2; exit 2 ;;
    esac
    recipe "$out/build-command.txt" python3 "$guard" --seconds "$(remaining_budget 900)" --rss-gib 12 -- \
        pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
        "${cpu_flags[@]}" --target-accelerator "$arch" \
        training/checks/cross_vendor_resume.mojo -o "$out/resume-driver"
    python3 "$helper" snapshot --source-root . --build-command-file "$out/build-command.txt" --output "$out/source.json"
    run_recipe "$out/build-command.txt" "$out/build.log" "$out/build-exit-code.txt"
    # Root-retained build binding is required before another campaign may reuse
    # this driver. No import or model is invoked by this host metadata block.
    python3 - "$out" "$vendor" "$arch" <<'PY'
import hashlib, json, pathlib, sys
root, vendor, arch = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()
if (root / 'build-exit-code.txt').read_text().strip() != '0': raise ValueError('build did not exit zero')
record = dict(schema='mojolearn.training.resume-build.v1', vendor=vendor, architecture=arch,
              files_sha256={name:digest(root/name) for name in
              ('resume-driver','source.json','build-command.txt','build.log','build-exit-code.txt')})
with (root/'build.json').open('x') as f: json.dump(record,f,sort_keys=True,indent=2); f.write('\n')
PY
else
    # Copy exact already-built artifacts after checking the retained hash set.
    # The source checkout is checked below before any resumed model work.
    python3 - "$build_from" "$out" "$vendor" "$arch" <<'PY'
import hashlib, json, os, pathlib, shutil, sys
old, new = map(pathlib.Path,sys.argv[1:3]); vendor, arch = sys.argv[3:5]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()
record=json.loads((old/'build.json').read_text())
if record.get('schema')!='mojolearn.training.resume-build.v1' or record.get('vendor')!=vendor or record.get('architecture')!=arch:
    raise ValueError('retained build vendor/architecture mismatch')
expected={'resume-driver','source.json','build-command.txt','build.log','build-exit-code.txt'}
if set(record['files_sha256'])!=expected: raise ValueError('incomplete retained build')
for name in sorted(expected):
    source=old/name
    if not source.is_file() or source.is_symlink() or digest(source)!=record['files_sha256'][name]:
        raise ValueError('retained build artifact mismatch: '+name)
    with source.open('rb') as src,(new/name).open('xb') as dst: shutil.copyfileobj(src,dst,1048576)
    if digest(new/name)!=record['files_sha256'][name]: raise ValueError('build changed while copying')
if (new/'build-exit-code.txt').read_text().strip()!='0': raise ValueError('nonzero original build')
os.chmod(new/'resume-driver',0o700)
with (new/'build.json').open('x') as f: json.dump(record,f,sort_keys=True,indent=2); f.write('\n')
PY
    # Preserve actual transferred bytes locally. The native helper then creates
    # its sealed immutable capture before decode and hashes those loaded bytes.
    python3 - "$incoming" "$out/incoming.ckptbin" <<'PY'
import os,stat,sys
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
try:
    before=os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size!=161008: raise ValueError('incoming must be fixed161008-byte checkpoint')
    with os.fdopen(fd,'rb',closefd=False) as f: raw=f.read(161009)
    after=os.fstat(fd)
    if len(raw)!=161008 or (before.st_mtime_ns,before.st_ctime_ns)!=(after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('incoming changed during capture')
finally: os.close(fd)
with open(sys.argv[2],'xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
PY
fi

run_leg() {
    local action=$1 budget=$2 input=${3:-}
    local stem="$out/$action"
    # Fail before device work if sources moved during the build or between
    # legs. record() independently repeats this check after successful work.
    python3 - "$out/source.json" <<'PY'
import json, pathlib, sys
from tools.training_cross_vendor_resume import source_inventory, sha_bytes, canonical
snapshot=json.loads(pathlib.Path(sys.argv[1]).read_text())
if snapshot.get('source_sha256') != sha_bytes(canonical(snapshot.get('files'))):
    raise ValueError('invalid retained source snapshot')
if source_inventory('.') != snapshot['files']:
    raise ValueError('source changed since retained build; refusing device work')
PY
    recipe "$stem.command.txt" env MOJOLEARN_TRAIN_EXPECT_VENDOR="$vendor" \
        MOJOLEARN_TRAIN_RESUME_ACTION="$action" MOJOLEARN_TRAIN_RESUME_INPUT="$input" \
        MOJOLEARN_TRAIN_RESUME_OUTPUT="$stem.ckptbin" \
        python3 "$guard" --seconds "$(remaining_budget "$budget")" --rss-gib 12 -- pixi run "$out/resume-driver"
    local rc=0
    if run_recipe "$stem.command.txt" "$stem.log" "$stem.shell-exit.txt"; then rc=0; else rc=$?; fi
    python3 "$helper" exit-record --exit-code "$rc" --run-command-file "$stem.command.txt" \
        --run-log "$stem.log" --binary "$out/resume-driver" --checkpoint "$stem.ckptbin" \
        --output "$stem.exit.json"
    ((rc == 0)) || return "$rc"
    local input_args=()
    [[ -z $input ]] || input_args=(--input-checkpoint "$input")
    python3 "$helper" record --source-root . --snapshot "$out/source.json" \
        --binary "$out/resume-driver" --checkpoint "$stem.ckptbin" \
        --build-log "$out/build.log" --run-log "$stem.log" --runtime-info "$out/runtime.txt" \
        --run-command-file "$stem.command.txt" --exit-code-file "$stem.exit.json" \
        --vendor "$vendor" --output-dir "$out/$action-evidence" "${input_args[@]}"
}

if [[ -n $incoming ]]; then
    run_leg resume16 600 "$out/incoming.ckptbin"
else
    run_leg continuous16 420
    run_leg head8 300
fi
printf '%s\n' 'Retained guarded local legs only. Cross-vendor comparison and effective negative control remain separate root jobs.' > "$out/COMPLETE.txt"
