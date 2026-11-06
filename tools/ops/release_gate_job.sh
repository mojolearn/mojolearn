#!/bin/bash
# tools/ops/release_gate_job.sh -- ONE NVIDIA/AMD release gate job on the box (plan item 4 of
# docs/plans/POST_0863_RELEASE_TOOLING.md). tools/ops/release_gate.py uploads this file verbatim beside
# its gate.env and launches it through the supported wrappers, which run it under
# tools/box_persist_run.sh (results to R2 from the box):
#   tools/nvidia_central.sh submit release-gate --cap 180 --results <res> <in>/release_gate_job.sh
#   tools/do_amd_steward.sh run <tag> <res> bash <in>/release_gate_job.sh
# It takes no arguments: gate.env (next to it) holds the exact build and gate argv.
#
# In order, all in the candidate commit's clean tree:
#   inputs   the wheels GATE_ARGV names exist (the Mac side checked every upload's sha256 on the box);
#   tree     GATE_TREE at GATE_SOURCE_COMMIT, clean (AMD: fetched here from GATE_FETCH_URL; NVIDIA: the
#            lane tree tools/nvidia_central.sh sync placed there);
#   gpu      the box's GPU is GATE_TARGET (sm_89 / gfx942), the target the autograd checks compile for;
#   runtime  AMD with adapters: the ROCm 7.2.3 runtime the ROCm torch build needs
#            (tools/amd_box_runtime_setup.sh, when /opt/rocm-7.2.3/lib is missing); python3 venv support;
#   pixi     the tree's locked default environment (the Mojo compiler the builder records);
#   build    the eight autograd executables: tools/gate_autograd_artifacts.py build (BUILD_ARGV; with a
#            retained set it reuses each proven-unchanged binary; a refused reuse falls back to a fresh
#            build, BUILD_FRESH_ARGV), bounded by GATE_BUILD_MINUTES;
#   overlay  when GATE_OVERLAY_SHA256 is set: the gate runner overlay (route-overlay.tgz, only GATE_OVERLAY
#            files) over the tree, AFTER the build (which reads the clean tree) and BEFORE the gate; every
#            file's sha256 before and after in out/route-overlay.txt; the files are restored on exit;
#   identity when GATE_IDENTITY_COLUMNS is set (tools/release.py --identity-on-wheels, Andrew 2026-11-05: ONE
#            execution of the identity lanes per device class, on the built wheels): tools/identity_wheel_record.py
#            installs the exact wheels (sha256 against GATE_IDENTITY_SHA256), selects the owed lanes by the
#            recorder's reuse rule and records them, every column on its own GPU(s) when the box has several,
#            into out/identity/ (identity-wheels.json + the column records); the gate below then runs with
#            --no-identity-lanes, so no lane runs twice;
#   gate     python3 tools/gate_smoke_box.py ... all (GATE_ARGV): release-light-v1, unchanged;
#   runtime-qualify  tools/release_runtime_installed.py run against GATE_WHEELS (runtime-<vendor> ledger
#            evidence, per unit: runtime-units-<class>/ receipts, earlier receipts from runtime-units-prior.tgz),
#            bounded by GATE_RUNTIME_MINUTES, recorded under out/runtime-<vendor>/, never the verdict; with
#            GATE_RUNTIME_CPU=1 (NVIDIA) also runtime-cpu (out/runtime-cpu/).
# On any exit the reports (GATE_FETCH) and the built autograd set are copied into
# $MOJOLEARN_RESULTS_DIR/out and the job's own record into $MOJOLEARN_RESULTS_DIR/job.
set -euo pipefail
IN="$(cd "$(dirname "$0")" && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?release_gate_job.sh runs under tools/box_persist_run.sh (nvidia_central.sh submit or do_amd_steward.sh run)}"
[ -f "$IN/gate.env" ] || { echo "release-gate-job: no $IN/gate.env" >&2; exit 2; }
# shellcheck disable=SC1091
. "$IN/gate.env"
[[ "$GATE_VENDOR" =~ ^(nvidia|amd)$ ]] || { echo "release-gate-job: bad GATE_VENDOR" >&2; exit 2; }
[[ "$GATE_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "release-gate-job: bad GATE_SOURCE_COMMIT" >&2; exit 2; }
[[ "$GATE_TREE" =~ ^/root/[A-Za-z0-9/_.-]+$ && "$GATE_WORK" =~ ^/root/results[A-Za-z0-9/_.-]+$ ]] \
    || { echo "release-gate-job: GATE_TREE / GATE_WORK must be plain /root paths (/root/results* for the work)" >&2; exit 2; }
[[ "${GATE_OVERLAY_SHA256:-}" =~ ^([0-9a-f]{64})?$ ]] || { echo "release-gate-job: bad GATE_OVERLAY_SHA256" >&2; exit 2; }
[[ "${GATE_RUNTIME_PRIOR_SHA256:-}" =~ ^([0-9a-f]{64})?$ ]] || { echo "release-gate-job: bad GATE_RUNTIME_PRIOR_SHA256" >&2; exit 2; }
[[ "${GATE_RUNTIME_CPU:-0}" =~ ^[01]$ ]] || { echo "release-gate-job: bad GATE_RUNTIME_CPU" >&2; exit 2; }
[[ "$GATE_BUILD_MINUTES" =~ ^[1-9][0-9]*$ ]] || { echo "release-gate-job: bad GATE_BUILD_MINUTES" >&2; exit 2; }
[[ "${GATE_JOB_CAP_MINUTES:-0}" =~ ^[0-9]+$ ]] || { echo "release-gate-job: bad GATE_JOB_CAP_MINUTES" >&2; exit 2; }
case "$GATE_VENDOR:$GATE_TARGET" in nvidia:sm_89|amd:gfx942) ;; *) echo "release-gate-job: unsupported vendor/target" >&2; exit 2 ;; esac
J="$RES/job"
mkdir -p "$J" "$RES/out"
[ ! -e "$GATE_WORK/out/gate-smoke.json" ] || { echo "release-gate-job: $GATE_WORK already holds a gate; a new job uses a new tag" >&2; exit 2; }
mkdir -p "$GATE_WORK"
exec > "$J/job.log" 2>&1
PHASE=inputs
export PATH="/root/.pixi/bin:/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
say() { echo "[$(date -u +%H:%M:%S) release-gate-job] $*"; }
# copy_reports SRC DST: the GATE_FETCH reports under SRC into DST. Only paths that exist enter the archive (a
# pattern that matches nothing, such as smoke.log when no smoke ran, is skipped, never a tar error), and
# autograd-smoke/tmp is excluded: pytest and comgr scratch that holds absolute symlinks, never a report
# (0.8.64: those symlinks aborted the Mac-side extraction). tools/test_release_gate_jobs.py runs this function.
copy_reports() {
    ( cd "$1" || exit 1
      shopt -s nullglob
      have=()
      for p in "${GATE_FETCH[@]}"; do
          for f in $p; do
              if [ -e "$f" ]; then have+=("$f"); fi
          done
      done
      if [ ${#have[@]} -gt 0 ]; then tar cf - --exclude=autograd-smoke/tmp "${have[@]}"; fi ) | tar xf - -C "$2"
}
OVERLAID=""
finish() {
    local code=$?
    trap - EXIT
    # the overlaid runner files go back to the source commit's bytes: the next job finds a clean tree
    if [ -n "$OVERLAID" ]; then
        ( cd "$GATE_TREE" && for f in $OVERLAID; do git checkout -q -- "$f" 2>/dev/null || rm -f "$f"; done ) \
            || say "restoring the overlaid files failed"
    fi
    # the reports, never the venv or the caches: the same report set the gate always brought home
    if [ -d "$GATE_WORK/out" ]; then
        copy_reports "$GATE_WORK/out" "$RES/out" || say "copying the reports failed"
    fi
    printf '{"vendor":"%s","version":"%s","source_commit":"%s","target":"%s","tag":"%s","phase":"%s","exit":%d,"job_script_sha256":"%s","gate_env_sha256":"%s","work":"%s"}\n' \
        "$GATE_VENDOR" "$GATE_VERSION" "$GATE_SOURCE_COMMIT" "$GATE_TARGET" "$GATE_TAG" "$PHASE" "$code" \
        "$(sha256sum "$IN/$(basename "$0")" | cut -d' ' -f1)" "$(sha256sum "$IN/gate.env" | cut -d' ' -f1)" "$GATE_WORK" > "$J/status.json"
    say "exit $code in phase $PHASE"
    exit "$code"
}
trap finish EXIT
trap 'exit 124' TERM INT
say "gate $GATE_VENDOR $GATE_VERSION at $GATE_SOURCE_COMMIT ($GATE_TAG); inputs $IN; work $GATE_WORK"
for ((i = 0; i < ${#GATE_ARGV[@]}; i++)); do
    if [ "${GATE_ARGV[$i]}" = --candidate-wheel ]; then
        w="${GATE_ARGV[$((i + 1))]}"; [ -f "$w" ] || { say "missing input $w"; exit 2; }
    fi
done
{ uname -a; nproc; free -g 2>/dev/null || true; df -h /root; } > "$J/box.txt" 2>&1 || true

PHASE=tree
if [ -n "$GATE_FETCH_URL" ]; then
    mkdir -p "$GATE_TREE" && cd "$GATE_TREE"
    [ -d .git ] || { git init -q . && git remote add origin "$GATE_FETCH_URL"; }
    git config --global --add safe.directory "$GATE_TREE" 2>/dev/null || true
    if [ "$(git rev-parse -q --verify HEAD 2>/dev/null || true)" != "$GATE_SOURCE_COMMIT" ]; then
        git fetch -q --depth=1 origin "$GATE_SOURCE_COMMIT"
        git checkout -q --detach -f "$GATE_SOURCE_COMMIT"
    fi
fi
cd "$GATE_TREE"
git config --global --add safe.directory "$GATE_TREE" 2>/dev/null || true
[ "$(git rev-parse HEAD)" = "$GATE_SOURCE_COMMIT" ] && [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || { say "$GATE_TREE is not a clean tree at $GATE_SOURCE_COMMIT (HEAD $(git rev-parse HEAD 2>/dev/null))"; exit 3; }
if [ "${GATE_REUSE:-0}" = 1 ]; then
    # the reuse proof reads the retained set's source revision from git; a shallow tree fetches it (best effort:
    # without it the builder refuses that reuse and compiles)
    prior=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("source_commit",""))' "$IN/autograd-prior/build-manifest.json" 2>/dev/null || true)
    if [[ "$prior" =~ ^[0-9a-f]{40}$ ]] && ! git cat-file -e "$prior^{commit}" 2>/dev/null; then
        git fetch -q --depth=1 origin "$prior" || say "could not fetch the retained set's source $prior; its entries rebuild"
    fi
fi

PHASE=gpu
if [ "$GATE_VENDOR" = nvidia ]; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
    gpu="sm_$cap"
else
    gpu=$(rocminfo | grep -m1 -o 'gfx[0-9a-f]*' || true)
fi
[ "$gpu" = "$GATE_TARGET" ] || { say "this box's GPU is $gpu, not $GATE_TARGET"; exit 3; }
say "GPU $gpu"

PHASE=runtime
if [ "$GATE_VENDOR" = amd ] && [ "$GATE_ADAPTERS" = 1 ] && [ ! -d /opt/rocm-7.2.3/lib ]; then
    say "installing the ROCm 7.2.3 runtime (tools/amd_box_runtime_setup.sh)"
    bash tools/amd_box_runtime_setup.sh > "$J/amd-runtime.log" 2>&1
fi
if ! python3 -c 'import ensurepip, venv' 2>/dev/null; then
    say "python3 lacks venv support; installing it"
    export DEBIAN_FRONTEND=noninteractive
    pyv=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    { apt-get -o DPkg::Lock::Timeout=300 update -qq && apt-get -o DPkg::Lock::Timeout=300 install -y -qq "python$pyv-venv"; } > "$J/apt.log" 2>&1
fi

PHASE=identity
irc=0
if [ -n "${GATE_IDENTITY_COLUMNS:-}" ]; then
    if [ "$GATE_VENDOR" = nvidia ]; then
        ngpu=$(nvidia-smi -L | grep -c '^GPU ')
    else
        ngpu=$(rocm-smi --showid --csv 2>/dev/null | grep -c '^card' || true)
        [ "$ngpu" -gt 0 ] || ngpu=1
    fi
    shas=(); for x in "${GATE_IDENTITY_SHA256[@]}"; do shas+=(--sha256 "$x"); done
    whl=(); for w in "${GATE_WHEELS[@]}"; do whl+=(--wheel "$w"); done
    mkdir -p "$GATE_WORK/out/identity"
    say "identity: columns $GATE_IDENTITY_COLUMNS on $ngpu GPU(s), $(nproc) core(s)"
    python3 tools/identity_wheel_record.py --columns "$GATE_IDENTITY_COLUMNS" \
        --label-device "$GATE_IDENTITY_LABEL_DEVICE" "${whl[@]}" "${shas[@]}" --gpus "$ngpu" \
        --out "$GATE_WORK/out/identity" --venv "$GATE_WORK/identity-venv" > "$J/identity.log" 2>&1 || irc=$?
    grep -E '^identity-wheels: (busy plan|.*REUSED|DONE|FAILED)' "$J/identity.log" | head -n 12 || true
    say "identity record exited $irc"
fi

PHASE=pixi
PIXI=$(command -v pixi || echo /root/.pixi/bin/pixi)
[ -x "$PIXI" ] || { curl -fsSL https://pixi.sh/install.sh | bash > "$J/pixi-install.log" 2>&1; PIXI=/root/.pixi/bin/pixi; }
"$PIXI" install --locked -e default > "$J/pixi-default.log" 2>&1
# pixi activation supplies MODULAR_HOME (tools/gha_autograd_box.sh): the build runs under `pixi run`, with the
# environment's own python; the gate runs on the system python3, outside the environment, as before
PYTHON=$GATE_TREE/.pixi/envs/default/bin/python3
{ "$PIXI" --version; "$GATE_TREE/.pixi/envs/default/bin/mojo" --version; } > "$J/compiler.txt" 2>&1

PHASE=build
mkdir -p "$GATE_WORK/out"
say "building the autograd executables: ${BUILD_ARGV[*]}"
brc=0
timeout "${GATE_BUILD_MINUTES}m" "$PIXI" run -e default "$PYTHON" "${BUILD_ARGV[@]}" > "$J/autograd-build.log" 2>&1 || brc=$?
if [ "$brc" != 0 ] && [ "${GATE_REUSE:-0}" = 1 ] && [ "$brc" != 124 ]; then
    say "the build with the retained set exited $brc; building every entry fresh"
    mv "$GATE_WORK/out/autograd-bin" "$GATE_WORK/autograd-bin.reuse-refused" 2>/dev/null || true
    brc=0
    timeout "${GATE_BUILD_MINUTES}m" "$PIXI" run -e default "$PYTHON" "${BUILD_FRESH_ARGV[@]}" > "$J/autograd-build-fresh.log" 2>&1 || brc=$?
fi
[ "$brc" = 0 ] || { say "the autograd build exited $brc"; tail -n 20 "$J"/autograd-build*.log; exit "$brc"; }
say "autograd executables built: $GATE_WORK/out/autograd-bin"

PHASE=overlay
if [ -n "${GATE_OVERLAY_SHA256:-}" ]; then
    # the gate runner overlay (tools/ops/release_gate.py GATE_OVERLAY): checked, then applied over the tree
    [ -f "$IN/route-overlay.tgz" ] && [ "$(sha256sum "$IN/route-overlay.tgz" | cut -d' ' -f1)" = "$GATE_OVERLAY_SHA256" ] \
        || { say "the route overlay is not the recorded bytes $GATE_OVERLAY_SHA256"; exit 4; }
    ro="$GATE_WORK/route-overlay"
    rm -rf "$ro" && mkdir -p "$ro" && tar -xzf "$IN/route-overlay.tgz" -C "$ro"
    [ -z "$(cd "$ro" && find . ! -type f ! -type d)" ] || { say "refusing a link or special file in the route overlay"; exit 4; }
    files=$(cd "$ro" && find . -type f | sed 's|^\./||' | LC_ALL=C sort)
    [ -n "$files" ] || { say "the route overlay is empty"; exit 4; }
    for f in $files; do
        case " ${GATE_OVERLAY[*]} " in *" $f "*) ;; *) say "the route overlay carries $f, not a gate runner file"; exit 4 ;; esac
    done
    for f in $files; do
        if [ -f "$f" ]; then echo "before $f $(sha256sum "$f" | cut -c1-64)"; else echo "before $f absent"; fi
    done > "$GATE_WORK/out/route-overlay.txt"
    OVERLAID=$files
    tar -xzf "$IN/route-overlay.tgz" -C "$GATE_TREE"
    for f in $files; do echo "after $f $(sha256sum "$f" | cut -c1-64)"; done >> "$GATE_WORK/out/route-overlay.txt"
    echo "overlay_sha256=$GATE_OVERLAY_SHA256" >> "$GATE_WORK/out/route-overlay.txt"
    cp "$IN/route-overlay.json" "$GATE_WORK/out/route-overlay.json"
    say "gate runner overlay applied: $(echo $files)"
fi

PHASE=gate
say "gate: python3 ${GATE_ARGV[*]}"
grc=0
python3 "${GATE_ARGV[@]}" > "$J/gate.log" 2>&1 || grc=$?
tail -n 5 "$J/gate.log"

PHASE=runtime-qualify
# runtime-<vendor> evidence for the ledger (tools/release_runtime_installed.py run: fresh venv, the exact
# candidate wheels, the frozen source's own tests). Recorded and fetched (out/runtime-<vendor>/), bounded by
# GATE_RUNTIME_MINUTES; its exit never changes this job's verdict above (the smoke's), but tools/release.py
# gate_ok fails the gate unless runtime-<vendor>/runtime-report.json says passed.
# PER UNIT (docs/release/RUNTIME_QUALIFICATION.md): every test file is one unit with its own budget and receipt.
# The receipts are appended to $RES/out/runtime-units-<class>/ as each unit ends (tools/box_persist_run.sh
# uploads $RES every few minutes, so an interrupted job keeps them), and the newest passing receipts of earlier
# gates come in as runtime-units-prior.tgz (GATE_RUNTIME_PRIOR_SHA256): a unit whose wheel bytes or source
# closure are unchanged on this device class is reused, not run. A source commit whose producer predates the
# unit flags runs it without them.
# GATE_RUNTIME_CPU=1 (the NVIDIA job): runtime-cpu as well, the same producer with --vendor cpu
# (MOJOLEARN_VENDOR=cpu for the qualifier only; never timed), after runtime-<vendor>, in what the cap leaves.
UNIT_FLAGS=0
grep -q -- '--unit-store' tools/release_runtime_installed.py 2>/dev/null && UNIT_FLAGS=1
# A prior set that is not the recorded bytes, or holds a link, is dropped (every unit then runs); it never
# changes the job's verdict, which is the smoke's.
rp="$GATE_WORK/runtime-units-prior"
if [ -n "${GATE_RUNTIME_PRIOR_SHA256:-}" ] && [ "$UNIT_FLAGS" = 1 ]; then
    if [ -f "$IN/runtime-units-prior.tgz" ] \
            && [ "$(sha256sum "$IN/runtime-units-prior.tgz" | cut -d' ' -f1)" = "$GATE_RUNTIME_PRIOR_SHA256" ]; then
        rm -rf "$rp" && mkdir -p "$rp" && tar -xzf "$IN/runtime-units-prior.tgz" -C "$rp" || rm -rf "$rp"
        if [ -d "$rp" ] && [ -n "$(cd "$rp" && find . ! -type f ! -type d)" ]; then
            say "refusing a link or special file in runtime-units-prior; every unit runs"; rm -rf "$rp"
        fi
    else
        say "runtime-units-prior.tgz is not the recorded bytes $GATE_RUNTIME_PRIOR_SHA256; every unit runs"
    fi
fi
# run_runtime VENDOR CLASS: one runtime producer run in the minutes the job cap still leaves
run_runtime() {
    local vendor=$1 class=$2 rt_min=$GATE_RUNTIME_MINUTES left rrc=0
    # A capped job (NVIDIA submit --cap) keeps 10 minutes after the qualifier to copy and persist reports.
    if [ "${GATE_JOB_CAP_MINUTES:-0}" != 0 ]; then
        left=$(( GATE_JOB_CAP_MINUTES - SECONDS / 60 - 10 ))
        [ "$left" -ge "$rt_min" ] || rt_min=$left
    fi
    if [ "$rt_min" -le 20 ]; then
        say "runtime qualifier ($vendor) skipped: only $rt_min minutes left under the job cap"
        return 0
    fi
    local extra=()
    if [ "$UNIT_FLAGS" = 1 ]; then
        extra=(--device-class "$class" --unit-store "$RES/out/runtime-units-$class")
        if [ -d "$rp/$class" ]; then extra+=(--prior-units "$rp/$class"); fi
    fi
    # The qualifier's own outer cap sits 10 minutes inside the outer bound (venv, install, report), so it stops
    # itself and writes runtime-report.json and the unit receipts instead of being killed.
    timeout "${rt_min}m" python3 tools/release_runtime_installed.py run --vendor "$vendor" \
        --source-root "$GATE_TREE" --source-commit "$GATE_SOURCE_COMMIT" "${wheel_args[@]}" \
        --timeout $(( (rt_min - 10) * 60 )) ${extra[@]+"${extra[@]}"} \
        --out "$GATE_WORK/out/runtime-$vendor" > "$J/runtime-qualify-$vendor.log" 2>&1 || rrc=$?
    say "runtime qualifier ($vendor, $class) exited $rrc in ${rt_min}m bound (recorded under out/runtime-$vendor)"
    grep -m 3 -E '^RUNTIME (units|vendor)' "$J/runtime-qualify-$vendor.log" || tail -n 3 "$J/runtime-qualify-$vendor.log"
}
if [ "${GATE_RUNTIME_MINUTES:-0}" != 0 ] && [ "${#GATE_WHEELS[@]}" -gt 0 ]; then
    wheel_args=(); for w in "${GATE_WHEELS[@]}"; do wheel_args+=(--wheel "$w"); done
    run_runtime "$GATE_VENDOR" "$GATE_VENDOR-$GATE_TARGET"
    if [ "${GATE_RUNTIME_CPU:-0}" = 1 ]; then
        run_runtime cpu "cpu-linux-$(uname -m)"
    fi
fi
PHASE=complete
# the one box job's verdict: the device checks' (the gate) and the identity record's
[ "$grc" != 0 ] || grc=$irc
exit "$grc"
