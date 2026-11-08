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
#   measuresmoke-setup  in the background, while the phases below run (network and disk bound): the measurement
#            smoke's two venvs (the exact wheels + MEASURE_SMOKE_PIP; the fixture generator's CPU torch), and after
#            the overlay the plan's data from R2 (measuresmoke-data.json); the smoke itself runs last;
#   identity when GATE_IDENTITY_COLUMNS is set (tools/release.py --identity-on-wheels, Andrew 2026-11-05: ONE
#            execution of the identity lanes per device class, on the built wheels): tools/identity_wheel_record.py
#            installs the exact wheels (sha256 against GATE_IDENTITY_SHA256), selects the owed lanes by the
#            recorder's reuse rule and records them, every column on its own GPU(s) when the box has several,
#            into out/identity/ (identity-wheels.json + the column records); the gate below then runs with
#            --no-identity-lanes, so no lane runs twice;
#   pixi     the tree's locked default environment (the Mojo compiler the builder records);
#   build    the eight autograd executables: tools/gate_autograd_artifacts.py build (BUILD_ARGV; with a
#            retained set it reuses each proven-unchanged binary; a refused reuse falls back to a fresh
#            build, BUILD_FRESH_ARGV), bounded by GATE_BUILD_MINUTES;
#   overlay  when GATE_OVERLAY_SHA256 is set: the gate runner overlay (route-overlay.tgz, only GATE_OVERLAY
#            files) over the tree, AFTER the build (which reads the clean tree) and BEFORE the gate; every
#            file's sha256 before and after in out/route-overlay.txt; the files are restored on exit;
#   shardcheck  ALWAYS (no change detection), right after the overlay (the tool is a GATE_OVERLAY file):
#            tools/logical_shard_identity.py, the one-GPU multi-device gate (1, 2 and 3 logical devices on this
#            card, same digests) on the exact wheels (+ numpy for the harness), into out/shardcheck/;
#   gate     python3 tools/gate_smoke_box.py ... all (GATE_ARGV): release-light-v1, unchanged;
#   runtime-qualify  tools/release_runtime_installed.py run against GATE_WHEELS (runtime-<vendor> ledger
#            evidence, per unit: runtime-units-<class>/ receipts, earlier receipts from runtime-units-prior.tgz),
#            bounded by GATE_RUNTIME_MINUTES, recorded under out/runtime-<vendor>/, never the verdict; with
#            GATE_RUNTIME_CPU=1 (NVIDIA) also runtime-cpu (out/runtime-cpu/);
#   hard-done  the job's verdict is settled (identity, shardcheck, device checks): out/gate-hard.json, the reports
#            copied into $MOJOLEARN_RESULTS_DIR/out and uploaded AT ONCE as a `hard` snapshot
#            (tools/box_persist_run.sh --snapshot ... hard), so tools/ops/release_gate.py brings them home and the
#            release goes on (identity-admit) while the box runs the smoke;
#   measuresmoke  LAST (Andrew 2026-11-08: "run immediately after the verification and not be something separate"):
#            tools/release_measurement_smoke.py, every recipe of the measurement plan once at its smoke shape through
#            tools/bench_board.py --smoke, our GPU arm only, on the exact wheels, into out/measuresmoke/. SOFT: its
#            outcome never changes this job's exit; tools/release.py judges the summary when the final results land.
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
[[ "${GATE_MEASURE_SMOKE_MINUTES:-90}" =~ ^[0-9]+$ ]] || { echo "release-gate-job: bad GATE_MEASURE_SMOKE_MINUTES" >&2; exit 2; }
[[ "${GATE_MEASURE_SMOKE_MAX_GB:-8}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "release-gate-job: bad GATE_MEASURE_SMOKE_MAX_GB" >&2; exit 2; }
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
# DISK LINES (0.8.66 run t: the 80 GB pod filled during the measurement smoke and nothing said how): the free space
# of the work dir's filesystem after every phase, "DISK free=<GB> after <phase>" (tools/test_release_gate_jobs.py)
disk_free_gb() { df -Pk "$GATE_WORK" 2>/dev/null | awk 'NR == 2 { printf "%.1f", $4 / 1048576 }'; }
disk_line() { say "DISK free=$(disk_free_gb)GB after $1"; }
# copy_reports SRC DST: the GATE_FETCH reports under SRC into DST. Only paths that exist enter the archive (a
# pattern that matches nothing, such as smoke.log when no smoke ran, is skipped, never a tar error), and
# autograd-smoke/tmp is excluded: pytest and comgr scratch that holds absolute symlinks, never a report
# (0.8.64: those symlinks aborted the Mac-side extraction). Nor a venv: the runtime qualifier builds its fresh venv
# under out/runtime-<vendor>/qualify/run/venv, and "runtime-*" used to copy it (6.3 GB at 0.8.66 run t) into the
# results dir beside the original. tools/test_release_gate_jobs.py runs this function.
copy_reports() {
    ( cd "$1" || exit 1
      shopt -s nullglob
      have=()
      for p in "${GATE_FETCH[@]}"; do
          for f in $p; do
              if [ -e "$f" ]; then have+=("$f"); fi
          done
      done
      if [ ${#have[@]} -gt 0 ]; then tar cf - --exclude=autograd-smoke/tmp --exclude=venv --exclude='venv-*' --exclude=.venv "${have[@]}"; fi ) | tar xf - -C "$2"
}
OVERLAID=""
finish() {
    local code=$?
    trap - EXIT
    if [ -n "${SAMPLER:-}" ]; then kill "$SAMPLER" 2>/dev/null || true; fi
    # the measurement smoke's background setup (phase measuresmoke-setup), when the job ends before the smoke
    for p in ${MS_VPID:-} ${MS_FPID:-} ${MS_DPID:-}; do kill "$p" 2>/dev/null || true; done
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
disk_line start
# THE PLAN'S DISK NEED (tools/ops/gate_disk_plan.py job-gb, written by tools/ops/release_gate.py): what this job writes
# (venvs, reports and their results copy, the measurement smoke under its cap). A box with less free is refused
# here, before anything is installed, never filled halfway (0.8.66 run t).
[[ "${GATE_DISK_NEED_GB:-0}" =~ ^[0-9]+$ ]] || { say "bad GATE_DISK_NEED_GB"; exit 2; }
if [ "${GATE_DISK_NEED_GB:-0}" -gt 0 ]; then
    free_gb=$(disk_free_gb); free_gb=${free_gb%.*}
    if [ -n "$free_gb" ] && [ "$free_gb" -lt "$GATE_DISK_NEED_GB" ]; then
        say "DISK REFUSED: $GATE_WORK has ${free_gb} GB free; this gate job needs $GATE_DISK_NEED_GB GB (tools/ops/gate_disk_plan.py). Free the box (old /root/results-release-gate/* work dirs) or rent a bigger disk (tools/dev_pod.sh DISK_GB)"
        exit 3
    fi
fi
for ((i = 0; i < ${#GATE_ARGV[@]}; i++)); do
    if [ "${GATE_ARGV[$i]}" = --candidate-wheel ]; then
        w="${GATE_ARGV[$((i + 1))]}"; [ -f "$w" ] || { say "missing input $w"; exit 2; }
    fi
done
{ uname -a; nproc; free -g 2>/dev/null || true; df -h /root; } > "$J/box.txt" 2>&1 || true
# BOX FACTS (release evidence, 2026-11-06; tools/release_evidence_set.py box-facts): the GPU and CPU model, the
# core count and memory in job/box-facts.txt, and every 30 s the cores more than half busy over the interval, of
# the total, in job/busy-cores.tsv (epoch, busy, total), so the committed evidence states busy cores of total.
{
    echo "nproc=$(nproc)"
    echo "cpu_model=$(grep -m 1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
    echo "mem_total_kb=$(awk '/^MemTotal/ {print $2}' /proc/meminfo 2>/dev/null)"
    echo "kernel=$(uname -r)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu=/'
    fi
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showproductname 2>/dev/null | grep -E 'Card (Series|series|SKU)' | sed 's/^/gpu=/'
    fi
} > "$J/box-facts.txt" 2>/dev/null || true
busy_cores_sampler() {
    local prev cur
    prev="$(grep -E '^cpu[0-9]+ ' /proc/stat)" || return 0
    while sleep 30; do
        cur="$(grep -E '^cpu[0-9]+ ' /proc/stat)" || return 0
        paste -d' ' <(echo "$prev") <(echo "$cur") | awk -v t="$(date -u +%s)" '
            { n = NF / 2; t0 = 0; t1 = 0
              for (i = 2; i <= n; i++) { t0 += $i; t1 += $(n + i) }
              d = t1 - t0; idle = ($(n + 5) + $(n + 6)) - ($5 + $6)
              if (d > 0 && (d - idle) * 2 >= d) busy++; total++ }
            END { printf "%s\t%d\t%d\n", t, busy, total }'
        prev="$cur"
    done
}
busy_cores_sampler >> "$J/busy-cores.tsv" 2>/dev/null &
SAMPLER=$!

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

PHASE=measuresmoke-setup
# THE MEASUREMENT SMOKE'S SETUP, IN THE BACKGROUND (network and disk bound, 20-45 min in all at 0.8.66): its two
# venvs install while identity, the build, shardcheck, the device checks and the runtime qualifier run, so the smoke
# that runs last (phase measuresmoke) starts at once. Its own venvs, never the identity venv: the exact wheels + the
# harness packages MEASURE_SMOKE_PIP names (harness only, never the runtime), and the CPU torch the fixture generator
# pins (tools/neural_fixtures_box.sh; never the measured venv, never raced). The plan's data starts after the
# overlay (its stager is a GATE_OVERLAY file).
MS_PY="$GATE_WORK/measure-venv/bin/python"
MS_FPY="$GATE_WORK/fixture-venv/bin/python"
MS_VPID=""; MS_FPID=""; MS_DPID=""
if [ "${#GATE_WHEELS[@]}" -gt 0 ]; then
    { python3 -m venv "$GATE_WORK/measure-venv" \
      && "$MS_PY" -m pip install -q --no-cache-dir "${GATE_WHEELS[@]}" \
      && "$MS_PY" -m pip install -q --no-cache-dir "${MEASURE_SMOKE_PIP[@]}"; } > "$J/measure-venv.log" 2>&1 &
    MS_VPID=$!
    { python3 -m venv "$GATE_WORK/fixture-venv" \
      && "$MS_FPY" -m pip install -q --no-cache-dir "${MEASURE_SMOKE_FIXTURE_PIP[@]}" \
           --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple; } \
        > "$J/fixture-venv.log" 2>&1 &
    MS_FPID=$!
    say "measuresmoke-setup: the measured venv and the fixture venv install in the background"
fi

disk_line measuresmoke-setup
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
    # RUN EVERYTHING by default (Andrew 2026-11-05); GATE_REUSE_UNCHANGED=1 (release.py --reuse-unchanged) opts in
    reuse=(); [ "${GATE_REUSE_UNCHANGED:-0}" = 1 ] && reuse=(--reuse-unchanged)
    python3 tools/identity_wheel_record.py --columns "$GATE_IDENTITY_COLUMNS" ${reuse[@]+"${reuse[@]}"} \
        --label-device "$GATE_IDENTITY_LABEL_DEVICE" "${whl[@]}" "${shas[@]}" --gpus "$ngpu" \
        --out "$GATE_WORK/out/identity" --venv "$GATE_WORK/identity-venv" > "$J/identity.log" 2>&1 || irc=$?
    grep -E '^identity-wheels: (busy plan|.*REUSED|DONE|FAILED)' "$J/identity.log" | head -n 12 || true
    say "identity record exited $irc"
fi

disk_line identity
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
if [ "${GATE_BUILD_AUTOGRAD:-1}" = 0 ]; then
    say "no compiled autograd check is owed (units plan): nothing is built"
else
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
fi

disk_line build
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

# the measurement smoke's data is staged by the smoke itself, group by group, and deleted after the last group that
# reads it (tools/release_measurement_smoke.py run --data-plan; 0.8.66 run t: the whole plan's data and scratch at
# once filled the pod), never here in the background for the whole plan
if [ ! -f "$IN/measuresmoke-data.json" ]; then
    say "measuresmoke-setup: no measuresmoke-data.json input; the plan's data must already be at its pins"
fi

disk_line overlay
PHASE=shardcheck
# THE ONE-GPU MULTI-DEVICE GATE (tools/logical_shard_identity.py, Andrew 2026-11-06). UNCONDITIONAL: it runs in
# every release gate job, whatever changed; no change detection, selector or reuse decides whether it runs. The
# fixed lane list runs the public multi-device entries with 1, 2 and 3 logical devices on this one card against
# the exact wheel bytes (the identity venv when the identity phase installed them, else a fresh venv of
# GATE_WHEELS) and records out/shardcheck/shardcheck.json; any lane not MATCH fails the job. It is not the real
# two-GPU check: that stays tools/two_device_checks.sh.
# It runs AFTER the overlay phase: the tool is a GATE_OVERLAY file (tools/ops/release_gate.py), so the box runs the
# same LANES / COUNTS / SCHEMA the Mac judges with (tools/release_identity_wheels.py imports the Mac's copy), and a
# tool fix after the freeze reaches the box without a refreeze.
# NumPy: the wheels declare no dependencies (python/pyproject.toml; numpy is the `numpy`/`verify` extra) and the
# tool's worker builds its fixtures and the verifier digest with NumPy (harness only, never the runtime), so the
# chosen venv always gets it; the identity venv's later users (tools/release_check_units.py) are unaffected by it.
src=0
shpy="$GATE_WORK/identity-venv/bin/python"
if [ ! -x "$shpy" ]; then
    shpy="$GATE_WORK/shard-venv/bin/python"
    if [ "${#GATE_WHEELS[@]}" -gt 0 ]; then
        { python3 -m venv "$GATE_WORK/shard-venv" \
          && "$shpy" -m pip install -q --no-cache-dir "${GATE_WHEELS[@]}"; } > "$J/shard-venv.log" 2>&1 || src=$?
    else
        say "shardcheck: no wheels to install (GATE_WHEELS is empty): FAIL"
        src=2
    fi
fi
if [ "$src" = 0 ]; then
    "$shpy" -m pip install -q --no-cache-dir 'numpy>=1.26.4' >> "$J/shard-venv.log" 2>&1 || src=$?
    [ "$src" = 0 ] || say "shardcheck: installing numpy into $shpy exited $src (see shard-venv.log): FAIL"
fi
if [ "$src" = 0 ]; then
    mkdir -p "$GATE_WORK/out/shardcheck"
    python3 tools/logical_shard_identity.py --python "$shpy" --vendor "$GATE_VENDOR" \
        --out "$GATE_WORK/out/shardcheck" > "$J/shardcheck.log" 2>&1 || src=$?
    grep -E '^SHARDCHECK' "$J/shardcheck.log" | head -n 16 || true
fi
say "shardcheck exited $src"

disk_line shardcheck
PHASE=gate
grc=0
if [ "${GATE_UNITS:-0}" = 1 ]; then
    # THE DEVICE CHECKS AS UNITS (tools/release_check_units.py): exactly the units the Mac's plan owes, in the
    # identity phase's venv (the same installed wheel bytes); no gate stage list, nothing twice
    whl=(); for w in "${GATE_WHEELS[@]}"; do whl+=(--wheel "$w"); done
    say "check units: python3 tools/release_check_units.py run --plan $IN/units-plan.json"
    python3 tools/release_check_units.py run --plan "$IN/units-plan.json" --python "$GATE_WORK/identity-venv/bin/python" \
        "${whl[@]}" --bin-dir "$GATE_WORK/out/autograd-bin" --version "$GATE_VERSION" \
        --source-commit "$GATE_SOURCE_COMMIT" --out "$GATE_WORK/out/units" > "$J/units.log" 2>&1 || grc=$?
    grep -E '^check-units:' "$J/units.log" | head -n 5 || true
else
    say "gate: python3 ${GATE_ARGV[*]}"
    python3 "${GATE_ARGV[@]}" > "$J/gate.log" 2>&1 || grc=$?
    tail -n 5 "$J/gate.log"
fi

disk_line gate
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
MS_RESERVE=0
[ "${#GATE_WHEELS[@]}" -eq 0 ] || MS_RESERVE=${GATE_MEASURE_SMOKE_MINUTES:-90}
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
    # A capped job (NVIDIA submit --cap) keeps 10 minutes after the qualifier to copy and persist reports, and the
    # measurement smoke's ceiling for the smoke that runs after it (MS_RESERVE, phase measuresmoke).
    if [ "${GATE_JOB_CAP_MINUTES:-0}" != 0 ]; then
        left=$(( GATE_JOB_CAP_MINUTES - SECONDS / 60 - 10 - MS_RESERVE ))
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
        # the one run under --identity-on-wheels runs every unit unless --reuse-unchanged opted in
        if [ -n "${GATE_IDENTITY_COLUMNS:-}" ] && [ "${GATE_REUSE_UNCHANGED:-0}" != 1 ]; then extra+=(--redo); fi
    fi
    # The qualifier's own outer cap sits 10 minutes inside the outer bound (venv, install, report), so it stops
    # itself and writes runtime-report.json and the unit receipts instead of being killed.
    # EVERY CORE (2026-11-07): the qualifier runs its units side by side, one per core of this pod (affinity and
    # cgroup quota) capped by free memory, and logs "QUALIFY ... workers=" and busy cores of total every 60 s.
    # Nothing here pins it: no MOJOLEARN_QUALIFY_JOBS, no taskset, no thread caps.
    timeout "${rt_min}m" python3 tools/release_runtime_installed.py run --vendor "$vendor" \
        --source-root "$GATE_TREE" --source-commit "$GATE_SOURCE_COMMIT" "${wheel_args[@]}" \
        --timeout $(( (rt_min - 10) * 60 )) ${extra[@]+"${extra[@]}"} \
        --out "$GATE_WORK/out/runtime-$vendor" > "$J/runtime-qualify-$vendor.log" 2>&1 || rrc=$?
    say "runtime qualifier ($vendor, $class) exited $rrc in ${rt_min}m bound (recorded under out/runtime-$vendor)"
    grep -m 3 -E '^RUNTIME (units|vendor)' "$J/runtime-qualify-$vendor.log" || tail -n 3 "$J/runtime-qualify-$vendor.log"
    # the workers it ran and its last busy-cores sample (every report states busy cores of total)
    local ql="$GATE_WORK/out/runtime-$vendor/qualify.log"
    if [ -f "$ql" ]; then grep -m 1 '^QUALIFY units=' "$ql" || true; grep '^QUALIFY busy' "$ql" | tail -n 1 || true; fi
}
if [ "${GATE_RUNTIME_MINUTES:-0}" != 0 ] && [ "${#GATE_WHEELS[@]}" -gt 0 ]; then
    wheel_args=(); for w in "${GATE_WHEELS[@]}"; do wheel_args+=(--wheel "$w"); done
    run_runtime "$GATE_VENDOR" "$GATE_VENDOR-$GATE_TARGET"
    if [ "${GATE_RUNTIME_CPU:-0}" = 1 ]; then
        run_runtime cpu "cpu-linux-$(uname -m)"
    fi
fi
disk_line runtime-qualify
PHASE=hard-done
# the one box job's verdict: the device checks' (the gate) and the identity record's
[ "$grc" != 0 ] || grc=$irc
# ... and the one-GPU multi-device gate's (phase shardcheck, unconditional)
[ "$grc" != 0 ] || grc=$src
# Settled: the reports go home NOW (a `hard` ring-slot upload; the manifest stays `running`), so the release judges
# the HARD parts and goes on while this box runs the smoke. gate-hard.json is the marker tools/ops/release_gate.py
# waits for; the final upload at exit carries the same reports plus out/measuresmoke/.
printf '{"schema":"mojolearn.release-gate-hard.v1","vendor":"%s","version":"%s","source_commit":"%s","tag":"%s","exit":%d,"utc":"%s"}\n' \
    "$GATE_VENDOR" "$GATE_VERSION" "$GATE_SOURCE_COMMIT" "$GATE_TAG" "$grc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$GATE_WORK/out/gate-hard.json"
copy_reports "$GATE_WORK/out" "$RES/out" || say "copying the reports failed"
persist_hard() {
    local runner=${MOJOLEARN_PERSIST_RUNNER:-} json=${MOJOLEARN_PERSIST_JSON:-} i
    # a runner that predates the env: the job's registration names its json (tools/box_persist_run.sh register)
    if [ -z "$json" ] && [ -n "${MOJOLEARN_PERSIST_TAG:-}" ] && [ -f "/root/box-persist/active/$MOJOLEARN_PERSIST_TAG.env" ]; then
        json=$(sed -n 's/^JSON=//p' "/root/box-persist/active/$MOJOLEARN_PERSIST_TAG.env" | head -n 1)
    fi
    [ -n "$runner" ] || runner=/root/box-persist/bin/box_persist_run.sh
    if [ -z "$json" ] || [ ! -f "$json" ] || [ ! -f "$runner" ]; then
        say "hard-done: no persist runner in reach; the next periodic upload carries the reports"
        return 0
    fi
    for i in 1 2 3; do
        if bash "$runner" --snapshot "$json" "$RES" hard; then
            say "hard-done: the reports are uploaded (exit $grc); the measurement smoke runs next"
            HARD_HOME=1
            return 0
        fi
        sleep 20
    done
    say "hard-done: the immediate upload failed 3 times; the next periodic upload carries the reports"
}
HARD_HOME=0
persist_hard
disk_line hard-done
# THE GATE'S VENVS GO ONCE ITS REPORTS ARE HOME (0.8.66 run t: identity, shard and runtime-qualifier venvs stayed on
# the pod through the smoke, which then found no room). Only after the hard snapshot is confirmed uploaded; nothing
# after this point uses them (the smoke has its own measured and fixture venvs).
if [ "$HARD_HOME" = 1 ]; then
    for v in "$GATE_WORK/identity-venv" "$GATE_WORK/shard-venv" "$GATE_WORK"/out/runtime-*/qualify/run/venv \
             "$RES"/out/runtime-*/qualify/run/venv; do
        if [ -d "$v" ] && [ ! -L "$v" ]; then rm -rf "$v"; fi
    done
    disk_line "dropping the gate's venvs (reports home)"
else
    say "hard-done: the reports are not confirmed home; the gate's venvs stay until the job ends"
fi

PHASE=measuresmoke
# THE MEASUREMENT SMOKE (tools/release_measurement_smoke.py, Andrew 2026-11-06: "make the smoke on the measurements a
# test as part of our pre release process"; 2026-11-08: "run immediately after the verification and not be something
# separate"). UNCONDITIONAL: every release gate job runs it, whatever changed; no change detection, selector or reuse.
# Every recipe of the frozen measurement plan (bench_board.plan_races, identical) runs ONCE through
# `tools/bench_board.py --smoke` (the board's smoke rows and small neural shape), OUR GPU arm only (no opponent
# installs, no opponent store), against the exact wheel bytes, in parallel batches on this one GPU. Judged on
# completion and on the fields the board ingests (finite quality, digest, provenance), never on speed; no timing from
# it reaches a board. out/measuresmoke/measuresmoke.json; any recipe not PASS (or declared unsupported) fails it.
# LAST in the job, after the verdict is home (phase hard-done): SOFT (tools/release_verdicts.py), so its exit never
# changes this job's; --strict on the Mac waits for it and stops on it. The tool is a GATE_OVERLAY file (the Mac's
# copy runs). Its ceiling is GATE_MEASURE_SMOKE_MINUTES, clipped on a capped job to what the cap leaves.
mrc=0
mout="$GATE_WORK/out/measuresmoke"
mkdir -p "$mout"
fpy=()
ms_min=${GATE_MEASURE_SMOKE_MINUTES:-90}
if [ "${GATE_JOB_CAP_MINUTES:-0}" != 0 ]; then
    left=$(( GATE_JOB_CAP_MINUTES - SECONDS / 60 - 10 ))
    [ "$left" -ge "$ms_min" ] || ms_min=$left
fi
if [ "${#GATE_WHEELS[@]}" -eq 0 ]; then
    say "measuresmoke: no wheels to install (GATE_WHEELS is empty): FAIL"
    mrc=2
elif [ "$ms_min" -le 5 ]; then
    say "measuresmoke: only $ms_min minutes left under the job cap: not run (no summary; the release records it SOFT)"
    mrc=124
else
    wait "$MS_VPID" || { mrc=$?; say "measuresmoke: the venv install exited $mrc (see measure-venv.log): FAIL"; }
    if wait "$MS_FPID"; then
        fpy=(--fixture-python "$MS_FPY")
    else
        say "measuresmoke: the fixture venv did not install (see fixture-venv.log); the layer recipes will FAIL"
    fi
    if [ -n "$MS_DPID" ]; then
        drc=0; wait "$MS_DPID" || drc=$?
        grep -E '^MEASURESMOKE data' "$J/measuresmoke-data.log" | head -n 12 || true
        [ "$drc" = 0 ] || { say "measuresmoke: staging the plan's data exited $drc: FAIL"; [ "$mrc" != 0 ] || mrc=$drc; }
    fi
    MS_VPID=""; MS_FPID=""; MS_DPID=""
fi
msdata=()
if [ -f "$IN/measuresmoke-data.json" ]; then msdata=(--data-plan "$IN/measuresmoke-data.json" --home "$HOME"); fi
if [ "$mrc" = 0 ]; then
    # run by the measured venv's python (the plan imports the board harness, which reads NumPy); its plan line
    # (workers and the bound that set them) and its busy-cores samples (cores busy of total, the first after two
    # minutes) reach this job log as they happen; the whole output stays in measuresmoke.log
    set +e
    "$MS_PY" tools/release_measurement_smoke.py run --vendor "$GATE_VENDOR" --python "$MS_PY" \
        --data-root "$HOME/datasets/gbm-bench" --ceiling $(( ms_min * 60 )) \
        --out "$mout" --work "$GATE_WORK/measuresmoke-work" ${fpy[@]+"${fpy[@]}"} ${msdata[@]+"${msdata[@]}"} \
        --max-work-gb "${GATE_MEASURE_SMOKE_MAX_GB:-8}" 2>&1 \
        | tee "$J/measuresmoke.log" | grep --line-buffered -E '^MEASURESMOKE (plan|busy|disk|vendor=)'
    mrc=${PIPESTATUS[0]}
    set -e
    grep -E '^MEASURESMOKE ' "$J/measuresmoke.log" | grep -v -E '^MEASURESMOKE (plan|busy|disk|data|vendor=)' | head -n 20 || true
fi
say "measuresmoke exited $mrc (SOFT: the job's exit stays $grc)"
disk_line measuresmoke
# the smoke's own venvs and scratch: its summary and logs are in out/ and the job dir; the next job on this pod
# (a new tag, a new work dir) finds the room
rm -rf "$GATE_WORK/measure-venv" "$GATE_WORK/fixture-venv" "$GATE_WORK/measuresmoke-work"
disk_line "dropping the smoke's venvs and scratch"

PHASE=complete
exit "$grc"
