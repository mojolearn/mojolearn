#!/bin/bash
# tools/ops/native_checks_steps.sh <step> -- THE STEP BODIES OF .github/workflows/native-checks.yml, in one script that
# the workflow and the rented CPU box (tools/ops/cpu_box_checks_job.sh, tools/cpu_central.sh check) both call, so the
# check logic has one copy (lane fx-cpu-box-checks, 2026-11-19; before, the bodies were inline YAML and the parse body
# the workflow env PARSE_GROUP_SH).
#
# Run it with the checkout UNDER TEST as the working directory, from whichever tree holds this script: on GitHub the
# workflow's own commit (checked out sparse at .wf/, github.workflow_sha), on the box the box's tools tree. So the
# workflow on main still runs on any older ref it checks out, as the inline PARSE_GROUP_SH did.
#
#   plan          IN_SUITE all|checks|host-parse|apple-parse, IN_FILTER, IN_COLUMN -> plan.json planned.txt host.json
#                 device.json apple.json in the working directory; the job outputs go to $GITHUB_OUTPUT when it is set
#   parse-group   KIND host|device|apple-fast|apple-identical|apple-host, STEM, MEMBERS, KEY, PARSE_LIMIT_SECONDS;
#                 the apple kinds also read CLS (the plan's flags class: default|metal1|host) and APPLE_ACCEL; writes
#                 PARSE_OUT (default out): <KEY>.log, result-<KEY>.txt, memory-<KEY>.txt; exit = the parse's rc
#
# THE APPLE KINDS (lane fx-apple-fast-ci, 2026-11-20): the Apple sets of bindings/merged_build.sh on macOS, elaborated.
# apple-fast is the legacy fast set (NO -D MOJOLEARN_NUMERIC_IDENTICAL: the `afn_*` routes, POLICY_BUILD_LEGACY_FAST),
# apple-identical the Apple column of the identical set, apple-host the CPU column on the Apple CPU. Flags are those
# of merged_build.sh's Darwin branch: --target-cpu apple-m1; a `metal1` file adds --target-accelerator metal:1; a
# `default` file passes NO accelerator on the release Mac (the compiler reads the Mac's GPU). A GitHub-hosted Mac is
# a VM whose GPU the compiler does not know ("Unknown GPU architecture detected", wheel-ci.yml's header), so here a
# `default` file names its accelerator: APPLE_ACCEL (default apple-m2, the M2 Pro that records the Apple column;
# `native` passes no flag, exactly the release line). On a Linux runner the same kinds are CROSS-elaborated
# (--target-triple arm64-apple-macosx11.0): no Apple machine at all; whether the compiler accepts that is what
# `suite=apple-parse` of native-checks.yml was written to find out. Either way this is parse, type check and
# elaboration only: no AIR, no metallib, no link, nothing run.
#   build         IN_COLUMN; writes NC_OUT (default out)/{results,logs} and NC_BIN (default bin)/run_<column>
#   shard         TASKS (check names), IN_TIMEOUT, IN_COLUMN; runs NC_BIN/run_<column> once per name into NC_OUT
#   summary       SOURCE_SHA; MERGED (default merged) holds results/ and planned.txt -> SUMMARY.md, REPORT.txt
#   verdict       DEVICE_MATRIX, RETRY_MATRIX; PARSE_DIR (default parse) holds device-parse[-retry]-<KEY>/result-<KEY>.txt;
#                 prints the device groups' table; exit 1 when a group's last attempt did not pass
# $GITHUB_STEP_SUMMARY is appended to when it is set.
step=${1:-}
GH_SUMMARY=${GITHUB_STEP_SUMMARY:-/dev/null}

step_plan() {
    set -euo pipefail
    case "$IN_SUITE" in all|checks|host-parse|apple-parse) ;; *) echo "suite must be all, checks, host-parse or apple-parse" >&2; exit 2 ;; esac
    # The Apple sets (suite apple-parse only, never part of `all`): fast and identical device files and the host
    # files of the Darwin plan, the byte LM in (merged_build.sh's Darwin default). APPLE_SETS picks the sets;
    # APPLE_UNITS_PER_JOB deals a file's units into parts of at most that many (0 = whole files), as PARSE_SPLIT does.
    echo '[]' > apple.json
    if [ "$IN_SUITE" = apple-parse ]; then
        python3 "$(dirname "$0")/apple_fast_check.py" plan --sets "${APPLE_SETS:-fast,identical}" \
            --units-per-job "${APPLE_UNITS_PER_JOB:-0}" > apple.json
    fi
    if [ "$IN_SUITE" = host-parse ] || [ "$IN_SUITE" = apple-parse ]; then
        echo '{"count": 0, "shards": 0, "matrix": {"include": []}}' > plan.json; : > planned.txt
    else
        python3 tools/ops/native_checks_report.py plan --column "$IN_COLUMN" --filter "$IN_FILTER" \
            --tasks-out planned.txt > plan.json
    fi
    if [ "$IN_SUITE" = checks ] || [ "$IN_SUITE" = apple-parse ]; then
        echo '[]' > host.json; echo '[]' > device.json
    else
        # the release identical device groups (the release sets export MOJOLEARN_PACKAGE_BYTE_LM=1). PARSE_SPLIT: a
        # group parsed as N jobs, its units dealt round-robin (the parse of each part covers its units' code; the
        # optimised release build of the whole group stays the check of the group as one module). Only the trees
        # group: the one that lost runners and the slowest by 5 minutes (see the workflow's header).
        python3 tools/merged_bindings.py plan --dir identical --platform linux --byte-lm 1 --layout release \
            | python3 -c '
import json, sys
PARSE_SPLIT = {"_mojolearn_native_trees": 2}
out = []
for line in sys.stdin:
    if not line.strip():
        continue
    stem, units = line.split()[0], line.split()[2].split(",")
    n = min(PARSE_SPLIT.get(stem, 1), len(units))
    for i in range(n):
        key = stem if n == 1 else f"{stem}_p{i + 1}of{n}"
        out.append(dict(stem=stem, key=key, members=",".join(units[i::n])))
print(json.dumps(out))' > device.json
        # the release host groups, the plan bindings/merged_build.sh host builds on Linux (byte LM off on Linux)
        python3 tools/merged_bindings.py plan --dir host --platform linux --byte-lm 0 --layout release \
            | python3 -c 'import json,sys; print(json.dumps([dict(stem=l.split()[0], key=l.split()[0], members=l.split()[2]) for l in sys.stdin if l.strip()]))' > host.json
    fi
    local o=${GITHUB_OUTPUT:-/dev/null}
    echo "host_matrix=$(cat host.json)" >> "$o"
    echo "host_count=$(python3 -c 'import json;print(len(json.load(open("host.json"))))')" >> "$o"
    python3 -c 'import json;print(len(json.load(open("host.json"))),"host groups to parse")'
    echo "device_matrix=$(cat device.json)" >> "$o"
    echo "device_count=$(python3 -c 'import json;print(len(json.load(open("device.json"))))')" >> "$o"
    python3 -c 'import json;print(len(json.load(open("device.json"))),"device groups to parse")'
    echo "apple_matrix=$(cat apple.json)" >> "$o"
    echo "apple_count=$(python3 -c 'import json;print(len(json.load(open("apple.json"))))')" >> "$o"
    python3 -c 'import json;print(len(json.load(open("apple.json"))),"Apple parse jobs")'
    echo "count=$(python3 -c 'import json;print(json.load(open("plan.json"))["count"])')" >> "$o"
    echo "matrix=$(python3 -c 'import json;print(json.dumps(json.load(open("plan.json"))["matrix"]))')" >> "$o"
    echo "sha=$(git rev-parse HEAD)" >> "$o"
    python3 -c 'import json;p=json.load(open("plan.json"));print(p["count"],"checks in",p["shards"],"shards")'
}

# One group's parse, shared by host-parse, device-parse and device-parse-retry (and the CPU box). Inputs: KIND
# host|device, STEM (the merged module), MEMBERS (its units, or one part of them), KEY (stem, or stem_p<i>of<n>).
step_parse_group() {
    set +e -u
    local out=${PARSE_OUT:-out}
    mkdir -p "$out"
    sys=$(uname -s)
    cores=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)
    target="--target-cpu x86-64-v3 --target-features +avx,+avx2,+bmi,+bmi2,+cx16,+f16c,+fma,+lzcnt,+movbe,+popcnt,+sahf,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave"
    [ "$(uname -m)" = x86_64 ] || target=""
    case "$KIND" in
        # bindings/merged_build.sh host: the host set's defines and the wheel's x86-64-v3 baseline
        host) flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU"; label="host-parse $KEY" ;;
        # bindings/merged_build.sh gpu on the cuda sm_89 leg: IDENTICAL, the NVIDIA column, one accelerator
        device) flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA --target-accelerator sm_89"; label="device-parse $KEY sm_89" ;;
        # bindings/merged_build.sh on macOS (the header of this file): the Apple CPU, the file's flags class
        apple-fast|apple-identical|apple-host)
            target="--target-cpu apple-m1"
            # off macOS: cross-elaborate for the Apple triple (the floor merged_build.sh links for, 11.0)
            [ "$sys" = Darwin ] || target="--target-triple arm64-apple-macosx11.0 --target-cpu apple-m1"
            accel=${APPLE_ACCEL:-apple-m2}
            # no Mac, no GPU to read: `native` can only mean a named accelerator there
            [ "$sys" = Darwin ] || [ "$accel" != native ] || accel=apple-m2
            case "${CLS:-default}" in
                metal1) accel_flags="--target-accelerator metal:1" ;;
                host) accel_flags="" ;;
                default) accel_flags="--target-accelerator $accel"; [ "$accel" != native ] || accel_flags="" ;;
                *) echo "CLS must be default, metal1 or host, got '${CLS:-}'" >&2; exit 2 ;;
            esac
            case "$KIND" in
                apple-fast) flags="$accel_flags" ;;
                apple-identical) flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1 $accel_flags" ;;
                apple-host) flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU" ;;
            esac
            label="$KIND $KEY (${CLS:-default}; ${accel_flags:-no accelerator flag}; $sys)" ;;
        *) echo "KIND must be host, device, apple-fast, apple-identical or apple-host, got '$KIND'" >&2; exit 2 ;;
    esac
    limit=${PARSE_LIMIT_SECONDS:-1320}
    start=$(date +%s)
    result() {  # kind rc: the line device-parse-triage and device-parse-verdict read
        printf 'key=%s stem=%s kind=%s rc=%s seconds=%s %s\n' "$KEY" "$STEM" "$1" "$2" "$(( $(date +%s) - start ))" \
            "$(cat "$out/memory-$KEY.txt" 2>/dev/null)" > "$out/result-$KEY.txt"
    }
    python3 tools/generated_sources.py prepare --repo "$PWD" < /dev/null > "$out/generated-sources.log" 2>&1 \
        || { tail -n 20 "$out/generated-sources.log"; result setup 1; exit 1; }
    python3 tools/merged_bindings.py root --stem "$STEM" --members "$MEMBERS" --out "$out/$KEY.mojo" < /dev/null \
        || { result setup 1; exit 1; }
    # memory: host RAM in use (Linux: MemTotal - MemAvailable; macOS: active + wired + compressed pages) and swap in
    # use, sampled every 5 s; a line every minute
    cg="/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2>/dev/null)"
    mem_sample() {  # prints "<used kB> <swap used kB>"
        if [ "$sys" = Darwin ]; then
            vm_stat | awk -v ps="$(sysctl -n hw.pagesize)" '/^Pages active/{a=$3} /^Pages wired down/{w=$4} /^Pages occupied by compressor/{c=$5} END{printf "%d ", (a + w + c) * ps / 1024}'
            sysctl -n vm.swapusage | awk '{v=$6; sub(/M$/, "", v); printf "%d\n", v * 1024}'
        else
            awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} /^SwapTotal:/{st=$2} /^SwapFree:/{sf=$2} END{print t-a, st-sf}' /proc/meminfo
        fi
    }
    (
        peak=0; swpeak=0; n=0
        while :; do
            read -r used_kb sw_kb < <(mem_sample)
            used=$(( ${used_kb:-0} * 1024 )); sw=$(( ${sw_kb:-0} * 1024 ))
            (( used > peak )) && peak=$used
            (( sw > swpeak )) && swpeak=$sw
            printf 'host_used_peak_bytes=%s swap_used_peak_bytes=%s\n' "$peak" "$swpeak" > "$out/memory-$KEY.txt"
            (( n % 12 == 0 )) && echo "memory t=$(( n * 5 ))s used_gib=$(( used >> 30 )) swap_gib=$(( sw >> 30 )) peak_gib=$(( peak >> 30 )) swap_peak_gib=$(( swpeak >> 30 ))"
            n=$(( n + 1 )); sleep 5
        done
    ) &
    sampler=$!
    # timeout signals the compile's whole process group (pixi and mojo), KILL 60 s later if it ignores TERM. macOS
    # ships no `timeout`: the same rule there in a few lines of Python (own session, TERM, 60 s, KILL, rc 124).
    limited() {
        if command -v timeout > /dev/null 2>&1; then
            timeout -k 60 "$@"
        else
            python3 -c '
import os, signal, subprocess, sys
p = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    sys.exit(p.wait(timeout=float(sys.argv[1])))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    sys.exit(124)' "$@"
        fi
    }
    echo "$label: mojo build -j $cores --emit llvm --no-optimization $flags $target" | tee "$out/flags-$KEY.txt"
    limited "$limit" pixi run mojo build -j "$cores" --emit llvm --no-optimization $flags \
        $target -I . -I bindings "$out/$KEY.mojo" -o "$out/$KEY.ll" > "$out/$KEY.log" 2>&1
    rc=$?
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    kind=compile
    if [ $rc -eq 124 ]; then
        kind=timeout; echo "::error::$label passed its ${limit}s limit (stuck or paging): killed"
    elif [ $rc -eq 137 ]; then
        kind=killed; echo "::error::$label was killed (SIGKILL: the out-of-memory killer or the limit's KILL)"
    # Parse, type check and elaboration are done when lowering starts. Without optimisation the lowering refuses
    # llvm.masked.gather with a non-constant alignment (the optimised release build folds it): a group whose only
    # errors are those is elaborated, so it passes (2026-11-16: _mojolearn_native_host and _native_x_host, both
    # built green by the release leg at 26709df246).
    elif [ $rc -ne 0 ] && grep -q 'error:' "$out/$KEY.log" && ! grep -E 'error:' "$out/$KEY.log" \
        | grep -q -v -E "failed to legalize operation 'pop.call_llvm_intrinsic'|llvm.masked.gather alignment must be a constant|could not lower funcs to LLVM|LowerToLLVMPipeline"; then
        echo "$label: elaborated; unoptimised lowering refused masked.gather only (not a source error)"
        rc=0
    fi
    [ $rc -eq 0 ] && kind=pass
    mem="$(cat "$out/memory-$KEY.txt" 2>/dev/null) cgroup_peak_bytes=$(cat "$cg/memory.peak" 2>/dev/null || echo unavailable) cgroup_swap_peak_bytes=$(cat "$cg/memory.swap.peak" 2>/dev/null || echo unavailable)"
    echo "$mem" > "$out/memory-$KEY.txt"
    result "$kind" "$rc"
    rm -f "$out/$KEY.ll"
    echo "$label rc=$rc kind=$kind seconds=$(( $(date +%s) - start )) members=$MEMBERS $mem" | tee -a "$GH_SUMMARY"
    grep -m 20 -E 'error:' "$out/$KEY.log"
    tail -n 5 "$out/$KEY.log"
    exit "$rc"
}

step_build() {
    set +e -u
    local out=${NC_OUT:-out} bindir=${NC_BIN:-bin}
    mkdir -p "$out/results" "$out/logs" "$bindir"
    defines="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    [ "$IN_COLUMN" = cpu ] && defines="$defines -D MOJOLEARN_COLUMN_CPU=1"
    # the device column counts fitted uploads for resident_second_call (core/resident_upload_count.mojo)
    [ "$IN_COLUMN" = cpu ] || defines="$defines -D MOJOLEARN_RESIDENT_UPLOAD_COUNT=1"
    # The wheel's Linux x86-64 baseline (bindings/build_mamba.sh TARGET_FLAGS). Without it `mojo build`
    # targets the build runner's own chip, and the shards that land on an older runner trap on its first
    # instruction: at 6906a59111 (run 43722561124) shards 0 and 4 SIGILLed on all 49 of their checks (rc 132,
    # 0-1 s, same main+0x7b06) while the other four shards passed every check.
    target="--target-cpu x86-64-v3 --target-features +avx,+avx2,+bmi,+bmi2,+cx16,+f16c,+fma,+lzcnt,+movbe,+popcnt,+sahf,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave"
    [ "$(uname -m)" = x86_64 ] || target=""
    pixi run mojo --version > "$out/compiler-version.log" 2>&1
    start=$(date +%s)
    # The gitignored generated modules (the archive Unicode table the archive_* checks import, the tokenizer
    # table), written exactly as the binding builds write them (bindings/merged_build.sh).
    python3 tools/generated_sources.py prepare --repo "$PWD" < /dev/null > "$out/logs/generated-sources.log" 2>&1
    rc=$?
    if [ "$rc" = 0 ]; then
        pixi run mojo build $defines $target -I . -o "$bindir/run_$IN_COLUMN" "checks/native_tests/run_$IN_COLUMN.mojo" \
            > "$out/logs/build-$IN_COLUMN.log" 2>&1
        rc=$?
    else
        cp "$out/logs/generated-sources.log" "$out/logs/build-$IN_COLUMN.log"
    fi
    python3 tools/ops/native_checks_report.py result "build-$IN_COLUMN" --log "$out/logs/build-$IN_COLUMN.log" \
        --rc "$rc" --seconds "$(( $(date +%s) - start ))" --out-dir "$out/results"
    echo "build rc=$rc"
    exit "$rc"
}

step_shard() {
    set +e -u  # GitHub's default bash runs with -e: a failing check must not end the shard
    local out=${NC_OUT:-out} bindir=${NC_BIN:-bin}
    mkdir -p "$out/results" "$out/logs"
    bin="$bindir/run_$IN_COLUMN"
    case "$bin" in /*) ;; *) bin="./$bin" ;; esac
    chmod +x "$bin"
    for name in $TASKS; do
        start=$(date +%s)
        timeout -k 30 "${IN_TIMEOUT:-900}" pixi run "$bin" "$name" > "$out/logs/$name.log" 2>&1
        rc=$?
        python3 tools/ops/native_checks_report.py result "$name" --log "$out/logs/$name.log" --rc "$rc" \
            --seconds "$(( $(date +%s) - start ))" --out-dir "$out/results"
        echo "$name rc=$rc"
    done
    exit 0
}

step_summary() {
    set +e -u
    local merged=${MERGED:-merged}
    printf 'source: %s\n' "${SOURCE_SHA:-unknown}" > notes.txt
    [ -z "${SUMMARY_NOTES:-}" ] || cat "$SUMMARY_NOTES" >> notes.txt
    python3 tools/ops/native_checks_report.py summary "$merged/results" --planned "$merged/planned.txt" \
        --notes notes.txt --out SUMMARY.md
    rc=$?
    cat SUMMARY.md >> "$GH_SUMMARY"
    python3 tools/ops/native_checks_report.py report "$merged/results" > REPORT.txt || true
    head -n 60 REPORT.txt
    exit "$rc"
}

# The device groups' verdict: each group's last attempt (the retry when there was one) must pass.
step_verdict() {
    PARSE_DIR=${PARSE_DIR:-parse} GH_SUMMARY=$GH_SUMMARY python3 - <<'PY'
import json, os, pathlib
def result(art, key):
    f = pathlib.Path(os.environ["PARSE_DIR"]) / art / f"result-{key}.txt"
    if not f.exists():
        return {"kind": "missing"}
    return dict(x.split("=", 1) for x in f.read_text().split() if "=" in x)
retried = {e["key"] for e in json.loads(os.environ.get("RETRY_MATRIX") or "[]")}
bad = 0
lines = ["| group | attempt | kind | rc | seconds | host RAM peak GiB | swap peak GiB |", "|---|---|---|---|---|---|---|"]
for e in json.loads(os.environ["DEVICE_MATRIX"]):
    k = e["key"]
    first = result(f"device-parse-{k}", k)
    last = result(f"device-parse-retry-{k}", k) if k in retried else first
    gib = lambda v: f"{int(v) / 2**30:.1f}" if str(v).isdigit() else "-"
    lines.append(f"| {k} | {'retry' if k in retried else 'first'} (first: {first.get('kind')}) | "
                 f"{last.get('kind')} | {last.get('rc', '-')} | {last.get('seconds', '-')} | "
                 f"{gib(last.get('host_used_peak_bytes'))} | {gib(last.get('swap_used_peak_bytes'))} |")
    bad += last.get("kind") != "pass"
text = "\n".join(lines)
print(text)
with open(os.environ["GH_SUMMARY"], "a") as o:
    o.write("device-parse verdict\n\n" + text + "\n")
raise SystemExit(f"device-parse: {bad} group(s) failed on their last attempt" if bad else 0)
PY
}

case "$step" in
    plan) step_plan ;;
    parse-group) step_parse_group ;;
    build) step_build ;;
    shard) step_shard ;;
    summary) step_summary ;;
    verdict) step_verdict ;;
    *) sed -n 2,36p "$0"; exit 2 ;;
esac
