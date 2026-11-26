#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One BATCH of coverage sabotage arms on one box (lane precise-coverage-0859, 2026-10-26).
#
# `python3 tools/coverage_sabotage.py batch-plan --audit A.json --json PLAN.json` groups the
# unproven arms of tools/identity_lanes/coverage_sabotage.tsv into conflict-free batches: no
# arm's lanes can be moved by another arm's patch in the same batch, and the patches apply
# together. One sabotaged build per batch then proves every arm in it:
#
#   1. clean      the batch's lanes on this tree's built bindings, BOTH SIDES IN ONE PROCESS through the
#                 public device= (identity_break --side both: the GPU side to <tag>_device.json, the CPU
#                 side to <tag>_cpu.json; an installation without a GPU records --side cpu only). No
#                 MOJOLEARN_VENDOR child (verify through device=, 2026-11-24).
#   2. sabotaged  every patch of the batch applied, the batch's bindings rebuilt, the lanes again
#   3. restored   the patches reverted, the saved clean native files put back, the lanes again
#   4. check      tools/coverage_sabotage.py review (binds the arm's committed review, if any, to
#                 the three records of the side it faults) then check-sides per arm (--batch-bindings):
#                 the arm is CAUGHT when its lanes read DIVERGENT on the side it faults (its `side`
#                 column) and IDENTICAL to clean on the restored build, and every lane stays put on
#                 the other side (the fault sits in one side's code)
#
# usage: bash tools/coverage_sabotage_batch.sh <plan.json> <batch id>
# env:   PYTHON        the interpreter (default: `pixi run -e default python` when pixi and
#                      pixi.toml are present, else python3)
#        BUILD_PREFIX  a command prefix for the binding builds (default: `pixi run -e default`
#                      when pixi is present, else none)
#        OUT           the output directory (default: $HOME/mojolearn-sabotage/<batch id>)
#        FIXTURES      identity_break fixtures (default: base)
#        SAB_SCRATCH   where the clean native files are saved for the restore (default
#                      $HOME/mojolearn-sabotage-scratch; one <batch id> directory, deleted on exit). Never
#                      under OUT: the job persists OUT, and the tar of every native (about 120 MB a batch)
#                      made the nvc1-0003 results 5.5 GB, so its final upload never landed before the cap.
# Output: one line per arm, `SABOTAGE batch=<id> arm=<arm> side=<cpu|gpu|cpu+gpu> verdict=CAUGHT|GPU-OWED|MISSED`,
# and $OUT/summary.txt; the records and per-arm check logs stay in $OUT.
# The tree is left as it was found (patches reverted, native files restored) on every exit path.
set -u
PLAN=${1:?usage: coverage_sabotage_batch.sh <plan.json> <batch id>}
BATCH=${2:?usage: coverage_sabotage_batch.sh <plan.json> <batch id>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${OUT:-$HOME/mojolearn-sabotage/$BATCH}
FIXTURES=${FIXTURES:-base}
mkdir -p "$OUT"
SCRATCH=${SAB_SCRATCH:-$HOME/mojolearn-sabotage-scratch}/$BATCH
rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
if [ -n "${PYTHON:-}" ]; then
    read -r -a PY <<< "$PYTHON"
elif command -v pixi > /dev/null 2>&1 && [ -f pixi.toml ]; then
    PY=(pixi run -e default python)
else
    PY=(python3)
fi
if [ -n "${BUILD_PREFIX+x}" ]; then
    read -r -a BP <<< "$BUILD_PREFIX"
elif command -v pixi > /dev/null 2>&1 && [ -f pixi.toml ]; then
    BP=(pixi run -e default)
else
    BP=()
fi
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }
: > "$OUT/summary.txt"

field() { python3 - "$PLAN" "$BATCH" "$1" <<'EOF'
import json, sys
plan, bid, key = sys.argv[1:4]
b = [x for x in json.load(open(plan))["batches"] if x["id"] == bid]
if not b:
    sys.exit(f"no batch {bid} in {plan}")
v = b[0][key]
print(",".join(v) if isinstance(v, list) else v)
EOF
}
ARMS=$(field arms) || exit 2
PATCHES=$(field patches) || exit 2
BUILDS=$(field builds) || exit 2
LANES=$(field lanes) || exit 2
COLUMNS=$(field columns) || exit 2
log "BATCH $BATCH arms=$ARMS builds=$BUILDS columns=$COLUMNS lanes=$LANES head=$(git rev-parse --short HEAD)"
# A plan older than the arm table proves nothing: in nvc1-0003 gbdt-1 and gbdt-2 ran the plan's lanes, the
# arms had gained lanes since, and check refused nine arms ("requested lanes do not cover this arm") after
# the builds. So the batch stops here, before any build, when an arm's patch, bindings or lanes are not the
# plan's (lane sab-gpu-missed, 2026-11-23); `coverage_sabotage.py batch-plan` writes a current plan.
if ! stale=$(python3 - "$PLAN" "$BATCH" <<'EOF'
import json, sys
sys.path.insert(0, "tools")
import coverage_sabotage as cs
plan, bid = sys.argv[1:3]
b = [x for x in json.load(open(plan))["batches"] if x["id"] == bid][0]
table, bad = cs.arms(), []
for arm in b["arms"]:
    a = table.get(arm)
    if a is None:
        bad.append(arm + ": not in the arm table")
        continue
    if a["patch"] not in b["patches"]:
        bad.append(arm + ": patch " + a["patch"])
    bad += [arm + ": build " + x for x in a["bindings"] if x not in b["builds"]]
    bad += [arm + ": lane " + x for x in a["lanes"] if x not in b["lanes"]]
print("; ".join(bad))
sys.exit(1 if bad else 0)
EOF
); then
    log "PLAN STALE: $stale"; rm -rf "$SCRATCH"; exit 6
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    log "TREE NOT CLEAN"; rm -rf "$SCRATCH"; exit 4
fi
# NATIVE_DIRS: where the builds write native files (default python/mojolearn; add a host set dir if
# the box keeps one outside the tree)
# The development layout (tools/development_build.py) loads its groups through the index _mojolearn_merged.json
# beside them; a sabotaged development build re-points that index at the sabotaged group, so the index is part of
# the native state snapshotted and restored with the binaries (2026-11-23, nn-integrate: without it a tree built in
# the development layout kept loading the clean groups and every arm read unmoved).
natives() { find ${NATIVE_DIRS:-python/mojolearn} \( -name '*.so' -o -name '*.dylib' -o -name '_mojolearn_merged.json' \) -type f 2>/dev/null | sort; }
if command -v sha256sum > /dev/null 2>&1; then HASH=(sha256sum); else HASH=(shasum -a 256); fi
snap() { natives | while read -r f; do "${HASH[@]}" "$f"; done; }
natives > "$OUT/clean_natives.txt"
snap > "$OUT/clean_natives.sha"
tar -cf "$SCRATCH/clean_natives.tar" -T "$OUT/clean_natives.txt"

# Both sides run in ONE identity_break process (the lanes' device="cpu" and device="gpu" calls): the GPU
# side to <tag>_device.json, then the CPU side to <tag>_cpu.json. An installation without a GPU records the
# CPU side only (its GPU side reads not applicable; a gpu arm cannot be proven there).
GPU_SIDE=$(MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" "${PY[@]}" -c \
    'import mojolearn as ml; print("no" if ml.vendor() == "cpu" else "yes")' 2> "$OUT/side_probe.err" | tail -n 1)
log "GPU side in this installation: ${GPU_SIDE:-probe failed}"
run_cols() { # $1 = clean | sab | restored
    local tag=$1
    if [ "$GPU_SIDE" = yes ]; then
        MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" "${PY[@]}" tools/identity_break.py \
            --lanes "$LANES" --fixtures "$FIXTURES" --repeats 1 --side both \
            --json "$OUT/${tag}_device.json" --cpu-json "$OUT/${tag}_cpu.json" \
            > "$OUT/${tag}_sides.log" 2>&1
    else
        MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" "${PY[@]}" tools/identity_break.py \
            --lanes "$LANES" --fixtures "$FIXTURES" --repeats 1 --side cpu --json "$OUT/${tag}_cpu.json" \
            > "$OUT/${tag}_sides.log" 2>&1
    fi
    local rc=$?
    log "RUN $tag sides=$([ "$GPU_SIDE" = yes ] && echo both || echo cpu) rc=$rc"
}

APPLIED=0
restore() {
    if [ "$APPLIED" = 1 ]; then
        # shellcheck disable=SC2046
        git apply -R $(python3 -c 'import sys; print(" ".join(reversed(sys.argv[1].split(","))))' "$PATCHES") \
            && log "REVERTED" || log "REVERT FAILED"
        APPLIED=0
    fi
    natives | while read -r f; do grep -qxF "$f" "$OUT/clean_natives.txt" || rm -f "$f"; done
    tar -xf "$SCRATCH/clean_natives.tar"
    if [ "$(snap)" = "$(cat "$OUT/clean_natives.sha")" ]; then log "NATIVE RESTORED"; else log "NATIVE RESTORE FAILED"; fi
}
# A per-batch time limit (the job's `timeout`) sends TERM to this whole process group: the builds and the
# runs die, this shell stops them all and exits, and the EXIT trap restores the tree and the natives.
# No `wait` here: a child that ignores TERM must not hold the restore (the job's KILL follows after its grace).
# shellcheck disable=SC2046
stop_all() { log "STOPPED (signal): killing the batch's builds and runs"; kill $(jobs -p) 2>/dev/null; exit 143; }
RESTORED=0
trap '[ "$RESTORED" = 1 ] || restore; rm -rf "$SCRATCH"' EXIT
trap 'stop_all' INT TERM

run_cols clean
# shellcheck disable=SC2046
if ! git apply $(echo "$PATCHES" | tr ',' ' '); then
    log "APPLY FAILED"; exit 3
fi
APPLIED=1
# PARALLEL PLAN (AGENTS.md hard rules; lane sabotage-gpu, 2026-11-23: this loop was serial at
# MOJOLEARN_COMPILE_JOBS=1, one core of a rented box busy): the batch's binding builds are independent
# (each writes its own .so), so they all start at once and share the allocation's cores
# (SAB_CORES, default alloc_cores of tools/ops/box_cores_lib.sh: the cgroup quota, not the host's CPUs),
# each with MOJOLEARN_COMPILE_JOBS = cores. (2026-11-23, nvc1-0003: at cores / builds the builds of a batch kept
# mean 1.7 to 3.2 of 14 cores busy; one binding's `mojo build -j` does not fill its share, so each build may use
# every core and the kernel shares them. -j is a thread cap, not a reservation.)
. "$ROOT/tools/ops/box_cores_lib.sh"
CORES=${SAB_CORES:-$(alloc_cores)}
NB=$(echo "$BUILDS" | tr ',' '\n' | grep -c .)
CJ=$CORES; [ "$CJ" -lt 1 ] && CJ=1
log "BUILDS $NB at once, MOJOLEARN_COMPILE_JOBS=$CJ each, cores $CORES"
one_build() {
    local b=$1 t rc
    t=$(date +%s)
    # A host build refuses MOJOLEARN_GPU_ARCHS ("a CPU build takes no MOJOLEARN_GPU_ARCHS"): the GPU job exports
    # it for the device builds, and in nvc1-0003 all 28 host builds of the plan failed on it in 0 s (BUILD-FAIL).
    # No arm builds a `_fast` binding: the identical lanes never load one, and the only one an arm named
    # (build_tokenizer_fast) builds on macOS alone (lane sab-gpu-missed, 2026-11-23).
    if [[ $b == *_host ]]; then
        env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_COMPILE_JOBS=$CJ MOJOLEARN_TARGET_COLUMN=cpu "${BP[@]}" bash "bindings/$b.sh" > "$OUT/build_$b.log" 2>&1; rc=$?
    else
        env MOJOLEARN_COMPILE_JOBS=$CJ MOJOLEARN_NUMERIC_MODE=identical "${BP[@]}" bash "bindings/$b.sh" > "$OUT/build_$b.log" 2>&1; rc=$?
    fi
    echo "$rc" > "$OUT/build_$b.rc"
    log "BUILD $b rc=$rc secs=$(( $(date +%s) - t ))"
}
# PREBUILT (lane sab-triage-1, 2026-11-26; tools/sabotage_prebuild.py): the GPU job compiles every batch's patched
# bindings ahead, in scratch worktrees over all the allocation's cores, while the batches before it run. The batch
# then copies its changed natives over the tree's instead of compiling; the patches above are applied all the same and
# the restore below removes the copies exactly as it removes a build's outputs.
PRE=${SAB_PREBUILT:+$SAB_PREBUILT/$BATCH}
if [ -n "$PRE" ] && [ -f "$PRE/ready" ]; then
    for b in ${BUILDS//,/ }; do
        cp "$PRE/build_$b.rc" "$OUT/build_$b.rc" 2>/dev/null || echo 99 > "$OUT/build_$b.rc"
        cp "$PRE/build_$b.log" "$OUT/build_$b.log" 2>/dev/null
        log "BUILD $b rc=$(cat "$OUT/build_$b.rc") secs=$(cat "$PRE/build_$b.secs" 2>/dev/null || echo 0) (prebuilt)"
    done
    [ -d "$PRE/out" ] && cp -a "$PRE/out/." "$ROOT/"
    log "PREBUILT natives placed: $(find "$PRE/out" -type f 2> /dev/null | wc -l) files"
else
    bsampler=$(busy_sampler "$OUT/busy_cores_build.txt")
    for b in ${BUILDS//,/ }; do one_build "$b" & done
    wait
    kill "$bsampler" 2>/dev/null
    log "BUSY build: $(busy_of "$OUT/busy_cores_build.txt") of $CORES cores"
fi
brc=0
for b in ${BUILDS//,/ }; do
    rc=$(cat "$OUT/build_$b.rc" 2>/dev/null || echo 99)
    if [ "$rc" != 0 ]; then
        brc=$rc
        grep -m 3 -B 2 -A 8 -i 'error' "$OUT/build_$b.log" | cut -c1-240 | head -30 >> "$OUT/summary.txt"
    fi
done
[ $brc = 0 ] && run_cols sab
restore
RESTORED=1
if [ $brc != 0 ]; then
    for arm in ${ARMS//,/ }; do echo "SABOTAGE batch=$BATCH arm=$arm side=all verdict=BUILD-FAIL" | tee -a "$OUT/summary.txt"; done
    exit 5
fi
run_cols restored
miss=0
for arm in ${ARMS//,/ }; do
    side=$("${PY[@]}" tools/coverage_sabotage.py list | awk -F'\t' -v a="$arm" '$1==a {print $5}')
    sideargs=()
    for col in cpu device; do
        [ -f "$OUT/clean_$col.json" ] || continue
        s=$([ $col = cpu ] && echo cpu || echo gpu)
        sideargs+=(--$s-clean "$OUT/clean_$col.json" --$s-sabotaged "$OUT/sab_$col.json" --$s-restored "$OUT/restored_$col.json")
        # A committed review (tools/identity_lanes/coverage_sabotage_reviews.json) binds to the faulted side's
        # records only when the sabotaged cell refused with the reviewed first line under the reviewed patch bytes.
        case "+$side+" in *+$s+*)
            [ -f "$OUT/evidence_${arm}_$col.json" ] || "${PY[@]}" tools/coverage_sabotage.py review --arm "$arm" \
                --clean "$OUT/clean_$col.json" --sabotaged "$OUT/sab_$col.json" --restored "$OUT/restored_$col.json" \
                --out "$OUT/evidence_${arm}_$col.json" > "$OUT/review_${arm}_$col.txt" 2>&1
            [ -f "$OUT/evidence_${arm}_$col.json" ] && sideargs+=(--$s-evidence "$OUT/evidence_${arm}_$col.json");;
        esac
    done
    # An installation without a GPU never ran the GPU side: a cpu+gpu arm whose CPU side passes reads GPU-OWED
    # (its GPU half is the GPU plan's), never CAUGHT and never MISSED for the side it did not run (lane w3f7-b).
    nogpu=(); [ "$GPU_SIDE" = yes ] || nogpu=(--no-gpu)
    "${PY[@]}" tools/coverage_sabotage.py check-sides --arm "$arm" "${sideargs[@]}" ${nogpu[@]+"${nogpu[@]}"} \
        --batch-bindings "$BUILDS" > "$OUT/check_${arm}.txt" 2>&1
    v=$(tail -n 1 "$OUT/check_${arm}.txt" | awk '{print $3}')
    [ "$v" = CAUGHT ] || [ "$v" = GPU-OWED ] || miss=1
    echo "SABOTAGE batch=$BATCH arm=$arm side=${side:-?} verdict=${v:-ERROR}" | tee -a "$OUT/summary.txt"
    # The check's reasons ride with a miss in summary.txt (lane w3f5-e, 2026-11-24: gate 3cda8a05b3's copies
    # kept only summary.txt, so eight MISSED arms carried no cause: an unstable clean lane, an unmoved
    # fixture or a binary change outside the declared closure could not be told apart). Each reason line is kept to
    # 4000 characters (lane w3-fix-9, 2026-11-25: the 300-character cut left gate d1641666a4's summary with
    # "AssertionError: check-generic-" and no class named, and the per-arm check files were not kept).
    [ "$v" = CAUGHT ] || grep -e '^INVALID' -e '^OWED' -e 'moved' -e 'unmoved' "$OUT/check_${arm}.txt" | grep -v '^LANES ' | cut -c1-4000 | head -n 12 \
        | sed "s/^/  CHECK $arm: /" >> "$OUT/summary.txt"
    # The per-lane outcome lines whole (lane w3f7-b): a cut list cannot say which lanes to correct.
    [ "$v" = CAUGHT ] || grep -e '^LANES ' "$OUT/check_${arm}.txt" | sed "s/^/  CHECK $arm: /" >> "$OUT/summary.txt"
done
log "DONE"
exit $miss
