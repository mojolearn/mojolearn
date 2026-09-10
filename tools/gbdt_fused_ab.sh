#!/bin/sh
# Native symmetric Logloss A/B. Builds are isolated from installed bindings.
# This checks model/prediction/loss fingerprints, then reports fit medians.
# Parse the complete driver before execution; live edits must not change
# shell read offsets halfway through a long build/benchmark.
main() {
set -eu
cd "$(dirname "$0")/.."
# Hold the build lock across timing too, so parallel compilations cannot
# distort host scheduling or thermals during this window.
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]; then
    exec tools/with_build_lock.sh sh tools/gbdt_fused_ab.sh "$@"
fi
out=${1:-bench/results/gbdt_fused_ab}
mkdir -p "$out"
out=$(cd "$out" && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/gbdt-fused-ab.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM
for mode in fast identical; do
    for arm in baseline fused; do
        set --
        if [ "$mode" = identical ]; then
            set -- "$@" -D MOJOLEARN_NUMERIC_IDENTICAL=1
        fi
        if [ "$arm" = fused ]; then
            set -- "$@" -D MOJOLEARN_2030_FUSED_EST_MOVE=1
        else
            set -- "$@" -D MOJOLEARN_2030_NO_FUSED_EST_MOVE=1
        fi
        tools/with_build_lock.sh pixi run mojo build -I . "$@" \
            checks/gbdt_fused_fit_check.mojo -o "$build_dir/$mode-$arm" \
            > "$out/$mode-$arm.build.log" 2>&1
    done
done
tools/bench_lock.sh acquire gbdt-fused-ab 'native full-fit baseline/fused' '5 minutes'
trap 'tools/bench_lock.sh release; rm -rf "$build_dir"' EXIT HUP INT TERM
# Reverse the arm order on the second pass to expose thermal/order effects.
for pass in 1 2; do
    arms='baseline fused'
    if [ "$pass" = 2 ]; then arms='fused baseline'; fi
    for mode in fast identical; do
        for rows in 65537 1000003; do
            for arm in $arms; do
                MOJOLEARN_GBDT_AB_ROWS=$rows "$build_dir/$mode-$arm" > "$out/$mode-$arm.$rows.$pass.log" 2>&1
            done
        done
    done
done
tools/bench_lock.sh release
rm -rf "$build_dir"
trap - EXIT HUP INT TERM
python3 - "$out" <<'PY'
from pathlib import Path
import statistics
import sys
root = Path(sys.argv[1])
for rows in (65537, 1000003):
    for mode in ('fast', 'identical'):
        fingerprints = set()
        times = {}
        for arm in ('baseline', 'fused'):
            times[arm] = []
            for run in (1, 2):
                lines = (root / f'{mode}-{arm}.{rows}.{run}.log').read_text().splitlines()
                if f'numeric_mode {mode.upper()}' not in lines:
                    raise SystemExit(f'Compiled mode mismatch: {mode} {arm} {run}')
                expected_arm = 'True' if arm == 'fused' else 'False'
                if not any(line.startswith(f'fused {expected_arm} ') for line in lines):
                    raise SystemExit(f'Compiled fused flag mismatch: {mode} {arm} {run}')
                hashes = [line.split()[-1] for line in lines if line.startswith('fingerprint ')]
                samples = [float(line.split()[-1]) for line in lines if line.startswith('fit_ms ')]
                if len(hashes) != 6 or len(samples) != 5:
                    raise SystemExit(f'Incomplete fit run: {mode} {arm} {run}')
                fingerprints.update(hashes)
                times[arm].extend(samples)
            print(mode, rows, arm, 'median_ms', statistics.median(times[arm]), 'samples', times[arm])
        if len(fingerprints) != 1:
            raise SystemExit(f'Fingerprint mismatch in {mode}: {sorted(fingerprints)}')
        baseline = statistics.median(times['baseline'])
        fused = statistics.median(times['fused'])
        print(mode, rows, 'IDENTICAL fingerprints', next(iter(fingerprints)), 'fused_speedup', baseline / fused)
PY

}
main "$@"
