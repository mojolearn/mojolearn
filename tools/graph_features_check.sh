#!/usr/bin/env bash
# Qualification only. Never a performance timer or an implicit build.
# Save complete logs and reports under /root/results-* on rented machines;
# retain the directory on the Mac and R2 before teardown.
set -euo pipefail
mode=${1:?source or installed}
target=${2:?cpu, metal, cuda or hip}
output=${3:?absolute evidence directory}
test_python=${MOJOLEARN_TEST_PYTHON:?set the Python executable to qualify}
root=$(cd "$(dirname "$0")/.." && pwd)
case "$output" in /*) ;; *) echo 'evidence path must be absolute' >&2; exit 2 ;; esac
mkdir -p "$output"
git -C "$root" rev-parse HEAD > "$output/controller-source-head.txt"
export MOJOLEARN_NUMERIC_MODE=identical
case "$target" in
  cpu|cuda|hip) export MOJOLEARN_VENDOR="$target" ;;
  metal) unset MOJOLEARN_VENDOR ;;
  *) echo "unknown target: $target" >&2; exit 2 ;;
esac
args=(verify-graph --vendor "$target" --out "$output/report.json")
if [[ $# -ge 4 ]]; then args+=(--expect "$4"); fi
case "$mode" in
  source)
    export PYTHONPATH="$root/python"
    cd "$root"
    ;;
  installed)
    unset PYTHONPATH
    args+=(--require-installed)
    # A neutral cwd plus module path checks prevent checkout import evidence
    # from being presented as an installed wheel result.
    cd "$output"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
set +e
"$test_python" -m mojolearn "${args[@]}" > "$output/verify.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$output/verify.rc"
"$test_python" - "$output" "$rc" <<'PY'
import collections
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
report = json.loads((p / 'report.json').read_text()) if (p / 'report.json').exists() else {}
cases = report.get('cases', {})
summary = {'exit_code': int(sys.argv[2]), 'backend': report.get('backend', {}),
           'expected_cases': len(report.get('expected_cases', [])), 'observed_cases': len(cases),
           'counts': dict(collections.Counter(row['status'] for row in cases.values())),
           'failures': {name: row.get('error', '') for name, row in cases.items() if row['status'] != 'passed'},
           'digests': sum(len(row.get('outputs', [])) for row in cases.values()),
           'comparison': report.get('comparison'), 'comparator_controls': report.get('comparator_controls')}
(p / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print('GRAPH', summary['exit_code'], summary['counts'], 'cases', summary['observed_cases'],
      '/', summary['expected_cases'], 'digests', summary['digests'])
PY
exit "$rc"
