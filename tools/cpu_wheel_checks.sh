#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tools/cpu_wheel_checks.sh WHEEL SHA256 SOURCE_COMMIT OUT [RUNTIME_PRIOR_DIR]
#
# THE CPU CHECKS OF THE ONE RELEASE RUN (tools/release.py --identity-on-wheels, Andrew 2026-11-05: every check
# that needs no GPU card runs on GitHub's free runners), against the built core wheel, from a clean tree at
# SOURCE_COMMIT. .github/workflows/cpu-identity-gate.yml (mode wheel, job wheel-checks) runs it; the cpu identity
# column itself runs in the same workflow's wheel-column matrix (tools/identity_wheel_record.py --columns cpu).
# In order, each into OUT/<name>/ with its own rc in OUT/checks.json:
#   install    a fresh venv, the wheel after its sha256 check, `import mojolearn` reads vendor cpu and the
#              wheel's identity_columns/COMMIT == SOURCE_COMMIT;
#   correctness  tools/qualify_verifier_wheel.py --scope cpu-only --no-identity-lanes (install guard, API
#              coverage, the CPU loaded-language-model check; no identity lane: the cpu column ran those);
#   runtime    tools/release_runtime_installed.py run --vendor cpu (per-unit receipts; a unit whose wheel bytes or
#              closure are unchanged since its last good cpu receipt in RUNTIME_PRIOR_DIR is reused, not run);
#   autograd   tools/autograd_smoke.py run --vendor cpu (the Python entries through the host bindings; the
#              compiled device-check executables stay on the GPU box);
#   adapters   tools/adapter_gate.py's smoke with --device cpu (torch and MAX on the CPU).
# Exit 0 only when every check passed. Every check runs even after one failed (one pass, all reports).
set -uo pipefail
[ $# -ge 4 ] || { echo "usage: cpu_wheel_checks.sh WHEEL SHA256 SOURCE_COMMIT OUT [RUNTIME_PRIOR_DIR]" >&2; exit 2; }
WHEEL=$1; WANT=$2; COMMIT=$3; OUT=$4; PRIOR=${5:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "cpu_wheel_checks: bad SOURCE_COMMIT" >&2; exit 2; }
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || { echo "cpu_wheel_checks: the tree is not at $COMMIT" >&2; exit 2; }
got=$(sha256sum "$WHEEL" | cut -d' ' -f1)
[ "$got" = "$WANT" ] || { echo "cpu_wheel_checks: $WHEEL is $got, not the release's $WANT" >&2; exit 2; }
mkdir -p "$OUT"
unset PYTHONPATH
export MOJOLEARN_VENDOR=cpu MOJOLEARN_NUMERIC_MODE=identical
PY=${PYTHON:-python3}
declare -A RC
check() { local name=$1; shift; mkdir -p "$OUT/$name"; "$@" > "$OUT/$name/log.txt" 2>&1; RC[$name]=$?
          echo "cpu-checks: $name rc=${RC[$name]}"; }

venv="$OUT/install/venv"
install() {
    "$PY" -m venv --clear "$venv" && "$venv/bin/python" -m pip install -q --no-cache-dir "$WHEEL" numpy \
    && "$venv/bin/python" -c 'import pathlib, sys, mojolearn
assert mojolearn.vendor() == "cpu", mojolearn.vendor()
c = (pathlib.Path(mojolearn.__file__).parent / "identity_columns" / "COMMIT").read_text().strip()
assert c == sys.argv[1], (c, sys.argv[1]); print("INSTALLED", mojolearn.__version__, c)' "$COMMIT"
}
check install install
version=$("$venv/bin/python" -c 'import mojolearn; print(mojolearn.__version__)' 2>/dev/null || echo unknown)
check correctness "$PY" tools/qualify_verifier_wheel.py --installed-python "$venv/bin/python" \
    --expected-version "$version" --expected-source-commit "$COMMIT" --scope cpu-only --no-identity-lanes \
    --output "$OUT/correctness/report"
prior=(); [ -n "$PRIOR" ] && [ -d "$PRIOR" ] && prior=(--prior-units "$PRIOR")
check runtime "$PY" tools/release_runtime_installed.py run --vendor cpu --source-root "$ROOT" \
    --source-commit "$COMMIT" --wheel "$WHEEL" --device-class "cpu-linux-$(uname -m)" \
    --unit-store "$OUT/runtime/units" ${prior[@]+"${prior[@]}"} --out "$OUT/runtime/report"
check autograd "$PY" tools/autograd_smoke.py run --vendor cpu --bin-dir "$OUT/autograd/none" \
    --python "$venv/bin/python" --out "$OUT/autograd/autograd-smoke.json"
check adapters "$PY" -c 'import sys; sys.path.insert(0, "tools"); import adapter_gate as AG
rec = AG.run("cpu", sys.argv[1], sys.argv[2], [sys.argv[3]])
print(rec.get("status"), rec.get("why", "")); sys.exit(0 if rec.get("ok") else 1)' "$PY" "$OUT/adapters/smoke" "$WHEEL"

ok=1
{ printf '{"schema":"mojolearn.cpu-wheel-checks.v1","source_commit":"%s","wheel_sha256":"%s","checks":{' "$COMMIT" "$WANT"
  sep=""; for n in install correctness runtime autograd adapters; do printf '%s"%s":%d' "$sep" "$n" "${RC[$n]}"; sep=,
      [ "${RC[$n]}" = 0 ] || ok=0; done
  printf '}}\n'; } > "$OUT/checks.json"
[ "$ok" = 1 ] && echo "cpu-checks: PASSED" || echo "cpu-checks: FAILED (see $OUT/checks.json)"
[ "$ok" = 1 ]
