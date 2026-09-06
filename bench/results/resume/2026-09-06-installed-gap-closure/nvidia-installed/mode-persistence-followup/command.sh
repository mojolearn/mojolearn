#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cores=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$
D=/root/installed-candidate/mode-persistence-followup
R=/root/mojolearn-frozen
mkdir -p "$D"
[[ $(wc -l < /root/installed-candidate/candidate/qualification-normalized/results.tsv) = 24 ]]
cd /tmp
"$R/.pixi/envs/gbmbench/bin/python3" -m venv --system-site-packages "$D/venv"
PY="$D/venv/bin/python"
"$PY" -m pip install --disable-pip-version-check --no-deps /root/installed-candidate/candidate/normalized/*.whl > "$D/install.log" 2>&1
"$PY" - "$D" "$R" <<'PY'
import hashlib,json,sys,sysconfig
from pathlib import Path
out,root=map(Path,sys.argv[1:]);package=Path(sysconfig.get_path('purelib'))/'mojolearn'
installed=package/'ensemble.py';original=installed.read_bytes();assert original==(root/'python/mojolearn/ensemble.py').read_bytes()
original_hash=hashlib.sha256(original).hexdigest()
installed.write_bytes(Path('/tmp/mojolearn-new-ensemble.py').read_bytes())
proof=json.loads(Path('/root/installed-candidate/candidate/build/build-provenance.json').read_text())
for member,digest in proof['extensions'].items():
 p=package.parent/member
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,member
for source,target in [('/tmp/mojolearn-new-ensemble.py','ensemble.py'),('/tmp/mojolearn-new-ordered-gate.py','ordered_rmse_surface_check.py'),('/tmp/mojolearn-mode-persistence-followup.sh','command.sh')]:
 (out/target).write_bytes(Path(source).read_bytes())
record={'schema':'mojolearn.numeric-mode-source-overlay.v1','scope':'New Python serialization wrapper on built CUDA candidate with failed full qualification; modified installed Python payload, not a release-wheel qualification','source_commit':'29a8c848','base_native_source':proof['source_commit'],'base_wheel_sha256':json.loads(Path('/root/installed-candidate/candidate/qualification-normalized/wheel-audit.json').read_text())['sha256'],'original_ensemble_sha256':original_hash,'ensemble_sha256':hashlib.sha256(installed.read_bytes()).hexdigest(),'gate_sha256':hashlib.sha256((out/'ordered_rmse_surface_check.py').read_bytes()).hexdigest(),'native_extensions_unchanged':len(proof['extensions'])}
(out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
PY
: > "$D/results.tsv"
status=0
for mode in fast deterministic identical; do
  rc=0
  MOJOLEARN_NUMERIC_MODE=$mode "$PY" "$D/ordered_rmse_surface_check.py" > "$D/$mode.log" 2>&1 || rc=$?
  printf '%s\t%s\n' "$mode" "$rc" >> "$D/results.tsv"
  [[ "$rc" = 0 ]] || status=1
done
printf '%s\n' "$status" > "$D/exit_code"
rm -rf "$D/venv"
exit "$status"
