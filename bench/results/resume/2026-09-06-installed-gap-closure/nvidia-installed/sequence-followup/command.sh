#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MOJOLEARN_COMPILE_JOBS=2 MAX_JOBS=2
cores=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$
R=/root/mojolearn-sequence-followup
BASE=/root/mojolearn-frozen
D=/root/installed-candidate/sequence-followup
mkdir -p "$D"
[[ $(wc -l < /root/installed-candidate/candidate/qualification-normalized/results.tsv) = 24 ]]
"$BASE/.pixi/envs/gbmbench/bin/python3" -m venv --system-site-packages "$D/venv"
PY="$D/venv/bin/python"
"$PY" -m pip install --disable-pip-version-check --no-deps /root/installed-candidate/candidate/normalized/*.whl > "$D/install.log" 2>&1
PKG=$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib")+"/mojolearn")')
export LD_LIBRARY_PATH="$PKG/.libs:${LD_LIBRARY_PATH:-}"
cp /tmp/mojolearn-sequence-followup.patch "$D/source.patch"
cp /tmp/mojolearn-sequence-followup.sh "$D/command.sh"
cp /root/installed-candidate/candidate/build/build-provenance.json "$D/base-build-provenance.json"
: > "$D/results.tsv"
cd "$R"
for mode in fast deterministic identical; do
  defs=()
  case "$mode" in deterministic) defs=(-D MOJOLEARN_NUMERIC_DETERMINISTIC=1);; identical) defs=(-D MOJOLEARN_NUMERIC_IDENTICAL=1);; esac
  dest="$PKG/cuda/sm_89"
  [[ "$mode" = fast ]] || dest="$dest/$mode"
  for binding in mamba transformer; do
    rc=0
    timeout -k 10 180 "$BASE/.pixi/envs/gbmbench/bin/mojo" build -j 2 --emit shared-lib --target-cpu x86-64-v3 "${defs[@]}" -I . -I bindings "bindings/_mojolearn_$binding.mojo" -o "$D/$mode-$binding.so" > "$D/$mode-$binding-build.log" 2>&1 || rc=$?
    printf 'build\t%s\t%s\t%s\n' "$binding" "$mode" "$rc" >> "$D/results.tsv"
    [[ "$rc" = 0 ]] || continue
    cp "$D/$mode-$binding.so" "$dest/_mojolearn_$binding.so"
    rc=0
    (cd /tmp && MOJOLEARN_NUMERIC_MODE=$mode PYTHONUNBUFFERED=1 timeout -k 5 45 "$PY" "$BASE/python/mojolearn/tests/test_${binding}_surface.py") > "$D/$mode-$binding.log" 2>&1 || rc=$?
    printf 'gate\t%s\t%s\t%s\n' "$binding" "$mode" "$rc" >> "$D/results.tsv"
  done
done
"$PY" - "$D" "$PKG" <<'PY'
from pathlib import Path
import json,hashlib,sys
out,pkg=map(Path,sys.argv[1:]);rows=[r.split('\t') for r in (out/'results.tsv').read_text().splitlines()]
record={'schema':'mojolearn.sequence-source-overlay.v1','scope':'Six rebuilt bindings over frozen CUDA candidate; not a full45 rebuilt/qualified release wheel','base_source_commit':'eb835021dcd79a59a7e8f78c754a75db3c1fea83','status':'PASSED' if len(rows)==12 and all(r[-1]=='0' for r in rows) else 'FAILED','results':rows,'retained_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},'installed_extension_hashes':{str(p.relative_to(pkg)):hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.rglob('_mojolearn*.so')}}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
PY
rm -rf "$D/venv"
