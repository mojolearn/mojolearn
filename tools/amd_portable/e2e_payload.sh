#!/bin/bash
# End-to-end prototype check on the AMD box, with the probe executable standing
# in for a binding: inject -> extraction build -> extract payload -> audit ->
# runtime materialize through COMGR (python/mojolearn/amd_portable.py) for
# gfx942 -> run the patched executable -> bits vs the native run. Also builds
# the payload for gfx90a/gfx950 through the same COMGR call (compile only).
set -u
H=/root/lq/amd-portable; E=$H/e2e; W=$H/wt
export PATH=/root/.pixi/bin:$PATH
rm -rf $E; mkdir -p $E/src $E/native $E/extraction
cp $H/pkernels.mojo $H/probe_native.mojo $E/src/
python3 $W/packaging/linux/amd_portable_payload.py inject $E/src > $E/inject.log 2>&1 || { echo "inject rc=$?"; exit 1; }
cat $E/inject.log
pixi run --manifest-path /root/mojolearn/pixi.toml mojo build -j 1 -I $E/src -o $E/extraction/probe.so $E/src/probe_native.mojo > $E/build_extraction.log 2>&1
echo "extraction build rc=$?"
cp $H/probe_native $E/native/probe.so
MV=$(pixi run --manifest-path /root/mojolearn/pixi.toml mojo --version 2>/dev/null)
python3 $W/packaging/linux/amd_portable_payload.py extract $E/native $E/extraction $E/payload \
  --llvm-bin /opt/rocm/llvm/bin --repo $W --mojo-version "$MV" > $E/extract.log 2>&1; echo "extract rc=$?"; tail -n 3 $E/extract.log
python3 $W/packaging/linux/amd_portable_payload.py audit $E/payload > $E/audit.log 2>&1; echo "audit rc=$?"; tail -n 1 $E/audit.log
cd $W/python && MOJOLEARN_AMD_PORTABLE_CACHE=$E/cache python3 - > $E/materialize.log 2>&1 <<'PY'
import json, hashlib
from pathlib import Path
from mojolearn import amd_portable as ap
root = Path("/root/lq/amd-portable/e2e/payload")
raw = (root / ap.MANIFEST).read_bytes()
actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in root.rglob("*") if p.is_file() and p.name != ap.MANIFEST}
doc = ap.validate_manifest(json.loads(raw), actual)
mh = hashlib.sha256(raw).hexdigest()
print("comgr", ap.comgr_version())
for gfx in ("gfx942", "gfx90a", "gfx950", "gfx942:sramecc+:xnack-"):
    try:
        out, rec = ap.materialize(root, doc, mh, gfx)
        print("MATERIALIZED", gfx, rec["isa"], out)
    except ap.AmdPortableError as exc:
        print("REFUSED", gfx, str(exc)[:300])
for gfx in ("gfx1100",):
    try:
        ap.family_target(doc, gfx); print("UNEXPECTED accept", gfx)
    except ap.AmdPortableError as exc:
        print("REFUSED", gfx, "(expected):", str(exc)[:120])
PY
echo "materialize rc=$?"; grep -E "comgr|MATERIALIZED|REFUSED|Error" $E/materialize.log | cut -c1-260
P=$(ls -d $E/cache/*/gfx942/comgr-*/probe.so)
$P > $E/run.out 2> $E/run.err; rc=$?
cmp -s $H/native.out $E/run.out && m=MATCH || m="DIFFER($(diff $H/native.out $E/run.out | grep -c '^<'))"
echo "E2E patched probe on gfx942 rc=$rc vs native: $m"
