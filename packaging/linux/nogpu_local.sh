#!/usr/bin/env bash
# GATE (d), THE HONEST NO-GPU TEST, RUN ON THE MAC.
#
#   bash packaging/linux/nogpu_local.sh bench/results/wheels/<stamp>-<vendor>/wheels/sets/<vendor>
#
# Builds a package root out of the tracked `.py` files and ONE fetched
# binary set, then imports it inside `python:3.12-slim` on linux/amd64 with
# NO device passed through, and requires the import to FAIL naming the
# device nodes it looked for, `MOJOLEARN_VENDOR`, and NO SUPPORTED GPU FOUND.
# One `docker run --cpus 2`. Nothing is compiled, rented or published.
#
# WHY IT EXISTS RATHER THAN LIVING ONLY IN `nogpu.py` ON THE BOX. Both GPU
# boxes are the WRONG PLACE to ask "what happens with no GPU":
#
#   * the RunPod pod is a container, so `unshare` is refused by seccomp and
#     there is no docker daemon inside it; nogpu.py records NOT TESTED.
#   * the DigitalOcean droplet is a VM, so `unshare` runs -- and on
#     2026-08-30 it ran and reported FAIL, which was the GATE being wrong,
#     not the library. nogpu.py's namespace route bind-mounts /dev/null over
#     each device node, and A BIND MOUNT LEAVES THE PATH EXISTING. The probe
#     asks `os.path.exists("/dev/kfd")`, which is still True when /dev/kfd
#     IS /dev/null. That route cannot pass by construction, whatever the
#     library does. See the note above the namespace block in nogpu.py.
#
# A box with no GPU is the one thing neither rented box can be, and this
# Mac can produce it for the price of a container. Verified 2026-08-30
# against the MI325X's hip set: the import raised
# `ImportError: mojolearn: NO SUPPORTED GPU FOUND ON THIS BOX` and printed
# the whole probe table with every path absent and every library not
# loadable.
set -euo pipefail
SET="${1:?path to a fetched set directory, e.g. .../wheels/sets/hip}"
VENDOR="$(basename "$SET")"
case "$VENDOR" in cuda|hip) ;; *) echo "set directory must be named cuda or hip, not $VENDOR"; exit 2 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "$(ls "$SET"/*.so 2>/dev/null)" ] || { echo "no .so in $SET"; exit 2; }

ROOT="$(mktemp -d)"; trap 'rm -rf "$ROOT"' EXIT
mkdir -p "$ROOT/mojolearn"
cp "$REPO"/python/mojolearn/*.py "$ROOT/mojolearn/"
cp -R "$SET" "$ROOT/mojolearn/$VENDOR"
rm -f "$ROOT/mojolearn/$VENDOR/manifest.json" "$ROOT/mojolearn/$VENDOR/readback.txt"
echo "package root: $(find "$ROOT" -name '*.so' | wc -l | tr -d ' ') extensions, vendor $VENDOR"

OUT="$ROOT/import.txt"
set +e
docker run --rm --cpus 2 --platform linux/amd64 -v "$ROOT:/pkg:ro" python:3.12-slim \
  sh -c 'cd /pkg && PYTHONPATH=/pkg python -c "import mojolearn; print(\"IMPORTED\", mojolearn.vendor())"' \
  >"$OUT" 2>&1
RC=$?
set -e
cat "$OUT"
echo "--- verdict ---"
FAIL=0
[ "$RC" -ne 0 ] || { echo "FAIL: the import SUCCEEDED on a box with no device"; FAIL=1; }
for want in "NO SUPPORTED GPU FOUND" "MOJOLEARN_VENDOR"; do
  grep -qF "$want" "$OUT" || { echo "FAIL: the refusal never says \"$want\""; FAIL=1; }
done
# Every path and library the probe consults must be NAMED in the refusal, so
# a user with a device the probe missed can see what it looked for. The list
# is read out of the library rather than retyped here.
for tok in $(python3 -c "
import sys; sys.path.insert(0, '$REPO/python')
import importlib.util as u
s = u.spec_from_file_location('_b', '$REPO/python/mojolearn/_backend.py'); m = u.module_from_spec(s); s.loader.exec_module(m)
for v, spec in m._PROBE.items(): print(' '.join(spec['paths']), ' '.join(spec['libs']))"); do
  grep -qF "$tok" "$OUT" || { echo "FAIL: the refusal never names $tok"; FAIL=1; }
done
[ "$FAIL" = 0 ] && echo "nogpu_local PASS (import refused, rc=$RC, every probed name printed)" || echo "nogpu_local FAIL"
exit "$FAIL"
