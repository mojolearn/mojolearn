#!/usr/bin/env bash
# auditwheel show + repair, then twine check, on the packed Linux wheel.
# RUNS ON THE MAC, inside docker, `--cpus 2`, ONE container at a time, on a
# FINISHED wheel. Nothing here compiles or fits.
#
#   bash packaging/linux/audit.sh python/dist/mojolearn-X.Y.Z-py3-none-linux_x86_64.whl \
#        bench/results/wheels/<stamp>-nvidia/sets/cuda/manifest.json \
#        bench/results/wheels/<stamp>-amd/sets/hip/manifest.json
#
# Leaves beside the wheel, under python/dist/audit/:
#   show.txt           `auditwheel show` verbatim: THE ARTIFACT (gate c). It
#                      names the manylinux level the wheel actually meets and
#                      every external library it found.
#   repair.txt         `auditwheel repair` verbatim
#   repaired/          the retagged wheel, the one that goes to TestPyPI
#   twine.txt          `twine check` on the repaired wheel
#
# THE TAG IS WHATEVER `show` MEASURED. pack_wheel.py wrote linux_x86_64 on
# purpose (PyPI refuses it), and `--plat` below is read out of show.txt, not
# typed. If the MAX runtime needs a glibc newer than 2.28, the tag says so
# and docs/LINUX_WHEEL.md is corrected to the measured level, not the other
# way round.
#
# --exclude, TWICE OVER. Driver-side libraries (libcuda.so.1, libamdhip64,
# libhsa-runtime64, ...) are read from the manifests' driver_libs_not_staged
# and excluded: they belong to the user's driver and must not be bundled.
# The STAGED MAX libraries are ALSO excluded by name, because auditwheel
# repair otherwise copies every library it can resolve into mojolearn.libs/
# with a hashed name, which would ship the closure twice; the closure is
# already in the wheel with RUNPATHs that resolve it (stage_libs.py).
# Whether auditwheel honors that for libraries resolved through $ORIGIN is
# one of the assumptions this file makes unrun: read repair.txt and the
# repaired wheel's listing (`unzip -l`) for a mojolearn.libs/ directory.
#
# THE SPLIT LINUX PACKAGES (python/mojolearn/gpu_plugins.py): run it ONCE PER
# WHEEL, the core `mojolearn-*` and each plugin `mojolearn_nvidia-*` /
# `mojolearn_amd-*`, all into the same audit/ directory (repaired/ then holds
# the whole set). The logs of the core (and of the combined wheel) keep their
# names, show.txt, repair.txt, twine.txt; a plugin's are show-<dist>.txt,
# repair-<dist>.txt and twine-<dist>.txt. A PLUGIN'S RUNTIME LIVES IN THE
# CORE: its bindings' DT_NEEDED MAX libraries resolve through RUNPATH to
# mojolearn/.libs/, which only the core ships, so inside the plugin alone
# auditwheel cannot locate them and `repair` would refuse. They are excluded
# by name: every file under the core's mojolearn/.libs/ is added to the
# --exclude list, read from the core wheel of the same version BESIDE the
# plugin (the packer writes the set into one directory); a plugin with no core
# beside it is refused. And any top-level <name>.libs/ directory in a repaired
# wheel (mojolearn.libs/, mojolearn_nvidia.libs/, ...) means auditwheel grafted
# a copy of something the closure already carries, which fails the run.
set -euo pipefail
WHL="${1:?wheel}"; shift
[ $# -ge 1 ] || { echo "give every set's manifest.json"; exit 2; }
IMAGE="${MOJOLEARN_AUDIT_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"
OUT="$(cd "$(dirname "$WHL")" && pwd)/audit"; mkdir -p "$OUT/repaired"
WHLABS="$(cd "$(dirname "$WHL")" && pwd)/$(basename "$WHL")"

BASE=$(basename "$WHL")
DIST=${BASE%%-*}
case "$DIST" in
  mojolearn) SUFFIX="" ;;
  mojolearn_nvidia|mojolearn_amd) SUFFIX="-$DIST" ;;
  *) echo "not a mojolearn wheel: $BASE"; exit 2 ;;
esac

EXCL=""
if [ -n "$SUFFIX" ]; then
  VER=$(printf '%s' "$BASE" | cut -d- -f2)
  CORE=$(ls -1 "$(dirname "$WHLABS")"/mojolearn-"$VER"-*.whl 2>/dev/null | head -1 || true)
  [ -n "$CORE" ] || { echo "$BASE is a split plugin: its runtime is the core's mojolearn/.libs/, and no mojolearn-$VER-*.whl sits beside it to name them"; exit 2; }
  for lib in $(python3 -c 'import sys,zipfile; print(" ".join(sorted({n.rsplit("/",1)[-1] for n in zipfile.ZipFile(sys.argv[1]).namelist() if n.startswith("mojolearn/.libs/") and not n.endswith("/")})))' "$CORE"); do
    case " $EXCL " in *" $lib "*) ;; *) EXCL="$EXCL $lib" ;; esac
  done
  [ -n "$EXCL" ] || { echo "the core beside $BASE ($(basename "$CORE")) ships no mojolearn/.libs/"; exit 2; }
  echo "plugin: the core's runtime ($(basename "$CORE")) is excluded by name"
fi
for m in "$@"; do
  for lib in $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(" ".join(d["driver_libs_not_staged"] + [s["name"] for s in d["staged_libs"]]))' "$m"); do
    case " $EXCL " in *" $lib "*) ;; *) EXCL="$EXCL $lib" ;; esac
  done
done
EXARGS=""; for l in $EXCL; do EXARGS="$EXARGS --exclude $l"; done
echo "excluding: $EXCL"

docker run --rm --cpus 2 --platform linux/amd64 \
  -v "$WHLABS:/w/$(basename "$WHL"):ro" -v "$OUT:/out" "$IMAGE" \
  sh -c "auditwheel show /w/$(basename "$WHL")" 2>&1 | tee "$OUT/show$SUFFIX.txt"
PLAT=$(grep -oE 'manylinux_[0-9]+_[0-9]+_x86_64' "$OUT/show$SUFFIX.txt" | sort -u | tail -1 || true)
[ -n "$PLAT" ] || { echo "auditwheel show named no manylinux tag; read $OUT/show$SUFFIX.txt"; exit 1; }
echo "measured platform tag: $PLAT"

docker run --rm --cpus 2 --platform linux/amd64 \
  -v "$WHLABS:/w/$(basename "$WHL"):ro" -v "$OUT:/out" "$IMAGE" \
  sh -c "auditwheel repair --plat $PLAT $EXARGS -w /out/repaired /w/$(basename "$WHL")" 2>&1 | tee "$OUT/repair$SUFFIX.txt"
ls -la "$OUT/repaired"
set -- "$OUT"/repaired/"$DIST"-*.whl
[ -f "$1" ] || { echo "auditwheel repair left no $DIST wheel in $OUT/repaired; read $OUT/repair$SUFFIX.txt"; exit 1; }
# `grep -c` EXITS 1 WHEN THE COUNT IS 0, and 0 is the wanted answer. Under
# `set -euo pipefail` this line killed the script on success, right before
# `twine check`, and left no twine.txt behind while the run still looked
# clean because the caller was reading a pipeline's exit code. Counted with
# `|| true` and asserted separately, so a NON-zero count is the failure it
# should always have been rather than a line of output nobody read.
for r in "$@"; do
  n=$(unzip -Z1 "$r" | grep -cE '^[^/]+\.libs/' || true)
  echo "top-level *.libs entries (want 0): $n   $(basename "$r")"
  [ "$n" = 0 ] || { echo "auditwheel repair grafted libraries into a top-level *.libs/ directory (a second copy of the staged closure); the --exclude list is incomplete"; exit 1; }
done

docker run --rm --cpus 2 --platform linux/amd64 -v "$OUT/repaired:/r:ro" python:3.12-slim \
  sh -c "pip install -q twine >/dev/null 2>&1 && twine check /r/$DIST-*.whl" 2>&1 | tee "$OUT/twine$SUFFIX.txt"
