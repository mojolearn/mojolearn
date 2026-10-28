#!/bin/bash
# Lane overhead-mamba-bwd: build the before arm's tokenizer host (origin/main
# 30928487f1 = this tree with before.patch applied) next to the after arm the
# leg's --build produced, then run the light same-bits/same-refusals fixture
# on both arms on the CPU column. Run through the CPU pod leg tool with
# --build tokenizer,neural,mamba,core,linalg; $LEG_OUT comes back to the Mac.
set -u
OUT=${LEG_OUT:?}
REPO=$(pwd)
JOBS=$(nproc)
export MOJOLEARN_COMPILE_JOBS=$JOBS
AFTER_HOST=$REPO/python/mojolearn/host
ls -la "$AFTER_HOST" > "$OUT/after-host.ls" 2>&1
sha256sum "$AFTER_HOST"/*.so > "$OUT/after-host.sha256" 2>&1
BEFORE_SRC=$HOME/before-src
rm -rf "$BEFORE_SRC"; mkdir -p "$BEFORE_SRC"
tar -C "$REPO" --exclude=./python/mojolearn/host --exclude=./.pixi -cf - . | tar -C "$BEFORE_SRC" -xf -
ln -s "$REPO/.pixi" "$BEFORE_SRC/.pixi"
(cd "$BEFORE_SRC" && patch -p1 < "$REPO/experiments/overhead-mamba-bwd/before.patch") > "$OUT/before-patch.log" 2>&1
echo "patch_rc=$?" >> "$OUT/before-patch.log"
BEFORE_HOST=$HOME/before-host
mkdir -p "$BEFORE_HOST"
START=$(date +%s)
(cd "$BEFORE_SRC" && MOJOLEARN_TOKENIZER_HOST_OUTDIR=$BEFORE_HOST sh bindings/build_tokenizer_host.sh) > "$OUT/build-before-tokenizer.log" 2>&1
RC=$?
echo "before_tokenizer_rc=$RC seconds=$(( $(date +%s) - START )) jobs=$JOBS" > "$OUT/build-before-tokenizer.rc"
for f in "$AFTER_HOST"/*_host.so; do
  case "$(basename "$f")" in _mojolearn_tokenizer_host.so) ;; *) cp "$f" "$BEFORE_HOST/" ;; esac
done
sha256sum "$BEFORE_HOST"/*.so > "$OUT/before-host.sha256" 2>&1
pixi run -e default mojo --version > "$OUT/compiler.txt" 2>&1
for arm in before after; do
  dir=$BEFORE_HOST; [ $arm = after ] && dir=$AFTER_HOST
  MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$dir PYTHONPATH=$REPO/python \
    pixi run -e default python tools/tests/overhead_mamba_bwd_fixture.py --out "$OUT/fixture-$arm.json" > "$OUT/fixture-$arm.log" 2>&1
  echo "fixture_${arm}_rc=$?" >> "$OUT/fixture.rc"
done
pixi run -e default python tools/tests/overhead_mamba_bwd_fixture.py --compare "$OUT/fixture-before.json" "$OUT/fixture-after.json" > "$OUT/compare.log" 2>&1
echo "compare_rc=$?" >> "$OUT/fixture.rc"
cat "$OUT/fixture.rc" "$OUT/build-before-tokenizer.rc"
