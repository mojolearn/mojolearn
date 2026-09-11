#!/bin/sh
# gbdt-finish lane (2026-09-11 night), H100 on RunPod. RUNS ON THE POD from
# /root/mojolearn. Every set is built from ONE source tree (the lane branch)
# and differs only by the restore defines, so a before/after pair is the
# switch and nothing else:
#   baseline  -D MOJOLEARN_2634_CTR_PREP_OFF=1 -D MOJOLEARN_2635_LINEAR_BOUNDS=1 -D MOJOLEARN_2636_SERIAL_STAGING=1
#   a2634     -D MOJOLEARN_2635_LINEAR_BOUNDS=1 -D MOJOLEARN_2636_SERIAL_STAGING=1   (2634 on)
#   both      -D MOJOLEARN_2636_SERIAL_STAGING=1                                     (2634, 2635 on)
#   all       no define                                                              (2634, 2635, 2636 on)
# Datasets and the GPU opponents install in the background at t=0.
set -u
R=/root/mojolearn; OUT=/root/trees_out; L=$OUT/logs
mkdir -p "$L" /root/bins
cd "$R" || exit 9
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
AB="sh tools/trees_identical_ab.sh"
note() { echo "$* $(date -u +%H:%M:%S)" | tee -a "$OUT/setup.txt"; }
note start commit="$(cat "$R/SHIPPED_COMMIT.txt" 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
{ nproc; grep -m1 'model name' /proc/cpuinfo; uname -a; python3 --version; } > "$OUT/box.txt" 2>&1
(
  python3 -m pip install --no-input -q --disable-pip-version-check catboost xgboost scikit-learn pandas pyarrow > "$L/pip.log" 2>&1; note pip=$?
  python3 -c "import catboost, xgboost, sklearn, numpy; print('catboost', catboost.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)" > "$OUT/versions.txt" 2>&1
  timeout -k 30 1800 python3 tools/speed_gbdt_arm.py --download taxi > "$L/download_taxi.log" 2>&1; note download_taxi=$?
  : > "$OUT/taxi.done"
  # The Istella-S fetch is 472 MB from library.istella.it and ran at about
  # 150 KB/s on 2026-09-11, so a 40-minute cap TIMED OUT (exit 124) with
  # 410 of 472 MB on disk. `data.done` gates the timed cells, so it is
  # written ONLY on success: a released gate with no data cost this lane
  # three rounds of Istella-S cells. `--download istella` does not resume
  # (any existing file counts as complete), so a partial file is finished
  # with `curl -C -` before re-running it.
  timeout -k 30 5400 python3 tools/speed_gbdt_arm.py --download istella > "$L/download_istella.log" 2>&1; note download_istella=$?
  if [ "$(tail -1 "$L/download_istella.log" | grep -c 'istella decoded')" = 1 ]; then
    : > "$OUT/data.done"
  else
    note istella_MISSING_no_data_done
  fi
) &
[ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$L/pixi_get.log" 2>&1
timeout -k 30 1500 pixi install > "$L/pixi_install.log" 2>&1; note pixi_install=$?
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
timeout -k 30 1500 bash bindings/build.sh > "$L/build_base.log" 2>&1; note build_base=$?
MOJOLEARN_EXTRA_DEFINES="-D MOJOLEARN_2634_CTR_PREP_OFF=1 -D MOJOLEARN_2635_LINEAR_BOUNDS=1 -D MOJOLEARN_2636_SERIAL_STAGING=1" \
  timeout -k 30 1500 bash bindings/build_gbdt.sh > "$L/build_gbdt_baseline.log" 2>&1; note build_gbdt_baseline=$?
mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
sha256sum /root/bins/baseline/*.so > "$OUT/bins_baseline.sha256"
: > "$OUT/build.done"
# the after sets, while Istella-S downloads
$AB build a2634 gbdt -D MOJOLEARN_2635_LINEAR_BOUNDS=1 -D MOJOLEARN_2636_SERIAL_STAGING=1; note build_a2634=$?
$AB build both gbdt -D MOJOLEARN_2636_SERIAL_STAGING=1; note build_both=$?
$AB build all gbdt; note build_all=$?
# host checks, 2635 on (default) and restored, each twice-compared by output
for arm in on off; do
  _d=""; [ "$arm" = off ] && _d="-D MOJOLEARN_2635_LINEAR_BOUNDS=1"
  # shellcheck disable=SC2086
  timeout -k 30 900 pixi run mojo run -I . $_d checks/binarization_check.mojo > "$L/check_binarization.$arm.log" 2>&1; note check_binarization_$arm=$?
  # shellcheck disable=SC2086
  timeout -k 30 900 pixi run mojo run -I . $_d checks/greedy_log_sum_check.mojo > "$L/check_greedylogsum.$arm.log" 2>&1; note check_greedylogsum_$arm=$?
done
cmp -s "$L/check_binarization.on.log" "$L/check_binarization.off.log"; note binarization_output_equal=$?
cmp -s "$L/check_greedylogsum.on.log" "$L/check_greedylogsum.off.log"; note greedylogsum_output_equal=$?
: > "$OUT/builds.done"
wait
note setup_done
: > "$OUT/setup.done"
