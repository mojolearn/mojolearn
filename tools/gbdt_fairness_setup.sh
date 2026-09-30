#!/bin/sh
# lane gbdt-fairness: set an H100 pod up to RE-TEST the GBDT speed claim
# (ours IDENTICAL 0.437x of CatBoost GPU on taxi gbdt-symmetric). RUNS ON THE
# POD from /root/mojolearn.
#
#   nohup sh tools/gbdt_fairness_setup.sh > /root/fair_out/setup_console.log 2>&1 &
#
# Deliberately SMALLER than tools/trees_identical_remote.sh: this lane times
# gbdt only, against CatBoost GPU only, so cuML, LightGBM-CUDA and the rf/trees
# bindings are not built or installed. Nothing here downloads a dataset: the
# two npz fixtures are staged from R2 by the Mac (tools/dataset_store.sh stage)
# and verified against bench/results/dataset_store/manifest.tsv, so the bytes
# are the pinned ones rather than a fresh decode.
set -u
R=/root/mojolearn
OUT=/root/fair_out
L="$OUT/logs"
mkdir -p "$L"
cd "$R" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
note() { echo "$* $(date -u +%H:%M:%S)" | tee -a "$OUT/setup.txt"; }

note start commit="$(cat "$R/SHIPPED_COMMIT.txt" 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
{ nproc; grep -m1 'model name' /proc/cpuinfo; uname -a; python3 --version; } > "$OUT/box.txt" 2>&1

# The opponent, in the background: CatBoost is the only library this lane
# times against. torch is already in the image and is one of the three device
# drains the probe uses, so it is not installed here.
(
  timeout -k 30 900 python3 -m pip install --no-input -q --disable-pip-version-check \
      catboost > "$L/pip.log" 2>&1
  note pip_catboost=$?
  python3 -c "import catboost, numpy; print('catboost', catboost.__version__, 'numpy', numpy.__version__)" \
      > "$OUT/versions.txt" 2>&1
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
      >> "$OUT/versions.txt" 2>&1
  : > "$OUT/pip.done"
) &

[ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$L/pixi_get.log" 2>&1
timeout -k 30 1500 pixi install > "$L/pixi_install.log" 2>&1; note pixi_install=$?
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

# IDENTICAL tier, the tier every number in the claim was taken in. The base
# binding first because `import mojolearn` reaches it; then gbdt, the one this
# lane actually times.
timeout -k 30 1500 bash bindings/build.sh > "$L/build_base.log" 2>&1; note build_base=$?
timeout -k 30 1500 bash bindings/build_gbdt.sh > "$L/build_gbdt.log" 2>&1; note build_gbdt=$?
ls -la python/mojolearn/identical/ > "$OUT/bindings_listing.txt" 2>&1
sha256sum python/mojolearn/identical/*.so > "$OUT/so_sha256.txt" 2>&1

MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$R/python" timeout -k 10 300 python3 -c \
  "import mojolearn; m = mojolearn.GradientBoosting(); b = m._bind('_mojolearn_gbdt'); print('import OK', m.numeric_mode_used(), m.vendor_used(), b.gbdt_per_round_paths())" \
  > "$L/import_identical.log" 2>&1
note import_identical=$?

wait
note setup_done
: > "$OUT/setup.done"
