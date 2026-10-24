#!/usr/bin/env bash
# External qualification controller for an owned rented GPU; never product code.
set -euo pipefail
out=${1:?absolute retained result directory}
python=${2:?test venv python}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
mkdir -p "$out"
phase() {
  local name=$1 budget=$2 rc
  shift 2
  set +e
  timeout --signal=TERM --kill-after=15 "$budget" "$@" > "$out/$name.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$out/$name.rc"
  echo "$name: exit $rc"
  return "$rc"
}
phase torch 600 "$python" tools/check_native_adapters.py --out "$out/native"
phase training 600 "$python" tools/check_native_training_adapters.py --extension "$out/native/mojolearn_torch.so" --out "$out/training"
mkdir -p "$out/host-staging"
cp "$out/native/libmojolearn_adapter.so" "$out/host-staging/"
phase staging-build-test 600 "$python" tools/check_native_adapters.py --out "$out/host-staging" --host-staging
phase staging-training 600 "$python" tools/check_native_training_adapters.py --extension "$out/host-staging/mojolearn_torch.so" --out "$out/staging-training" --reference "$out/training"
phase max-regression 600 pixi run python tools/check_max_native_adapters.py --library "$out/mojolearn_ops.mojoc" --native-library "$out/native/libmojolearn_adapter.so" --out "$out/max-regression"
phase max-training 600 pixi run python tools/check_max_training_adapters.py --library "$out/mojolearn_ops.mojoc" --native-library "$out/native/libmojolearn_adapter.so" --torch-results "$out/training" --out "$out/max-training"
phase training-faults 600 "$python" tools/check_training_adapter_faults.py --qualified-out "$out/native" --out "$out/training-faults"
phase staging-training-faults 600 "$python" tools/check_training_adapter_faults.py --qualified-out "$out/host-staging" --out "$out/staging-training-faults" --host-staging
phase max-training-fault 600 pixi run python tools/check_max_training_fault.py --library "$out/mojolearn_ops.mojoc" --native-library "$out/native/libmojolearn_adapter.so" --torch-results "$out/training" --out "$out/max-training-fault"
