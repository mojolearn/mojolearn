#!/usr/bin/env bash
# Install a GPU torch into a THROWAWAY venv on a rented box and run the
# vendor-library gemm arm against it.
#
# WHY A VENV AND NOT THE PIXI ENVIRONMENT. `pixi.toml`'s `skgpu` feature pins
# pytorch, scikit-learn, numpy and scipy exactly, on purpose: every recorded
# timing in `bench/results/` was taken under that solve, and a benchmark whose
# baseline moved is not a baseline. Installing a CUDA or ROCm wheel into it to
# answer one question would move it for every past number. So this venv is
# built beside it, used once, and left on a box that is about to be destroyed.
#
# WHICH WHEEL. Chosen from what the box actually has, never from the vendor
# name passed in: `nvidia-smi` or `rocminfo` decides. A guess here produces a
# CPU wheel, and a CPU wheel produces a perfectly good millisecond for the
# wrong device -- which is the exact failure `tools/vendor_gemm_price.py`
# refuses on, so this script cannot silently succeed either way.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
VENV="${VENDOR_TORCH_VENV:-/root/.venv-vendor-torch}"
MAXMACS="${VENDOR_MAX_MACS:-5e10}"
REPEATS="${VENDOR_REPEATS:-10}"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  INDEX="https://download.pytorch.org/whl/cu129"
  echo "[vendor-torch] NVIDIA detected: $(nvidia-smi -L | head -1)"
elif command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
  INDEX="https://download.pytorch.org/whl/rocm6.4"
  echo "[vendor-torch] AMD detected: $(rocm-smi --showproductname 2>/dev/null | grep -i 'card series' | head -1)"
else
  echo "[vendor-torch] REFUSED: neither nvidia-smi nor rocminfo/rocm-smi on this box."
  exit 3
fi

# `python3 -m venv` is NOT present on a minimal Ubuntu 24.04: the stdlib
# `venv` module ships in a separate `python3-venv` package and the failure is
# a bare non-zero exit with no useful message. Measured on the DigitalOcean
# H100 image, 2026-08-25, leg 15, where it cost the whole vendor arm.
if ! python3 -m venv "$VENV" >/dev/null 2>&1; then
  echo "[vendor-torch] python3 -m venv unavailable; installing python3-venv"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq python3-venv python3-pip >/dev/null 2>&1
  python3 -m venv "$VENV" >/dev/null 2>&1 || {
    echo "[vendor-torch] venv still unavailable after apt-get; REFUSING rather"
    echo "[vendor-torch] than installing torch into the system python, which"
    echo "[vendor-torch] would be a mutation of a box whose pixi environment"
    echo "[vendor-torch] every other measurement on this leg depends on."
    exit 4
  }
fi
"$VENV/bin/pip" install -q --upgrade pip >/dev/null 2>&1
echo "[vendor-torch] installing torch from $INDEX (this is the long step)"
"$VENV/bin/pip" install -q torch --index-url "$INDEX" || { echo "[vendor-torch] pip install failed"; exit 5; }

"$VENV/bin/python" - <<'PY'
import torch
print("[vendor-torch] torch", torch.__version__,
      "cuda", torch.version.cuda, "hip", getattr(torch.version, "hip", None),
      "available", torch.cuda.is_available())
PY

"$VENV/bin/python" tools/vendor_gemm_price.py \
  --repeats "$REPEATS" --warmup 3 --max-macs "$MAXMACS" \
  --out "${VENDOR_PRICE_OUT:-$(ls -td bench/results/e1/*/ 2>/dev/null | head -1)vendor_price.json}"
rc=$?
echo "[vendor-torch] vendor_gemm_price exit=$rc"
exit $rc
