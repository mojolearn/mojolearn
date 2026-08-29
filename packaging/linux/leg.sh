#!/usr/bin/env bash
# ONE COMMAND PER VENDOR, from the Mac: rent the box, build that vendor's
# three sets on it, run the on-box gates, fetch, and file the result under
# bench/results/wheels/<stamp>-<vendor>/.
#
#   bash packaging/linux/leg.sh nvidia            build + gates on RunPod
#   bash packaging/linux/leg.sh amd               build + gates on DigitalOcean
#   MOJOLEARN_WHEEL_VERSION=X.Y.Z bash packaging/linux/leg.sh nvidia install
#   MOJOLEARN_WHEEL_VERSION=X.Y.Z bash packaging/linux/leg.sh amd install
#                                                 the TestPyPI install smoke
#
# This is a thin wrapper over the two existing legs and changes nothing
# about their safety story (lease, dead-man, terminate-and-verify): it only
# sets the phase-9 knobs that make the bootstrap run ONE diag script and
# nothing else, then moves the fetched diag/ directory somewhere with a
# name the packer expects. Run it with `nohup ... & disown` from a session
# that will outlive it; killing the owning session fires the leg's EXIT trap
# and terminates the box mid-build (2026-08-29, pod 5to0fmyf6y0b39).
#
# BEFORE RENTING, by hand: `ls bench/results/runpod_leases/` and the RunPod
# pod list must show no live mojolearn pod for nvidia; the DigitalOcean
# account holds ONE GPU droplet at a time for amd. Another session may own
# one; wait for it.
set -uo pipefail
VENDOR="${1:?nvidia|amd}"; MODE="${2:-build}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
case "$MODE" in
  build)   DIAG=packaging/linux/leg_diag.sh ;;
  install) DIAG=packaging/linux/leg_diag_install.sh; : "${MOJOLEARN_WHEEL_VERSION:?set MOJOLEARN_WHEEL_VERSION=X.Y.Z}" ;;
  *) echo "mode must be build or install"; exit 2 ;;
esac
STAMP="$(date +%Y-%m-%d_%H%M%S)"
DEST="bench/results/wheels/$STAMP-$VENDOR"
BEFORE="$(ls -d bench/results/e1/*/ 2>/dev/null | sort)"
say() { echo "[$(date +%T) linux-leg $VENDOR] $*"; }

case "$VENDOR" in
  nvidia)
    export MOJOLEARN_RUNPOD_KEY_FILE="${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key}"
    export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat "$MOJOLEARN_RUNPOD_KEY_FILE")}"
    export MOJOLEARN_GEMM_LEG_E1_PHASES=9
    export MOJOLEARN_GEMM_LEG_P9_ONLY_DIAG=1
    export MOJOLEARN_GEMM_LEG_P9_DIAG="$DIAG"
    # The arithmetic: 60 min lease, 600 s fetch reserve in the leg, so the
    # bootstrap's work bound is about 3000 s; a cold pixi install takes 5 to
    # 8 min before phase 9 starts. 2400 s for the diag leaves the fetch its
    # reserve. Thirty builds must fit in that or the leg comes home partial.
    export MOJOLEARN_GEMM_LEG_P9_DIAG_TIMEOUT="${MOJOLEARN_GEMM_LEG_P9_DIAG_TIMEOUT:-2400}"
    # Driver >= 580 so MAX compiles natively; the ptxas escape hangs iforest.
    export MOJOLEARN_GEMM_LEG_CUDA="${MOJOLEARN_GEMM_LEG_CUDA:-\"13.0\"}"
    # A build leg is CPU bound: prefer a box with cores over the cheapest GPU.
    export MOJOLEARN_GEMM_LEG_GPU_NVIDIA="${MOJOLEARN_GEMM_LEG_GPU_NVIDIA:-NVIDIA H100 PCIe}"
    export MOJOLEARN_WHEEL_VERSION="${MOJOLEARN_WHEEL_VERSION:-}" MOJOLEARN_WHEEL_INDEX="${MOJOLEARN_WHEEL_INDEX:-testpypi}"
    say "renting; diag=$DIAG"
    tools/gemm_remote_leg.sh nvidia --payload phase8 --rent; RC=$? ;;
  amd)
    export MOJOLEARN_E1_PHASES=9
    export MOJOLEARN_P9_ONLY_DIAG=1
    export MOJOLEARN_P9_DIAG="$DIAG"
    # e2_remote_leg.sh bounds the bootstrap at lease minus FETCH_RESERVE (420 s)
    # minus provisioning; 2400 s for the diag keeps under it after a cold pixi.
    export MOJOLEARN_P9_DIAG_TIMEOUT="${MOJOLEARN_P9_DIAG_TIMEOUT:-2400}"
    export MOJOLEARN_WHEEL_VERSION="${MOJOLEARN_WHEEL_VERSION:-}" MOJOLEARN_WHEEL_INDEX="${MOJOLEARN_WHEEL_INDEX:-testpypi}"
    say "renting; diag=$DIAG"
    bash tools/e2_remote_leg.sh amd "${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"; RC=$? ;;
  *) echo "vendor must be nvidia or amd"; exit 2 ;;
esac
say "leg exited rc=$RC; locating the fetched diag/"

NEW=""
for d in $(ls -d bench/results/e1/*/ 2>/dev/null | sort); do
  case "$BEFORE" in *"$d"*) continue ;; esac
  [ -f "$d/diag/SUMMARY.txt" ] && NEW="$d"
done
if [ -z "$NEW" ]; then
  say "no new bench/results/e1/*/diag/SUMMARY.txt came home. Read the leg's own output above."
  exit 1
fi
mkdir -p "$DEST"
cp -R "$NEW/diag/." "$DEST/"
echo "source_e1_dir=$NEW" >> "$DEST/SUMMARY.txt"
if [ "$MODE" = build ]; then
  for tgz in "$DEST"/wheels/sets/*.tar.gz; do
    [ -f "$tgz" ] || continue
    ( cd "$DEST/wheels/sets" && tar xzf "$(basename "$tgz")" )
  done
fi
say "filed under $DEST"
cat "$DEST/SUMMARY.txt"
exit $RC
