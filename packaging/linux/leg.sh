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
    # DIGITALOCEAN IS THE DEFAULT FOR NVIDIA TOO SINCE 2026-09-02, and the
    # reason is a measured one. RunPod legs died four times that morning --
    # a 276 MB archive over this Mac's 68 KB/s uplink, a dropped ssh
    # mid-upload, and a phase-9 run that built 2 of 15 bindings -- while the
    # DigitalOcean path landed both an MI325X and an H100 the same
    # afternoon, and it is the path that got the clone fix (the box fetches
    # the commit from GitHub in about three seconds). DO carries
    # gpu-h100x1-80gb in nyc2. Set MOJOLEARN_LINUX_LEG_NVIDIA_VIA=runpod for
    # the old route, which is kept below unchanged.
    if [ "${MOJOLEARN_LINUX_LEG_NVIDIA_VIA:-do}" = "do" ]; then
      export MOJOLEARN_E1_PHASES=9
      export MOJOLEARN_P9_ONLY_DIAG=1
      export MOJOLEARN_P9_DIAG="$DIAG"
      export MOJOLEARN_P9_DIAG_TIMEOUT="${MOJOLEARN_P9_DIAG_TIMEOUT:-2400}"
      export MOJOLEARN_WHEEL_VERSION="${MOJOLEARN_WHEEL_VERSION:-}" MOJOLEARN_WHEEL_INDEX="${MOJOLEARN_WHEEL_INDEX:-testpypi}"
      say "renting an H100 on DigitalOcean; diag=$DIAG"
      bash tools/e2_remote_leg.sh nv "${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"; RC=$?
      break_nvidia_case=1
    fi
    if [ "${break_nvidia_case:-0}" = "1" ]; then : ; else
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
    tools/gemm_remote_leg.sh nvidia --payload phase8 --rent; RC=$?
    fi ;;
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

# WHICH NEW DIRECTORY IS *OURS*.
#
# 2026-08-30: the nvidia and amd legs were run at the same time, and this
# loop took whichever new directory sorted last. The nvidia leg filed the
# AMD leg's result as its own and reported "build FAILED, no vendor" while
# its own diag on disk read vendor=cuda, build_rc=0, 30 binaries staged and
# the identical smoke passing all 29 lanes. A leg that misreports a partial
# SUCCESS as a total failure is worse than one that just fails.
#
# The fetched directory names carry the vendor (`*-runpod-nvidia`,
# `*-mojolearn-e2-amd`), so candidates are filtered by it. Two legs for the
# same vendor still must not overlap, and that is what the pre-flight lease
# check at the top of this file is for.
case "$VENDOR" in
  # A DigitalOcean nvidia leg files as `*-mojolearn-e2-nv`, a RunPod one as
  # `*-runpod-nvidia`; match the route that actually ran rather than the word.
  nvidia) if [ "${MOJOLEARN_LINUX_LEG_NVIDIA_VIA:-do}" = "do" ]; then VTOKEN=e2-nv; else VTOKEN=nvidia; fi ;;
  amd)    VTOKEN=amd ;;
esac
NEW=""
NMATCH=0
for d in $(ls -d bench/results/e1/*/ 2>/dev/null | sort); do
  case "$BEFORE" in *"$d"*) continue ;; esac
  case "$d" in *"$VTOKEN"*) ;; *)
      say "ignoring $d: a new directory, but not this leg's vendor ($VTOKEN)"
      continue ;;
  esac
  [ -f "$d/diag/SUMMARY.txt" ] || continue
  NEW="$d"; NMATCH=$((NMATCH + 1))
done
if [ "$NMATCH" -gt 1 ]; then
  say "WARNING: $NMATCH new $VTOKEN directories with a diag/SUMMARY.txt;"
  say "  taking the last ($NEW). Another leg for this vendor may have been"
  say "  running concurrently, which the pre-flight check exists to prevent."
fi
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
