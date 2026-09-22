#!/bin/bash
# tools/release_linux_build.sh -- THE LINUX RELEASE BUILD, ONE CPU BOX, NO GPU.
#
#   bash tools/release_linux_build.sh <40-hex commit>            # dry run
#   bash tools/release_linux_build.sh <40-hex commit> --rent     # build
#
# Builds cuda/sm_90a, cuda/sm_89 and hip/gfx942 on ONE RunPod CPU pod running
# the pinned Ubuntu 22.04 image (the AMD leg's container image, GCC 11.4, ld
# 2.38, patchelf 0.17.2), with no GPU present: Mojo compiles each set ahead of
# time from --target-accelerator alone (PTX for NVIDIA, a code object for
# AMD). Each set goes through the same tools/release061_remote_build.sh the GPU
# legs run, with MOJOLEARN_RELEASE_NO_DEVICE=1. The pod is rented, guarded,
# fetched and deleted by tools/runpod_cpu_leg.sh (Mac dead-man, on-pod
# watchdog, verified teardown).
#
# PROVEN at d181d9792 (0.8.14), 2026-09-22: every shipped binary of all three
# sets byte-identical (sha256) to the three GPU-box builds of the same commit
# (docs/RELEASE_CHECKLIST.md section 2c has the numbers).
#
# Output (default ~/mojolearn-evidence/releases/<commit>/linux-cpu-box/<stamp>):
#   cuda-sm_90a/release-build  cuda-sm_89/release-build  hip-gfx942/release-build
# each the tree a GPU leg writes (build/sets/<vendor>/<arch>/, build/
# build-provenance.json), so pack_wheel.py takes them unchanged; the last
# lines printed are the pack command. LEG/ holds the runner's record
# (timings.tsv with the spend, teardown.txt, route.txt).
#
# Options:
#   --out DIR        output root (must not exist)
#   --vcpu N         pod vCPUs (default 32)      --flavors LIST (default cpu5g,cpu3g)
#   --jobs N         MOJOLEARN_BUILD_JOBS (default: vcpu / 2, 16 GiB per job)
#   --archs 'A B'    subset, default 'sm_90a sm_89 gfx942'
#   --lease MIN      on-pod self-delete (default 120)
#   --no-bincache    build every extension from source (the R2 binding cache
#                    is ON by default; tools/bincache.py, partition
#                    none/<image>; a hit is placed only after its archive's
#                    key, fields and every file's sha256 verify)
#   --rent           actually rent (otherwise a dry run that creates nothing)
#
# The shipped source is the commit's TRACKED tree (a detached worktree made
# here and removed afterwards). The route's own files (this build machinery:
# tools/release061_remote_build.sh, tools/cpu_build_guard.py,
# tools/release_linux_cpu_box.sh, tools/bincache.py) come from THIS checkout
# and are overlaid on the box, so an older commit can be built by the current
# route; none of them is in the build's source inventory, and route.txt
# records each one's sha256.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE='rocm/dev-ubuntu-22.04@sha256:a3850e6638c6c390436ef1aacd72fd1359af36083ac823d5136818206998c484'
COMMIT="${1:-}"
shift || true
OUT=""; VCPU=32; FLAVORS="cpu5g,cpu3g"; JOBS=""; ARCHS="sm_90a sm_89 gfx942"; LEASE=120; BINCACHE=1; RENT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out) shift; OUT="${1:-}" ;;
        --vcpu) shift; VCPU="${1:-}" ;;
        --flavors) shift; FLAVORS="${1:-}" ;;
        --jobs) shift; JOBS="${1:-}" ;;
        --archs) shift; ARCHS="${1:-}" ;;
        --lease) shift; LEASE="${1:-}" ;;
        --no-bincache) BINCACHE=0 ;;
        --rent) RENT=1 ;;
        -h|--help) sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
die() { printf 'REFUSED: %s\n' "$*" >&2; exit 1; }
printf '%s' "$COMMIT" | grep -Eq '^[0-9a-f]{40}$' || die "a full 40-hex commit is required"
git -C "$ROOT" cat-file -e "$COMMIT^{commit}" 2>/dev/null || die "commit $COMMIT is not in this repository (git fetch first)"
printf '%s' "$VCPU" | grep -Eq '^[0-9]+$' || die "--vcpu must be a number"
[ -n "$JOBS" ] || JOBS=$((VCPU / 2))
printf '%s' "$JOBS" | grep -Eq '^[0-9]+$' && [ "$JOBS" -ge 1 ] && [ "$JOBS" -le 16 ] || die "--jobs must be 1..16"
for a in $ARCHS; do case "$a" in sm_90a|sm_89|gfx942) ;; *) die "unknown arch $a" ;; esac; done
SHORT=$(printf '%s' "$COMMIT" | cut -c1-9)
[ -n "$OUT" ] || OUT="${MOJOLEARN_EVIDENCE_ROOT:-$HOME/mojolearn-evidence}/releases/$COMMIT/linux-cpu-box/$(date -u +%Y-%m-%d_%H%M%S)"
[ ! -e "$OUT" ] || die "$OUT exists"

# ------------------------------------------------------------ the source
WT="${MOJOLEARN_RELEASE_WORKTREE_ROOT:-$HOME/mojolearn-wt}/release-linux-build-$SHORT-$$"
cleanup() { [ -d "$WT" ] && git -C "$ROOT" worktree remove --force "$WT" > /dev/null 2>&1; rm -rf "$TMPF"; }
TMPF=$(mktemp -d "${TMPDIR:-/tmp}/release-linux-build.XXXXXX")
trap cleanup EXIT
git -C "$ROOT" worktree add --detach "$WT" "$COMMIT" > /dev/null 2>&1 || die "cannot add a worktree at $COMMIT"
[ "$(git -C "$WT" rev-parse HEAD)" = "$COMMIT" ] || die "worktree HEAD is not $COMMIT"

# ------------------------------------------------------------ the route overlay
OVERLAY="tools/release061_remote_build.sh tools/cpu_build_guard.py tools/release_linux_cpu_box.sh tools/bincache.py"
( cd "$ROOT" && tar czf "$TMPF/overlay.tgz" $OVERLAY ) || die "overlay tarball"
# None of these may be in the build's source inventory (linux_surface_qualification.sh's walk).
for f in $OVERLAY; do
    case "$f" in *.mojo|bindings/*|packaging/linux/*|python/mojolearn/*|tokenizer/tools/*|tools/linux_surface_qualification.sh)
        die "overlay file $f is in the source inventory" ;; esac
done
B64=$(base64 < "$TMPF/overlay.tgz" | tr -d '\n')
cat > "$TMPF/cmd.sh" <<EOF
set -u
cd /root/mojolearn
echo '$B64' | base64 -d > /root/route-overlay.tgz
mkdir -p "\$LEG_OUT/release"
{ echo "overlay_sha256=\$(sha256sum /root/route-overlay.tgz | cut -c1-64)"
  for f in $OVERLAY; do printf 'before %s %s\n' "\$f" "\$( [ -f "\$f" ] && sha256sum "\$f" | cut -c1-64 || echo absent)"; done
} > "\$LEG_OUT/release/overlay.txt"
tar xzf /root/route-overlay.tgz -C /root/mojolearn
for f in $OVERLAY; do printf 'after %s %s\n' "\$f" "\$(sha256sum "\$f" | cut -c1-64)"; done >> "\$LEG_OUT/release/overlay.txt"
export MOJOLEARN_BINCACHE_OUT="\$LEG_OUT/bincache" MOJOLEARN_BINCACHE=$BINCACHE
MOJOLEARN_BUILD_JOBS=$JOBS bash tools/release_linux_cpu_box.sh "\$LEG_OUT/release" $ARCHS
EOF
bash -n "$TMPF/cmd.sh" || die "command script"

echo "== release_linux_build: $COMMIT, archs [$ARCHS], $VCPU vCPU ($FLAVORS), jobs $JOBS, bincache=$BINCACHE =="
echo "   image $IMAGE"
echo "   out $OUT"
RARGS="--lane rel-$SHORT --worktree $WT --image $IMAGE --vcpu $VCPU --flavors $FLAVORS --disk 80 --lease $LEASE --jobs $JOBS --cmd-file $TMPF/cmd.sh --out $OUT/LEG"
[ "$BINCACHE" = 1 ] && RARGS="$RARGS --release-bincache"
[ "$BINCACHE" = 1 ] || RARGS="$RARGS --no-bincache"
if [ "$RENT" != 1 ]; then
    # shellcheck disable=SC2086
    bash "$ROOT/tools/runpod_cpu_leg.sh" $RARGS
    exit $?
fi
mkdir -p "$OUT"
T0=$(date +%s)
# shellcheck disable=SC2086
bash "$ROOT/tools/runpod_cpu_leg.sh" $RARGS --rent
LEG_RC=$?
T1=$(date +%s)
R="$OUT/LEG/remote/leg_out/release"
rc=0
for a in $ARCHS; do
    case "$a" in sm_*) v=cuda ;; *) v=hip ;; esac
    if [ -f "$R/$v-$a/release-build/build/build-provenance.json" ] && [ "$(cat "$R/$v-$a/release-build/exit_code" 2>/dev/null)" = 0 ]; then
        mv "$R/$v-$a" "$OUT/$v-$a"
        echo "   $v/$a: OK $OUT/$v-$a/release-build"
    else
        echo "   $v/$a: FAILED (see $R/$v-$a/release-build-console.log)"
        rc=1
    fi
done
echo "wall_seconds=$((T1 - T0)) leg_exit=$LEG_RC" | tee "$OUT/wall.txt"
grep '^spend' "$OUT/LEG/timings.tsv" 2>/dev/null | tee -a "$OUT/wall.txt"
[ "$LEG_RC" = 0 ] && [ "$rc" = 0 ] || exit 1
echo
echo "Pack (docs/RELEASE_CHECKLIST.md section 3):"
S=""; P=""
for a in $ARCHS; do
    case "$a" in sm_*) v=cuda ;; *) v=hip ;; esac
    S="$S --set $OUT/$v-$a/release-build/build/sets/$v"; P="$P --build-proof $OUT/$v-$a/release-build/build/build-provenance.json"
done
echo "  python3 packaging/linux/pack_wheel.py --profile release-linux3$S$P --out <dist>"
