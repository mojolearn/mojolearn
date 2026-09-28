#!/bin/sh
# lane/cnn-apple (2026-09-28): one Apple speed request for the cnn family.
# Run by `tools/apple_steward.py submit --kind speed --builds bindings/build_x_cnn.sh`
# in the steward's worktree at the commit.
#   CNN_PARTS   "speed entries plans" (default all three)
#   CNN_PLANS_ONLY  rows of gemm_plans.mojo to run (default all)
set -u
PARTS=${CNN_PARTS:-"speed entries plans"}
MODE=${MOJOLEARN_NUMERIC_MODE:-identical}
echo "CNN-APPLE commit $(git rev-parse --short HEAD) host $(hostname) $(sysctl -n machdep.cpu.brand_string 2>/dev/null) mode $MODE"
rc=0
py_parts=""
for p in $PARTS; do case $p in speed|entries) py_parts="$py_parts $p" ;; esac; done
if [ -n "$py_parts" ]; then
    # shellcheck disable=SC2086
    pixi run -e default python tools/apple_speed_cnn/profile.py "$MODE" . $py_parts || rc=1
fi
case " $PARTS " in *" plans "*)
    d=$(mktemp -d "${TMPDIR:-/tmp}/cnn-plans.XXXXXX")
    mf="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    [ "$MODE" = fast ] && mf=""
    # shellcheck disable=SC2086
    pixi run mojo build -j 2 $mf -I . tools/apple_speed_cnn/gemm_plans.mojo -o "$d/plans" && "$d/plans" || rc=1
    rm -rf "$d" ;;
esac
exit $rc
