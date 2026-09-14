# Workstream D fixes on a GPU box, HALF B of two (lane/expose-d-fixes).
# Fits one 60-minute lease: check-mixture, check-kernel-methods and
# check-cholesky, the three checks half A does not run. They are `mojo run`
# of source (pixi.toml's check-* tasks) and need no binding, so this half
# builds nothing.
#
# check-kernel-methods is the unknown: on the first MI300X leg at 2b2f568b0
# it started 5 minutes into the body and had printed nothing when the
# 60-minute cap hit, so whether it was still compiling or still running is
# not on the record (its output is a pipe and was never flushed). It is
# given the largest bound here and runs AFTER check-mixture, so a hang
# costs only itself. check-cholesky passed on that leg and nothing on this
# branch touches cholesky/, so it runs last with a short bound. Each step
# prints its elapsed seconds.
set -u
cd /root/mojolearn 2>/dev/null || cd "$(pwd)"
OUT=/root/gemm_leg_out; mkdir -p "$OUT"
[ -n "${MOJOLEARN_COMMIT:-}" ] && echo "$MOJOLEARN_COMMIT" > commit.txt
[ -f commit.txt ] && echo "commit $(cat commit.txt)"

bounded() {
    _secs=$1; _name=$2; shift 2
    _t0=$(date +%s)
    if command -v timeout > /dev/null 2>&1; then
        timeout -k 20 "$_secs" "$@" > "$OUT/$_name.log" 2>&1
    else
        "$@" > "$OUT/$_name.log" 2>&1
    fi
    _rc=$?
    echo "== $_name exit $_rc seconds $(( $(date +%s) - _t0 )) (bound $_secs)"
    [ "$_rc" = 124 ] && echo "   $_name HIT ITS BOUND"
    return $_rc
}
quiet() { grep -v "warning:\|^ *\^\|^    var\|^Imported\|^Included\|note:\|^ *~" "$1" | tail -"${2:-3}" | cut -c1-240; }

bounded 900 check-mixture pixi run check-mixture
quiet "$OUT/check-mixture.log" 4
bounded 1800 check-kernel-methods pixi run check-kernel-methods
quiet "$OUT/check-kernel-methods.log" 6
bounded 300 check-cholesky pixi run check-cholesky
quiet "$OUT/check-cholesky.log" 3
exit 0
