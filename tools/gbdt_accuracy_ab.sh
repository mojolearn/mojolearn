#!/bin/sh
# DEVIATION 2043 A/B: is FAST-with-the-fused-one-byte-kernel FASTER *and* NO
# WORSE ON HELD-OUT ACCURACY than FAST as shipped, on a 32-lane column?
#
# THE DATASET IS HIGGS AND THE SCRIPT FETCHES IT ITSELF.
#
#   HIGGS, 11,000,000 x 28, binary classification, from the UCI archive:
#   https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
#   2.6 GB compressed, decoded once into $GBM_BENCH_DATA/higgs/higgs_speed.npz
#   (about 1.3 GB of float32). Both the download and the decode are done by
#   `tools/speed_gbdt_arm.py --download higgs`, which this script calls in its
#   own named step BEFORE any arm is built or timed -- see `load_higgs`
#   (tools/speed_gbdt_arm.py:522) for the split. The split is gbm-bench's: the
#   LAST 500,000 rows are the test set at every rung and `--rows` caps the
#   TRAINING rows from the front, so two rungs are nested prefixes of one
#   problem scored on identical held-out rows.
#
#   This box is expected to be a freshly created rented droplet that cloned
#   the repo at a commit and has no dataset store, so nothing here assumes a
#   cache. It assumes only network access to the UCI archive. If that fetch
#   fails, `load_with_fallback` (tools/speed_gbdt_arm.py:625) drops to the
#   SYNTHETIC binary fixture and says so on stderr, and every emitted line
#   then carries `synthclf-...` in its shape tag rather than `higgs-...`. A
#   synthetic row is NOT an answer to this question and the summary below
#   marks it `shape=NOT-HIGGS` so it cannot be misread.
#
#   ROWS DEFAULT TO 2,000,000, above [[mojolearn-large-data-timing-floor]]'s
#   1,000,000-row floor for any tree timing. 1,000,000 is the floor itself and
#   is the smallest value that may be passed here.
#
# WHY THIS SCRIPT EXISTS AND WHY IT IS NOT `tools/fast_replication_ab.sh`.
#
# That harness (DEVIATION 2040/2041/2042) compares two FAST builds and
# REFUSES to report a ratio when the two output hashes differ, because those
# candidates are bit-neutral scheduling changes and a moved hash there means
# the change was a tradeoff rather than a win. DEVIATION 2043 IS THE
# TRADEOFF CASE BY CONSTRUCTION: the fused arm accumulates dithered
# fixed-point Int32 and the PASS(8) ladder accumulates warp-private float32,
# so the bits MUST move and a harness that refuses on that can never rule on
# it. FAST is explicitly not bit-reproducible ([[fast-is-not-identical]]),
# so the gate for flipping a FAST default here is not "bits equal" -- it is
# "FASTER, and held-out accuracy NOT DEGRADED". Nothing in this repository
# had ever measured the second half; this script is that measurement.
#
# It therefore prints the hashes as a RECORD and never as a gate, and it
# carries an accuracy column that `fast_replication_ab.sh` does not have,
# because `bench/lanes_price_main.mojo`'s gbdt lane hashes the trained model
# and has no held-out score at all.
#
# THE THREE ARMS, and each is a BUILD, not a flag:
#
#   base   FAST as shipped.                bash bindings/build_gbdt.sh
#          On a 32-lane column `greedy_one_byte_fixed_for`
#          (checks/kernel_matrix.mojo:1526) returns False, so the one-byte
#          blocks fall through to CatBoost's PASS(8) ladder: gridDim.z =
#          stat_count, TWO walks over the compressed index per level.
#   pin    FAST + -D MOJOLEARN_2043_FAST_FUSED_ONE_BYTE=1
#          (checks/kernel_matrix.mojo:1607). The row returns True, the fused
#          two-stat 8-bit kernel runs -- one walk, one launch -- and with it
#          come the magnitude reduce, `acc_i32` and the fixed bridge that
#          IDENTICAL already carries. The arm built is "FAST with IDENTICAL's
#          one-byte histogram stage", which is the right comparison for this
#          question and, as the row's own comment says, the wrong one for a
#          shipping decision taken on time alone.
#   ident  IDENTICAL. MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_gbdt.sh,
#          which lands the binary under python/mojolearn/identical/ where
#          python/mojolearn/_backend.py picks it up when the SAME variable is
#          set in the environment at import. It is here as the third corner
#          of the observed inversion (0.757 FAST vs 0.693 IDENTICAL on an
#          H100) and as the accuracy reference the other two are read
#          against, NOT as a candidate default.
#
# WHY THE ARMS ARE STAGED BY COPYING A .so RATHER THAN BUILT SIDE BY SIDE.
# `base` and `pin` are both FAST, so `_backend.tier_dir()` gives them the
# SAME directory and they cannot coexist under it. Each is built once, then
# stashed under $OUT/so/, and the one an arm needs is copied back to
# python/mojolearn/_mojolearn_gbdt.so immediately before that arm's process
# starts. Every round is a fresh process that loads the binding at import, so
# a copy between processes is a complete arm swap. `ident` needs no copy: it
# lives in its own directory and is selected by the environment.
#
# THE ARMS ALTERNATE ROUND BY ROUND, base then pin then ident, and no arm
# takes its second round before every arm has taken its first. That is
# `fast_replication_ab.sh`'s rule and it is not a style choice: a rented box
# may throttle mid-run (this repository has measured one drifting 1.7x in
# twenty minutes) and blocks give the first arm's cold clocks against the
# last arm's hot ones with no way to tell.
#
# THE WARM-UP IS THE HARNESS'S OWN AND IS NOT COUNTED. Each process runs one
# untimed fit and emits FSPEED-WARMUP, then MOJOLEARN_SPEED_ROUNDS=1 timed
# fit and emits FSPEED. This script medians the FSPEED lines only; the
# WARMUP prefix is a different line and never enters the median.
#
#   MOJOLEARN_GACC_OUT=<dir> sh tools/gbdt_accuracy_ab.sh
#
# Knobs, all optional:
#   MOJOLEARN_GACC_OUT        results directory
#   MOJOLEARN_GACC_ROUNDS     timed rounds per arm (default 3, median taken)
#   MOJOLEARN_GACC_ROWS       training rows (default 2000000, floor 1000000)
#   MOJOLEARN_GACC_LANE       gbdt-symmetric (default) | gbdt-depthwise |
#                             gbdt-lossguide. SymmetricTree is the default
#                             because `use_pointwise_searcher` defaults False
#                             (gbdt/train.mojo:482), so the symmetric learner
#                             runs the GREEDY subsets searcher and IS the
#                             dispatch this row keys.
#   MOJOLEARN_GACC_PYTHON     interpreter for the arms (default python3, the
#                             one bindings/build_gbdt.sh smoke-tests against)
#   MOJOLEARN_GACC_SKIP_BUILD 1 -> reuse $OUT/so/ and python/mojolearn/identical/
#   MOJOLEARN_GACC_SKIP_FETCH 1 -> assume the dataset cache is already there
#   MOJOLEARN_GACC_SPEED_MARGIN   how much faster PIN must be to pass the
#                             speed half of the gate (default 0.02 = 2%)
#   MOJOLEARN_GACC_AUC_TOL    how much AUC PIN may lose and still pass the
#                             accuracy half (default 0.0005)
#   MOJOLEARN_GACC_LL_TOL     how much logloss PIN may gain (default 0.0005)
#   GBM_BENCH_DATA            dataset store (default ~/datasets/gbm-bench)
set -u

OUT="${MOJOLEARN_GACC_OUT:-bench/results/gbdt_accuracy_ab/$(date +%Y-%m-%d_%H%M%S)}"
ROUNDS="${MOJOLEARN_GACC_ROUNDS:-3}"
ROWS="${MOJOLEARN_GACC_ROWS:-2000000}"
LANE="${MOJOLEARN_GACC_LANE:-gbdt-symmetric}"
# THE INTERPRETER MUST HAVE NUMPY AND A BARE RENTED BOX'S python3 DOES NOT.
#
# The 2026-09-03 H100 leg died three separate ways on one cause: the higgs
# fetch, `build_gbdt.sh`'s own smoke gate and every timed round all raised
# `ModuleNotFoundError: No module named 'numpy'` under the system python3 of
# a freshly created droplet.
#
# The interpreter must be the DEFAULT pixi environment's -- not the `bench`
# feature's and not the system's. The binding exports
# `PyInit__mojolearn_gbdt` (bindings/_mojolearn_gbdt.mojo:441), so it is a
# CPython extension bound to the ABI of the interpreter it was built
# against. `bench` is Python 3.14 and would not load the .so it is being
# asked to measure; the default env is 3.13 and carries numpy 2.5.2 already,
# so the ABI-correct choice is also the one that fixes the import. It is a
# MULTI-WORD command, invoked through `py` rather than as "$PY", because
# quoting a multi-word value looks for a binary with spaces in its name.
PY="${MOJOLEARN_GACC_PYTHON:-pixi run -e gbmbench python}"
# shellcheck disable=SC2086  # $PY is deliberately word-split
py() { $PY "$@"; }
# The builds run their smoke gates in their own shells, so they are told
# too -- otherwise the build fails on numpy while the harness does not.
MOJOLEARN_PYTHON="${MOJOLEARN_GACC_BUILD_PYTHON:-pixi run -e gbmbench python}"
export MOJOLEARN_PYTHON
SPEED_MARGIN="${MOJOLEARN_GACC_SPEED_MARGIN:-0.02}"
AUC_TOL="${MOJOLEARN_GACC_AUC_TOL:-0.0005}"
LL_TOL="${MOJOLEARN_GACC_LL_TOL:-0.0005}"
AB_DEFINE="MOJOLEARN_2043_FAST_FUSED_ONE_BYTE"
GBM_BENCH_DATA="${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}"
export GBM_BENCH_DATA

# THE FLOOR IS ENFORCED, NOT DOCUMENTED. [[mojolearn-large-data-timing-floor]]
# forbids measuring or deciding tree performance below 1,000,000 rows, and a
# rule no spelling enforces is a comment
# ([[mojolearn-contract-clause-must-be-enforced]]).
if [ "$ROWS" -lt 1000000 ] 2>/dev/null; then
    echo "!! MOJOLEARN_GACC_ROWS=$ROWS is below the 1,000,000-row tree timing floor; refusing" >&2
    exit 2
fi

mkdir -p "$OUT/so" "$OUT/logs"

echo "== DEVIATION 2043 accuracy A/B: lane $LANE, higgs rows $ROWS, $ROUNDS rounds -> $OUT =="
{
    echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "head $(git rev-parse HEAD 2>/dev/null || echo nogit)"
    echo "host $(hostname -s 2>/dev/null || echo unknown)"
    echo "lane $LANE"
    echo "rows $ROWS"
    echo "rounds $ROUNDS"
    echo "ab_define $AB_DEFINE"
    echo "dataset higgs (UCI 00280/HIGGS.csv.gz) store $GBM_BENCH_DATA"
    echo "python $PY ($($PY -V 2>&1))"
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "mem $(sysctl -n vm.swapusage 2>/dev/null)"
        echo "device apple (no CUDA); this A/B answers a 32-lane NVIDIA question"
    else
        echo "mem $(free -m 2>/dev/null | awk '/^Mem:/{printf "used %sM avail %sM; ",$3,$7} /^Swap:/{printf "swap used %sM",$3}')"
        echo "device $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || rocm-smi --showproductname 2>/dev/null | head -3 || echo unknown)"
    fi
} > "$OUT/env.txt"
cat "$OUT/env.txt"

# --------------------------------------------------------------------------
# The dataset, as its own named step, before any build and outside every
# timer. Left inside an arm it would be paid inside a per-arm budget, and an
# arm killed mid-download leaves no cache for the next one.
# --------------------------------------------------------------------------
# THE FETCH RUNS *BESIDE* THE BUILDS, NOT BEFORE THEM.
#
# The lease is ONE HOUR and it is not negotiable ([[rented-gpus-self-expire]]:
# a rented GPU expires on its own after an hour, in code). Serially this leg
# does not fit: a 2.6 GB download plus an 11-million-row gzip-CSV decode, then
# three mojo binding builds, then the timed rotation. The fetch is network and
# single-core CPU; the builds are compiler CPU. They do not contend for the
# GPU and neither needs the other's output, so the fetch is backgrounded here
# and JOINED after the builds, immediately before the first fit that needs it.
#
# It is joined with an explicit `wait` and its exit status is recorded, so a
# failed download is a stated failure and not a silent fallback to the
# synthetic fixture -- `load_with_fallback` would otherwise substitute synth
# rows and the run would look like it had answered the question.
FETCH_PID=""
if [ "${MOJOLEARN_GACC_SKIP_FETCH:-0}" != "1" ]; then
    echo "== fetch higgs IN BACKGROUND (2.6 GB + gzip csv decode), joined after the builds =="
    if command -v timeout > /dev/null 2>&1; then
        ( pixi install -e gbmbench > "$OUT/logs/install.gbmbench.log" 2>&1
          timeout -k 30 3600 $PY tools/speed_gbdt_arm.py --download higgs ) \
            > "$OUT/logs/download.higgs.log" 2>&1 &
    else
        ( pixi install -e gbmbench > "$OUT/logs/install.gbmbench.log" 2>&1
          py tools/speed_gbdt_arm.py --download higgs ) \
            > "$OUT/logs/download.higgs.log" 2>&1 &
    fi
    FETCH_PID=$!
    echo "fetch_pid=$FETCH_PID (backgrounded)"
    # The 2026-09-03 second leg fell back to synthetic rows because the
    # interpreter had no pandas. `gbmbench` has it, but the environment has to
    # EXIST before the fetch can import from it, and `pixi run -e` installs on
    # first use -- which inside a backgrounded subshell would race the fetch.
    # It is installed ahead of the fetch, inside the same background job.
fi

# --------------------------------------------------------------------------
# The builds. `build_estimators.sh` first because `build_gbdt.sh` links
# against it (the same order tools/do_speed_leg.sh's remote body uses).
# --------------------------------------------------------------------------
build_fail=0
if [ "${MOJOLEARN_GACC_SKIP_BUILD:-0}" != "1" ]; then
    echo "== build estimators (FAST) =="
    bash bindings/build_estimators.sh > "$OUT/logs/build.estimators.log" 2>&1 \
        || { echo "!! estimators build failed"; tail -20 "$OUT/logs/build.estimators.log"; build_fail=1; }

    echo "== build BASE  (FAST as shipped) =="
    bash bindings/build_gbdt.sh > "$OUT/logs/build.base.log" 2>&1 \
        || { echo "!! BASE build failed"; tail -20 "$OUT/logs/build.base.log"; build_fail=1; }
    cp python/mojolearn/_mojolearn_gbdt.so "$OUT/so/base.so" 2>/dev/null \
        || { echo "!! BASE .so not produced"; build_fail=1; }

    echo "== build PIN   (FAST, -D ${AB_DEFINE}=1) =="
    MOJOLEARN_EXTRA_DEFINES="-D ${AB_DEFINE}=1" \
        bash bindings/build_gbdt.sh > "$OUT/logs/build.pin.log" 2>&1 \
        || { echo "!! PIN build failed"; tail -20 "$OUT/logs/build.pin.log"; build_fail=1; }
    cp python/mojolearn/_mojolearn_gbdt.so "$OUT/so/pin.so" 2>/dev/null \
        || { echo "!! PIN .so not produced"; build_fail=1; }

    # THE FAST DIRECTORY IS LEFT HOLDING **base**, not pin. A diagnostic
    # binary installed under the shipped name is how a later run in this
    # checkout silently measures the wrong arm.
    cp "$OUT/so/base.so" python/mojolearn/_mojolearn_gbdt.so 2>/dev/null || true

    echo "== build IDENT (IDENTICAL -> python/mojolearn/identical/) =="
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_estimators.sh \
        > "$OUT/logs/build.ident.estimators.log" 2>&1 \
        || { echo "!! IDENT estimators build failed"; tail -20 "$OUT/logs/build.ident.estimators.log"; }
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_gbdt.sh \
        > "$OUT/logs/build.ident.log" 2>&1 \
        || { echo "!! IDENT build failed"; tail -20 "$OUT/logs/build.ident.log"; build_fail=1; }
fi

ls -la "$OUT/so" python/mojolearn/_mojolearn_gbdt.so \
    python/mojolearn/identical/_mojolearn_gbdt.so > "$OUT/bindings_listing.txt" 2>&1 || true
if [ "$build_fail" != "0" ]; then
    echo "!! at least one build failed; the table below will say NO DATA where it did"
fi

# --------------------------------------------------------------------------
# The rotation. One process per (round, arm); arms alternate.
# --------------------------------------------------------------------------
run_arm() {
    # $1 arm name, $2 round index
    _arm="$1"; _r="$2"
    _log="$OUT/logs/$_arm.r$_r.log"
    case "$_arm" in
        base|pin)
            if [ ! -f "$OUT/so/$_arm.so" ]; then
                echo "  $_arm r$_r SKIPPED (no $OUT/so/$_arm.so)"; return 1
            fi
            cp "$OUT/so/$_arm.so" python/mojolearn/_mojolearn_gbdt.so || return 1
            _mode=fast
            ;;
        ident)
            if [ ! -f python/mojolearn/identical/_mojolearn_gbdt.so ]; then
                echo "  ident r$_r SKIPPED (no python/mojolearn/identical/_mojolearn_gbdt.so)"; return 1
            fi
            _mode=identical
            ;;
        *) return 1 ;;
    esac
    MOJOLEARN_NUMERIC_MODE="$_mode" \
    MOJOLEARN_SPEED_SIZE=shipped \
    MOJOLEARN_SPEED_ROUNDS=1 \
    MOJOLEARN_SPEED_BUDGET_S="${MOJOLEARN_SPEED_BUDGET_S:-1800}" \
    MOJOLEARN_SPEED_DEADLINE_S="${MOJOLEARN_SPEED_DEADLINE_S:-3600}" \
        py bench/speed/forest_speed_arm.py \
            --lane "$LANE" --dataset higgs --rows "$ROWS" --ours-only \
        > "$_log" 2>&1
    # `$?` IS CAPTURED BEFORE ANYTHING ELSE RUNS. Reading it inside an echo
    # that also carries a command substitution is a coin flip about expansion
    # order, and the value it loses is the one that says the arm died.
    _rc=$?
    echo "  $_arm r$_r exit=$_rc $(grep -c '^FSPEED ' "$_log" 2>/dev/null) timed lines"
    return 0
}

# JOIN THE BACKGROUNDED FETCH. Nothing below may run on synthetic rows
# without saying so, so the status is recorded and echoed before any fit.
if [ -n "$FETCH_PID" ]; then
    echo "== joining the higgs fetch =="
    wait "$FETCH_PID"; FETCH_RC=$?
    echo "download_higgs_exit=$FETCH_RC" >> "$OUT/env.txt"
    echo "higgs fetch exit=$FETCH_RC"
    [ "$FETCH_RC" -ne 0 ] && echo "WARNING: the higgs fetch FAILED (exit $FETCH_RC). Every arm below will" \
        && echo "         fall back to the SYNTHETIC fixture and every line will say NOT-HIGGS."
    tail -3 "$OUT/logs/download.higgs.log" 2>/dev/null
fi

echo "== rotation: $ROUNDS rounds x (base, pin, ident), one process each =="
for r in $(seq 1 "$ROUNDS"); do
    for arm in base pin ident; do
        run_arm "$arm" "$r"
    done
done

# Leave the checkout holding the SHIPPED fast binary, whatever happened above.
cp "$OUT/so/base.so" python/mojolearn/_mojolearn_gbdt.so 2>/dev/null || true

# --------------------------------------------------------------------------
# The table. Everything below reads the harness's own contract lines:
#
#   FSPEED lane=<l> arm=ours shape=<tag> round=<i> ms=<f> hash=<16 hex|->
#   FSPEED-ACC lane=<l> arm=ours metric=<logloss|auc|rmse|accuracy> value=<f>
#
# The fields are `k=v` tokens in an order this parser does NOT assume; each
# value is pulled out by its key. `fast_replication_ab.sh` was bitten once by
# a positional parser that read the SIZE column as the seconds.
# --------------------------------------------------------------------------
med() {  # med <arm>  -> median wall SECONDS over the arm's timed rounds
    cat "$OUT/logs/$1".r*.log 2>/dev/null \
        | grep '^FSPEED ' \
        | awk '{for(i=1;i<=NF;i++) if(index($i,"ms=")==1){print substr($i,4)/1000.0}}' \
        | sort -n \
        | awk '{a[NR]=$1} END{if(NR)printf "%.4f", (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
}
# THE ACCURACY OF A *FAST* ARM IS A DISTRIBUTION, NOT A VALUE.
#
# FAST makes no run-to-run promise -- `python/mojolearn/_backend.py:14` says
# in as many words that the same fit on the same box may return different
# bits on two runs, and that on the histogram lanes it measurably does. So
# `base` has its OWN held-out spread across rounds, and a single sample of
# `pin` compared against a single sample of `base` under a 0.0005 tolerance
# would be reading that spread as a treatment effect.
#
# The three accessors below therefore report the arm's MEDIAN and its
# OBSERVED RANGE over the rounds, and the verdict prints base's range beside
# the delta so the tolerance can be judged against the noise it has to clear.
# `metric` (median) is what the gate uses; `metric_lo`/`metric_hi` are what
# make the gate's answer readable. One `FSPEED-ACC` line is emitted per
# process and each round is its own process, so N rounds give N samples.
_metric_vals() {  # _metric_vals <arm> <name> -> one numeric value per line
    cat "$OUT/logs/$1".r*.log 2>/dev/null \
        | grep '^FSPEED-ACC ' | grep "metric=$2 " \
        | awk '{for(i=1;i<=NF;i++) if(index($i,"value=")==1) print substr($i,7)}'
}
metric() {  # metric <arm> <name> -> MEDIAN over the arm's rounds
    _metric_vals "$1" "$2" | sort -n \
        | awk '{v[n++]=$1} END{if(n==0)exit; if(n%2)printf "%s", v[(n-1)/2];
                else printf "%.9g", (v[n/2-1]+v[n/2])/2}'
}
metric_lo() { _metric_vals "$1" "$2" | sort -n | awk 'NR==1{printf "%s", $1}'; }
metric_hi() { _metric_vals "$1" "$2" | sort -n | awk 'END{if(NR)printf "%s", $1}'; }
metric_n()  { _metric_vals "$1" "$2" | grep -c . ; }
hashes() {  # hashes <arm> -> the distinct prediction digests, comma joined
    cat "$OUT/logs/$1".r*.log 2>/dev/null \
        | grep '^FSPEED ' \
        | awk '{for(i=1;i<=NF;i++) if(index($i,"hash=")==1){print substr($i,6)}}' \
        | sort -u | tr '\n' ',' | sed 's/,$//'
}
shapes() {  # shapes <arm> -> the distinct shape tags seen
    cat "$OUT/logs/$1".r*.log 2>/dev/null \
        | grep '^FSPEED ' \
        | awk '{for(i=1;i<=NF;i++) if(index($i,"shape=")==1){print substr($i,7)}}' \
        | sort -u | tr '\n' ',' | sed 's/,$//'
}

printf 'arm\tmode\tshape\tmed_s\tauc\tlogloss\thash\n' > "$OUT/ab.tsv"
echo
echo "== per-arm lines (machine readable; one per arm) =="
for arm in base pin ident; do
    m=$(med "$arm");        [ -z "$m" ] && m="-"
    a=$(metric "$arm" auc); [ -z "$a" ] && a="-"
    l=$(metric "$arm" logloss); [ -z "$l" ] && l="-"
    alo=$(metric_lo "$arm" auc); [ -z "$alo" ] && alo="-"
    ahi=$(metric_hi "$arm" auc); [ -z "$ahi" ] && ahi="-"
    an=$(metric_n "$arm" auc)
    h=$(hashes "$arm");     [ -z "$h" ] && h="-"
    s=$(shapes "$arm");     [ -z "$s" ] && s="-"
    case "$arm" in
        base)  lbl="FAST" ;;
        pin)   lbl="FAST+2043" ;;
        ident) lbl="IDENTICAL" ;;
    esac
    # A shape tag that is not higgs means load_with_fallback substituted the
    # synthetic fixture. Say so IN THE LINE; a synth number must never be
    # read as a higgs number.
    case "$s" in
        higgs-*) shp="$s" ;;
        -)       shp="-" ;;
        *)       shp="NOT-HIGGS:$s" ;;
    esac
    echo "GACC arm=$arm mode=$lbl lane=$LANE rows=$ROWS rounds=$ROUNDS shape=$shp med_s=$m auc=$a auc_n=$an auc_range=$alo-$ahi logloss=$l hash=$h"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$lbl" "$shp" "$m" "$a" "$l" "$h" >> "$OUT/ab.tsv"
done

echo
echo "== verdict (the gate is FASTER *and* ACCURACY NOT DEGRADED, never bits) =="
bm=$(med base); ba=$(metric base auc); bl=$(metric base logloss)
blo=$(metric_lo base auc); bhi=$(metric_hi base auc); bn=$(metric_n base auc)
echo "base AUC over $bn round(s): median $ba, range ${blo:--}..${bhi:--}  (tolerance $AUC_TOL)"
for arm in pin ident; do
    m=$(med "$arm"); a=$(metric "$arm" auc); l=$(metric "$arm" logloss)
    if [ -z "$bm" ] || [ -z "$m" ] || [ -z "$ba" ] || [ -z "$a" ]; then
        echo "GACC-VERDICT arm=$arm vs=base time_ratio=- d_auc=- d_logloss=- gate=NO DATA"
        continue
    fi
    # `have_ll` is a FLAG and not a sentinel value. An awk that is handed a
    # sentinel has to divide by zero to make a NaN, and division by zero is
    # FATAL in awk -- the verdict line would vanish and the run would look
    # like it had produced no comparison at all.
    have_ll=1
    [ -z "$bl" ] && have_ll=0
    [ -z "$l" ] && have_ll=0
    awk -v arm="$arm" -v bm="$bm" -v m="$m" -v ba="$ba" -v a="$a" \
        -v bl="${bl:-0}" -v l="${l:-0}" -v have_ll="$have_ll" \
        -v blo="${blo:-}" -v bhi="${bhi:-}" -v bn="${bn:-0}" \
        -v sm="$SPEED_MARGIN" -v at="$AUC_TOL" -v lt="$LL_TOL" 'BEGIN{
        ratio = (bm+0 > 0) ? (m+0)/(bm+0) : 0;
        dauc  = (a+0) - (ba+0);
        dll   = (l+0) - (bl+0);
        faster = (ratio > 0 && ratio < 1.0 - (sm+0));
        auc_ok = (dauc >= -(at+0));
        ll_ok  = (have_ll+0 == 0) ? 1 : (dll <= (lt+0));
        # BASE'"'"'S OWN BAND IS THE NOISE FLOOR. If base'"'"'s rounds spread wider
        # than the tolerance, a delta inside that band is not a measured
        # degradation and the tolerance is not the thing being tested --
        # say so rather than printing a confident FLIP or HOLD over noise.
        band = (bn+0 >= 2 && blo != "" && bhi != "") ? (bhi+0) - (blo+0) : -1;
        noisy = (band >= 0 && band > (at+0));
        if (arm != "pin")                   gate = "REFERENCE (not a candidate default)";
        else if (noisy && !auc_ok)          gate = "INCONCLUSIVE (base AUC band " band " exceeds tol " (at+0) ")";
        else if (faster && auc_ok && ll_ok) gate = "FLIP";
        else if (!faster)                   gate = "HOLD (not faster by the margin)";
        else if (!auc_ok)                   gate = "HOLD (AUC degraded)";
        else                                gate = "HOLD (logloss degraded)";
        if (have_ll+0 == 0)
            printf "GACC-VERDICT arm=%s vs=base time_ratio=%.4f d_auc=%+.6f d_logloss=- gate=%s\n",
                   arm, ratio, dauc, gate;
        else
            printf "GACC-VERDICT arm=%s vs=base time_ratio=%.4f d_auc=%+.6f d_logloss=%+.6f gate=%s\n",
                   arm, ratio, dauc, dll, gate;
    }'
done

echo
cat "$OUT/ab.tsv"
echo
echo "THE HASHES ARE EXPECTED TO DIFFER between base and pin and that is NOT a"
echo "defect: the fused arm accumulates dithered fixed-point Int32 where the"
echo "PASS(8) ladder accumulates warp-private float32. They are recorded, never"
echo "gated. A ratio here is a result only on a row whose shape= says higgs."
echo
echo "results in $OUT"

# EXIT NON-ZERO WHEN THERE IS NO ANSWER.
#
# The 2026-09-03 leg printed a table of dashes and returned 0, so the remote
# runner logged EXTRA-gbdt-accuracy-ab-EXIT=0 over a run in which nothing
# built, nothing downloaded and nothing was measured. A harness that reports
# success on total failure is worse than one that crashes: it costs a lease
# AND the trust in every other green line beside it.
rc=0
[ "$build_fail" != "0" ] && rc=2
if [ -z "$(med base)" ] || [ -z "$(med pin)" ]; then
    echo "!! NO COMPARISON PRODUCED: base and pin must both have timed rounds"
    rc=3
fi
# THE SYNTHETIC FIXTURE IS NOT AN ANSWER AND MUST NOT EXIT GREEN.
# The 2026-09-03 leg fell back to synthclf-720000x100 -- which is also BELOW
# the 1,000,000-row floor this script enforces on its own input -- and still
# returned 0. Worse, base and pin came back BIT-EQUAL on it: the fused arm
# never engaged, because a fixture whose features do not reach >128 bins
# never takes the one-byte branch the define switches. A run that cannot
# reach the arm under test is a failed run, not a null result.
case "$(shapes base)" in
    higgs-*) ;;
    *) echo "!! NOT HIGGS: the fixture fell back to synthetic rows, which"
       echo "   neither reaches the arm under test nor clears the row floor"
       rc=4 ;;
esac
[ "$rc" != "0" ] && echo "!! gbdt_accuracy_ab FAILED (rc=$rc)"
exit $rc
