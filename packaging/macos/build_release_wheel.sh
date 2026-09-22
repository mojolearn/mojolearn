#!/bin/sh
# Build the macOS arm64 wheel: ten extensions per numeric tier, staged MAX
# runtime, re-signed, packed.
#
# WHY STAGING IS NOT OPTIONAL. `otool -L` on the freshly built extension shows
#
#     @rpath/libKGENCompilerRTShared.dylib
#     @rpath/libAsyncRTMojoBindings.dylib
#
# Those live in the pixi environment. A wheel that ships without them imports
# fine on THIS machine, where the rpath still resolves, and fails on every
# other with a dyld error naming a path the user has never heard of. That is
# the worst kind of packaging bug: invisible to the person who built it.
#
# So they are copied next to the extension, the rpath is repointed at
# @loader_path/.dylibs, and the result is re-signed -- macOS invalidates the
# signature the moment install_name_tool rewrites a load command, and an
# unsigned dylib is killed on load on Apple silicon rather than merely warned
# about.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$here"

ENV_LIB="$here/.pixi/envs/default/lib"
PKG="$here/python/mojolearn"
DYLIBS="$PKG/.dylibs"
STAMP=$(mktemp "${TMPDIR:-/tmp}/mojolearn-release-stamp.XXXXXX")
trap 'rm -f "$STAMP"' EXIT INT TERM

# Refreshed every build. These are COPIES of the repository root's files, and
# a stale copy in a published wheel is a wrong LICENSE or a wrong README on
# PyPI. python/.gitignore keeps them out of the checkout.
cp "$here/LICENSE" "$here/NOTICE" "$here/README.md" "$here/python/"
# CITATION.cff goes INSIDE the package, not beside pyproject.toml: it is
# shipped as package data so `pip install` carries the machine-readable
# citation. Attribution that lives only in the git repository does not
# travel with the artifact, and the third way people acquire this library
# is neither a clone nor a pip install but a copied file.
cp "$here/CITATION.cff" "$here/python/mojolearn/"

# EVERY EXTENSION IN THE WHEEL IS BUILT HERE. `pyproject.toml`'s
# package-data globs `*.so`, so an extension that is NOT built here is not
# absent from the wheel -- it is shipped STALE, whatever happens to be sitting
# in the working tree from an earlier build. That is how `_mojolearn.so` came
# to ship an artifact predating `eval_x`/`eval_y`, where every `fit` raised
# "takes 6 positional arguments but 8 were given", and how
# `_mojolearn_estimators.so` shipped as a ZERO-KERNEL artifact on 2026-08-22
# while this file said "every extension" and built two of three.
#
# AS OF 2026-08-24 THE LIST IS TEN EXTENSIONS IN TWO NUMERIC MODES. The
# FAST set lands at python/mojolearn/*.so and the IDENTICAL set at
# python/mojolearn/identical/*.so; python/mojolearn/_backend.py loads the
# identical set when MOJOLEARN_NUMERIC_MODE=identical is set at import
# (checks/numerics.mojo reads the build define). Both sets ship in ONE
# wheel. The list below is THE list: bindings/build_*.sh that is not named
# here does not ship, and a name here with no script fails the build.
# 2026-08-24: five bindings were added at once (svm/isolation-forest,
# solver/hierarchy, metrics/spectral, holtwinters/tsa, and the linalg GEMM
# surface). They are listed here because a build script that is not named
# here does not ship, and python/mojolearn/_backend.py now knows all eleven.
# 2026-09-01: `_mojolearn_gp` (gaussian_process) joined as the THIRTEENTH,
# the day the estimator left `_NOT_YET` (commit 22a5b550), added to BOTH
# lists below the same day its binding was written.
# 2026-09-02: `_mojolearn_mamba` (FOURTEENTH) and `_mojolearn_transformer`
# (FIFTEENTH) added to BOTH lists. The mamba pair was FOUND MISSING while
# the transformer binding was being written ([[fix-docs-on-discovery]]):
# it had shipped a Python surface on 2026-09-01 with these two lists never
# updated, which is exactly the ships-STALE failure the paragraph above
# describes -- a release cut in between would have carried the mamba
# classes and no binary, or a stale one. The transformer binding COMPILED
# FOR THE FIRST TIME 2026-09-02, on the M4 and on APPLE ONLY -- no NVIDIA
# or AMD box has built it -- and its Python surface gate
# (python/mojolearn/tests/test_transformer_surface.py) is green in all
# three tiers on that one box (transformer/README.md's PyPI-surface
# ledger). Listing it here means a release build ships it fresh or fails
# loudly, never silently omits it.
#
# DELIBERATELY NOT MIRRORED IN tools/e1_bootstrap.sh, AND THAT ASYMMETRY IS
# THE POINT. That script builds bindings on a RENTED GPU under a work bound.
# Phase 8 drives every lane gate through `mojo run` on the lane's own driver,
# not through a Python binding, so the five new extensions buy a leg nothing
# and would roughly double its binding-build time. A leg that spends its
# lease compiling and comes home with an empty lanes/ has bought nothing at
# all. Add a binding there only when a phase actually imports it.
BUILD_SCRIPTS="build_gbdt.sh build_rf.sh build_trees.sh"
EXT_NAMES="_mojolearn_gbdt _mojolearn_rf _mojolearn_trees"

# ONE TIER RULE (DEVIATION 2490, 2026-09-10): the three TREE lanes above
# build in every tier MODES names. EVERY OTHER BINDING is identical only and
# lives in the two lists below, built and gated for the identical tier alone
# the way the byte LM always was. Cross-vendor bitwise identity is the
# product; a fast tier ships only where it has a measured win over the
# opponent's own CPU, and outside trees it has none (the reasoning and the
# M4 numbers are on `_TIERED` in python/mojolearn/_backend.py). The
# identical-only build scripts exit 2 on any other MOJOLEARN_NUMERIC_MODE.
# Before this the split was neural-only (three lanes, 2026-09-10 morning);
# the reason for THOSE was different (their fused kernels were gated on the
# identical contract, so the lower tiers were slower) and no longer matters.
IDENTICAL_ONLY_SCRIPTS="build.sh build_estimators.sh build_svm.sh build_solver.sh build_metrics.sh build_preprocessing.sh build_tsa.sh build_linalg.sh build_arima.sh build_gp.sh build_training.sh build_mamba.sh build_transformer.sh build_kernel_methods.sh build_mixture.sh build_hdbscan.sh build_resample.sh build_ivf.sh build_embedding.sh"
IDENTICAL_ONLY_NAMES="_mojolearn _mojolearn_estimators _mojolearn_svm _mojolearn_solver _mojolearn_metrics _mojolearn_preprocessing _mojolearn_tsa _mojolearn_linalg _mojolearn_arima _mojolearn_gp _mojolearn_training _mojolearn_mamba _mojolearn_transformer _mojolearn_kernel_methods _mojolearn_mixture _mojolearn_hdbscan _mojolearn_resample _mojolearn_ivf _mojolearn_embedding"
# THE RELEASE PROFILE IS THE DEFAULT (2026-09-22). Until 0.8.14 this read
# ${MOJOLEARN_PACKAGE_BYTE_LM:-0}: only release-provenance.yml set it to 1, so
# the checklist's local command built a wheel with ZERO host bindings and the
# smoke failed on `host/_mojolearn_neural_host.so is not built`. A published
# wheel always carries the byte LM and every host binding, so 1 is the
# default; MOJOLEARN_PACKAGE_BYTE_LM=0 remains an explicit opt-out for a
# legacy or bisect build. It is exported so verify_wheel.sh (run at the end)
# checks the same profile this script built.
PACKAGE_BYTE_LM=${MOJOLEARN_PACKAGE_BYTE_LM:-1}
case "$PACKAGE_BYTE_LM" in 0|1) ;; *) echo 'MOJOLEARN_PACKAGE_BYTE_LM must be 0 or 1' >&2; exit 2 ;; esac
export MOJOLEARN_PACKAGE_BYTE_LM="$PACKAGE_BYTE_LM"
echo "== byte LM and host bindings: MOJOLEARN_PACKAGE_BYTE_LM=$PACKAGE_BYTE_LM"
unset MOJOLEARN_BYTE_LM_OUTDIR
# THE HOST (CPU) BINDINGS, one per family, under python/mojolearn/host/.
# Since 0.8.6 (the packaging lane, 2026-09-14) every host family the manifest
# declares ships in the wheel, and WHICH families is READ from
# python/mojolearn/host_surface.py rather than written here;
# packaging/check_ext_lists.py fails this file if it ever carries a host list
# of its own. Until 0.8.5 the byte LM's was the only one, named by hand in
# four places. They build with the release profile (PACKAGE_BYTE_LM=1), in
# the identical tier alone, with no accelerator target, pinned to the CPU
# column, and each is read back below.
HOST_FAMILIES=$(python3 python/mojolearn/host_surface.py --wheel-families) || exit 2
HOST_NAMES=$(python3 python/mojolearn/host_surface.py --wheel-bindings) || exit 2
[ -n "$HOST_FAMILIES" ] && [ -n "$HOST_NAMES" ] || { echo 'the manifest names no wheel host family' >&2; exit 2; }
if [ "$PACKAGE_BYTE_LM" = 1 ]; then
    # bindings/build_host_family.sh refuses an existing output rather than
    # overwriting it, so a host .so left by an earlier local build fails the
    # build instead of being reused; remove them first, as build_sets.sh does.
    for n in $HOST_NAMES; do rm -f "$PKG/host/$n.so"; done
    # bindings/build_byte_lm.sh refuses the same way ("byte LM: output already
    # exists"), and on 2026-09-21 a byte LM left by an earlier local build
    # failed the 0.8.12 wheel build 13 minutes in. It is rebuilt below like
    # every other extension, so the stale copy goes first; the refusal inside
    # build_byte_lm.sh stays, for anyone running it by hand. A CI checkout
    # never has the file.
    rm -f "$PKG/identical/_mojolearn_byte_lm.so"
fi

# THE PER-SCRIPT GATES ARE OFF HERE, AND THE REASON IS A CLEAN CHECKOUT.
# Each bindings/build_*.sh ends by copying python/mojolearn/ aside and
# importing the package to fit on the binary it just built. The package
# __init__ imports EVERY binding, so that gate needs all five extensions to
# exist already; in the shared working tree they always did, and in a clean
# checkout of a tag (the only honest place to build a release from) the
# first gate fails on the fourth missing .so before anything runs. Found
# 2026-08-23 on the first clean-tree build. So this script builds all
# fifteen binaries gate-off (it said "eleven" until 2026-09-01 and
# "thirteen" until 2026-09-02, in each case some bindings after that count
# was true) and then runs THE release gate,
# verify_wheel.sh, which
# installs the finished wheel into a clean venv under every claimed
# interpreter and fits every estimator family in EVERY SHIPPED numeric mode
# (three since 2026-08-29; this line said BOTH when there were two). That
# is strictly more than the per-script gates check, and it runs on the
# artifact that ships rather than on a copy of the tree.
#
# THE TWO LISTS ABOVE ARE NOT THE SAME LIST AND BOTH MUST NAME EVERY
# EXTENSION. BUILD_SCRIPTS is what runs; EXT_NAMES is what is then CHECKED
# for existence and for being newer than this script's start. An extension
# built but not named in EXT_NAMES ships STALE rather than absent, silently,
# which is how `_mojolearn.so` once shipped predating eval_x/eval_y.
# THE TIERS THIS WHEEL CARRIES. `fast` lives at python/mojolearn/*.so and
# every other tier one directory down under its own name, which is the layout
# python/mojolearn/_backend.py loads from.
#
# THE DEFAULT IS THREE TIERS, 2026-08-29. It read "TWO TIERS AND NOT THREE"
# until the determinism lane closed the same day; that sentence is deleted
# rather than softened, because a stale default here silently ships a wheel
# whose deterministic tier raises from a missing-binary stub.
#
# WHAT CLOSED THE LANE, and it is not a pin count. A tier is a PROMISE, and
# the promise is measured by tools/repeat_run_stability.py, which runs one
# fit repeatedly in one process and compares RAW OUTPUT BYTES. Taken at one
# commit on all three vendors on 2026-08-29:
#
#     column                 fast                     deterministic
#     Apple M4 (Metal)       MOVED in 8 of 10 tries   STABLE 10/10
#     NVIDIA RTX 4090 (CUDA) MOVED, 24 answers in 24  STABLE, 1 in 12
#     AMD MI325X (HIP)       MOVED, 6 answers in 24   STABLE, 1 in 12
#
# bench/results/stability/RESULTS.md is the record. The pin side is 15 files
# keyed to PIN_DETERMINISM; the determinism class is small because this tree
# uses no float atomicAdd anywhere, which is why the middle tier is cheap.
#
# THE MIDDLE TIER IS NOT THE TOP ONE WEARING A HAT. Its hashes DIFFER across
# vendors on 10 of 12 comparable lanes, gemm-vendor among them -- it keeps
# the vendor matmul and its speed and buys no cross-vendor identity.
#
# To cut a two-tier wheel anyway (a hotfix, a bisect), name the tiers:
#     MOJOLEARN_RELEASE_MODES="fast identical" ./packaging/macos/build_release_wheel.sh
# and keep verify_wheel.sh's copy of the variable set the same for one release
# or the verifier fails the wheel for lacking a tier it was never asked to
# build. pyproject.toml's package-data glob carries all three directories.
MODES="${MOJOLEARN_RELEASE_MODES:-fast deterministic identical}"
echo "== numeric tiers in this wheel: $MODES"
if [ "$PACKAGE_BYTE_LM" = 1 ]; then
    case " $MODES " in *' identical '*) ;; *) echo 'Byte LM requires the identical tier' >&2; exit 2 ;; esac
fi
# Refuse stale unsupported-mode artifacts rather than silently packaging them.
for byte_path in "$PKG/_mojolearn_byte_lm.so" "$PKG/deterministic/_mojolearn_byte_lm.so"; do
    [ ! -e "$byte_path" ] && [ ! -L "$byte_path" ] || { echo 'Byte LM is IDENTICAL only' >&2; exit 2; }
done
if [ "$PACKAGE_BYTE_LM" = 0 ]; then
    [ ! -e "$PKG/identical/_mojolearn_byte_lm.so" ] && [ ! -L "$PKG/identical/_mojolearn_byte_lm.so" ] || {
        echo 'Legacy build refuses an unrequested byte LM binary' >&2; exit 2;
    }
fi

# DEVIATION 2501: the builds run MOJOLEARN_BUILD_JOBS at a time through
# xargs -P; each (mode, script) pair writes its own log, printed in full when
# it finishes so the transcript reads as before. Every pair has its own output
# directory and mktemp scratch, so the pairs are independent. The identical
# tier is listed first because it holds the most scripts, and the pairs are
# then queued heaviest first (ordered_pairs below). Any failure fails the
# build after the running pairs finish (xargs exits 123).
#
# THE DEFAULT IS FOUR BUILDS OF ONE COMPILER WORKER (2026-09-21, measured).
# The 0.8.13 macOS build with an empty compile cache ran 2666 s. It compiled
# one extension at a time with one worker: it ran under `tools/mac_slot.py
# run`, which pins MOJOLEARN_BUILD_JOBS=1 and MOJOLEARN_COMPILE_JOBS=1 and
# overrides whatever this script defaults to (the log's third line says
# "1 at a time"). The 61 per-extension compile times (build_seconds in the
# local cache's manifests) sum to 2371 s, the longest is the identical mamba
# binding at 322.5 s, and the remaining 295 s are staging, packing, the audit
# and the five interpreter runs. The 0.8.12 build (908 s,
# ~/mojolearn-evidence/releases/0.8.12/mac-build-r5.log) was ALSO one at a
# time with one worker, so its 16 minutes were not a parallel default: its
# compiles were faster on that run, not more of them at once.
#
# Scheduling those 61 measured times on P single-worker builds, heaviest
# first (the order below), gives a compile wall of 1186 s at P=2, 792 s at
# 3, 595 s at 4 and 476 s at 5; with the 295 s around it, P=4 is about 890 s
# (15 minutes) cold, where the 2 x 2 default of 63089d263 would have been
# about 25 minutes even at full speed. Four workers stay inside the Apple
# release budget of five cores (Andrew, 2026-09-21; the M4 has 4P + 6E cores
# and 16 GB), and at about 1.2 GB peak per mojo compile they need about
# 5 GB. One worker each, not two: the measurements are all at one worker, and
# 4 x 2 would be eight compiler threads on a five core budget.
#
# UNDER mac_slot, pass --slots 4: it then holds four of the five Mac slots
# and hands the build MOJOLEARN_BUILD_JOBS=4 (compiler workers stay 1):
#
#     python3 tools/mac_slot.py --slots 4 run -- ./packaging/macos/build_release_wheel.sh
#
# Without --slots, mac_slot still pins one build at a time.
#
# THE LOCAL COMPILE CACHE. With MOJOLEARN_BINCACHE_DIR set every pair runs
# through `tools/bincache.py build`, keyed on the binding's import closure,
# every build script, pixi.lock, the Mojo and Xcode toolchains, the mode, the
# environment and the checkout path, with the one .so it writes DECLARED
# (the pairs share a tree, so outputs are never inferred). An unchanged
# binding is placed from the cache, verified byte for byte against the
# archive's manifest, and every gate below runs on it exactly as on a fresh
# build. Unset, nothing changes. The header of tools/bincache.py has the key.
BUILD_JOBS="${MOJOLEARN_BUILD_JOBS:-4}"
# One compiler worker per build unless the caller says otherwise (the
# bindings/build_*.sh default is 2); see the measurement above.
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-1}"
case "$BUILD_JOBS" in ''|*[!0-9]*|0) echo 'MOJOLEARN_BUILD_JOBS must be 1..16' >&2; exit 2;; esac
[ "$BUILD_JOBS" -le 16 ] || { echo 'MOJOLEARN_BUILD_JOBS must be 1..16' >&2; exit 2; }
BUILD_LOGS=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-release-builds.XXXXXX")
build_pairs() {
    for mode in $MODES; do
        [ "$mode" = identical ] || continue
        for script in $BUILD_SCRIPTS; do printf '%s %s\n' "$mode" "$script"; done
        for script in $IDENTICAL_ONLY_SCRIPTS; do printf '%s %s\n' "$mode" "$script"; done
        if [ "$PACKAGE_BYTE_LM" = 1 ]; then printf '%s %s\n' "$mode" build_byte_lm.sh; fi
        # The host bindings. Identical only, and built with no accelerator
        # target, which is why they are named here rather than added to a
        # tier list: none is a tier member or a vendor member.
        if [ "$PACKAGE_BYTE_LM" = 1 ]; then
            for f in $HOST_FAMILIES; do printf '%s %s\n' "$mode" "build_${f}_host.sh"; done
        fi
    done
    for mode in $MODES; do
        [ "$mode" = identical ] && continue
        for script in $BUILD_SCRIPTS; do printf '%s %s\n' "$mode" "$script"; done
    done
}
# HEAVIEST FIRST. The seconds are the 0.8.13 cold compile times (one worker,
# build_seconds in the cache manifests); every pair not named weighs 0 and
# keeps its place (a stable sort). A stale entry only moves wall time: the
# pairs are independent. Measured order-versus-heaviest-first at four builds:
# 664 s against 595 s of compile.
pair_weight() {
    case "$1 $2" in
        "identical build_mamba.sh") echo 323 ;;
        "identical build_byte_lm.sh") echo 182 ;;
        "identical build_hdbscan.sh") echo 182 ;;
        "fast build_gbdt.sh") echo 157 ;;
        "identical build_transformer.sh") echo 136 ;;
        "identical build_kernel_methods.sh") echo 111 ;;
        "identical build_ivf.sh") echo 110 ;;
        "identical build_gbdt.sh") echo 87 ;;
        "deterministic build_gbdt.sh") echo 81 ;;
        "identical build_mixture.sh") echo 71 ;;
        "identical build_resample.sh") echo 58 ;;
        "identical build_gbdt_host.sh") echo 52 ;;
        *) echo 0 ;;
    esac
}
ordered_pairs() {
    build_pairs | while read -r mode script; do
        printf '%s %s %s\n' "$(pair_weight "$mode" "$script")" "$mode" "$script"
    done | sort -s -k1,1nr | cut -d' ' -f2-
}
[ "$(ordered_pairs | sort)" = "$(build_pairs | sort)" ] || { echo 'build order lost or added a pair' >&2; exit 2; }
echo "== building $(build_pairs | wc -l | tr -d ' ') extensions, $BUILD_JOBS at a time, $MOJOLEARN_COMPILE_JOBS compiler worker(s) each (logs in $BUILD_LOGS)"
ordered_pairs | xargs -P "$BUILD_JOBS" -n 2 sh -c '
    logs=$1; mode=$2; script=$3; log="$logs/${mode}_${script%.sh}.log"
    # The byte LM build keeps its own gate on, as it always did here.
    skip=1; [ "$script" = build_byte_lm.sh ] && skip=
    # THE ONE FILE THIS PAIR WRITES, declared for the compile cache. Named
    # the way the gates below name it: build.sh is _mojolearn, build_X.sh is
    # _mojolearn_X, a host shim is host/_mojolearn_X_host, the byte LM lives
    # in identical/, and the fast tier is the package directory itself.
    case "$script" in
        build_*_host.sh) f="${script#build_}"; output="python/mojolearn/host/_mojolearn_${f%_host.sh}_host.so" ;;
        build_byte_lm.sh) output="python/mojolearn/identical/_mojolearn_byte_lm.so" ;;
        build.sh) ext=_mojolearn ;;
        *) e="${script#build_}"; ext="_mojolearn_${e%.sh}" ;;
    esac
    case "$script" in build_*_host.sh|build_byte_lm.sh) ;; *)
        if [ "$mode" = fast ]; then output="python/mojolearn/$ext.so"; else output="python/mojolearn/$mode/$ext.so"; fi ;;
    esac
    runner=bash
    if [ -n "${MOJOLEARN_BINCACHE_DIR:-}" ]; then
        runner="python3 tools/bincache.py build"
        export MOJOLEARN_BINCACHE_OUTPUTS="$output" MOJOLEARN_BINCACHE_SHELL=bash
        # The cache never replaces an existing file, so a .so left by an
        # earlier local build would force a compile. Every build script
        # replaces its output anyway, and the staleness gate below requires a
        # file newer than this run, so removing it first loses nothing.
        rm -f "$output"
    fi
    # A host build must never see an accelerator target: it has no device
    # code, and MOJOLEARN_GPU_ARCHS reaching it is the one way it can be
    # silently wrong. Both output directory variables (the per-family one and
    # the shared MOJOLEARN_HOST_OUTDIR) are unset for the same reason the byte
    # LM build unsets its own (NO APOSTROPHE IN THIS BLOCK: it is one
    # single-quoted sh -c string), so it lands in the package tree. It compiles the CPU
    # column only and refuses any other MOJOLEARN_TARGET_COLUMN by name, so
    # the column is pinned to cpu here (packaging/linux/build_sets.sh does the
    # same; the Linux legs export the GPU column to every build).
    case "$script" in build_*_host.sh)
        fam="${script#build_}"; fam="${fam%_host.sh}"
        FAM=$(printf "%s" "$fam" | tr "a-z" "A-Z")
        if MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_SKIP_BUILD_GATE=$skip MOJOLEARN_TARGET_COLUMN=cpu \
             env -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_HOST_OUTDIR -u "MOJOLEARN_${FAM}_HOST_OUTDIR" \
             $runner "./bindings/$script" > "$log" 2>&1; then
            { echo "== $script ($mode) OK"; cat "$log"; }
        else
            { echo "== $script ($mode) FAILED"; cat "$log"; }
            exit 1
        fi
        exit 0 ;;
    esac
    if MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_SKIP_BUILD_GATE=$skip $runner "./bindings/$script" > "$log" 2>&1; then
        { echo "== $script ($mode) OK"; cat "$log"; }
    else
        { echo "== $script ($mode) FAILED"; cat "$log"; }
        exit 1
    fi' build_one_pair "$BUILD_LOGS" 2>&1 || { echo "== at least one extension failed to build (logs in $BUILD_LOGS)" >&2; exit 1; }

# THE FILES THE REST OF THIS SCRIPT GATES: EXT_NAMES x MODES (three tree
# bindings x three tiers = nine) plus IDENTICAL_ONLY_NAMES in identical/
# alone (thirteen), twenty-two on the default build as of DEVIATION 2490. Built above or absent, never
# stale: every one is checked for existence and for being newer than this
# script's start, so a build script that silently left the old file in place
# fails here instead of shipping.
# One entry per extension per tier. `fast` is the package directory itself and
# every other tier is a subdirectory of the same name, so this loop does not
# need to know which tiers exist, only what MODES says.
ALL_SOS=""
for n in $EXT_NAMES; do
    for mode in $MODES; do
        if [ "$mode" = "fast" ]; then
            ALL_SOS="$ALL_SOS $PKG/$n.so"
        else
            ALL_SOS="$ALL_SOS $PKG/$mode/$n.so"
        fi
    done
done
# The identical-only lanes are gated in ONE tier, whatever MODES says.
case " $MODES " in *" identical "*)
    for n in $IDENTICAL_ONLY_NAMES; do ALL_SOS="$ALL_SOS $PKG/identical/$n.so"; done ;;
esac
if [ "$PACKAGE_BYTE_LM" = 1 ]; then
    # Every host binding is gated for existence and staleness like every
    # other extension, but each is asked a different question: it must report
    # 'cpu' as its vendor AND as its kernel-matrix column, because a GPU
    # vendor here would mean the CPU-only build saw an accelerator target, and
    # a GPU column would mean the build box's GPU name was folded into a
    # vendor-neutral binary (the 0.8.5 freeze caught exactly that: the NVIDIA
    # and AMD legs' copies differed by 43 bytes and the packer refused the
    # wheel). They live beside the tiers, in host/, which is where the
    # runtime looks rather than through _backend.binding().
    for n in $HOST_NAMES; do
        ALL_SOS="$ALL_SOS $PKG/host/$n.so"
        pixi run -e pkg python - "$PKG/host/$n.so" "$n" <<'PYHOST'
import importlib.util, json, sys
path, name = sys.argv[1], sys.argv[2]
prefix = name[len('_mojolearn_'):]
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert getattr(module, prefix + '_numeric_mode')() == 1, name + ' must be IDENTICAL'
assert getattr(module, prefix + '_vendor')() == 'cpu', name + ' must report cpu'
assert str(getattr(module, prefix + '_column')()) == 'cpu', name + ' must compile as the CPU column'
assert not bool(getattr(module, prefix + '_sabotage')()), name + ' is a sabotage build'
print(json.dumps(dict(extension=name, native_vendor='cpu', column='cpu',
    numeric_mode=1, supported_modes=['identical'],
    unsupported_modes=['fast', 'deterministic'])))
PYHOST
    done
    ALL_SOS="$ALL_SOS $PKG/identical/_mojolearn_byte_lm.so"
    pixi run -e pkg python - "$PKG/identical/_mojolearn_byte_lm.so" <<'PYBYTE'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('_mojolearn_byte_lm', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.byte_lm_numeric_mode() == 1
assert module.byte_lm_vendor() == 'metal'
assert module.byte_lm_profile() == 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
print(json.dumps(dict(extension='_mojolearn_byte_lm', native_vendor='metal', numeric_mode=1,
    profile=module.byte_lm_profile(), supported_modes=['identical'], unsupported_modes=['fast', 'deterministic'])))
PYBYTE
fi
for so in $ALL_SOS; do
    [ -f "$so" ] || { echo "ERROR: $so was not produced" >&2; exit 1; }
    [ "$so" -nt "$STAMP" ] || { echo "ERROR: $so predates this build (stale)" >&2; exit 1; }
done

# AND NOTHING ELSE. The loop above walks ALL_SOS, so it can only ever check a
# file it already expects; a .so that is on NO list is neither required nor
# refused. pyproject.toml's package-data globs `*.so`, `deterministic/*.so` and
# `identical/*.so` unconditionally, so such a file is not absent from the wheel,
# it SHIPS -- built by a different commit, for a different layout, with nothing
# anywhere naming it.
#
# This is not hypothetical. On 2026-09-12 this tree held eighteen extensions
# dated to the 0.6.0 build, nine in `fast` and nine in `deterministic`, from
# before DEVIATION 2490 narrowed the lower tiers to the three tree lanes. Every
# gate in this script passed over them in silence. PyPI was spared only because
# the published wheel is built by release-provenance.yml on a FRESH CHECKOUT,
# where .gitignore means they do not exist -- luck of the build environment, not
# a check. A local build had no such luck, and this is the check.
EXPECTED_SOS=$(mktemp "${TMPDIR:-/tmp}/mojolearn-expected-sos.XXXXXX")
for so in $ALL_SOS; do printf '%s\n' "${so#"$PKG/"}"; done | sort -u > "$EXPECTED_SOS"
UNEXPECTED_SOS=$( (cd "$PKG" && find . -name '*.so') | sed 's|^\./||' | sort -u \
    | grep -Fxv -f "$EXPECTED_SOS" || true )
rm -f "$EXPECTED_SOS"
if [ -n "$UNEXPECTED_SOS" ]; then
    echo "ERROR: $PKG holds extensions this build does not expect." >&2
    echo "       They are not absent from the wheel; package-data globs *.so," >&2
    echo "       so they SHIP STALE. Remove them and build again:" >&2
    printf '%s\n' "$UNEXPECTED_SOS" | sed 's|^|           '"$PKG"'/|' >&2
    exit 1
fi
echo "package tree: exactly the $(printf '%s\n' $ALL_SOS | sort -u | wc -l | tr -d ' ') expected extensions, no strays"

# The FULL transitive closure, walked rather than sampled. See
# packaging/macos/stage_dylibs.py: reading only the extension's direct
# dependencies staged 2 dylibs when the real closure is 4, and shipped a wheel
# that failed on install with "Library not loaded:
# @rpath/libMSupportGlobals.dylib, referenced from .dylibs/libAsyncRT...".
# It also verifies statically that nothing is left unresolved, which is the
# only form of this check that means anything on the build machine.
# EVERY EXTENSION OF EVERY TIER IN ONE CALL, because there is one `.dylibs`
# and the
# script wipes it before staging. Separate calls would leave earlier closures
# deleted -- invisibly, because on THIS machine the original rpath still
# resolves into the pixi environment. The identical/ set sits one directory
# down and gets @loader_path/../.dylibs; the stager computes that per file.
# shellcheck disable=SC2086
pixi run -e pkg python "$here/packaging/macos/stage_dylibs.py" \
    $ALL_SOS "$ENV_LIB"
pixi run -e pkg python "$here/packaging/portable_math/stage.py" "$PKG/.dylibs" \
    --receipt "$here/portable-math-build.json"



# THE TAG AND THE BINARY MUST AGREE, and nothing else checks this.
# The wheel filename is what pip compares before it tries to load anything, so
# a tag above the binary's floor turns installable Macs away and a tag below it
# installs onto Macs where the extension cannot load. Both are silent on the
# machine that built the wheel.
# CHECKED FOR EVERY EXTENSION, not just the first: one wheel carries one tag,
# and the tag is only honest if it is the floor of EVERYTHING inside.
TAG_MINOS=$(grep -E '^DEFAULT_MACOS_TARGET' "$here/python/setup.py" | sed 's/[^0-9.]//g')
for so in $ALL_SOS; do
    BIN_MINOS=$(otool -l "$so" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
    if [ "$BIN_MINOS" != "$TAG_MINOS" ]; then
        echo "ERROR: $(basename "$so") minos $BIN_MINOS but setup.py tags $TAG_MINOS" >&2
        echo "       MACOS_FLOOR in bindings/build.sh and bindings/build_gbdt.sh" >&2
        echo "       and DEFAULT_MACOS_TARGET in python/setup.py must all match" >&2
        exit 1
    fi
    echo "macOS floor: ${so#$PKG/} minos $BIN_MINOS == wheel tag $TAG_MINOS"
done

# THE ISA BASELINE, WHICH NO HEADER CAN SEE. arm64 Mach-O cpusubtype stays
# ARM64_ALL whatever --target-cpu was, so this has to disassemble. Gates the
# wheel: a binary carrying bf16, i8mm or SME instructions SIGILLs on the Macs
# the macosx_11_0 tag invites in.
# shellcheck disable=SC2086
pixi run -e pkg python "$here/packaging/isa_baseline.py" $ALL_SOS

# NO GPU KERNELS, NO WHEEL. A build on a machine without a usable Apple GPU
# emits the host half and silently no Metal shader code, exits 0, and produces
# a wheel that imports and then dies on the first fit. That shipped once, as
# TestPyPI 0.1.0a2. See packaging/macos/check_gpu_embedded.py.
#
# EVERY EXTENSION EXCEPT THE HOST BINDINGS (DEVIATION 2680). Those are built
# with `env -u MOJOLEARN_GPU_ARCHS`, contain no device code by design and read
# back vendor 'cpu', so they have ZERO AIR markers and this gate refuses them
# correctly. The exclusion is HERE, at the call site, and NOT inside
# check_gpu_embedded.py: that script exists because 0.1.0a2 shipped a host-only
# build that passed every other gate, so teaching it to skip a file whose name
# looks host-like would reopen the hole it was written to close. The exclusion
# is by the exact names the manifest lists, never by a host-like pattern.
# Every other consumer of ALL_SOS still sees the host bindings, and must:
# stage_dylibs.py wipes and rebuilds the one .dylibs directory, the minos loop
# keeps the wheel tag honest as the floor of everything inside, and
# isa_baseline.py is a HOST-code check, which is precisely what they are.
GPU_SOS=""
for so in $ALL_SOS; do
    is_host=0
    for n in $HOST_NAMES; do
        [ "$so" = "$PKG/host/$n.so" ] && is_host=1
    done
    [ "$is_host" = 1 ] || GPU_SOS="$GPU_SOS $so"
done
# shellcheck disable=SC2086
pixi run -e pkg python "$here/packaging/macos/check_gpu_embedded.py" $GPU_SOS

# THE IDENTITY PAYLOAD (the packaging lane, 2026-09-14; docs/VERIFY.md).
# `python -m mojolearn verify` needs the one card comparator and
# mojolearn/reference_cards/; `python -m mojolearn identity` needs
# tools/identity_break.py, the three training GPU columns the manifest
# names, and a commit witness (an installed wheel has no git, and
# identity_break refuses to write a column with an empty commit). All are
# COPIES made here of files that live once in git, so the wheel has no
# second implementation of anything; python/.gitignore keeps the copies out
# of the checkout. The Linux packer (packaging/linux/pack_wheel.py) reads
# the same sources straight into the archive.
rm -f "$PKG/_identity_trace_diff.py" "$PKG/_identity_break.py"
rm -rf "$PKG/identity_columns"
cp "$here/tools/identity_trace_diff.py" "$PKG/_identity_trace_diff.py"
cp "$here/tools/identity_break.py" "$PKG/_identity_break.py"
python3 "$here/tools/verification_ctr_payload.py" --output "$PKG/verify_reference/ctr_models" || exit 1
RECORD=$(python3 python/mojolearn/host_surface.py --training-gpu-column-record) || exit 1
mkdir -p "$PKG/identity_columns/$RECORD"
for col in $(python3 python/mojolearn/host_surface.py --training-gpu-columns); do
    [ -f "$here/$col" ] || { echo "ERROR: the manifest names $col, which is not in this checkout" >&2; exit 1; }
    cp "$here/$col" "$PKG/identity_columns/$RECORD/$(basename "$col")"
done
COMMIT=$(git -C "$here" rev-parse HEAD) || { echo "ERROR: no commit witness; the wheel's identity columns need one" >&2; exit 1; }
printf '%s\n' "$COMMIT" > "$PKG/identity_columns/COMMIT"
echo "identity payload: $(ls "$PKG/identity_columns/$RECORD" | wc -l | tr -d ' ') columns from $RECORD, commit $COMMIT"

cd "$here/python"
rm -rf dist build ./*.egg-info
pixi run -e pkg python -m build --wheel --no-isolation
python3 "$here/tools/wheel_api_audit.py" --require-complete \
    --output "$here/python/dist/API-macos.json" "$here/python/dist/"*.whl

echo "wheel:"
ls -la "$here/python/dist"/*.whl

# THE GATE. Not optional and not a separate step you may forget: a wheel
# that this script produced and that has not passed verify_wheel.sh is a
# wheel that imported on the build machine and nothing else, and that shape
# of artifact has shipped broken twice (TestPyPI 0.1.0a1, 0.1.0a2).
cd "$here"
./packaging/macos/verify_wheel.sh
