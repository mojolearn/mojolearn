#!/bin/sh
# Build the CPython extension into python/mojolearn/_mojolearn.so.
# Run from anywhere; requires pixi.
#
# Two include paths, both required. `-I .` is the mojolearn package root, so
# `cluster.estimator` and `neighbors.estimator` resolve. `-I bindings` is this
# directory: `_mojolearn.mojo` is the entry point, and if this project grows
# capability modules beside it the way mojotrees' bindings did, they resolve
# as top-level imports only when this directory is on the path. It costs
# nothing now and its absence would fail confusingly later.
#
# packaging/macos/build_release_wheel.sh runs this script rather than
# repeating the command, so the flags live in exactly one place.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

# THE TARGET IS PINNED TO A PORTABLE BASELINE, NOT LEFT AT THE HOST.
#
# `mojo build` defaults --target-cpu and --target-features to WHATEVER CHIP
# RAN THE COMPILER. On this development machine that is:
#
#   --target-cpu apple-m4
#   --target-features ...,+bf16,+i8mm,+sme,+sme-f64f64,+sme-i16i64,+sme2
#
# +sme and +sme2 do not exist on M1, M2 or M3; +bf16 and +i8mm do not exist on
# M1. LLVM does not need to be ASKED to use an enabled feature -- it emits
# bfdot, smmla and bfmmla from ordinary loops once the bit is set -- so a
# host-built wheel SIGILLs on older Apple silicon, at the instruction, inside
# the extension, with no diagnostic a user can act on.
#
# Nothing downstream catches it either. arm64 Mach-O `cpusubtype` stays
# ARM64_ALL whatever -mcpu was, so a native-built wheel LOOKS portable to any
# header-reading check, and verifying it on the machine that built it proves
# nothing at all.
#
# apple-m1 is the oldest Apple silicon: measured 2026-08-20, a probe built at
# this baseline runs correctly on the M4, so the baseline is forward-compatible
# in the direction that matters.
#
# THE COST: a baseline artifact may be slower than a native one. mojotrees
# measured that difference by disassembly at 719,057 native instructions
# against 718,585 at apple-m1, of which 483 were `nop`. It has NOT been
# re-measured for mojolearn's kernels, and it should be before any published
# timing is attributed to this wheel rather than to a local build.
#
# THERE IS NO --target-accelerator FLAG HERE AND THERE MUST NOT BE.
#
# This script carried `--target-accelerator metal:1` from the first commit.
# `metal:1` is not a target the compiler knows -- `mojo build
# --print-supported-accelerators` lists `apple-m1` .. `apple-m5-metal4` and no
# `metal:1` -- but that is the smaller half of the finding. Measured
# 2026-08-21, on this exact source, with `--target-cpu apple-m1` held fixed:
#
#   (no --target-accelerator)          113 AIR blobs   every kernel loads
#   --target-accelerator metal:1         0 AIR blobs   nothing loads
#   --target-accelerator apple-m1        0 AIR blobs   nothing loads
#
# An AIR blob is a compiled Metal function embedded in the binary; the count
# is `strings -a <so> | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' | sort -u |
# wc -l`, which is what verify_kernels below does. PASSING THE FLAG AT ALL,
# right value or wrong, suppresses ahead-of-time Metal compilation entirely
# and the extension dies at the first launch with
#
#   Failed to create Metal function: <mangled kernel name>
#
# With the flag gone, `--target-cpu apple-m1` alone still produces `minos 11.0`
# and the portable CPU baseline this comment is about, so nothing is lost.
TARGET_FLAGS="--target-cpu apple-m1"

# THE macOS FLOOR, AND IT DEFAULTED ABSURDLY HIGH.
#
# `mojo build` takes its deployment target from the host SDK. On this machine
# that produced an extension with `minos 26.0` -- a wheel installable only on
# macOS 26, released weeks ago, which is very nearly nobody.
#
# The MAX runtime dylibs this links are built at `minos 11.0`. Modular shipped
# them wide; only our own compile step was narrow. Measured 2026-08-20:
# MACOSX_DEPLOYMENT_TARGET of 11.0, 13.0 and 14.0 each produce exactly that
# minos, so the floor is ours to choose.
#
# 11.0 (Big Sur) is chosen because it is the FIRST macOS that runs on Apple
# silicon at all. There is no Apple silicon Mac that can run anything older,
# so this is the widest floor that means anything, and it matches the dylibs.
#
# setup.py's DEFAULT_MACOS_TARGET must equal this or the wheel TAG and the
# BINARY disagree, which is a published lie in one of two directions: too low
# and it installs where it cannot load, too high and it is refused by Macs
# that could have run it. packaging/macos/build_release_wheel.sh checks that
# they agree by reading the Mach-O header after the build.
MACOS_FLOOR="11.0"
export MACOSX_DEPLOYMENT_TARGET="$MACOS_FLOOR"

# THE ENTRY FILE IS COMPILED UNDER A DIFFERENT BASENAME, AND THAT IS NOT
# COSMETIC.
#
# Mojo 1.0.0 (ed45d567) decides how many kernels to compile ahead of time from
# THE BASENAME OF THE FILE IT IS HANDED. Measured 2026-08-21: three
# byte-identical copies of bindings/_mojolearn.mojo (md5
# ebdff6092a117fbd7e836bead0883f12) in one directory, built with one command
# that differed only in which of them it named:
#
#   copyml2.mojo        113 AIR blobs   everything loads
#   _mojolearn.mojo      29 AIR blobs   GBDT dies, k-means and k-NN load
#   mojolearn_ext.mojo    0 AIR blobs   nothing loads
#
# Same directory, same flags, same include paths, same second, same bytes.
# `-j 1` does not change it, a different `-o` name does not change it, an
# empty MODULAR_HOME cache does not change it, and editing the source does not
# change it: the outcome tracks the basename and nothing else. `_mojolear`
# gets 84 where `_mojolearn` gets 67 on a smaller variant, so it is not a rule
# about leading underscores or about any readable property of the name.
#
# This is a toolchain defect, it is upstream, and it has no fix here. What it
# has is a workaround: hand the compiler a name that lands well, and CHECK the
# artifact afterwards rather than trusting that it did. The name below is
# measured, not chosen -- and because the lottery may move when the source
# changes, ALT_STEMS are tried in turn and verify_kernels decides.
#
# The output file keeps its real name. CPython only requires that the .so be
# called _mojolearn.so and export PyInit__mojolearn; the SOURCE it was compiled
# from is invisible to the import machinery, and both of those are unchanged.
PRIMARY_STEM="copyml2"
ALT_STEMS="mlext_a mlext_b mlext_c mlext_d"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn. The name of the file must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function", which is the least helpful error in the toolchain.

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM

# COUNTING BLOBS IS A PRE-FILTER, NOT THE GATE.
#
# The first version of this function asked only whether each subsystem had at
# least one AIR blob, and the 29-blob build PASSED IT: the stem that loses
# GBDT keeps exactly one `gbdt_` blob out of 85, so presence-of-one is
# satisfied by the artifact that shipped broken for weeks. Measured
# 2026-08-21 against the three reproducer builds, per subsystem:
#
#             gbdt  cluster  neighbors  core
#   good        85       22          4     2
#   29-blob      1       22          4     2
#   0-blob       0        0          0     0
#
# So the cheap filter needs a floor that sits between 1 and 85 (10 below), and
# even then it is only a way to skip smoke-testing a hopeless build. THE GATE
# IS run_smoke BELOW, because the failure being guarded against is precisely
# that a .so imports cleanly and dies at the first launch.
#
# `_gpu_shared_mem` is a prefix the compiler puts on the blob symbol; it is not
# part of the Metal function name.
air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

kernels_plausible() {
    _air=$(air_blobs "$1")
    # subsystem:floor -- floors are a filter, never the proof
    for _pair in gbdt:10 cluster:5 neighbors:1 core:1; do
        _sub=${_pair%%:*}
        _min=${_pair#*:}
        _n=$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)
        if [ "$_n" -lt "$_min" ]; then
            printf '  %s: %s AIR blobs, want at least %s\n' \
                "$_sub" "$_n" "$_min" >&2
            return 1
        fi
    done
    return 0
}

# THE REAL GATE: import the extension and launch one kernel from each of the
# three estimators. Every broken build in this bug's history imported fine and
# raised "Failed to create Metal function" at the first launch, so nothing
# short of launching proves anything. Kept tiny on purpose -- 512x3, two
# trees -- because this runs on every build and the GPU is shared.
run_smoke() {
    MOJOLEARN_SMOKE_SO="$1" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn.so"))
sys.path.insert(0, tmp)
import numpy as np
import mojolearn.ensemble as ens, mojolearn.cluster as clu
import mojolearn.neighbors as nbr
X = np.random.default_rng(0).random((512, 3), dtype=np.float32)
m = ens.GradientBoosting(loss="RMSE", n_estimators=2, max_depth=3,
                         border_count=16).fit(X, X[:, 0])
m.predict(X)

# ONE LOSS PER KERNEL FAMILY, because a family the smoke test never fits
# is a family whose kernels can be missing from the artifact while this
# gate says PASS. That is not hypothetical: for one build this test fit
# RMSE only, `MultiClassOneVsAll`'s kernels were absent, every floor and
# the smoke test passed, and the failure surfaced in a user's fit.
#
#   RMSE                pointwise_target_kernel
#   Logloss             cross_entropy_kernel
#   MAE                 the EXACT estimator -- segmented sort, scan,
#                       need-weights, binary search
#   MultiClass          multilogit_val_and_first_der / second_der_row
#   MultiClassOneVsAll  one_vs_all_val_and_first_der / second_der
#
# Add a row here whenever a kernel family is added. Two trees each; the
# point is to LAUNCH them, not to fit anything.
yb = (X[:, 0] > 0.5).astype(np.float32)
ens.GradientBoosting(loss="Logloss", n_estimators=2, max_depth=3,
                     border_count=16).fit(X, yb).predict(X)
ens.GradientBoosting(loss="MAE", n_estimators=2, max_depth=3,
                     border_count=16).fit(X, X[:, 0]).predict(X)
yc = (X[:, 0] * 3).astype(np.int32).clip(0, 2).astype(np.float32)
for mc in ("MultiClass", "MultiClassOneVsAll"):
    mm = ens.GradientBoosting(loss=mc, n_estimators=2, max_depth=3,
                              border_count=16).fit(X, yc)
    mm.predict(X)
    mm.predict_proba(X)

clu.KMeans(n_clusters=4, random_state=0).fit(X)
nbr.NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X[:2])
shutil.rmtree(tmp, ignore_errors=True)
PY
}

built=""
count=""
for stem in $PRIMARY_STEM $ALT_STEMS; do
    src="$tmpdir/$stem.mojo"
    cp bindings/_mojolearn.mojo "$src"
    out="$tmpdir/$stem.so"

    # shellcheck disable=SC2086  # TARGET_FLAGS is deliberately word-split
    pixi run mojo build --emit shared-lib \
        $TARGET_FLAGS \
        -I . -I bindings \
        "$src" \
        -o "$out"

    if kernels_plausible "$out" && run_smoke "$out"; then
        count=$(air_blobs "$out" | wc -l | tr -d ' ')
        mv "$out" python/mojolearn/_mojolearn.so
        built="$stem"
        break
    fi
    printf 'kernel emission incomplete for stem "%s"; retrying\n' "$stem" >&2
done

if [ -z "$built" ]; then
    printf '%s\n' \
      "FAILED: no candidate basename produced a working extension." \
      "Every attempt either dropped a subsystem's Metal functions or died" \
      "at the first kernel launch with 'Failed to create Metal function'." \
      "Add another stem to ALT_STEMS and re-run; see the basename comment" \
      "above for why this is a lottery rather than a bug in the source." >&2
    exit 1
fi

echo "built python/mojolearn/_mojolearn.so ($count AIR blobs, stem $built)"
