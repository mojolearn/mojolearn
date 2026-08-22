#!/bin/sh
# Build the CPython extension into python/mojolearn/_mojolearn.so.
# Run from anywhere; requires pixi.
#
# THIS EXTENSION IS k-MEANS AND k-NN. GBDT moved to its own extension in
# `bindings/_mojolearn_gbdt.mojo` / `bindings/build_gbdt.sh`; DBSCAN, PCA,
# tSVD and OLS live in `bindings/_mojolearn_estimators.mojo`. Each is built
# separately so an independently changing binding does not become a merge
# point, and all three land in one wheel.
#
# Two include paths, both required. `-I .` is the mojolearn package root, so
# `cluster.estimator` and `neighbors.estimator` resolve. `-I bindings` is this
# directory: capability modules placed beside the entry point resolve as
# top-level imports only when this directory is on the path.
#
# packaging/macos/build_release_wheel.sh runs this script rather than
# repeating the command, so the flags live in exactly one place.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

# ============================================================================
# MACOSX_DEPLOYMENT_TARGET IN THE ENVIRONMENT SUPPRESSES AHEAD-OF-TIME METAL
# COMPILATION ENTIRELY. THAT -- NOT THE ENTRY FILE'S BASENAME -- IS THE BUG.
# ============================================================================
#
# **THIS FILE PREVIOUSLY ARGUED, AT LENGTH AND WITH NUMBERS, THAT THE COUNT OF
# COMPILED METAL FUNCTIONS DEPENDED ON THE BASENAME OF THE ENTRY FILE.** It
# compiled a copy under a measured stem, verified the artifact, and retried
# over a list of eleven alternates. That explanation is WRONG and the whole
# apparatus is gone; PORTING.md 70 has been corrected to match.
#
# Measured 2026-08-21 on `bindings/_mojolearn_gbdt.mojo`, one variable at a
# time, WITH THE COMPILER CACHE CLEARED BEFORE EACH BUILD -- the step every
# earlier experiment omitted, and the reason they all read wrong:
#
#   MACOSX_DEPLOYMENT_TARGET   --target-cpu    AIR blobs
#   11.0                       apple-m1            0
#   12.0                       apple-m1            0
#   (unset)                    apple-m1          141
#   11.0                       (host)              0
#   (unset)                    (host)            141
#
# `--target-cpu apple-m1` is innocent. The VALUE of the deployment target is
# innocent -- 12.0 fails exactly as 11.0 does. SETTING THE VARIABLE AT ALL is
# what does it: with it set, `mojo build` emits an EMPTY 134-byte metallib for
# every kernel and embeds nothing, and the extension then imports cleanly and
# dies at the first launch with "Failed to create Metal function".
#
# Four different basenames built cold with the variable set all give 0; the
# same four built cold with it unset all give 141. So this script compiles the
# entry file UNDER ITS OWN NAME, IN PLACE, and there is no stem loop.
#
# WHY EVERY EARLIER MEASUREMENT DISAGREED: THE CACHE.
#
# `$MODULAR_HOME/cache/.mojo_cache` is content-addressed and ITS KEY DOES NOT
# INCLUDE THE DEPLOYMENT TARGET. An empty metallib produced by a build with
# the variable set is therefore served to every later build that hashes to the
# same key, whatever ITS flags are. When this was found that cache held 40,772
# files, of which 20,682 were 134-byte empty metallibs.
#
# That one fact explains the entire history. A "good" basename was one whose
# kernels happened to hash to keys still holding REAL metallibs from an older
# configuration; a "bad" one hashed to poisoned keys. The famous decline --
# 113 blobs, then 101, then 86, then 84 across four points in the history --
# was not the module outgrowing a budget. It was CACHE ATTRITION: each source
# change invalidated more of the surviving real entries, each rebuild replaced
# them with empties, and nothing could refill them because every build set the
# variable. Those counts were archaeology, not compilation.
#
# **Clear the cache before any build whose kernel count you intend to
# believe**, or the number is fiction. With a warm cache, `12.0` measured 141
# and looked like the fix; it was reading back the `(unset)` build from one
# minute earlier.
#
# ----------------------------------------------------------------------------
# THE macOS FLOOR IS SET AT THE LINKER INSTEAD, WHICH KEEPS BOTH PROPERTIES
# ----------------------------------------------------------------------------
#
# The deployment target still has to be low. `mojo build` takes it from the
# host SDK, which here means `minos 26.0` -- a wheel installable only on macOS
# 26, which is very nearly nobody. The MAX runtime dylibs this links are built
# at `minos 11.0`; only our own compile step was narrow. 11.0 (Big Sur) is the
# FIRST macOS that runs on Apple silicon at all, so it is the widest floor
# that means anything, and it matches the dylibs.
#
# `ld -platform_version macos 11.0 <sdk>` stamps LC_BUILD_VERSION exactly as
# the environment variable would, and the Metal compile step never sees it.
# Measured on a cold cache: every kernel AND `minos 11.0`, together.
#
# setup.py's DEFAULT_MACOS_TARGET must equal MACOS_FLOOR or the wheel TAG and
# the BINARY disagree, which is a published lie in one of two directions: too
# low and it installs where it cannot load, too high and it is refused by Macs
# that could have run it. THIS SCRIPT AND bindings/build_gbdt.sh MUST AGREE
# WITH EACH OTHER TOO -- the two .so files land in one wheel under one tag,
# and the tag is the lower bound of what is inside it.
MACOS_FLOOR="11.0"

# NOT EXPORTED, DELIBERATELY -- see above. Unset it if the caller had it set,
# because inheriting it from an outer shell silently reproduces the bug.
unset MACOSX_DEPLOYMENT_TARGET

if [ "$(uname)" = "Darwin" ]; then
    MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
else
    MACOS_SDK=""  # linux arm (E1, 2026-08-22): no Mach-O, no Metal SDK
fi
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""

# THE CPU TARGET IS PINNED TO A PORTABLE BASELINE, NOT LEFT AT THE HOST.
#
# `mojo build` defaults --target-cpu and --target-features to WHATEVER CHIP
# RAN THE COMPILER. On this development machine that is `apple-m4` with
# +bf16, +i8mm, +sme, +sme-f64f64, +sme-i16i64, +sme2.
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
# THERE IS NO --target-accelerator FLAG HERE AND THERE MUST NOT BE. Measured
# 2026-08-21 with --target-cpu apple-m1 held fixed: no flag gives every
# kernel, `--target-accelerator metal:1` gives 0 and `--target-accelerator
# apple-m1` gives 0. `metal:1` is not a target the compiler knows -- `mojo
# build --print-supported-accelerators` lists `apple-m1` .. `apple-m5-metal4`
# -- but that is the smaller half: PASSING THE FLAG AT ALL, right value or
# wrong, suppresses ahead-of-time Metal compilation.
TARGET_FLAGS="--target-cpu apple-m1"
[ "$(uname)" = "Darwin" ] || TARGET_FLAGS=""
# Explicit kernel-matrix column: MOJOLEARN_TARGET_COLUMN=apple|nvidia|amd|amd_rdna
# becomes a -D define that overrides TARGET_COLUMN autodetection.
COLUMN_DEFINE=""
if [ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]; then
    COLUMN_DEFINE="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
fi

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn. The name of the file must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function", which is the least helpful error in the toolchain.

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn.so"

# shellcheck disable=SC2086  # both flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn.mojo \
    -o "$out"

# COUNTING BLOBS IS A PRE-FILTER, NOT THE GATE.
#
# The first version of this function asked only whether each subsystem had at
# least one AIR blob, and a known-broken artifact PASSED IT: the build that
# lost GBDT kept exactly 1 of 85 `gbdt_` blobs, so presence-of-one was
# satisfied by the artifact that shipped broken for weeks. The floor has to
# sit well above 1 -- and even then it is only a way to skip smoke-testing a
# hopeless build. THE GATE IS run_smoke BELOW, because the failure being
# guarded against is precisely that a .so imports cleanly and dies at the
# first launch.
#
# `gbdt` is no longer in this list because it is no longer in this module.
# `bindings/build_gbdt.sh` carries its own floor, measured on its own module.
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
    [ "$(uname)" = "Darwin" ] || return 0  # linux gate is run_smoke
    _air=$(air_blobs "$1")
    # subsystem:floor -- floors are a filter, never the proof. A good build
    # measured 22 cluster, 4 neighbors and 2 core; a suppressed one has 0 of
    # everything.
    for _pair in cluster:15 neighbors:3 core:1; do
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

# THE MACH-O FLOOR IS READ BACK, NOT ASSUMED. The linker flag is the only
# thing setting it now, and a silently dropped `-Xlinker` would publish a
# wheel whose tag and binary disagree -- exactly the failure the flag exists
# to prevent.
minos_matches() {
    [ "$(uname)" = "Darwin" ] || return 0  # linux gate is run_smoke
    _got=$(otool -l "$1" \
        | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
    if [ "$_got" != "$MACOS_FLOOR" ]; then
        printf '  minos is %s, want %s\n' "$_got" "$MACOS_FLOOR" >&2
        return 1
    fi
    return 0
}

# THE REAL GATE: import the extension and launch one kernel from each
# estimator it carries. Every broken build in this bug's history imported fine
# and raised "Failed to create Metal function" at the first launch, so nothing
# short of launching proves anything. Kept tiny on purpose -- 512x3 -- because
# this runs on every build and the GPU is shared.
#
# THERE ARE NO GBDT ROWS HERE ANY MORE. They moved with the code, to
# `bindings/build_gbdt.sh`, which fits one model per loss family. A smoke test
# for kernels that are not in the artifact would be a gate that cannot fail.
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
import mojolearn.cluster as clu
import mojolearn.neighbors as nbr
X = np.random.default_rng(0).random((512, 3), dtype=np.float32)
clu.KMeans(n_clusters=4, random_state=0).fit(X)
nbr.NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X[:2])
shutil.rmtree(tmp, ignore_errors=True)
PY
}

if ! kernels_plausible "$out" || ! minos_matches "$out" || ! run_smoke "$out"; then
    printf '%s\n' \
      "" \
      "FAILED: the extension did not come out complete." \
      "" \
      "THE FIRST THING TO SUSPECT IS THE COMPILER CACHE, not this source." \
      "\$MODULAR_HOME/cache/.mojo_cache is content-addressed and its key does" \
      "NOT include the macOS deployment target, so ONE build made with" \
      "MACOSX_DEPLOYMENT_TARGET set poisons those keys with empty 134-byte" \
      "metallibs and every later build reads them back. Find them with" \
      "" \
      "  find \"\$MODULAR_HOME/cache/.mojo_cache\" -type f -size -200c \\\\" \
      "    -exec sh -c 'head -c4 \"\$1\" | grep -q MTLB && echo \"\$1\"' _ {} \\;" \
      "" \
      "and if there are any, move the whole cache aside and build again." \
      "The cache is regenerable; a poisoned one is not detectable any other" \
      "way, because the build succeeds and the artifact merely does not work." \
      "" \
      "See the comment at the top of this script and PORTING.md 70." >&2
    exit 1
fi

count=$(air_blobs "$out" | wc -l | tr -d ' ')
mv "$out" python/mojolearn/_mojolearn.so
echo "built python/mojolearn/_mojolearn.so ($count AIR blobs, minos $MACOS_FLOOR)"
