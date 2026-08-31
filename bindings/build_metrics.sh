#!/bin/sh
# Build the metrics + spectral-clustering CPython extension into
# python/mojolearn/_mojolearn_metrics.so. Run from anywhere; requires pixi.
#
# This script is bindings/build_estimators.sh's structure, on purpose. Every
# non-obvious line in it is there because a build without it shipped a broken
# artifact once, and the reasons are worth reading before changing anything:
#
# 1. IT MUST NOT SET `MACOSX_DEPLOYMENT_TARGET` IN THE ENVIRONMENT. That
#    suppresses ahead-of-time Metal compilation entirely: `mojo build` writes
#    an empty 134-byte metallib per kernel and embeds nothing, so the
#    extension imports cleanly and dies at the first launch with "Failed to
#    create Metal function". Measured 2026-08-21 on a cold cache, one
#    variable, same file and flags: variable set -> 0 AIR blobs, variable
#    unset -> 141. The VALUE is innocent (12.0 fails exactly as 11.0 does);
#    setting it AT ALL is what does it. The floor is passed to the LINKER
#    instead, which stamps LC_BUILD_VERSION exactly as the variable would
#    while the Metal compile step never sees it.
#
#    AND THE COMPILER CACHE IS WHY EVERY EARLIER MEASUREMENT DISAGREED.
#    `$MODULAR_HOME/cache/.mojo_cache` is content-addressed and its key does
#    NOT include the deployment target, so ONE build made with the variable
#    set poisons those keys with empty metallibs and every later build reads
#    them back whatever ITS flags are. Clear the cache before any build whose
#    kernel count you intend to believe, or the number is fiction. See the
#    long write-up at the top of bindings/build.sh.
#
# 2. IT MUST HAVE A GATE. build_estimators.sh once built, printed
#    "built ...", and exited 0 whatever came out, which is how a kernel-less
#    artifact shipped silently for hours. A blob floor is only a pre-filter;
#    the real gate is LAUNCHING a kernel from each lane. Both are below and
#    this script refuses to install an artifact that fails either.
#
# WHAT IS IN THIS EXTENSION, and what each half is worth:
#
#   metrics/   accuracy, rand, adjusted rand, entropy, mutual information,
#              homogeneity, completeness, v-measure, r2, KL divergence,
#              silhouette (score and samples) and trustworthiness. CERTIFIED
#              bit-identical Apple M4 <-> NVIDIA H100 <-> AMD MI325X at leg 11
#              (E3_RESULTS.md round 11, section 7, 34 stages, commit 144aa5b).
#   spectral/  SpectralClustering, both cuVS overloads. HAS RUN ON ONE APPLE
#              M4 AND NOWHERE ELSE (IDENTICAL_SPECTRAL_CONTRACT.md section 10:
#              "no cross-vendor result of any kind"). It is not in
#              tools/e1_bootstrap.sh phase 8 and tools/e3_round_judge.sh
#              section 7 does not name it.
#
# A green build of this script says the kernels COMPILED AND LAUNCHED on this
# machine. It says nothing about either lane's cross-vendor status.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

# Must equal bindings/build.sh's MACOS_FLOOR and setup.py's
# DEFAULT_MACOS_TARGET: these .so files land in ONE wheel under ONE tag, and
# the tag is the lower bound of what is inside it. Disagreement is a
# published lie in one of two directions -- too low and the wheel installs
# where it cannot load, too high and Macs that could have run it refuse it.
MACOS_FLOOR="11.0"

# NOT EXPORTED, DELIBERATELY -- see (1) above. Unset it if the caller had it
# set, because inheriting it from an outer shell reproduces the bug silently.
unset MACOSX_DEPLOYMENT_TARGET

if [ "$(uname)" = "Darwin" ]; then
    MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
else
    MACOS_SDK=""  # linux arm (E1): no Mach-O, no Metal SDK
fi
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-metrics.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_metrics.so"

# `--target-cpu apple-m1` is the oldest Apple silicon: `mojo build` otherwise
# defaults to whatever chip ran the compiler, and a host-built wheel SIGILLs
# on older Macs at the instruction, inside the extension, with no diagnostic a
# user can act on. arm64 Mach-O `cpusubtype` stays ARM64_ALL whatever -mcpu
# was, so such a wheel LOOKS portable to any header-reading check.
#
# `--target-accelerator metal:1` is kept from build_estimators.sh's
# invocation, which is the script this one is modelled on. NOTE THE
# DISAGREEMENT WITH bindings/build.sh, which measured that passing
# --target-accelerator AT ALL gives 0 blobs and therefore passes none; two
# sibling scripts have disagreed about this flag since 2026-08-22 and both
# currently produce working artifacts. If the blob count printed at the end
# of this script is 0, DROP THIS FLAG FIRST.
# THE LINUX x86-64 CPU BASELINE. Without it `mojo build` targets WHATEVER
# CHIP THE BUILD BOX HAS, and on 2026-08-30 that shipped a 0.3.0 wheel whose
# HOST code used AVX-512. It died with SIGILL on an L40 whose host was an AMD
# EPYC 7773X, a Zen 3 part with no AVX-512, inside `cluster::estimator::
# kmeans_fit`, on
#
#     vandps 0x6d7ee(%rip){1to4},%xmm2,%xmm2
#
# where `{1to4}` is an EVEX embedded broadcast. There is no `cpuid` dispatch
# anywhere in these binaries, so it is unconditional. That breaks every AMD
# Zen 1, 2 and 3 host, most Intel consumer parts and every Xeon before
# Skylake-SP, which together are a large share of GPU servers. The GPU was
# never involved; the selector had correctly chosen sm_80 for that sm_89
# device.
#
# macOS has pinned `--target-cpu apple-m1` since 0.1.0 AND gates the result
# with `packaging/isa_baseline.py`. Linux had NEITHER. That is the whole bug.
#
# x86-64-v3 is AVX2, FMA, BMI2 and SSE4.2, so Haswell 2013 and Zen 1 2017
# onward, and it EXCLUDES AVX-512. Anything modern enough to hold a supported
# GPU clears it.
#
# aarch64 Linux keeps the empty flags it always had, because an x86 CPU name
# is meaningless there and its host CPU is not the variable that broke.
TARGET_FLAGS="--target-cpu apple-m1 --target-accelerator metal:1"
if [ "$(uname)" != "Darwin" ]; then
    case "$(uname -m)" in
        x86_64) TARGET_FLAGS="--target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3}" ;;
        *)      TARGET_FLAGS="" ;;   # linux arm: host cpu + its GPU
    esac
fi
# MOJOLEARN_GPU_ARCHS: ONE GPU architecture a LINUX set is compiled for
# (sm_80, gfx942, ...). Until 2026-08-30 ONLY bindings/build.sh read this
# variable, so a leg that set it built ONE binding for the asked-for
# architecture and every other binding for the build box's own device (the
# A40 leg's 27-sm_86 read-back, bench/results/wheels/LEGS_2026-08-30.md).
# The compiler takes EXACTLY ONE name (a comma list is rejected);
# packaging/linux/build_sets.sh reads the architecture back out of every
# binary and refuses a set that disagrees with what was asked.
if [ -n "${MOJOLEARN_GPU_ARCHS:-}" ] && [ "$(uname)" != "Darwin" ]; then
    case "$MOJOLEARN_GPU_ARCHS" in *,*)
        echo "MOJOLEARN_GPU_ARCHS='$MOJOLEARN_GPU_ARCHS': the compiler takes EXACTLY ONE architecture (measured 2026-08-30); run one build per architecture" >&2
        exit 2 ;;
    esac
    TARGET_FLAGS="$TARGET_FLAGS --target-accelerator $MOJOLEARN_GPU_ARCHS"
    echo "!! MOJOLEARN_GPU_ARCHS=$MOJOLEARN_GPU_ARCHS (--target-accelerator); read the architecture back out of the built .so"
fi
# Explicit kernel-matrix column: MOJOLEARN_TARGET_COLUMN=apple|nvidia|amd|amd_rdna
COLUMN_DEFINE=""
# THE NUMERIC MODE IS A BUILD DEFINE, NOT A SOURCE FLIP (2026-08-23).
# MOJOLEARN_NUMERIC_MODE=identical compiles with -D MOJOLEARN_NUMERIC_IDENTICAL=1
# (original/numerics.mojo reads it through is_defined, the same shape as the
# column define) and lands the binary under python/mojolearn/identical/, where
# python/mojolearn/_metrics_impl.py picks it up when the env var
# MOJOLEARN_NUMERIC_MODE=identical is set at import. Default is fast and the
# default location. The build-time smoke gates import the FAST package, so
# they are skipped for an identical build.
MODE_DEFINE=""
OUTDIR="python/mojolearn"
if [ "${MOJOLEARN_NUMERIC_MODE:-fast}" = "identical" ]; then
    MODE_DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    OUTDIR="python/mojolearn/identical"
    mkdir -p "$OUTDIR"
    export MOJOLEARN_SKIP_BUILD_GATE=1
elif [ "${MOJOLEARN_NUMERIC_MODE:-fast}" = "deterministic" ]; then
    # The MIDDLE tier: reproducible run to run on ONE device, with no
    # promise about a second one. It gets its own directory because it
    # is its own binary -- PIN_DETERMINISM is comptime, so a
    # deterministic build is different code from both neighbours, not
    # the identical build with a flag turned down.
    MODE_DEFINE="-D MOJOLEARN_NUMERIC_DETERMINISTIC=1"
    OUTDIR="python/mojolearn/deterministic"
    mkdir -p "$OUTDIR"
    export MOJOLEARN_SKIP_BUILD_GATE=1
elif [ "${MOJOLEARN_NUMERIC_MODE:-fast}" != "fast" ]; then
    echo "MOJOLEARN_NUMERIC_MODE must be fast, deterministic or identical, got '$MOJOLEARN_NUMERIC_MODE'" >&2
    exit 2
fi
if [ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]; then
    COLUMN_DEFINE="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
fi

# Two include paths, both required. `-I .` is the mojolearn package root, so
# `metrics.estimator` and `spectral.estimator` resolve. `-I bindings` is this
# directory, so a capability module placed beside the entry point resolves as
# a top-level import.
# shellcheck disable=SC2086
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_metrics.mojo \
    -o "$out"

# `_gpu_shared_mem` is a prefix the compiler puts on the blob symbol; it is
# not part of the Metal function name.
air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

# The gates below import the WHOLE python package, so during a from-scratch
# multi-binding build (a fresh linux box, E1) they fail on the siblings'
# not-yet-built .so files; and the AIR/otool checks are Mach-O-only. The
# caller that sets this owns end-to-end verification.
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ] || [ "$(uname)" != "Darwin" ]; then
    mv "$out" "$OUTDIR/_mojolearn_metrics.so"
    echo "built $OUTDIR/_mojolearn_metrics.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# ============================================================================
# A FLOOR, NOT A PROOF, AND A DELIBERATELY UNAMBITIOUS ONE.
# ============================================================================
# The failure this guards against is an artifact with ZERO kernels; the
# failure run_smoke guards against is an artifact that imports and dies at the
# first launch.
#
# 10 IS THE SIBLING'S NUMBER, NOT A MEASURED ONE FOR THIS MODULE, and it is
# set that way on purpose. This module was written without ever being built
# (the author of these files does not run builds), so any per-lane threshold
# invented here would be a guess that can FALSE-FAIL a good artifact, which is
# worse than a weak filter. bindings/build_estimators.sh proved 10 safe for a
# module with a comparable set of kernel families, and this one carries
# strictly more of them (metrics groups A-D, spectral's Laplacian/SpMV/Lanczos,
# plus the imported cluster k-means, core row-norm, gemm and neighbors kernels
# both lanes call).
#
# THE FOLLOW-UP THE OPERATOR OWNS: this script prints the real count and the
# per-prefix breakdown below. Read them once on a cold cache and raise this
# floor to a measured number, and add per-prefix floors in the shape of
# bindings/build.sh's `for _pair in cluster:15 neighbors:3 core:1` loop. Note
# WHY that has to be measured rather than guessed: build.sh records a
# known-broken artifact that kept exactly 1 of 85 `gbdt_` blobs and PASSED a
# presence-of-one check, so a per-prefix floor of 1 would be theatre.
AIR_FLOOR=10

count=$(air_blobs "$out" | wc -l | tr -d ' ')
if [ "$count" -lt "$AIR_FLOOR" ]; then
    printf 'FAILED: %s AIR blobs, want at least %s.\n' "$count" "$AIR_FLOOR" >&2
    printf 'If this is 0, check MACOSX_DEPLOYMENT_TARGET in the environment,\n' >&2
    printf 'then drop --target-accelerator from TARGET_FLAGS (bindings/build.sh\n' >&2
    printf 'measured that flag at 0 blobs), and then look in\n' >&2
    printf '$MODULAR_HOME/cache/.mojo_cache for empty 134-byte metallibs --\n' >&2
    printf 'one poisoned build serves them to every later one. Find them with\n' >&2
    printf '  find "$MODULAR_HOME/cache/.mojo_cache" -type f -size -200c \\\n' >&2
    printf '    -exec sh -c '"'"'head -c4 "$1" | grep -q MTLB && echo "$1"'"'"' _ {} \\;\n' >&2
    exit 1
fi

echo "  AIR blobs by prefix (READ THIS ONCE AND RAISE AIR_FLOOR TO A MEASURED NUMBER):"
air_blobs "$out" | sed -e 's/_[0-9a-f]\{16\}air$//' -e 's/_.*$//' \
    | sort | uniq -c | sed 's/^/    /'

# THE MACH-O FLOOR IS READ BACK, NOT ASSUMED. The linker flag is the only
# thing setting it, and a silently dropped `-Xlinker` would publish a wheel
# whose tag and binary disagree.
got=$(otool -l "$out" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
if [ "$got" != "$MACOS_FLOOR" ]; then
    printf 'FAILED: minos is %s, want %s.\n' "$got" "$MACOS_FLOOR" >&2
    exit 1
fi

# ============================================================================
# THE REAL GATE: import and LAUNCH from BOTH LANES IN THIS EXTENSION.
# ============================================================================
# Every broken build in this bug's history imported fine. These calls are
# THROUGH THE PYTHON WRAPPERS, not the raw bindings, because the extension's
# entry points take bare addresses plus a packed `params` list and a
# hand-rolled call here would encode that ABI a second time and drift from it.
#
# `_metrics_impl` and `_spectral_impl` are imported as submodules rather than
# through the package's export list, exactly as build_estimators.sh does, so
# this gate does not depend on `mojolearn/__init__.py` having been wired yet.
# WHEN THE OPERATOR WIRES `mojolearn.metrics` AND `mojolearn.cluster.
# SpectralClustering`, SWITCH THESE IMPORTS TO THE PUBLIC NAMES.
#
# Kept small -- 96 rows -- because this runs on every build and the GPU is
# shared. The spectral fit is the expensive one (a thick-restart Lanczos with
# max_iterations = 10n).
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_metrics.so"))
sys.path.insert(0, tmp)
import numpy as np

from mojolearn import _metrics_impl as M
from mojolearn import _spectral_impl as S

rng = np.random.default_rng(0)
n = 96
X = rng.random((n, 3), dtype=np.float32)
y = np.ascontiguousarray(X[:, 0])
yhat = y + np.float32(0.01) * rng.random(n, dtype=np.float32)
lt = (np.arange(n) * 7919 % 4).astype(np.int32)
lp = (np.arange(n) * 104729 % 4).astype(np.int32)

# ONE LAUNCH FROM EACH KERNEL GROUP IN THIS EXTENSION. Against a zero-kernel
# artifact every one of these raises "Failed to create Metal function".
M.accuracy_score(lt, lp)                       # group A, integer atomics
M.adjusted_rand_score(lt, lp)                  # group A, ARI's own matrix
M.mutual_info_score(lt, lp)                    # group A, contingency + host
M.r2_score(y, yhat)                            # group B, the pinned sum tree
M.kl_divergence(np.abs(y) + 1e-3, np.abs(yhat) + 1e-3)   # group B
M.silhouette_score(X, lt)                      # group C, the batched path
M.silhouette_samples(X, lt)                    # group C, per-sample
M.trustworthiness(X, X[:, :2], n_neighbors=5)  # group D, ranks + knn_search
print("  smoke: metrics groups A, B, C and D each launched")

# spectral: the kNN graph, the Laplacian, the Lanczos and the k-means, on the
# dataset path. A FAILURE HERE IS A REAL FINDING, not flakiness -- a restart
# breakdown or a disconnected graph is refused BY NAME by the ported code, so
# a raise names what went wrong.
sc = S.SpectralClustering(n_clusters=3, n_neighbors=10, random_state=0).fit(X)
assert sc.labels_.shape == (n,), sc.labels_.shape
print("  smoke: spectral clustering launched (%d columns of embedding)"
      % sc.embedding_.shape[1])
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_metrics.so"
echo "built $OUTDIR/_mojolearn_metrics.so ($count AIR blobs, minos $MACOS_FLOOR)"
