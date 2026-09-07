#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OUR side of the `umap` classical speed lane, through the public Python
API, in whatever numeric tier `MOJOLEARN_NUMERIC_MODE` selected at import.

    python3 bench/speed/umap_speed_arm.py --rounds 5 --size large
    MOJOLEARN_NUMERIC_MODE=identical \\
        python3 bench/speed/umap_speed_arm.py --rounds 5 --size wide

DEVIATION 2130. `umap` has no lane in `bench/speed/classical_speed_main.mojo`,
so unlike the other classical lanes our arm is not a compiled driver but
`mojolearn.UMAP` (`python/mojolearn/_umap_impl.py`) on the bindings the leg
built, the same way `bench/speed/seq_py_speed_arm.py` carries Mamba-2/3 and
`bench/speed/forest_speed_arm.py` carries the forests. The cost of that
choice is named: what is timed is what a user of this surface pays,
including the `as_f32_c` validation pass and the host-list marshalling the
binding does on both sides of the call.

THE MODE OF THIS ARM IS NOT CHOSEN HERE. No `numeric_mode=` is passed. The
tier is whatever the library loaded at import, and it is READ BACK from the
binary the estimator actually holds (`UMAP.numeric_mode_used()`, which
returns the tier directory of the loaded extension, never the environment
string) and printed on the header as `mode=FAST|DETERMINISTIC|IDENTICAL`,
exactly the discipline `forest_speed_arm.py` follows (DEVIATION 1896). A
run of this arm is therefore impossible to mislabel.

THE SAME BYTES AS THE VENDOR ARM, BY CONSTRUCTION. The fixture comes from
`tools/speed_cuml_arm.py::fixture("umap", size)` (DEVIATION 2134), the same
function the cuML arm calls, and the accuracy line comes from the same
`trustworthiness_subsample` helper on the same 10,000 rows (DEVIATION 2136).
The round loop is `tools/speed_cuml_arm.py::race`, so warm-up, timing,
synchronization and hashing are one implementation for both sides.

WHAT THE HASH MEANS HERE. `hash=` is FNV-1a64 over the embedding's bytes,
a WITHIN-ARM repeatability probe; ours and cuML's embeddings are two
different optimizers on two different neighbor graphs and are not supposed
to share bits. Under `identical` the hash is expected to repeat across
rounds; nothing here gates it (the gates live in `umap/checks/`).

BOUNDS OF THE SURFACE, READ FROM THE SOURCE ON 2026-09-07. `_umap_impl.py`
requires `n_neighbors >= 2`, `n_components in {2, 3}`, `n_samples >=
2 * n_components + 4`, `n_neighbors <= n_samples`, finite float32 input,
`local_connectivity == 1`, `metric='euclidean'`, `init='spectral'`, and a
`random_state >= 0` (ours cannot be unseeded; the default is 0, and 0 is
what this arm uses because passing anything is choosing). There is no
upper bound on rows or features: `umap/sparse_estimator.mojo` holds the
graph in CSR (O(n * n_neighbors)), and the exact neighbor search
(`neighbors/estimator.mojo::knn_search`) tiles queries against the index
(a `query_tile x n_index` distance tile, tile lowered to fit a capped
workspace). 200,000 x 64 at n_neighbors 15 is inside every bound; a fixture
that were not would be REFUSED here rather than shrunk.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

# `tools/` is not a package and never has been, so the shared arm module is
# imported by path. The repository root is two levels up from this file
# (bench/speed/ -> bench/ -> root), and `python/` goes on the path too so an
# in-repo, not-yet-installed `mojolearn` is importable exactly the way
# `bench/speed/forest_speed_arm.py` arranges it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_cuml_arm as spec           # noqa: E402

LANE = "umap"
ARM = "ours"


def _device_name():
    """The GPU by name. `spec.gpu_device_name()` asks cupy or torch, which
    are the vendor arm's runtimes and may be absent on a box that only has
    our bindings; `nvidia-smi` is the fallback, then the host platform, so
    a header is never printed without a device."""
    name = spec.gpu_device_name()
    if name != "unknown-gpu":
        return name
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
        lines = out.stdout.strip().splitlines()
        if out.returncode == 0 and lines:
            return lines[0].strip().replace(" ", "_")
    except (OSError, subprocess.SubprocessError):
        pass
    import platform
    return (platform.system() + "_" + platform.machine()).replace(" ", "_")


def _mode_label(model):
    """The tier read back from the loaded binary, upper-cased the way every
    FSPEED header spells it. Falls back to the environment's word only if
    the read-back itself fails, and says so."""
    try:
        return model.numeric_mode_used().upper()
    except Exception as e:                                # noqa: BLE001
        spec.note(LANE, ARM, "numeric_mode_used() failed (%r); mode label "
                             "is the ENVIRONMENT's word, not the binary's"
                  % (e,))
        env = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
        return {"identical": "IDENTICAL",
                "deterministic": "DETERMINISTIC"}.get(env, "FAST")


def build_parser():
    p = argparse.ArgumentParser(
        prog="umap_speed_arm",
        description="mojolearn.UMAP, in the numeric mode MOJOLEARN_NUMERIC_"
                    "MODE selected at import (read back from the binary and "
                    "printed on the header), on the same fixture the cuML "
                    "arm consumes")
    p.add_argument("--rounds", type=int,
                   default=int(os.environ.get("MOJOLEARN_SPEED_ROUNDS", "5")),
                   help="timed rounds after the one untimed warm-up")
    p.add_argument("--size", default=os.environ.get("MOJOLEARN_SPEED_SIZE",
                                                    "shipped"),
                   choices=spec.SIZES,
                   help="shipped/large = 200000x64, wide = 100000x512, "
                        "smoke = 5000x16 (DEVIATION 2135)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    size = args.size

    try:
        import mojolearn                                  # noqa: PLC0415
    except Exception as e:                                # noqa: BLE001
        # The EXPECTED failure on a box whose CUDA build of the metrics
        # binding did not land: a refusal, not a traceback, so the leg's
        # log reader sees a lane that ran and said why.
        spec.refuse(LANE, ARM, "import mojolearn failed: %s: %s"
                    % (e.__class__.__name__, " ".join(str(e).split())))
        return 0

    arrays, tag, prm = spec.fixture(LANE, size)
    x = np.ascontiguousarray(arrays["x"], dtype=np.float32)
    n, d = x.shape
    spec.note(LANE, ARM, "size=%s fixture_source=%s shape=%s"
              % (size, prm["_source"], tag))

    # REFUSE, NEVER SHRINK. Every bound the surface enforces is checked here
    # first so the refusal names the bound rather than arriving as a
    # ValueError from inside the timed region.
    k, comp = prm["n_neighbors"], prm["n_components"]
    bounds = [
        (k >= 2, "n_neighbors must be >= 2"),
        (comp in (2, 3), "n_components must be 2 or 3"),
        (n >= 2 * comp + 4, "n_samples must be >= 2*n_components+4"),
        (k <= n, "n_neighbors must be <= n_samples"),
        (prm["metric"] == "euclidean", "metric must be euclidean"),
        (bool(np.isfinite(x).all()), "input must be finite"),
    ]
    for ok, why in bounds:
        if not ok:
            spec.refuse(LANE, ARM, "OUT-OF-BOUNDS for mojolearn.UMAP: %s "
                                   "(fixture %s); not shrunk" % (why, tag))
            return 0

    try:
        model = mojolearn.UMAP(n_neighbors=k, n_components=comp,
                               min_dist=prm["min_dist"],
                               n_epochs=prm["n_epochs"],
                               metric=prm["metric"], init="spectral")
        mode = _mode_label(model)
    except Exception as e:                                # noqa: BLE001
        spec.refuse(LANE, ARM, "mojolearn.UMAP could not be constructed or "
                               "its tier read back: %s: %s"
                    % (e.__class__.__name__, " ".join(str(e).split())))
        return 0
    spec.note(LANE, ARM, "n_neighbors=%d n_components=%d min_dist=%g "
                         "n_epochs=%d metric=%s init=spectral random_state=0 "
                         "(the class default; ours cannot be unseeded)"
              % (k, comp, prm["min_dist"], prm["n_epochs"], prm["metric"]))
    last = {}

    def call():
        # A fresh estimator each round, as the vendor arm constructs one:
        # `fit_transform` on a fitted instance would refit anyway, and the
        # object's retained copies of X and the embedding (`transform`'s
        # model) must not accumulate across rounds. The binding returns a
        # HOST array, so the return is the synchronization (`race` also
        # drains cupy/torch, which are no-ops here).
        m = mojolearn.UMAP(n_neighbors=k, n_components=comp,
                           min_dist=prm["min_dist"], n_epochs=prm["n_epochs"],
                           metric=prm["metric"], init="spectral")
        emb = m.fit_transform(x)
        last["emb"] = emb
        return (emb,)

    try:
        spec.race(LANE, ARM, tag, args.rounds, size, _device_name(), call,
                  mode=mode, hash_max=spec.UMAP_HASH_MAX)
    except Exception as e:                                # noqa: BLE001
        spec.refuse(LANE, ARM, "raised inside the round loop: %s: %s"
                    % (e.__class__.__name__, " ".join(str(e).split())))
        return 0

    if "emb" in last:
        name, value = spec.trustworthiness_subsample(
            x, last["emb"], k, prm["trust_rows"])
        if name is None:
            spec.refuse(LANE, ARM, "FSPEED-ACC not emitted: %s" % value)
        else:
            spec.acc(LANE, ARM, name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
