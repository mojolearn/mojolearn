# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the six training primitives routed through
`mojolearn.training` (workstream D, 2026-09-14): `embedding_forward`,
`embedding_backward`, `rms_norm_forward`, `rms_norm_backward`,
`linear_forward`, `linear_backward`. They were exported by the shipped
training binding and implemented in `_training_impl.py` with no public
name; this file gates the ROUTE and the WIRING (the six names, their
argument checks, their shapes, and answers that are EXACT in float32 so a
wrong slot or a wrong count shows as a wrong number). The RMS-norm and linear
values against their exact and Float64 references moved to the native
check checks/native_tests/training_primitives_{cpu,gpu}.mojo (lane
tmig-training, 2026-11-14); this file keeps the 3-D shape spelling. The
arithmetic behind the embedding fold is gated by the embedding lane's own check and
the GEMM by the gemm lane's.

    cd python && python3 -m mojolearn.tests.test_training_primitives_surface

Exit 2 naming `bindings/build_training.sh` when unbuilt.
"""
import sys

import numpy as np

import mojolearn
from mojolearn import training as T
from mojolearn import _training_impl
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run

SIX = ("embedding_forward", "embedding_backward", "rms_norm_forward", "rms_norm_backward",
       "linear_forward", "linear_backward")


def _ints(shape, seed, lo=-4, hi=5):
    """Small integers as float32: every product and sum below is exact."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(lo, hi, size=shape).astype(np.float32))


def _bits_same(a, b):
    a, b = np.ascontiguousarray(np.asarray(a)), np.ascontiguousarray(np.asarray(b))
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def arm_route(rep):
    for name in SIX:
        rep.check("ROUTE", name in T.__all__, "%s in mojolearn.training.__all__" % name)
        rep.check("ROUTE", getattr(T, name) is getattr(_training_impl, name), "mojolearn.training.%s IS the _training_impl function" % name)


def arm_embedding(rep):
    w = _ints((11, 6), 1)
    ids = np.ascontiguousarray(np.array([3, 0, 3, 10, 7, 3, 0, 5], dtype=np.int32))
    y = np.asarray(T.embedding_forward(w, ids))
    rep.check("EMB", y.shape == (8, 6) and y.dtype == np.float32, "embedding_forward is (N, D) float32", (y.shape, y.dtype))
    rep.check("EMB", _bits_same(y, w[ids]), "embedding_forward is the gather, bit for bit")
    dy = _ints((8, 6), 2)
    dw = np.asarray(T.embedding_backward(dy, ids, 11))
    want = np.zeros((11, 6), np.float32)
    for i, t in enumerate(ids):
        want[t] += dy[i]
    rep.check("EMB", dw.shape == (11, 6), "embedding_backward is (V, D)", dw.shape)
    rep.check("EMB", _bits_same(dw, want), "embedding_backward is the scatter-add of small integers, exact, rows 3 and 0 folded over three and two positions")
    rep.check("EMB", np.all(dw[[1, 2, 4, 6, 8, 9]] == 0.0), "untouched vocabulary rows are zero")


def arm_rms_norm_shape(rep):
    x = _ints((5, 8), 3, 1, 4)
    w = np.ascontiguousarray(np.full((8,), 2.0, np.float32))
    y = np.asarray(T.rms_norm_forward(x, w, 0.0))
    rep.check("RMS", y.shape == (5, 8) and y.dtype == np.float32, "rms_norm_forward is (M, D) float32", (y.shape, y.dtype))
    y3 = np.asarray(T.rms_norm_forward(x.reshape(1, 5, 8), w, 0.0))
    rep.check("RMS", y3.shape == (1, 5, 8) and _bits_same(y3.reshape(5, 8), y), "a 3-D activation keeps its shape and its bits")


def arm_refusals(rep):
    w = _ints((4, 3), 8)
    ids = np.array([1, 2], np.int32)
    rep.raises("REFUSE", TypeError, "float32", "float64 weight", T.embedding_forward, w.astype(np.float64), ids)
    rep.raises("REFUSE", ValueError, "(V, D)", "a 1-D weight", T.embedding_forward, w[0], ids)
    rep.raises("REFUSE", ValueError, "row counts", "ids and dy of different lengths", T.embedding_backward, _ints((3, 3), 9), ids, 4)
    rep.raises("REFUSE", ValueError, "(N, D)", "a 1-D dy", T.embedding_backward, _ints((3,), 9), ids, 4)
    rep.raises("REFUSE", ValueError, "(D,)", "an rms weight of the wrong length", T.rms_norm_forward, _ints((2, 3), 1), _ints((4,), 1), 0.0)
    rep.raises("REFUSE", ValueError, "shapes differ", "rms backward with dy of another shape", T.rms_norm_backward, _ints((2, 4), 1), _ints((2, 3), 1), _ints((3,), 1), 0.0)
    rep.raises("REFUSE", ValueError, "K=", "linear_forward with mismatched K", T.linear_forward, _ints((2, 3), 1), _ints((4, 5), 1))
    rep.raises("REFUSE", ValueError, "(M, N)", "linear_backward with dc of the wrong shape", T.linear_backward, _ints((2, 2), 1), _ints((2, 3), 1), _ints((4, 3), 1))


def arm_provenance(rep):
    rep.check("PROVENANCE", T.numeric_mode_used() == mode(), "training.numeric_mode_used() is the process default")
    rep.check("PROVENANCE", "training" in mojolearn.__all__, "mojolearn.training exported")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_training", "build_training.sh")
    rep = Report("test_training_primitives_surface")
    return run("test_training_primitives_surface", [("ROUTE", arm_route), ("EMB", arm_embedding), ("RMS", arm_rms_norm_shape),
                                                    ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
