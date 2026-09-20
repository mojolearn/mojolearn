# SPDX-License-Identifier: Apache-2.0
"""Saved-model CPU inference for KMeans (lane/kmeans-save, 2026-09-16).

THE GAP THIS CLOSES. `KMeans.predict` and `KMeans.transform` shipped, and
`_mojolearn_core_host` exports both, but the class had no `save`, so the
GPU-train / save / CPU-infer route stopped at serialization and
`host_surface.SAVED_MODEL_INFERENCE_OWED` said so.

The format and refusal tests need no binding. The round-trip tests fit
through the bindings this process routes (a GPU set, or the reference host
set under `reference_training()` on a CPU-only install), save, load through
`mojolearn.host_model` and require the host answer to be the same bytes as
the fitted model's; they skip when the core host binding is not built.
Cross-vendor identity is tools/classical_host_gate.py's and
tools/identity_break.py's, not this file's.

A CHECK THAT CANNOT FAIL IS NOT A CHECK. `test_one_ulp_in_the_file_moves_the
_answer` rewrites one centroid element of a saved file by one ULP and
requires the same comparison to FIRE, naming the values that differ.
"""
import os
import tempfile
import zipfile

import numpy as np
import pytest

import mojolearn
from mojolearn import _backend, _classical_host, _serialize
from mojolearn._cpu_reference import reference_training
from mojolearn.cluster import _KMEANS_FORMAT


def _rows(n=512, d=6, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X


def _host_built():
    return os.path.exists(_backend.host_module_path("_mojolearn_core_host"))


def _probe(e, Xh):
    return (np.asarray(e.predict(Xh)), np.asarray(e.transform(Xh)))


def _bytes(outputs):
    return [(a.dtype.str, a.shape, a.tobytes()) for a in outputs]


# ------------------------------------------------------------------ format

def test_kmeans_format_is_a_host_model_format():
    assert _KMEANS_FORMAT == "mojolearn-kmeans-1"
    assert _KMEANS_FORMAT in _classical_host.CLASSICAL_FORMATS
    assert _classical_host._FORMATS[_KMEANS_FORMAT] == {"KMeans": _classical_host.HostKMeans}


def test_unfitted_kmeans_refuses_save():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="not fitted"):
            mojolearn.KMeans().save(os.path.join(tmp, "m.npz"))


def test_kmeans_is_a_declared_inference_lane_and_no_longer_owed():
    """lane/kmeans-save put `mojolearn-kmeans-1` in the classical host door;
    lane/classical-host-recordings (2026-09-16) declared the six fitted
    k-means lanes and recorded them, so `kmeans` left the owed registry.

    The assertion that would catch a regression is the pair: the lanes are
    DECLARED (so the gate covers them) and they are NOT owed (so nothing
    claims a recording is still missing). Dropping one k-means lane from
    `host_surface` fails the first; re-adding a `kmeans` note to
    SAVED_MODEL_INFERENCE_OWED fails the second.
    """
    from mojolearn import host_surface
    declared = host_surface.inference_lanes()
    for lane in ("kmeans", "kmeans-random", "kmeans-array", "kmeans-weighted",
                 "kmeans-sqrt", "kmeans-classic-pp"):
        assert lane in declared, f"{lane} is not a declared inference lane"
    assert "kmeans" not in host_surface.saved_model_inference_owed()


# ------------------------------------------------------------- round trip

CASES = [
    ("plain", lambda: mojolearn.KMeans(n_clusters=6, random_state=3)),
    ("sqrt", lambda: mojolearn.KMeans(n_clusters=6, random_state=3, metric="l2_sqrt_expanded")),
    ("random-start", lambda: mojolearn.KMeans(n_clusters=5, init="random", n_init=2, random_state=7)),
    ("classic-pp", lambda: mojolearn.KMeans(n_clusters=5, random_state=4, oversampling_factor=0.0)),
]


def _fit_and_save(make, X, path):
    with reference_training():
        est = make().fit(X)
    est.save(path)
    return est


@pytest.mark.parametrize("label,make", CASES)
def test_host_model_reproduces_the_fitted_model(label, make):
    if not _host_built():
        pytest.skip("mojolearn/host/_mojolearn_core_host.so is not built")
    X = _rows()
    Xt, Xh = X[:384], X[384:]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.npz")
        est = _fit_and_save(make, Xt, path)
        want = _bytes(_probe(est, Xh))
        host = mojolearn.host_model(path)
        assert type(host).__name__ == "HostKMeans" and host.estimator == "KMeans"
        assert host.vendor_used() == "cpu"
        got = _bytes(_probe(host, Xh))
        assert got == want
        # The fit's own attributes travel, and the host's predict on the
        # TRAINING rows is still `labels_` bit for bit.
        assert np.asarray(host.labels_).tobytes() == np.asarray(est.labels_).tobytes()
        assert np.asarray(host.predict(Xt)).tobytes() == np.asarray(est.labels_).tobytes()
        assert host.n_iter_ == est.n_iter_
        assert host.inertia_ == est.inertia_
        assert (host.sum_scale_, host.weight_scale_) == (est.sum_scale_, est.weight_scale_)
        # A host model saves under its own class name; every other member is
        # the same bytes.
        again = os.path.join(tmp, "again.npz")
        host.save(again)
        first = _serialize.read_npz(path, _classical_host.CLASSICAL_FORMATS)
        second = _serialize.read_npz(again, _classical_host.CLASSICAL_FORMATS)
        assert sorted(first) == sorted(second)
        assert _serialize.scalar_str(second, "estimator") == "HostKMeans"
        for member in sorted(first):
            if member == "estimator":
                continue
            a, b = first[member], second[member]
            if isinstance(a, (str, bytes, list)):
                assert a == b, member
            else:
                assert (a.dtype, tuple(a.shape), a.tobytes()) == (b.dtype, tuple(b.shape), b.tobytes()), member


def test_the_same_model_saves_to_the_same_bytes():
    X = _rows(256, 4, seed=2)
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "a.npz"), os.path.join(tmp, "b.npz")
        est = _fit_and_save(lambda: mojolearn.KMeans(n_clusters=4, random_state=1), X, a)
        est.save(b)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read()


def test_plain_load_round_trips_without_a_host_binding():
    X = _rows(256, 4, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.npz")
        est = _fit_and_save(lambda: mojolearn.KMeans(n_clusters=4, random_state=1), X, path)
        back = mojolearn.KMeans.load(path)
        assert np.asarray(back.cluster_centers_).tobytes() == np.asarray(est.cluster_centers_).tobytes()
        assert back.metric == "euclidean" and back.init == "k-means++"
        assert (back.n_clusters, back.n_features_in_) == (4, 4)
        # `_saved_mode` persists the tier that WOULD run, so a model whose
        # instance never set one saves the backend's default and loads with
        # it set. That is the point of saving it: the file names the tier.
        assert back.numeric_mode == (est.numeric_mode or _backend.default_mode())


# --------------------------------------------------- the check can fail

def _rewrite_member(src, dst, name, payload):
    """Copy `src` to `dst` replacing one npz member's raw data bytes."""
    with zipfile.ZipFile(src, "r") as zin:
        members = {m: zin.read(m) for m in zin.namelist()}
    members[name] = payload
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zout:
        for m in sorted(members):
            info = zipfile.ZipInfo(m, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zout.writestr(info, members[m])
    return dst


def _bump_one_ulp(npy_bytes, index=0):
    """The npy member with element `index` moved one FLOAT32 ULP away from
    zero. `math.nextafter` steps a double, which rounds back to the same
    float32 and would make this perturbation a no-op, so the step is taken
    in float32 and the test asserts the value actually moved."""
    descr, _f, _s, off = _serialize._parse_header(npy_bytes)
    assert descr in ("<f4", "=f4"), descr
    head, data = npy_bytes[:off], bytearray(npy_bytes[off:])
    lo = index * 4
    old = np.frombuffer(bytes(data[lo:lo + 4]), dtype="<f4")[0]
    new = np.nextafter(old, np.float32(np.inf if old >= 0 else -np.inf), dtype=np.float32)
    data[lo:lo + 4] = np.asarray(new, dtype="<f4").tobytes()
    return head + bytes(data), old, new


def test_one_ulp_in_the_file_moves_the_answer():
    """FAIL-FIRST. The comparison the round-trip test trusts is run against a
    file whose first centroid element is one ULP away, and it must FIRE,
    printing the values that differ rather than a count."""
    if not _host_built():
        pytest.skip("mojolearn/host/_mojolearn_core_host.so is not built")
    X = _rows()
    Xt, Xh = X[:384], X[384:]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.npz")
        _fit_and_save(lambda: mojolearn.KMeans(n_clusters=6, random_state=3), Xt, path)
        good = _bytes(_probe(mojolearn.host_model(path), Xh))
        with zipfile.ZipFile(path, "r") as z:
            bumped, old, new = _bump_one_ulp(z.read("centers.npy"))
        assert old != new, "nextafter moved nothing; the perturbation is not a perturbation"
        bad_path = _rewrite_member(path, os.path.join(tmp, "bumped.npz"), "centers.npy", bumped)
        bad = _bytes(_probe(mojolearn.host_model(bad_path), Xh))
        moved = [i for i, (g, b) in enumerate(zip(good, bad)) if g != b]
        # Name what moved, with values, not a count.
        report = []
        for i in moved:
            g = np.frombuffer(good[i][2], dtype=good[i][0]).reshape(good[i][1])
            b = np.frombuffer(bad[i][2], dtype=bad[i][0]).reshape(bad[i][1])
            where = np.argwhere(g != b)
            first = tuple(where[0])
            report.append(f"output {i}{good[i][1]} at {first}: {g[first]!r} -> {b[first]!r} "
                          f"({len(where)} of {g.size} elements moved)")
        assert moved, (
            f"one ULP on centroid element 0 ({old!r} -> {new!r}) changed NOTHING; "
            "this comparison cannot fail and proves nothing"
        )
        # Name the outputs that did NOT move too. One ULP on one centroid
        # need not change any label (a row's nearest center is decided by a
        # margin, not by the last bit), and reporting only the movers would
        # read as though it had.
        names = {0: "predict", 1: "transform"}
        still = [names[i] for i in range(len(good)) if i not in moved]
        print(f"centers[0] {old!r} -> {new!r}; " + "; ".join(report)
              + (f"; unmoved: {', '.join(still)}" if still else ""))


# ------------------------------------------------------------- refusals

def _saved(tmp, **kw):
    path = os.path.join(tmp, "m.npz")
    _fit_and_save(lambda: mojolearn.KMeans(n_clusters=4, random_state=1, **kw), _rows(256, 4, seed=3), path)
    return path


def test_a_truncated_file_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        path = _saved(tmp)
        cut = os.path.join(tmp, "cut.npz")
        with open(path, "rb") as fh:
            whole = fh.read()
        with open(cut, "wb") as fh:
            fh.write(whole[: len(whole) // 2])
        with pytest.raises(Exception) as e:
            mojolearn.KMeans.load(cut)
        assert not isinstance(e.value, AssertionError)


def test_a_file_of_another_format_refuses_by_its_tag():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "other.npz")
        _serialize.write_npz(path, {"format": "mojolearn-kmeans-0", "estimator": "KMeans"})
        with pytest.raises(ValueError, match="mojolearn-kmeans-0"):
            mojolearn.KMeans.load(path)
        with pytest.raises(ValueError):
            mojolearn.host_model(path)


def test_a_file_saved_by_another_estimator_refuses_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        path = _saved(tmp)
        arrays = _serialize.read_npz(path, _KMEANS_FORMAT)
        arrays["estimator"] = "DBSCAN"
        other = _serialize.write_npz(os.path.join(tmp, "other.npz"), arrays)
        with pytest.raises(ValueError, match="DBSCAN"):
            mojolearn.KMeans.load(other)


def test_a_centroid_count_that_disagrees_with_the_dimensionality_refuses():
    """The file says four centers of four features and carries three."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _saved(tmp)
        arrays = _serialize.read_npz(path, _KMEANS_FORMAT)
        c = np.asarray(arrays["centers"])
        arrays["centers"] = np.ascontiguousarray(c[:3])
        short = _serialize.write_npz(os.path.join(tmp, "short.npz"), arrays)
        with pytest.raises(ValueError, match=r"centers shape"):
            mojolearn.KMeans.load(short)


def test_a_meta_of_the_wrong_length_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        path = _saved(tmp)
        arrays = _serialize.read_npz(path, _KMEANS_FORMAT)
        arrays["meta"] = np.asarray(arrays["meta"])[:4]
        bad = _serialize.write_npz(os.path.join(tmp, "bad.npz"), arrays)
        with pytest.raises(ValueError, match="meta fields"):
            mojolearn.KMeans.load(bad)


def test_a_metric_name_that_disagrees_with_its_code_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        path = _saved(tmp)
        arrays = _serialize.read_npz(path, _KMEANS_FORMAT)
        arrays["metric"] = "l2_sqrt_expanded"          # the code member still says 0
        bad = _serialize.write_npz(os.path.join(tmp, "bad.npz"), arrays)
        with pytest.raises(ValueError, match="but its meta holds code 0"):
            mojolearn.KMeans.load(bad)


def test_an_unknown_metric_name_refuses_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        path = _saved(tmp)
        arrays = _serialize.read_npz(path, _KMEANS_FORMAT)
        arrays["metric"] = "mahalanobis"
        bad = _serialize.write_npz(os.path.join(tmp, "bad.npz"), arrays)
        with pytest.raises(ValueError, match="mahalanobis"):
            mojolearn.KMeans.load(bad)


def test_a_labels_count_that_disagrees_with_the_fit_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        path = _saved(tmp)
        arrays = _serialize.read_npz(path, _KMEANS_FORMAT)
        arrays["labels"] = np.asarray(arrays["labels"])[:-1]
        bad = _serialize.write_npz(os.path.join(tmp, "bad.npz"), arrays)
        with pytest.raises(ValueError, match="labels hold"):
            mojolearn.KMeans.load(bad)


def test_a_cast_is_refused_rather_than_performed():
    """`centers` saved as float64 is refused, never cast back down."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _saved(tmp)
        arrays = _serialize.read_npz(path, _KMEANS_FORMAT)
        arrays["centers"] = np.asarray(arrays["centers"]).astype(np.float64)
        bad = _serialize.write_npz(os.path.join(tmp, "bad.npz"), arrays)
        with pytest.raises(ValueError, match="refusing to cast"):
            mojolearn.KMeans.load(bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
