# SPDX-License-Identifier: Apache-2.0
"""Census fixtures: finalizers, unavailable fit_transform, module functions, cross_vendor, importers, host trees.

Testing tools for tools/ops/python_frames_census.py (fb4-census, 2026-11-17), never product: the smallest working
call of each public member verify's own fixtures do not reach. Shapes follow the tests (python/mojolearn/tests).
Loading this module imports only the standard library; each fixture takes `ml` (the installed mojolearn) and `cx`
(python_frames_census.FixtureContext).
"""
from pathlib import Path

#: the source checkout this tool runs from: the tests' stdlib-only ONNX writer lives there (not in the wheel)
_SOURCE = Path(__file__).resolve().parents[3]


def _xy(rows=32, cols=4):
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.random((rows, cols), dtype=np.float32)
    return X, (X[:, 0] > .5).astype(np.int64), (X[:, 0] * 2).astype(np.float32)


# ------------------------------------------------------------------ finalizers (measured by del)
def _fitted_del(cls_name, member, kind="X", **params):
    """Fit a tiny instance (a fit failure is ignored: the finalizer still runs on the instance), then del it."""
    def fixture(ml, cx):
        X, yc, yr = _xy(64, 8)
        model = getattr(ml, cls_name)(**params)
        try:
            model.fit(X) if kind == "X" else model.fit(X, yc if kind == "c" else yr)
        except Exception:
            pass
        box = {"m": model}
        model = None
        cx.delete(box, "m", member)
    return fixture


def host_forest_del(ml, cx):
    X, _, yr = _xy()
    path = str(cx.tmp / "forest")
    ml.RandomForestRegressor(n_estimators=1, max_depth=1, random_state=7).fit(X, yr).save(path)
    box = {"h": ml.HostForest.from_file(path)}
    cx.delete(box, "h", "mojolearn.HostForest.__del__")


# ------------------------------------------------------------------ host tree constructors
def host_forest_init(ml, cx):
    # tests/test_host_tree_native.py _forest_options: one split, two leaves, a regressor
    import numpy as np
    arrays = dict(offsets=np.asarray([0, 3], "<i4"), colid=np.asarray([0, -1, -1], "<i4"),
                  quesval=np.asarray([.5, 0, 0], "<f4"), left_child=np.asarray([1, -1, -1], "<i4"),
                  leaves=np.asarray([0, -.5, .75], "<f4"))
    arrays = {k: ml.Array.from_buffer(v) for k, v in arrays.items()}
    host = ml.HostForest(estimator="RandomForestRegressor", n_features=2, n_trees=1, num_outputs=1, classes=None,
                         device="fixture", inference_engine="sequential", **arrays)
    host.predict(np.zeros((2, 2), np.float32))


#: tests/test_host_model_native_router.py FLOAT_TEXT: one float feature, one depth-1 tree
_GBDT_TEXT = ("format mojolearn-model 2\nfeatures 1 0\ntrees 1\nlosses 0\n"
              "feature 0 folds 1 one_hot 0 type float nan as_is borders 1 0.5/3f000000\n"
              "tree 0 depth 1 dim 1 weights 0\nsplit 0 0 0 0\n"
              "leaf 0 0 0.25/3e800000\nleaf 0 1 0.75/3f400000\n")


def host_gbdt_init(ml, cx):
    import numpy as np
    host = ml.HostGBDT(loss="RMSE", text=_GBDT_TEXT, n_features_in=1, approx_dim=1, n_classes=-1,
                       numeric_mode=None, bias=None, estimator="GradientBoosting")
    host.predict(np.asarray([[0.], [1.]], np.float32))


def gbdt_adapter_load(ml, cx):
    # save/load of the scikit-learn adapters always refuse (no adapter archive format): the refusal is the call
    for name in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        try:
            getattr(ml, name).load(str(cx.tmp / "adapter"))
        except Exception:
            pass


# ------------------------------------------------------------------ fit_transform where there is no transform
_NO_TRANSFORM = ("BernoulliNB", "CategoricalNB", "ComplementNB", "GaussianNB", "MultinomialNB",
                 "QuadraticDiscriminantAnalysis", "MinCovDet", "EllipticEnvelope", "AlternatingLeastSquares")


def unavailable_fit_transform(ml, cx):
    # an available_if member the instance does not offer: its read refuses with AttributeError (served by the
    # class's native __getattribute__), measured by access
    for name in _NO_TRANSFORM:
        cx.access(getattr(ml, name)(), "fit_transform", f"mojolearn.{name}.fit_transform", expect=(AttributeError,))


# ------------------------------------------------------------------ module functions
def gpu_arch(ml, cx):
    ml.gpu_arch()


def numeric_mode(ml, cx):
    ml.numeric_mode()


def linalg_mode_profile(ml, cx):
    from mojolearn import linalg
    linalg.numeric_mode()
    linalg.profile()


# ------------------------------------------------------------------ cross_vendor (loopback)
def cross_vendor_tables(ml, cx):
    import numpy as np
    from mojolearn import cross_vendor as cv
    feed = cv.BatchFeed(np.arange(2 * 2 * 1 * 5, dtype=np.int32).reshape(2, 2, 1, 5), start_step=3)
    feed(3, 1)
    rates = cv.LearningRateTable([.1, .05], start_step=3)
    rates(4)
    raw = np.arange(4, dtype=np.float32)
    cv.state_hash(dict(parameters=raw, m=raw, v=raw, flags=np.zeros(1, np.int32)))


def _wire(sock, header, payload=b""):
    import json
    import struct
    raw = json.dumps(dict(header, nbytes=len(payload))).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw + payload)


def _read_wire(sock):
    import json
    import struct

    def exact(count):
        data = b""
        while len(data) < count:
            part = sock.recv(count - len(data))
            if not part:
                raise ConnectionError("census peer connection closed")
            data += part
        return data
    header = json.loads(exact(struct.unpack(">I", exact(4))[0]))
    return header, exact(header["nbytes"])


def _oracle_peer(port, shards, errors):
    """tests/test_cross_vendor.py's independent socket peer (a fixed gradient per shard, param -= total), run in a
    thread: the profiler measures the Coordinator in the main thread only."""
    import array
    import hashlib
    import socket
    try:
        param = array.array("f", [0.5] * 8)

        def digest():
            return hashlib.sha256(param.tobytes()).hexdigest()
        with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
            from mojolearn import cross_vendor as cv
            _wire(sock, dict(protocol=cv.PROTOCOL, name="census", vendor="oracle", shards=shards, completed=0,
                             state=digest(), n_total=len(param), chained=False))
            held, total, completed = [], None, 0
            while True:
                head, payload = _read_wire(sock)
                if head["cmd"] == "step":
                    grads = [array.array("f", [(k + 1) * .25 + i for i in range(len(param))]) for k in shards]
                    held = [g.tobytes() for g in grads]
                    _wire(sock, dict(step=head["step"], shards=shards, losses=[float(k) for k in shards]),
                          b"".join(held))
                elif head["cmd"] == "fold":
                    parts = ([payload] if payload else []) + held
                    values = array.array("f", parts[0])
                    for part in parts[1:]:
                        other = array.array("f", part)
                        for i in range(len(values)):
                            values[i] = values[i] + other[i]
                    total = values.tobytes()
                    _wire(sock, dict(step=head["step"]), total)
                elif head["cmd"] == "apply":
                    total = array.array("f", payload or total)
                    for i in range(len(param)):
                        param[i] -= total[i]
                    completed += 1
                    _wire(sock, dict(completed=completed, state=digest()))
                else:   # done or refuse
                    return
    except BaseException as exc:   # noqa: BLE001 - reported by the fixture
        errors.append(exc)


def cross_vendor_coordinator(ml, cx):
    import threading
    from mojolearn import cross_vendor as cv
    coord = cv.Coordinator(host="127.0.0.1", port=0, workers=1, logical_shards=4, steps=1,
                           accept_timeout=30, timeout=30)
    errors = []

    def peer():
        # the listener opens inside run(): connect once it is ready, as tests/test_cross_vendor.py's peers do
        # (connecting as soon as the Coordinator returned was refused, and run() then timed out at 30 s)
        if not coord.ready.wait(30):
            errors.append(TimeoutError("coordinator never listened"))
            return
        _oracle_peer(coord.bound_port, [0, 1, 2, 3], errors)
    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    coord.run()
    coord.drain_events()
    thread.join(40)
    if errors:
        raise errors[0]


def _byte_state(ml):
    # tests/test_parallel_byte_native.py state_fixture: the smallest byte-LM replica state
    import numpy as np
    from mojolearn import _backend, _byte_lm_impl
    shape = ml.ByteLanguageModelConfig(batch=1, length=4, d_model=8, n_heads=2, n_kv=1, head_dim=4,
                                       intermediate=16, n_layers=1, vocab_size=16)
    mode = _backend.default_mode()
    return dict(schema="mojolearn.small-byte-lm-state.v1", profile=_byte_lm_impl._runtime_profile(shape, mode),
                numeric_mode=mode, model_shape=shape.to_dict(), parameter_names=list(shape.parameter_names),
                parameter_shapes=[list(x) for x in shape.parameter_shapes], parameter_offsets=list(shape.offsets),
                parameters=np.zeros(shape.n_total, np.float32), m=np.zeros(shape.n_total, np.float32),
                v=np.zeros(shape.n_total, np.float32), flags=np.zeros(shape.n_tensors, np.int32),
                completed_steps=0, next_batch_index=0,
                config=dict(kind=2, lr=.125, beta1=.5, beta2=.75, eps=1e-8, weight_decay=.25, momentum=0.,
                            dampening=0., nesterov=False, max_norm=0.),
                data_schedule={"name": "parallel-native-independent", "offsets": [0, 4], "seed": 21})


def cross_vendor_worker(ml, cx):
    # tests/test_cross_vendor_admission_native.py: a Worker builds its trainer on the CPU; run() needs a GPU
    import numpy as np
    from mojolearn import cross_vendor as cv
    feed = cv.BatchFeed(np.zeros((1, 1, 1, 5), np.int32))
    worker = cv.Worker(_byte_state(ml), shards=[0], batches=feed, address=("127.0.0.1", 1))
    try:
        worker.drain_events()
    finally:
        worker.trainer.close()


# ------------------------------------------------------------------ importers
def _onnx_writer():
    import importlib.util
    path = _SOURCE / "python" / "mojolearn" / "tests" / "_onnx_fixture.py"
    spec = importlib.util.spec_from_file_location("_census_onnx_fixture", path)
    mod = importlib.util.module_from_spec(spec)
    # its @dataclass records resolve their module through sys.modules ('NoneType' object has no attribute
    # '__dict__' when the module was never registered)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def importers_read(ml, cx):
    from mojolearn import importers
    from mojolearn.importers import onnx
    fx = _onnx_writer().write_fixture("linear_layernorm_linear", str(cx.tmp))
    onnx.read(fx.path)
    importers.supported_ops()
    try:
        onnx.reader_error(1, "not an ONNX model", None, str(cx.tmp / "x.onnx"))
    except Exception:   # it raises the reader's error: that is the call
        pass


def importers_records(ml, cx):
    from mojolearn import importers
    importers.UnsupportedModel("census", (("relu", "Relu", "census fixture"),))


FIXTURES = {
    "ann_del": (lambda ml, cx: [_fitted_del(n, f"mojolearn.{n}.__del__", n_lists=2, n_probes=1)(ml, cx)
                                for n in ("IVFPQIndex", "IVFSQIndex", "IVFRaBitQIndex", "IVFIndex")],
                [f"mojolearn.{n}.__del__" for n in ("IVFPQIndex", "IVFSQIndex", "IVFRaBitQIndex", "IVFIndex")]),
    "cagra_del": (_fitted_del("CagraIndex", "mojolearn.CagraIndex.__del__", graph_degree=4,
                              intermediate_graph_degree=8, n_neighbors=2, itopk_size=8),
                  ["mojolearn.CagraIndex.__del__"]),
    "kernel_ridge_del": (_fitted_del("KernelRidge", "mojolearn.KernelRidge.__del__", kind="r"),
                         ["mojolearn.KernelRidge.__del__"]),
    "nearest_neighbors_del": (_fitted_del("NearestNeighbors", "mojolearn.NearestNeighbors.__del__", n_neighbors=2),
                              ["mojolearn.NearestNeighbors.__del__"]),
    "kneighbors_classifier_del": (_fitted_del("KNeighborsClassifier", "mojolearn.KNeighborsClassifier.__del__",
                                              kind="c", n_neighbors=2),
                                  ["mojolearn.KNeighborsClassifier.__del__"]),
    "kneighbors_regressor_del": (_fitted_del("KNeighborsRegressor", "mojolearn.KNeighborsRegressor.__del__",
                                             kind="r", n_neighbors=2),
                                 ["mojolearn.KNeighborsRegressor.__del__"]),
    "host_forest_del": (host_forest_del, ["mojolearn.HostForest.__del__"]),
    "host_forest_init": (host_forest_init, ["mojolearn.HostForest.__init__"]),
    "host_gbdt_init": (host_gbdt_init, ["mojolearn.HostGBDT.__init__"]),
    "gbdt_adapter_load": (gbdt_adapter_load, ["mojolearn.GradientBoostingClassifier.load",
                                              "mojolearn.GradientBoostingRegressor.load"]),
    "unavailable_fit_transform": (unavailable_fit_transform, [f"mojolearn.{n}.fit_transform" for n in _NO_TRANSFORM]),
    "gpu_arch": (gpu_arch, ["mojolearn.gpu_arch"]),
    "numeric_mode": (numeric_mode, ["mojolearn.numeric_mode"]),
    "linalg_mode_profile": (linalg_mode_profile, ["mojolearn.linalg.numeric_mode", "mojolearn.linalg.profile"]),
    "cross_vendor_tables": (cross_vendor_tables, ["mojolearn.cross_vendor.BatchFeed.__init__",
                                                  "mojolearn.cross_vendor.BatchFeed.__call__",
                                                  "mojolearn.cross_vendor.LearningRateTable.__init__",
                                                  "mojolearn.cross_vendor.LearningRateTable.__call__",
                                                  "mojolearn.cross_vendor.state_hash"]),
    "cross_vendor_coordinator": (cross_vendor_coordinator, ["mojolearn.cross_vendor.Coordinator.__init__",
                                                            "mojolearn.cross_vendor.Coordinator.run",
                                                            "mojolearn.cross_vendor.Coordinator.drain_events"]),
    "cross_vendor_worker": (cross_vendor_worker, ["mojolearn.cross_vendor.Worker.__init__",
                                                  "mojolearn.cross_vendor.Worker.drain_events"]),
    "importers_read": (importers_read, ["mojolearn.importers.onnx.read", "mojolearn.importers.supported_ops",
                                        "mojolearn.importers.onnx.reader_error"]),
    "importers_records": (importers_records, ["mojolearn.importers.UnsupportedModel.__init__"]),
}

NEEDS_GPU = {
    "mojolearn.cross_vendor.Worker.run": "the byte-LM cooperative worker trains on a GPU device; on the CPU wheel "
                                         "its trainer has no device step (tests/test_cross_vendor_worker_native.py "
                                         "skips the complete worker on vendor 'cpu')",
}

#: members no public operation runs, left unreached on purpose
WHY_NOT = {}
