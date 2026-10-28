"""Light Sequential admission fixture: same bits and same refusals.

Loads ONE compiled `_mojolearn_runtime_sequential_host` (the Sequential owner,
whose admission is under test) and one compiled `_mojolearn_x_sequence_host`
(the CPU-column numerical binding), plus the tokenizer host (Array owner and
shared provider types), straight from files, without importing the
mojolearn package (which needs the whole binary set). The three files come from
the environment:

    SEQ_RUNTIME_SO=<.../_mojolearn_runtime_sequential_host.so>
    SEQ_NUMERIC_SO=<.../_mojolearn_x_sequence_host.so>
    SEQ_TOKENIZER_SO=<.../_mojolearn_tokenizer_host.so>
    python3 tools/tests/sequential_admission_fixture.py <out.json> <scratch dir>

It prints one line per operation (`op digest`) and per refusal case
(`case ExceptionType: message`) and writes them to <out.json>. Run it once
with the owner built from the base commit and once with the head; the two
JSON files must agree on every digest and on every refusal except the cases
listed in NARROWED, which are refusals of fields the operation never reads:

    python3 tools/tests/sequential_admission_fixture.py --compare <before.json> <after.json>
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "mojolearn"


def _load_extension(full, path):
    loader = importlib.machinery.ExtensionFileLoader(full, str(path))
    spec = importlib.util.spec_from_loader(full, loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _sequential_class(runtime_so, numeric_so, tokenizer_so):
    # A bare package object over the real source directory: submodules load
    # from the tree, `mojolearn/__init__.py` (binary-set selection) never runs.
    package = types.ModuleType("mojolearn")
    package.__path__ = [str(PACKAGE)]
    sys.modules["mojolearn"] = package
    numeric = _load_extension("mojolearn._mojolearn_x_sequence_host", numeric_so)
    tokenizer = _load_extension("mojolearn._host._mojolearn_tokenizer_host", tokenizer_so)
    runtime = _load_extension("mojolearn._host._mojolearn_runtime_sequential_host", runtime_so)
    backend = types.ModuleType("mojolearn._backend")
    backend._DEFAULT_MODE = "identical"
    backend.default_mode = lambda: "identical"
    backend.vendor = lambda: str(numeric.x_sequence_vendor())

    def binding(name, mode):
        if mode != "identical":
            raise ImportError("fixture loads the identical CPU column only")
        return numeric

    backend.binding = binding
    # `_array` connects its container owner from the tokenizer host, which
    # also registers the shared native provider types the Sequential owner
    # returns (NeuralDescriptorProvider).
    hosts = {"_mojolearn_runtime_sequential_host": runtime, "_mojolearn_tokenizer_host": tokenizer}
    backend.load_host_module = lambda name: hosts[name]
    sys.modules["mojolearn._backend"] = backend
    package._backend = backend
    # The package connects its stdlib module registry at import; native
    # admission reads it (bindings/native_loaded_module.mojo).
    import mojolearn._native_loaded_modules  # noqa: F401
    from mojolearn import _x_sequence_model
    return _x_sequence_model.Sequential


def _digest(value):
    h = hashlib.sha256()

    def feed(v):
        if isinstance(v, np.ndarray):
            h.update(repr((v.dtype.str, v.shape)).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, (list, tuple)):
            h.update(b"[")
            for item in v:
                feed(item)
            h.update(b"]")
        elif isinstance(v, (bytes, bytearray)):
            h.update(bytes(v))
        else:
            h.update(repr(v).encode())

    feed(value)
    return h.hexdigest()


def _file(path):
    return pathlib.Path(path).read_bytes()


def main(out_path, scratch):
    Sequential = _sequential_class(os.environ["SEQ_RUNTIME_SO"], os.environ["SEQ_NUMERIC_SO"], os.environ["SEQ_TOKENIZER_SO"])
    scratch = pathlib.Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.random((32, 4), dtype=np.float32)
    y = (X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    labels = (X[:, 0] > X[:, 1]).astype(np.int64)
    digests = {}
    refusals = {}

    def regressor(optimizer):
        return Sequential("identical").linear(8).relu().linear(1).compile(loss="mse", optimizer=optimizer, lr=0.01)

    # Unfitted description-only reads (no optimizer state, no fit fields).
    bare = regressor("adam")
    digests["unfitted.layers"] = _digest(bare.layers)
    digests["unfitted.table"] = _digest(bare._table())
    digests["unfitted.program"] = _digest(bare._program(4))
    digests["unfitted.parameter_layers"] = _digest(bare._param_layers())
    digests["unfitted.sample"] = _digest(bare._sample_of(X))

    for optimizer in ("adam", "sgd"):
        m = regressor(optimizer).fit(X, y, epochs=3, batch_size=8, seed=0)
        tag = "mse_" + optimizer
        digests[tag + ".fit.params"] = _digest(m.params_)
        digests[tag + ".fit.loss_curve"] = _digest(list(m.loss_curve_))
        digests[tag + ".predict"] = _digest(m.predict(X))
        digests[tag + ".forward"] = _digest(m._forward(X, 0))
        digests[tag + ".layers"] = _digest(m.layers)
        digests[tag + ".table"] = _digest(m._table())
        digests[tag + ".program"] = _digest(m._program(4))
        digests[tag + ".parameter_layers"] = _digest(m._param_layers())
        ckpt = scratch / (tag + ".ckpt")
        m.save_checkpoint(str(ckpt))
        digests[tag + ".save_checkpoint"] = _digest(_file(ckpt))
        exported = scratch / (tag + ".export")
        m.export(str(exported))
        digests[tag + ".export"] = _digest(_file(exported))
        # A model loaded from an export holds no optimizer/training state.
        loaded = Sequential.load(str(exported))
        digests[tag + ".loaded_export.predict"] = _digest(loaded.predict(X))
        digests[tag + ".loaded_export.layers"] = _digest(loaded.layers)
        resumed = Sequential.from_checkpoint(str(ckpt)).fit(X, y, epochs=1, batch_size=8, seed=0, resume=True)
        digests[tag + ".resume.params"] = _digest(resumed.params_)

    for loss in ("cross_entropy", "bce_with_logits"):
        c = Sequential("identical").linear(8).relu().linear(2 if loss == "cross_entropy" else 1)
        c = c.compile(loss=loss, optimizer="adam", lr=0.01).fit(X, labels, epochs=3, batch_size=8, seed=0)
        digests[loss + ".fit.params"] = _digest(c.params_)
        digests[loss + ".predict"] = _digest(c.predict(X))
        digests[loss + ".predict_proba"] = _digest(c.predict_proba(X))

    def refuse(case, call):
        try:
            call()
        except Exception as error:  # noqa: BLE001 - the type is the record
            refusals[case] = type(error).__name__ + ": " + str(error)
        else:
            refusals[case] = "no refusal"

    fitted = regressor("adam").fit(X, y, epochs=1, batch_size=8, seed=0)

    class PyList(list):
        pass

    class PyArray(np.ndarray):
        pass

    class PyObject:
        pass

    class Sub(Sequential):
        pass

    def with_field(name, value):
        m = regressor("adam").fit(X, y, epochs=1, batch_size=8, seed=0)
        m.__dict__[name] = value
        return m

    refuse("predict.unfitted", lambda: regressor("adam").predict(X))
    refuse("predict_proba.unfitted", lambda: regressor("adam").predict_proba(X))
    refuse("predict.bad_dtype", lambda: fitted.predict(np.array([["a"] * 4] * 2)))
    refuse("predict.wrong_features", lambda: fitted.predict(X[:, :3]))
    refuse("predict.python_object_X", lambda: fitted.predict(PyObject()))
    refuse("predict.python_list_subclass_X", lambda: fitted.predict(PyList(X.tolist())))
    refuse("predict_proba.mse_model", lambda: fitted.predict_proba(X))
    def as_subclass():
        m = regressor("adam").fit(X, y, epochs=1, batch_size=8, seed=0)
        m.__class__ = Sub
        return m

    refuse("construct.subclass_model", lambda: Sub("identical"))
    refuse("predict.subclass_model", lambda: as_subclass().predict(X))
    refuse("layers.subclass_model", lambda: as_subclass().layers)
    refuse("fit.subclass_model", lambda: as_subclass().fit(X, y, epochs=1))
    # Fields each read operation actually uses: still refused.
    refuse("predict.layers_python_list", lambda: with_field("_layers", PyList(fitted._layers)).predict(X))
    refuse("layers.layers_python_list", lambda: with_field("_layers", PyList(fitted._layers)).layers)
    refuse("table.layers_python_list", lambda: with_field("_layers", PyList(fitted._layers))._table())
    refuse("parameter_layers.layers_python_list", lambda: with_field("_layers", PyList(fitted._layers))._param_layers())
    refuse("predict.params_ndarray_subclass", lambda: with_field("params_", fitted.params_.view(PyArray)).predict(X))
    refuse("predict.n_features_python_object", lambda: with_field("n_features_in_", PyObject()).predict(X))
    refuse("predict.fit_loss_python_object", lambda: with_field("_fit_loss", PyObject()).predict(X))
    refuse("predict.classes_python_object", lambda: with_field("classes_", PyObject()).predict(X))
    refuse("predict.y_1d_python_object", lambda: with_field("_y_1d", PyObject()).predict(X))
    refuse("predict.numeric_mode_python_object", lambda: with_field("numeric_mode", PyObject()).predict(X))
    # Fields only fit/save/state operations read: refused there (full walk).
    for name in ("_opt_m", "loss_curve_", "_options"):
        refuse("fit." + name + "_python_object", lambda name=name: with_field(name, PyObject()).fit(X, y, epochs=1, batch_size=8, seed=0, resume=True))
        refuse("save_checkpoint." + name + "_python_object", lambda name=name: with_field(name, PyObject()).save_checkpoint(str(scratch / "refused.ckpt")))
        refuse("predict." + name + "_python_object", lambda name=name: with_field(name, PyObject()).predict(X))
        refuse("layers." + name + "_python_object", lambda name=name: with_field(name, PyObject()).layers)

    for key in sorted(digests):
        print(key, digests[key])
    for key in sorted(refusals):
        print("REFUSAL", key, refusals[key][:160])
    pathlib.Path(out_path).write_text(json.dumps({"digests": digests, "refusals": refusals}, indent=1, sort_keys=True) + "\n")


# Read-only operations admit only the model fields they read
# (bindings/sequential_admission.mojo `_read_fields`). A Python-defined value
# in a field that predict/layers never read was refused by the full inventory
# walk and is not refused by the narrow admission; fit and save still refuse
# it (the `fit.*` and `save_checkpoint.*` cases stay equal).
NARROWED = tuple(op + "." + name + "_python_object" for op in ("predict", "layers") for name in ("_opt_m", "loss_curve_", "_options"))


def compare(before_path, after_path):
    before = json.loads(pathlib.Path(before_path).read_text())
    after = json.loads(pathlib.Path(after_path).read_text())
    failures = []
    if sorted(before["digests"]) != sorted(after["digests"]):
        failures.append("digest keys differ")
    for key in sorted(before["digests"]):
        same = before["digests"][key] == after["digests"].get(key)
        print("SAME_BITS" if same else "DIFFER", key)
        if not same:
            failures.append(key)
    for key in sorted(before["refusals"]):
        old, new = before["refusals"][key], after["refusals"].get(key)
        if key in NARROWED:
            ok = old.startswith("TypeError: ") and new == "no refusal"
            print("NARROWED" if ok else "UNEXPECTED", key, "|", old[:100], "->", new)
        else:
            ok = old == new and old != "no refusal"
            print("SAME_REFUSAL" if ok else "REFUSAL_DIFFERS", key, "|", old[:120])
        if not ok:
            failures.append(key)
    print("RESULT", "PASS" if not failures else "FAIL " + ",".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    main(sys.argv[1], sys.argv[2])
