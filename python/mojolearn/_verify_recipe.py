# SPDX-License-Identifier: Apache-2.0
"""Verify a user's ALGORITHM: a recipe, fitted from scratch on every device.

`model.check(X)` and `verify --check` compare one fitted model's answers on
the GPU and on the CPU. A recipe goes one step further back: it says HOW the
model is made (the estimator class or the layers of a `Sequential`, the
hyperparameters, the seed, the fit arguments and the data) and what to hash
(the fitted `digest_`, the predictions on stated rows, the score). Verify
fits it from scratch in a fresh process per device of this installation (the
CPU, and the GPU when there is one), compares every hash, prints IDENTICAL or
DIVERGENT with the first differing value, and writes a RECORD. A record made
on another machine or vendor is compared with `--against`: that is the
cross-vendor claim for the user's own algorithm, within one version.

    python -m mojolearn verify --recipe r.json                 # writes r.record.json
    python -m mojolearn verify --recipe r.json --against other.record.json
    python -m mojolearn verify --recipe r.json --self-test     # it can fail
    python -m mojolearn verify --recipe example:ridge          # a shipped recipe

THE RECIPE (one JSON object, `FORMAT`):

    {"format": "mojolearn.recipe.v1",
     "model": {"estimator": "Ridge", "params": {"alpha": 1.0}}
           or {"sequential": {"input_shape": [8, 4], "numeric_mode": "identical",
                              "layers": [["linear", 16], ["rms_norm"], ["attention", 2, {"kv_heads": 1}],
                                         ["residual_block", {"sequential": [["linear", 16]]}], ...],
                              "compile": {"loss": "mse", "optimizer": "sgd", "lr": 0.01}}}
             (or "description" and "options": the model's own layer table, which
             `recipe(model, ...)` writes from a live model),
     "seed": 0,                 the estimator's random_state, or Sequential.fit's seed
     "fit": {"epochs": 2},      keyword arguments of fit (plain values)
     "data": {"generator": "regression", "args": {"rows": 256, "features": 8, "test_rows": 64, "seed": 0}}
          or {"file": "r.data.npz", "sha256": "<hex of the file>"},   arrays X, y, X_test
     "outputs": {"digest": true, "predict": {"method": "predict", "on": "test"}, "score": true}}

Refused in words, never compared: a recipe in `fast` mode (no promise about
bits), a data file whose SHA-256 is not the recipe's, a record made with
another Mojolearn version (bits are per version), a record of another recipe
or other data.

This is verification tooling (AGENTS.md "The verify command is protected"):
it calls the public API as a user does, around product calls, never inside
one, and adds no product path. NumPy is imported inside the functions. The
shared describe and first-difference code is `_check_both`'s.
"""
import base64
import copy
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile

from . import _check_both as cb

FORMAT = "mojolearn.recipe.v1"
RECORD_FORMAT = "mojolearn.recipe-record.v1"
CHILD_FORMAT = "mojolearn.recipe-child.v1"

EXIT_IDENTICAL = 0
EXIT_DIVERGENT = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_CANNOT_RUN = 4

#: an output whose bytes are at most this many is kept in the record, so that
#: `--against` on another machine can name the first differing element
RECORD_VALUES_MAX = 1 << 20
#: the arrays a data file may hold
DATA_ARRAYS = ("X", "y", "X_test")
#: which rows an output is computed on
ROWS = ("train", "test")
#: a model's constructor parameter that a recipe never carries: verify decides
#: the device, and the seed is the recipe's own field
NOT_PARAMS = ("device",)
SEED_PARAM = "random_state"


class Refused(Exception):
    """The recipe or the record is refused; the message says why. Never a verdict."""


class CannotRun(Exception):
    """A device could not fit or answer the recipe; the message says why."""


def _np():
    import numpy
    return numpy


# --------------------------------------------------------------------------
# the data: built-in generators (NumPy's PCG64 and elementwise IEEE operations
# only, so the bytes are the same on every machine) or a file with its hash
# --------------------------------------------------------------------------

def _linear_target(np, X, rng, outputs=1):
    """`X @ w` column by column in a fixed order: no BLAS, whose summation
    order differs between machines."""
    w = rng.standard_normal((X.shape[1], outputs))
    acc = np.zeros((X.shape[0], outputs))
    for j in range(X.shape[1]):  # verify tooling: over the generator's columns
        acc = acc + X[:, j:j + 1].astype(np.float64) * w[j]
    return acc


def _gen_regression(np, rng, rows, features=8, noise=0.1):
    X = rng.standard_normal((rows, features)).astype(np.float32)
    y = _linear_target(np, X, rng)[:, 0] + noise * rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _gen_classification(np, rng, rows, features=8, classes=2):
    X = rng.standard_normal((rows, features)).astype(np.float32)
    y = np.argmax(_linear_target(np, X, rng, classes), axis=1).astype(np.int64)
    return X, y


def _gen_blobs(np, rng, rows, features=4, centers=3, spread=0.5):
    middle = 4.0 * rng.standard_normal((centers, features))
    label = rng.integers(0, centers, rows)
    X = (middle[label] + spread * rng.standard_normal((rows, features))).astype(np.float32)
    return X, label.astype(np.int64)


def _gen_categorical(np, rng, rows, numeric=4, categorical=2, levels=4, task="regression"):
    """`categorical` columns of integer codes 0..levels-1 (as float32) first,
    then `numeric` standard normal columns. The target adds a per-level effect
    of each coded column to a linear term of the numeric ones."""
    codes = rng.integers(0, levels, (rows, categorical))
    numbers = rng.standard_normal((rows, numeric))
    effect = rng.standard_normal((categorical, levels))
    target = _linear_target(np, numbers, rng)[:, 0]
    for j in range(categorical):  # verify tooling: over the generator's coded columns
        target = target + effect[j][codes[:, j]]
    X = np.concatenate([codes.astype(np.float32), numbers.astype(np.float32)], axis=1)
    if task == "classification":
        return X, (target > np.median(target)).astype(np.int64)
    if task != "regression":
        raise Refused(f"the categorical generator's task is 'regression' or 'classification', not {task!r}")
    return X, target.astype(np.float32)


def _gen_sequence(np, rng, rows, shape=(8, 4)):
    """Samples of `shape` (steps, channels) as rows of prod(shape) values, and
    one target per sample: a smooth function of the first and last steps."""
    width = 1
    for n in shape:  # verify tooling: over the sample's axes
        width *= int(n)
    X = rng.standard_normal((rows, width)).astype(np.float32)
    y = 0.5 * X[:, 0].astype(np.float64) - X[:, width - 1] + 0.25 * np.tanh(X[:, width // 2].astype(np.float64))
    return X, y.astype(np.float32).reshape(rows, 1)


GENERATORS = {
    "regression": _gen_regression,
    "classification": _gen_classification,
    "blobs": _gen_blobs,
    "categorical": _gen_categorical,
    "sequence": _gen_sequence,
}


def generate(name, args):
    """The arrays of a built-in generator: X, y and, with `test_rows`, X_test
    (the rows that follow the training rows in the same stream)."""
    np = _np()
    if name not in GENERATORS:
        raise Refused(f"no built-in generator {name!r}; the generators are {', '.join(sorted(GENERATORS))}")
    args = dict(args or {})
    seed = int(args.pop("seed", 0))
    rows = int(args.pop("rows", 256))
    test_rows = int(args.pop("test_rows", 0))
    try:
        X, y = GENERATORS[name](np, np.random.default_rng(seed), rows + test_rows, **args)
    except TypeError as exc:
        raise Refused(f"the {name} generator does not take these arguments: {exc}") from exc
    data = dict(X=np.ascontiguousarray(X[:rows]), y=np.ascontiguousarray(y[:rows]))
    if test_rows:
        data["X_test"] = np.ascontiguousarray(X[rows:])
    return data


def data_sha256(data):
    """SHA-256 over every array (name, dtype, shape, bytes) in `DATA_ARRAYS` order."""
    m = hashlib.sha256()
    for name in DATA_ARRAYS:
        if name in data:
            a = data[name]
            m.update(name.encode() + b"\0" + str(a.dtype).encode() + b"\0" + str(tuple(a.shape)).encode() + b"\0")
            m.update(a.tobytes())
    return m.hexdigest()


def file_sha256(path):
    m = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            m.update(block)
    return m.hexdigest()


def _data_path(spec, base):
    path = spec["file"]
    return path if os.path.isabs(path) else os.path.join(base or ".", path)


def load_data(spec, base=None):
    """The arrays a recipe's `data` names. A file is refused unless its
    SHA-256 is the recipe's."""
    np = _np()
    if "generator" in spec:
        return generate(spec["generator"], spec.get("args"))
    if "file" not in spec:
        raise Refused("the recipe's data names neither a generator nor a file")
    path = _data_path(spec, base)
    if not os.path.exists(path):
        raise Refused(f"the recipe's data file {path} does not exist")
    want = str(spec.get("sha256") or "")
    have = file_sha256(path)
    if want.lower() != have:
        raise Refused(f"the data file {path} has SHA-256 {have}, not the recipe's {want or '(none given)'}: "
                      "these are other data, so nothing is compared")
    with np.load(path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in DATA_ARRAYS if name in loaded.files}
    if "X" not in data:
        raise Refused(f"the data file {path} holds no array X")
    return data


# --------------------------------------------------------------------------
# the recipe: read, check, canonical hash
# --------------------------------------------------------------------------

def _plain(value, what):
    """`value` as JSON-plain Python values (a NumPy scalar or array becomes
    its Python value or list), or Refused naming `what`."""
    np = _np()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v, what) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _plain(v, what) for k, v in value.items()}
    raise Refused(f"{what} holds a {type(value).__name__}, which a recipe cannot carry (plain values only)")


def _modes(recipe):
    """Every numeric_mode the recipe names."""
    model = recipe.get("model") or {}
    found = []
    if "estimator" in model:
        found.append((model.get("params") or {}).get("numeric_mode"))
    seq = model.get("sequential")
    if isinstance(seq, dict):
        found.append(seq.get("numeric_mode"))
    return found


def check_recipe(recipe):
    """Raise Refused unless `recipe` is a recipe verify can run: the format,
    one model, data, at least one output, identical mode."""
    if not isinstance(recipe, dict) or recipe.get("format") != FORMAT:
        raise Refused(f"not a recipe: the format must be {FORMAT!r}")
    model = recipe.get("model")
    if not isinstance(model, dict) or (("estimator" in model) == ("sequential" in model)):
        raise Refused("the recipe's model names exactly one of 'estimator' (a class and its params) "
                      "or 'sequential' (a model of the public layers)")
    if "estimator" in model:
        params = model.get("params") or {}
        for name in NOT_PARAMS:
            if name in params:
                raise Refused(f"the recipe sets {name}=; a recipe runs on every device this installation has, "
                              f"so it leaves {name} out")
        if SEED_PARAM in params:
            raise Refused(f"the recipe sets {SEED_PARAM} in params; the seed is the recipe's own 'seed' field")
    else:
        seq = model["sequential"]
        if not isinstance(seq, dict) or (("layers" in seq) == ("description" in seq)):
            raise Refused("a sequential model gives exactly one of 'layers' (the method calls) or "
                          "'description' (the model's layer table)")
    for mode in _modes(recipe):
        if mode not in (None, "identical"):
            raise Refused(f"the recipe is set to numeric_mode={mode!r}, which makes no promise about bits; "
                          "a recipe is verified in identical mode only")
    data = recipe.get("data")
    if not isinstance(data, dict) or not ("generator" in data or "file" in data):
        raise Refused("the recipe's data names a built-in generator or a file with its sha256")
    outputs = recipe.get("outputs") or {}
    if not any(outputs.get(k) for k in ("digest", "predict", "score")):
        raise Refused("the recipe hashes nothing: outputs names digest, predict or score")
    predict = outputs.get("predict")
    if predict and isinstance(predict, dict) and predict.get("on", "train") not in ROWS:
        raise Refused(f"outputs.predict.on is 'train' or 'test', not {predict.get('on')!r}")
    fit = recipe.get("fit") or {}
    if not isinstance(fit, dict):
        raise Refused("the recipe's fit is an object of keyword arguments")
    if "seed" in fit and recipe.get("seed") is not None:
        raise Refused("the recipe gives the seed twice (seed and fit.seed); give it once, as seed")


def canonical(recipe):
    """The recipe as it decides bits: a data file is named by its SHA-256
    alone, so the same recipe in another directory has the same hash."""
    out = copy.deepcopy(recipe)
    data = out.get("data") or {}
    if "file" in data:
        out["data"] = {"sha256": str(data.get("sha256") or "").lower()}
    return out


def recipe_sha256(recipe):
    text = json.dumps(canonical(recipe), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------
# `mojolearn.recipe(model, X, y)`: a recipe from a live, unfitted model
# --------------------------------------------------------------------------

def _sequential_spec(model):
    fields = object.__getattribute__(model, "__dict__")
    layers = fields.get("_layers") or []
    options = fields.get("_options")
    if not layers:
        raise Refused("this Sequential has no layers")
    if options is None:
        raise Refused("this Sequential is not compiled: call compile(...) before recipe(...)")
    shape = fields.get("input_shape")
    return dict(numeric_mode=fields.get("numeric_mode"),
                input_shape=None if shape is None else [int(n) for n in shape],
                description=_plain([list(layer) for layer in layers], "the layer table"),
                options=_plain(dict(options), "the compile options"))


def recipe(model, X=None, y=None, *, data=None, fit_params=None, X_test=None, method="predict",
           predict_on=None, score=False, digest=True, path=None):
    """Describe how a model is made, as a recipe that `verify --recipe` fits
    from scratch on every device and compares bit for bit.

    In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and
    AMD GPUs. A recipe states your algorithm (the estimator and its
    hyperparameters, or the layers of a `Sequential` and its compile options;
    the seed; the fit arguments; the data) and what to hash: the fitted
    `digest_`, the predictions on stated rows and the score.
    `python -m mojolearn verify --recipe r.json` fits it in a fresh process on
    each device of this installation, says IDENTICAL or DIVERGENT with the
    first differing value, and writes a record that another machine compares
    with `--against`.

    Parameters
    ----------
    model : estimator or Sequential
        The model as you would fit it, not yet fitted: an estimator with its
        constructor parameters, or a compiled `Sequential`.
    X, y : array-like, optional
        The training data. They are written beside the recipe (`path` is then
        required) and named by their SHA-256.
    data : dict, optional
        Instead of `X` and `y`: a built-in generator, for example
        ``{"generator": "regression", "args": {"rows": 256, "features": 8,
        "test_rows": 64, "seed": 0}}``. Generators: regression,
        classification, blobs, categorical, sequence.
    fit_params : dict, optional
        Keyword arguments of `fit`, plain values only (for a `Sequential`:
        epochs, batch_size, shuffle, seed).
    X_test : array-like, optional
        Rows to predict on, written with `X` and `y`.
    method : str or None, default 'predict'
        The method whose answers are hashed; None hashes no predictions.
    predict_on : {'train', 'test'} or None
        Which rows `method` answers for. Default: 'test' when there are test
        rows, else 'train'.
    score : bool, default False
        Also hash `model.score` on the same rows (needs `y` for 'train').
    digest : bool, default True
        Hash the fitted model's `digest_`.
    path : str, optional
        Write the recipe there as JSON (and the data beside it, as
        ``<name>.data.npz``).

    Returns
    -------
    dict
        The recipe.

    Examples
    --------
    >>> import mojolearn
    >>> r = mojolearn.recipe(mojolearn.Ridge(alpha=1.0),
    ...                      data={"generator": "regression", "args": {"rows": 256, "test_rows": 64}},
    ...                      score=True, path="ridge.json")
    >>> # then: python -m mojolearn verify --recipe ridge.json
    """
    np = _np()
    fit = dict(_plain(dict(fit_params or {}), "fit_params"))
    seed = None
    if "_layers" in object.__getattribute__(model, "__dict__"):   # a Sequential: its layer table
        spec = {"sequential": _sequential_spec(model)}
        if "seed" in fit:
            seed = fit.pop("seed")
    else:
        get = getattr(model, "get_params", None)
        if not callable(get):
            raise Refused(f"{type(model).__name__} has no get_params, so its parameters cannot be written down")
        try:
            params = dict(get(deep=False))
        except TypeError:
            params = dict(get())
        for name in NOT_PARAMS:
            params.pop(name, None)
        if SEED_PARAM in params:
            seed = params.pop(SEED_PARAM)
        spec = {"estimator": type(model).__name__, "params": _plain(params, f"the parameters of {type(model).__name__}")}
    if data is not None and X is not None:
        raise Refused("give the data once: X and y, or data")
    if data is None:
        if X is None:
            raise Refused("recipe needs the data: X (and y), or data={'generator': ...}")
        if path is None:
            raise Refused("recipe with arrays needs path: the arrays are written beside the recipe and named by "
                          "their SHA-256")
        arrays = dict(X=np.ascontiguousarray(np.asarray(X)))
        if y is not None:
            arrays["y"] = np.ascontiguousarray(np.asarray(y))
        if X_test is not None:
            arrays["X_test"] = np.ascontiguousarray(np.asarray(X_test))
        for name, a in arrays.items():
            if a.dtype == object:
                raise Refused(f"{name} is an object array; a recipe's data are numeric arrays")
        stem = os.path.splitext(os.path.basename(path))[0]
        data_file = stem + ".data.npz"
        target = os.path.join(os.path.dirname(os.path.abspath(path)), data_file)
        np.savez(target, **arrays)
        data = {"file": data_file, "sha256": file_sha256(target)}
        has_test = X_test is not None
    else:
        data = _plain(dict(data), "data")
        has_test = bool((data.get("args") or {}).get("test_rows"))
    outputs = {"digest": bool(digest), "score": bool(score)}
    if method:
        outputs["predict"] = {"method": method, "on": predict_on or ("test" if has_test else "train")}
    out = {"format": FORMAT, "model": spec, "seed": _plain(seed, "the seed"), "fit": fit, "data": data,
           "outputs": outputs}
    check_recipe(out)
    if path is not None:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(out, stream, indent=1, sort_keys=True)
            stream.write("\n")
    return out


# --------------------------------------------------------------------------
# one device: build, fit, answer (in a fresh process)
# --------------------------------------------------------------------------

def _apply_calls(ml, model, calls):
    """Apply `[name, *args, {kwargs}]` layer calls; an argument
    `{"sequential": [...]}` is a sub-model of the same calls
    (residual_block, gated, parallel)."""
    for call in calls:  # verify tooling: over the recipe's layer calls
        if isinstance(call, str):
            call = [call]
        if not isinstance(call, list) or not call or not isinstance(call[0], str):
            raise Refused(f"a layer call is [name, args..., {{kwargs}}], not {call!r}")
        name, args, kwargs = call[0], list(call[1:]), {}
        if args and isinstance(args[-1], dict) and "sequential" not in args[-1]:
            kwargs = args.pop()
        if name.startswith("_") or not callable(getattr(model, name, None)) or name in (
                "fit", "compile", "predict", "predict_proba", "save_checkpoint", "export", "load_state_dict"):
            raise Refused(f"Sequential has no layer method {name!r}")
        args = [_apply_calls(ml, ml.Sequential(), a["sequential"]) if isinstance(a, dict) and "sequential" in a
                else a for a in args]
        model = getattr(model, name)(*args, **kwargs)
    return model


def build_model(ml, recipe):
    """The unfitted model the recipe describes, and the fit keyword arguments."""
    spec = recipe["model"]
    fit = dict(recipe.get("fit") or {})
    seed = recipe.get("seed")
    if "estimator" in spec:
        name = spec["estimator"]
        cls = getattr(ml, name, None) if not name.startswith("_") else None
        if cls is None or not isinstance(cls, type):
            raise Refused(f"mojolearn has no estimator class {name!r}")
        params = dict(spec.get("params") or {})
        if seed is not None:
            try:
                takes = SEED_PARAM in cls().get_params(deep=False)
            except Exception:
                takes = True   # the constructor below says so by name if it does not
            if not takes:
                raise Refused(f"{name} takes no {SEED_PARAM}, so the recipe's seed has nowhere to go; set seed null")
            params[SEED_PARAM] = seed
        try:
            return cls(**params), fit
        except Exception as exc:
            raise Refused(f"{name}(**params) refused: {type(exc).__name__}: {exc}"[:400]) from exc
    seq = spec["sequential"]
    shape = seq.get("input_shape")
    model = ml.Sequential(seq.get("numeric_mode"), input_shape=None if shape is None else tuple(shape))
    if "layers" in seq:
        model = _apply_calls(ml, model, seq["layers"])
        model = model.compile(**dict(seq.get("compile") or {}))
    else:
        # the model's own layer table and compile options, set as the
        # check's device twin sets them (Sequential._mojolearn_device_twin)
        fields = object.__getattribute__(model, "__dict__")
        fields["_layers"] = [tuple(e[:4]) + (tuple(e[4]),) if len(e) > 4 else tuple(e) for e in seq["description"]]
        fields["_options"] = dict(seq["options"])
    if seed is not None:
        fit["seed"] = seed
    return model, fit


def _rows(data, on):
    if on == "test":
        if "X_test" not in data:
            raise Refused("outputs ask for the test rows, but the data have no X_test (a generator's test_rows)")
        return data["X_test"]
    return data["X"]


def run_recipe(ml, recipe, data):
    """Fit the recipe on this process's device and answer its outputs.
    Returns `[(name, value)]`: the digest string, the prediction array, the
    score as a float64 array of one."""
    np = _np()
    cb.refuse_fast(ml)
    model, fit = build_model(ml, recipe)
    try:
        cb.refuse_fast(ml, model)
    except cb.NotChecked as exc:
        raise Refused(str(exc)) from exc
    args = (data["X"],) + ((data["y"],) if "y" in data else ())
    try:
        fitted = model.fit(*args, **fit)
    except Exception as exc:
        raise CannotRun(f"fit raised {type(exc).__name__}: {exc}"[:600]) from exc
    model = fitted if fitted is not None else model
    outputs = recipe.get("outputs") or {}
    answers = []
    if outputs.get("digest"):
        try:
            answers.append(("digest", str(model.digest_)))
        except AttributeError as exc:
            raise CannotRun(f"the fitted {type(model).__name__} has no digest_: {exc}"[:600]) from exc
    predict = outputs.get("predict")
    on = "train"
    if predict:
        predict = predict if isinstance(predict, dict) else {"method": "predict"}
        method, on = predict.get("method", "predict"), predict.get("on", "train")
        if not callable(getattr(model, method, None)):
            raise Refused(f"{type(model).__name__} has no method {method!r}")
        try:
            result = getattr(model, method)(_rows(data, on))
        except Refused:
            raise
        except Exception as exc:
            raise CannotRun(f"{method} raised {type(exc).__name__}: {exc}"[:600]) from exc
        for k, a in enumerate(cb.as_arrays(result)):
            answers.append((f"{method}[{k}]" if isinstance(result, (tuple, list)) else method, a))
    if outputs.get("score"):
        # on the training rows: they are the rows with targets
        try:
            value = model.score(*args)
        except Exception as exc:
            raise CannotRun(f"score raised {type(exc).__name__}: {exc}"[:600]) from exc
        answers.append(("score", np.asarray([value], dtype=np.float64)))
    return answers


def child_main(directory, ml=None):
    """One device of a recipe, in its own process: read request.json, fit,
    answer, and write result.json and one .npy per array output."""
    np = _np()
    answer = dict(format=CHILD_FORMAT, ran=False)
    try:
        with open(os.path.join(directory, "request.json"), encoding="utf-8") as stream:
            request = json.load(stream)
        if ml is None:
            import mojolearn as ml
        device = request["device"]
        vendor = ml.vendor()
        if (device == "cpu") != (vendor == "cpu"):
            raise CannotRun(f"the {device} process loaded the {vendor!r} binaries")
        data = load_data(request["recipe"]["data"], request.get("base"))
        answers = run_recipe(ml, request["recipe"], data)
        side = "cpu" if device == "cpu" else "gpu"
        env = cb.side_environment(ml, side)
        env.pop("side", None)
        outputs = []
        for k, (name, value) in enumerate(answers):
            if isinstance(value, str):
                outputs.append(dict(name=name, kind="digest", value=value))
            else:
                np.save(os.path.join(directory, f"out_{k}.npy"), value, allow_pickle=value.dtype == object)
                outputs.append(dict(name=name, kind="array", file=f"out_{k}.npy", **cb.describe_output(value)))
        answer.update(ran=True, device=device, data_sha256=data_sha256(data), env=env, outputs=outputs,
                      version=getattr(ml, "__version__", None), vendor=vendor)
    except Refused as exc:
        answer.update(refused=True, reason=str(exc))
    except Exception as exc:
        answer["reason"] = str(exc) if isinstance(exc, (CannotRun, cb.NotChecked)) else \
            f"{type(exc).__name__}: {exc}"[:600]
    temporary = os.path.join(directory, "result.json.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(answer, stream)
    os.replace(temporary, os.path.join(directory, "result.json"))
    return 0


def device_env(device):
    """The environment of a fresh process on `device`: the CPU binaries are
    selected with MOJOLEARN_VENDOR=cpu; the GPU process takes the
    installation's own."""
    env = dict(os.environ)
    if device == "cpu":
        env["MOJOLEARN_VENDOR"] = "cpu"
    elif env.get("MOJOLEARN_VENDOR") == "cpu":
        env.pop("MOJOLEARN_VENDOR")
    return env


def spawn_children(jobs):
    """Run one fresh process per (device, directory), all at once (the
    devices are independent), and return {device: (exit code, output tail)}."""
    # an isolated parent (`python -I`) starts isolated children: the same package, never the working directory's
    command = [sys.executable] + (["-I"] if sys.flags.isolated else []) + ["-m", "mojolearn._verify_recipe"]
    procs = {device: subprocess.Popen(command + [directory],
                                      env=device_env(device), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
             for device, directory in jobs}
    out = {}
    for device, proc in procs.items():
        text, _ = proc.communicate()
        out[device] = (proc.returncode, (text or "")[-600:])
    return out


def devices_of(ml):
    """The devices this installation has: the CPU, and the GPU when its
    binaries are not the CPU's."""
    return ["cpu"] if ml.vendor() == "cpu" else ["cpu", "gpu"]


def _encode(a):
    if a.dtype == object or a.nbytes > RECORD_VALUES_MAX:
        return None
    return base64.b64encode(a.tobytes()).decode("ascii")


def _decode(output):
    np = _np()
    if output.get("values") is None:
        return None
    raw = base64.b64decode(output["values"])
    return np.frombuffer(raw, dtype=np.dtype(output["dtype"])).reshape(output["shape"]).copy()


def fit_on_devices(ml, recipe, base=None, devices=None, spawn=None):
    """Fit the recipe from scratch in one fresh process per device. Returns
    `({device: result}, {device: [arrays]})`; a result has `ran` and either
    the outputs or the reason."""
    np = _np()
    devices = devices or devices_of(ml)
    results, arrays = {}, {}
    with tempfile.TemporaryDirectory(prefix="mojolearn_recipe_") as tmp:
        jobs = []
        for device in devices:  # verify tooling: one request per device
            directory = os.path.join(tmp, device)
            os.makedirs(directory)
            with open(os.path.join(directory, "request.json"), "w", encoding="utf-8") as stream:
                json.dump(dict(format=CHILD_FORMAT, device=device, recipe=recipe,
                               base=os.path.abspath(base) if base else None), stream)
            jobs.append((device, directory))
        ended = (spawn or spawn_children)(jobs)
        for device, directory in jobs:
            path = os.path.join(directory, "result.json")
            if not os.path.exists(path):
                code, tail = ended.get(device, (None, ""))
                results[device] = dict(ran=False, reason=f"the {device} process ended (exit {code}) without a "
                                                         f"result: {tail.strip()[-300:]}")
                continue
            with open(path, encoding="utf-8") as stream:
                answer = json.load(stream)
            if answer.get("format") != CHILD_FORMAT:
                answer = dict(ran=False, reason="the device process answered in another format")
            mine = []
            for output in answer.get("outputs", ()):
                if output["kind"] == "array":
                    a = np.load(os.path.join(directory, output.pop("file")), allow_pickle=output["dtype"] == "object")
                    output["values"] = _encode(a)
                    mine.append(a)
                else:
                    mine.append(None)
            results[device] = answer
            arrays[device] = mine
    return results, arrays


# --------------------------------------------------------------------------
# compare: every result against the first
# --------------------------------------------------------------------------

def _same(a, b):
    if a["name"] != b["name"] or a["kind"] != b["kind"]:
        return False
    if a["kind"] == "digest":
        return a["value"] == b["value"]
    return (a["sha256"], a["dtype"], a["shape"]) == (b["sha256"], b["dtype"], b["shape"])


def _difference(a, b, a_array, b_array):
    """What differs between two outputs of the same name, as words."""
    if a["kind"] == "digest":
        return f"digest_ {a['value']} against {b['value']}"
    if a_array is None:
        a_array = _decode(a)
    if b_array is None:
        b_array = _decode(b)
    if a_array is None or b_array is None:
        return (f"sha256 {a['sha256'][:16]}... against {b['sha256'][:16]}... (the values are not in the record: "
                f"an output over {RECORD_VALUES_MAX} bytes keeps its hash only)")
    first = cb._first_difference([a_array], [b_array])
    if first is None:
        return "the same bytes under different hashes (a record was edited)"
    if first["why"] in ("dtype", "shape"):
        return f"a different {first['why']}: {first['gpu']} against {first['cpu']}"
    index = ", ".join(str(i) for i in first["index"])
    return f"first differing element at [{index}] (flat {first['flat']}): {first['gpu']} against {first['cpu']}"


def compare_results(entries):
    """`entries` is `[(label, result, arrays or None)]`. IDENTICAL when every
    result answers every output with the same bytes as the first; else
    DIVERGENT with the first differing output and value."""
    ran = [e for e in entries if e[1].get("ran")]
    if len(ran) != len(entries):
        missing = [f"{label}: {r.get('reason')}" for label, r, _ in entries if not r.get("ran")]
        return dict(verdict="NOT RUN", identical=False, reason="; ".join(missing), compared=len(ran))
    ref_label, ref, ref_arrays = ran[0]
    for label, result, arrays in ran[1:]:
        names_a = [o["name"] for o in ref["outputs"]]
        names_b = [o["name"] for o in result["outputs"]]
        if names_a != names_b:
            return dict(verdict="DIVERGENT", identical=False, compared=len(ran),
                        first=dict(a=ref_label, b=label, output=None,
                                   detail=f"different outputs: {names_a} against {names_b}"))
        for k, (oa, ob) in enumerate(zip(ref["outputs"], result["outputs"])):
            if not _same(oa, ob):
                detail = _difference(oa, ob, ref_arrays[k] if ref_arrays else None, arrays[k] if arrays else None)
                return dict(verdict="DIVERGENT", identical=False, compared=len(ran),
                            first=dict(a=ref_label, b=label, output=oa["name"], detail=detail))
    outputs = len(ref["outputs"])
    return dict(verdict="IDENTICAL", identical=True, compared=len(ran), outputs=outputs)


def verdict_line(verdict):
    if verdict["verdict"] == "IDENTICAL":
        return (f"IDENTICAL: {verdict['compared']} fit(s) from scratch, {verdict['outputs']} output(s) each, "
                "the same bits")
    if verdict["verdict"] == "NOT RUN":
        return f"NOT RUN: {verdict['reason']}"
    first = verdict["first"]
    what = f"output {first['output']}" if first.get("output") else "the outputs"
    return f"DIVERGENT: {first['a']} and {first['b']} differ in {what}: {first['detail']}"


# --------------------------------------------------------------------------
# the record and --against
# --------------------------------------------------------------------------

def make_record(recipe, results, verdict, version, vendor):
    return dict(format=RECORD_FORMAT, recipe=recipe, recipe_sha256=recipe_sha256(recipe),
                data_sha256=next((r.get("data_sha256") for r in results.values() if r.get("ran")), None),
                version=version, vendor=vendor, machine=platform.machine(), system=platform.system(),
                made=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                results=results, verdict=verdict)


def check_against(mine, other):
    """Raise Refused unless `other` is a record of the same recipe, the same
    data and the same version as `mine`."""
    if not isinstance(other, dict) or other.get("format") != RECORD_FORMAT:
        raise Refused(f"the other file is not a recipe record (format {RECORD_FORMAT!r})")
    if other.get("version") != mine.get("version"):
        raise Refused(f"the other record was made with mojolearn {other.get('version')}, this one with "
                      f"{mine.get('version')}: bits are promised within one version, so two versions are not "
                      "compared")
    if other.get("recipe_sha256") != mine.get("recipe_sha256"):
        raise Refused("the other record is of another recipe (its recipe hash differs), so nothing is compared")
    if other.get("data_sha256") != mine.get("data_sha256"):
        raise Refused(f"the other record fitted other data (data hash {other.get('data_sha256')} against "
                      f"{mine.get('data_sha256')}), so nothing is compared")


def against_entries(mine, arrays, other, other_label="other"):
    entries = [(f"this machine {d}", r, arrays.get(d)) for d, r in mine["results"].items()]
    where = other.get("vendor") or "?"
    entries += [(f"{other_label} ({where}) {d}", r, None) for d, r in other["results"].items()]
    return entries


# --------------------------------------------------------------------------
# the self-test: the comparison must be able to fail
# --------------------------------------------------------------------------

def _nudged(results, arrays):
    """A copy of the results with one value moved by one ulp on the last
    device: the first float element of the first float array output, else one
    hex digit of the digest. Returns (results, arrays, what was moved)."""
    np = _np()
    results = copy.deepcopy(results)
    arrays = {d: list(a) for d, a in arrays.items()}
    device = list(results)[-1]
    outputs = results[device]["outputs"]
    for k, output in enumerate(outputs):
        a = arrays[device][k]
        if a is not None and a.dtype.kind == "f" and a.size:
            # the lowest bit of element 0's word: one ulp for a finite value, another payload for a NaN
            # (nextafter leaves a NaN as it is, and a model whose answers are NaN must still be able to fail)
            b = a.copy()
            words = b.reshape(-1).view(np.dtype(f"u{b.dtype.itemsize}"))
            words[0] ^= 1
            arrays[device][k] = b
            output.update(cb.describe_output(b))
            output["values"] = _encode(b)
            return results, arrays, f"{device} {output['name']} element 0 moved by one ulp"
    for output in outputs:
        if output["kind"] == "digest":
            v = output["value"]
            output["value"] = v[:-1] + ("1" if v[-1:] != "1" else "0")
            return results, arrays, f"{device} digest_ last hex digit changed"
    return results, arrays, None


def self_test(ml, recipe, base, results, arrays, version, vendor):
    """Four controls on a recipe that has run; each must come out as stated.
    Returns `[(control, passed, words)]`."""
    controls = []
    nudged, nudged_arrays, what = _nudged(results, arrays)
    if len(results) == 1:
        # one device here: the moved copy stands for a second machine
        nudged = {"cpu (moved copy)": nudged[list(nudged)[0]]}
        nudged_arrays = {"cpu (moved copy)": nudged_arrays[list(nudged_arrays)[0]]}
        entries = [(d, r, arrays.get(d)) for d, r in results.items()] + \
                  [(d, r, nudged_arrays.get(d)) for d, r in nudged.items()]
    else:
        entries = [(d, r, nudged_arrays.get(d)) for d, r in nudged.items()]
    verdict = compare_results(entries)
    controls.append(("one-ulp change is DIVERGENT", what is not None and verdict["verdict"] == "DIVERGENT",
                     f"{what}: {verdict_line(verdict)}"))
    fast = copy.deepcopy(recipe)
    if "estimator" in fast["model"]:
        fast["model"].setdefault("params", {})["numeric_mode"] = "fast"
    else:
        fast["model"]["sequential"]["numeric_mode"] = "fast"
    try:
        check_recipe(fast)
        controls.append(("fast recipe is refused", False, "a fast recipe was accepted"))
    except Refused as exc:
        controls.append(("fast recipe is refused", True, str(exc)))
    with tempfile.TemporaryDirectory(prefix="mojolearn_recipe_selftest_") as tmp:
        np = _np()
        data = load_data(recipe["data"], base)
        path = os.path.join(tmp, "data.npz")
        np.savez(path, **data)
        wrong = dict(copy.deepcopy(recipe), data={"file": path, "sha256": "0" * 64})
        try:
            load_data(wrong["data"], tmp)
            controls.append(("wrong data hash is refused", False, "a data file with another hash was accepted"))
        except Refused as exc:
            controls.append(("wrong data hash is refused", True, str(exc)))
    record = make_record(recipe, results, compare_results([(d, r, arrays.get(d)) for d, r in results.items()]),
                         version, vendor)
    other = dict(copy.deepcopy(record), version=f"{version}+another")
    try:
        check_against(record, other)
        controls.append(("another version is refused", False, "a record of another version was accepted"))
    except Refused as exc:
        controls.append(("another version is refused", True, str(exc)))
    return controls


# --------------------------------------------------------------------------
# shipped recipes: the covered set (`--recipe example:<name>`)
# --------------------------------------------------------------------------

def _est(name, params, data, seed=None, predict="predict", on="test", score=True):
    return {"format": FORMAT, "model": {"estimator": name, "params": params}, "seed": seed, "fit": {},
            "data": data, "outputs": {"digest": True, "predict": {"method": predict, "on": on}, "score": score}}


def _seq(shape, layers, data_rows=32, epochs=2, batch=16, seed=0):
    return {"format": FORMAT,
            "model": {"sequential": {"numeric_mode": "identical", "input_shape": list(shape), "layers": layers,
                                     "compile": {"loss": "mse", "optimizer": "sgd", "lr": 0.01}}},
            "seed": seed, "fit": {"epochs": epochs, "batch_size": batch, "shuffle": False},
            "data": {"generator": "sequence", "args": {"rows": data_rows, "test_rows": 16, "shape": list(shape),
                                                        "seed": 7}},
            "outputs": {"digest": True, "predict": {"method": "predict", "on": "test"}, "score": False}}


EXAMPLES = {
    "ridge": _est("Ridge", {"alpha": 1.0},
                  {"generator": "regression", "args": {"rows": 512, "features": 8, "test_rows": 64, "seed": 0}}),
    "logistic": _est("LogisticRegression", {"C": 1.0, "max_iter": 50},
                     {"generator": "classification", "args": {"rows": 512, "features": 8, "classes": 3,
                                                              "test_rows": 64, "seed": 1}}),
    "kmeans": _est("KMeans", {"n_clusters": 3, "n_init": 1, "max_iter": 20},
                   {"generator": "blobs", "args": {"rows": 600, "features": 4, "centers": 3, "test_rows": 60,
                                                   "seed": 2}}, seed=0, score=False),
    "gbdt-categorical": _est("GradientBoosting", {"n_estimators": 8, "max_depth": 3, "cat_features": [0, 1]},
                             {"generator": "categorical", "args": {"rows": 800, "numeric": 4, "categorical": 2,
                                                                   "levels": 5, "test_rows": 100, "seed": 3}},
                             seed=0, score=False),
    # residual(source) adds the output of layer `source`: the blocks' skips. (residual_block with a Sequential
    # body is refused by the Sequential input admission at 555135703f; reported, not worked around here.)
    "seq-transformer": _seq((8, 16), [
        ["linear", 16], ["rms_norm"], ["attention", 2, {"kv_heads": 1}], ["residual", 0],
        ["rms_norm"], ["linear", 32], ["silu"], ["linear", 16], ["residual", 3],
        ["flatten"], ["linear", 1]]),
    "seq-mamba2": _seq((8, 2), [["linear", 321], ["softplus"], ["ssd_scan", 1], ["flatten"], ["linear", 1]]),
    "seq-recurrent": _seq((8, 4), [["lstm", 8], ["gru", 6], ["flatten"], ["linear", 1]]),
    "seq-conv1d-residual": _seq((4, 8), [
        ["conv1d", 8, 3, {"padding": 1}], ["silu"],
        ["conv1d", 8, 3, {"padding": 1}], ["silu"], ["conv1d", 8, 3, {"padding": 1}], ["residual", 1],
        ["flatten"], ["linear", 1]]),
}

# nn-attention begin: classic attention (positional_encoding, the encoder and decoder layers, cross-attention)
EXAMPLES["seq-attention"] = _seq((6, 4), [
    ["linear", 8], ["positional_encoding", 8, {"dropout": 0.0}],
    ["transformer_encoder_layer", 8, 2, 16, {"dropout": 0.0}],
    ["transformer_decoder_layer", 8, 2, 16, {"dropout": 0.0, "memory": 0, "tgt_is_causal": True}],
    ["multihead_attention", 8, 2, {"kdim": 4, "vdim": 4, "source": -1}],
    ["flatten"], ["linear", 1]])
# nn-attention end


def load_recipe(name):
    """The recipe and the directory its data file is relative to: a JSON file,
    or `example:<name>` for a shipped one."""
    if name.startswith("example:"):
        key = name.split(":", 1)[1]
        if key not in EXAMPLES:
            raise Refused(f"no shipped recipe {key!r}; they are {', '.join(EXAMPLES)}")
        return copy.deepcopy(EXAMPLES[key]), None, "example-" + key
    with open(name, encoding="utf-8") as stream:
        try:
            value = json.load(stream)
        except ValueError as exc:
            raise Refused(f"{name} is not JSON: {exc}") from exc
    stem = name[:-5] if name.endswith(".json") else name
    return value, os.path.dirname(os.path.abspath(name)), stem


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------

def _result_lines(results):
    lines = []
    for device, r in results.items():
        if not r.get("ran"):
            lines.append(f"device    {device}: {'REFUSED' if r.get('refused') else 'NOT RUN'}: {r.get('reason')}")
            continue
        env = r.get("env") or {}
        lines.append(f"device    {device}: {env.get('device')} (binaries {r.get('vendor')}, "
                     f"mojolearn {r.get('version')}, {env.get('mode')} mode)")
        for o in r["outputs"]:
            if o["kind"] == "digest":
                lines.append(f"  {o['name']:<10} {o['value']}")
            else:
                shape = "x".join(str(n) for n in o["shape"]) or "scalar"
                lines.append(f"  {o['name']:<10} {shape} {o['dtype']} sha256 {o['sha256']}  "
                             f"first {', '.join(o['head'])}")
    return lines


def cmd_recipe(args):
    """`verify --recipe R [--against RECORD] [--record-out PATH] [--self-test] [--json]`."""
    json_out = getattr(args, "json", False)
    report = dict(format=RECORD_FORMAT + "-report", recipe=args.recipe)
    lines = [f"# mojolearn verify --recipe {args.recipe}", ""]

    def finish(code, **extra):
        report.update(extra, exit=code)
        if json_out:
            print(json.dumps(report, indent=1, sort_keys=True, default=str))
        else:
            print("\n".join(lines))
        return code

    try:
        recipe, base, stem = load_recipe(args.recipe)
        check_recipe(recipe)
        other = None
        if getattr(args, "against", None):
            with open(args.against, encoding="utf-8") as stream:
                other = json.load(stream)
            if not isinstance(other, dict) or other.get("format") != RECORD_FORMAT:
                raise Refused(f"{args.against} is not a recipe record (format {RECORD_FORMAT!r})")
            if other.get("recipe_sha256") != recipe_sha256(recipe):
                raise Refused(f"{args.against} is a record of another recipe (its recipe hash differs), "
                              "so nothing is compared")
    except Refused as exc:
        lines.append(f"REFUSED: {exc}")
        return finish(EXIT_REFUSED, reason=str(exc))
    except OSError as exc:
        print(f"USAGE: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        import mojolearn as ml
        cb.refuse_fast(ml)
    except cb.NotChecked as exc:
        lines.append(f"REFUSED: {exc}")
        return finish(EXIT_REFUSED, reason=str(exc))
    version, vendor = getattr(ml, "__version__", None), ml.vendor()
    devices = devices_of(ml)
    lines.append(f"recipe    sha256 {recipe_sha256(recipe)}")
    lines.append(f"devices   {', '.join(devices)} (one fresh process each, fitted from scratch)")
    results, arrays = fit_on_devices(ml, recipe, base, devices)
    refused = [r for r in results.values() if r.get("refused")]
    lines += _result_lines(results)
    if refused:
        lines += ["", f"REFUSED: {refused[0].get('reason')}"]
        return finish(EXIT_REFUSED, reason=refused[0].get("reason"), results=results)
    verdict = compare_results([(d, r, arrays.get(d)) for d, r in results.items()])
    record = make_record(recipe, results, verdict, version, vendor)
    out_path = getattr(args, "record_out", None) or stem + ".record.json"
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)
        stream.write("\n")
    if len(devices) == 1 and verdict["identical"]:
        lines += ["", "this installation has one device (the CPU), so there is no second device to compare here; "
                      "the record holds its result for a record of another machine or vendor (--against)"]
    else:
        lines += ["", "this installation: " + verdict_line(verdict)]
    lines.append(f"record    {out_path}")
    report.update(record=out_path, verdict=verdict, results=results)
    code = EXIT_IDENTICAL if verdict["identical"] else (EXIT_CANNOT_RUN if verdict["verdict"] == "NOT RUN"
                                                        else EXIT_DIVERGENT)
    if code == EXIT_CANNOT_RUN:
        return finish(code)
    if other is not None:
        try:
            check_against(record, other)
        except Refused as exc:
            lines.append(f"REFUSED --against {args.against}: {exc}")
            return finish(EXIT_REFUSED, reason=str(exc))
        cross = compare_results(against_entries(record, arrays, other, args.against))
        lines.append(f"against   {args.against}: " + verdict_line(cross))
        report["against"] = cross
        if not cross["identical"]:
            code = EXIT_DIVERGENT
    if getattr(args, "self_test", False):
        controls = self_test(ml, recipe, base, results, arrays, version, vendor)
        lines += ["", "self-test (each control must come out as stated):"]
        lines += [f"  {'PASS' if ok else 'FAIL'}  {name}: {words}" for name, ok, words in controls]
        passed = all(ok for _, ok, _ in controls)
        lines.append(f"SELF-TEST {'PASS' if passed else 'FAIL'}: {sum(ok for _, ok, _ in controls)} of "
                     f"{len(controls)} controls")
        report["self_test"] = [dict(control=n, passed=ok, words=w) for n, ok, w in controls]
        if not passed:
            code = EXIT_DIVERGENT
    return finish(code)


if __name__ == "__main__":
    # one device of a recipe: `python -m mojolearn._verify_recipe DIRECTORY`
    sys.exit(child_main(sys.argv[1]))
