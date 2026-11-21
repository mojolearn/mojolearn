# SPDX-License-Identifier: Apache-2.0
"""A receipt for a fit, and `reproduce`: fit again and compare with it.

`estimator.receipt(X, y)` after `fit` writes down what was fitted on what:
the Mojolearn version, the numeric mode, the class, every constructor
parameter, a fingerprint of the data, the hash of the fitted state and the
hash of what the model answers for `X`. `python -m mojolearn reproduce
RECEIPT.json --X FILE [--y FILE]` and `mojolearn.reproduce(receipt, X, y)`
build the same estimator, fit it on the data given now and say whether the
fit is the recorded one, and when it is not, which part differs.

The three user checks, so they do not overlap:

    python -m mojolearn verify   is the library right on this machine
    estimator.check(X)           do the GPU and the CPU agree on my data
    reproduce                    does my fit today equal my recorded fit

`identical` mode only: there the answer is one hash and no tolerance. `fast`
makes no promise about bits, so a receipt is refused there in words.

This is verification tooling: it runs around product calls and never inside
one. The fit, the save of the fitted model and the hash of the saved file run
in Mojo (`state_file_sha256` of the host core binding, the door `check`
already opens for the activity record); Python wires the call, reads the
receipt and prints. The data fingerprint and the prediction hash are the
forms `check` prints (`_check_both.describe_output`, `outputs_digest`): the
SHA-256 of the array bytes. There is no second hashing scheme here.

Like `_check_both.py` this module is handed the imported package as `ml`, so
the tool tests run it without a built binding.
"""
import json
import os
import sys
import tempfile

from . import _check_both as cb

RECEIPT_FORMAT = "mojolearn.receipt.v1"
FORMAT = "mojolearn.reproduce.v1"
#: the parts compared, in the order a difference is named
PARTS = ("version", "parameters", "data", "fitted_state", "predictions")
#: the native entry that hashes a saved model file (bindings/_mojolearn_core_host.mojo)
STATE_ENTRY = "state_file_sha256"
#: for an estimator with neither predict nor transform (it answers at fit):
#: the fitted attribute hashed in the answer's place, the first it has
FITTED_ANSWERS = ("labels_", "embedding_")

EXIT_SAME = 0
EXIT_DIFFERENT = 1
EXIT_USAGE = 2
EXIT_REFUSED_FAST = 3
EXIT_CANNOT_RUN = 4

FAST_REFUSAL = ("a receipt records bits, and only identical mode promises bits; "
                "there is no receipt and no reproduce in fast mode")


class NotReproducible(cb.NotChecked):
    """No receipt could be written, or nothing could be compared; the message
    says why. Never a verdict."""


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

def canonical(value, name="parameter"):
    """A constructor parameter as plain JSON: None, bool, int, float, str,
    lists, tuples (tagged, so the same tuple is rebuilt) and dictionaries with
    text keys. Anything else (an estimator, a callable, an array) has no bits
    to record and is refused by name."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [canonical(v, name) for v in value]}
    if isinstance(value, list):
        return [canonical(v, name) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise NotReproducible(f"{name} is a dictionary with keys that are not text")
        return {k: canonical(value[k], name) for k in sorted(value)}
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return canonical(item(), name)       # a NumPy scalar: its Python value
    raise NotReproducible(f"{name} is a {type(value).__name__}, not a plain value; a receipt needs "
                          "constructor parameters that are numbers, text, None, or lists of them")


def revive(value):
    """The parameter value `canonical` wrote."""
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(revive(v) for v in value["__tuple__"])
        return {k: revive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [revive(v) for v in value]
    return value


def _text(value):
    return json.dumps(value, sort_keys=True)


def parameters_of(est):
    """Every constructor parameter (`get_params`) in canonical order and form,
    and the device the estimator names. `device` says where a fit runs, not
    what is fitted: a receipt holds on any machine, so it is recorded beside
    the parameters and never compared."""
    raw = cb.estimator_spec(est)["params"]
    device = raw.pop("device", None) if _has_device(type(est)) else None
    cls = type(est).__name__
    return {name: canonical(raw[name], f"{cls}({name}=...)") for name in sorted(raw)}, device


def _has_device(cls):
    return bool(getattr(cls, "_mojolearn_device", False))


# --------------------------------------------------------------------------
# data fingerprint and prediction hash: the forms `check` prints
# --------------------------------------------------------------------------

def fingerprint(value, name):
    """shape, dtype, byte count and SHA-256 of the bytes of one input array
    (`_check_both.describe_output`, without the first values)."""
    arrays = cb.as_arrays(value)
    if len(arrays) != 1:
        raise NotReproducible(f"{name} is {len(arrays)} arrays; a receipt fingerprints one array for {name}")
    a = arrays[0]
    if a.dtype == object and a.ndim != 1:
        raise NotReproducible(f"{name} is not a numeric array (dtype object, shape {tuple(a.shape)}); "
                              "a receipt fingerprints the bytes of a dense array")
    described = cb.describe_output(a)
    return dict(shape=described["shape"], dtype=described["dtype"], sha256=described["sha256"],
                bytes=int(a.nbytes))


def data_fingerprint(X, y):
    """The fingerprints of X and of y (None when there is no y). The two
    arrays are independent, so they are hashed on two threads (SHA-256 over a
    large buffer runs without the interpreter lock); one array is one chain."""
    if y is None:
        return dict(X=fingerprint(X, "X"), y=None)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fx = pool.submit(fingerprint, X, "X")
        fy = pool.submit(fingerprint, y, "y")
        return dict(X=fx.result(), y=fy.result())


def default_method(est, method=None):
    """The method whose answer for X is hashed: the one named, else the first
    of `predict`, `transform` the estimator has, else None."""
    if method is not None:
        if not callable(getattr(est, method, None)):
            raise NotReproducible(f"{type(est).__name__} has no method {method!r}")
        return method
    return next((m for m in cb.DEFAULT_METHODS if callable(getattr(est, m, None))), None)


def prediction_hash(est, X, method=None, attribute=None):
    """What the fitted model answers for X, hashed as `check` hashes it. An
    estimator with neither predict nor transform answered at fit: its
    `labels_` or `embedding_` is hashed (`attribute` names it in the record)."""
    if attribute is None:
        method = default_method(est, method)
    if attribute is None and method is None:
        attribute = next((a for a in FITTED_ANSWERS if getattr(est, a, None) is not None), None)
        if attribute is None:
            return dict(present=False, reason=f"{type(est).__name__} has neither predict nor transform, "
                                              f"and no {' or '.join(FITTED_ANSWERS)}")
    if attribute is not None:
        value = getattr(est, attribute, None)
        if value is None:
            return dict(present=False, reason=f"{type(est).__name__} has no {attribute} after fit")
        arrays, method = cb.as_arrays(value), None
    else:
        arrays = cb.as_arrays(getattr(est, method)(X))
    outputs = []
    for a in arrays:
        described = cb.describe_output(a)
        outputs.append(dict(shape=described["shape"], dtype=described["dtype"], sha256=described["sha256"]))
    named = dict(method=method) if attribute is None else dict(attribute=attribute)
    return dict(present=True, sha256=cb.outputs_digest(arrays), outputs=outputs, **named)


# --------------------------------------------------------------------------
# the fitted-state hash: native save, native SHA-256 of the saved file
# --------------------------------------------------------------------------

def state_entry(ml):
    """The native function that hashes a saved model file, or None when this
    installation's binaries do not have it."""
    door = cb.activity_door(ml)
    entry = getattr(door, STATE_ENTRY, None)
    return entry if callable(entry) else None


def state_hash(ml, est):
    """The hash of the fitted state: the estimator saves itself with its own
    save (the deterministic native writer) into a scratch directory and the
    native runtime hashes the file. Equal fitted models write equal files, so
    this is one hash for the whole fitted state. `present` is False, with the
    reason, for an estimator with no save or an installation without the
    native hash entry; nothing is hashed in Python in its place.

    Shared: `check` can print the same hash for its two sides."""
    name = type(est).__name__
    form = cb.save_form(est)
    if form is None:
        return dict(present=False, reason=f"{name} has no save")
    entry = state_entry(ml)
    if entry is None:
        return dict(present=False, reason="this installation's binaries have no fitted-state hash entry")
    save, suffix = form
    with tempfile.TemporaryDirectory(prefix="mojolearn-receipt-") as directory:
        try:
            getattr(est, save)(os.path.join(directory, "model" + suffix))
        except Exception as exc:
            return dict(present=False, reason=f"{name}.{save} raised {type(exc).__name__}: {exc}"[:300])
        names = sorted(n for n in os.listdir(directory) if os.path.isfile(os.path.join(directory, n)))
        if not names:
            return dict(present=False, reason=f"{name}.{save} wrote no file")
        files = [dict(name=n, bytes=os.path.getsize(os.path.join(directory, n)),
                      sha256=str(entry(os.path.join(directory, n)))) for n in names]
        if len(files) == 1:
            return dict(present=True, form=save, sha256=files[0]["sha256"], bytes=files[0]["bytes"])
        # a save of several files: one hash over the list of names and hashes
        listing = os.path.join(directory, "mojolearn-receipt-listing")
        with open(listing, "w", encoding="utf-8", newline="\n") as fh:
            for f in files:
                fh.write(f"{f['name']} {f['sha256']}\n")
        return dict(present=True, form=save, sha256=str(entry(listing)),
                    bytes=sum(f["bytes"] for f in files), files=files)


# --------------------------------------------------------------------------
# the receipt
# --------------------------------------------------------------------------

def _mode_of(ml, est):
    mode = getattr(est, "numeric_mode", None)
    return mode if isinstance(mode, str) else ml.numeric_mode()


def make_receipt(ml, est, X, y=None, method=None, path=None):
    """The receipt of a fitted estimator: see the module docstring. Raises
    NotChecked (in words) in `fast` mode, for parameters that are not plain
    values, and for an estimator with neither a state hash nor an answer."""
    try:
        cb.refuse_fast(ml, est)
    except cb.NotChecked as exc:
        raise NotReproducible(f"{exc}. {FAST_REFUSAL[0].upper()}{FAST_REFUSAL[1:]}.") from exc
    parameters, device = parameters_of(est)
    data = data_fingerprint(X, y)
    predictions = prediction_hash(est, X, method)
    state = state_hash(ml, est)
    if not state["present"] and not predictions["present"]:
        raise NotReproducible(f"{type(est).__name__} has nothing to record: no fitted-state hash "
                              f"({state['reason']}) and no answer ({predictions['reason']})")
    receipt = dict(format=RECEIPT_FORMAT, mojolearn_version=str(ml.__version__), numeric_mode=_mode_of(ml, est),
                   estimator=type(est).__name__, parameters=parameters, data=data, fitted_state=state,
                   predictions=predictions, recorded_on=dict(vendor=str(ml.vendor()), device=device))
    if path is not None:
        save_receipt(receipt, path)
    return receipt


def save_receipt(receipt, path):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


def load_receipt(receipt):
    """A receipt from a dictionary or a JSON file, checked for its form."""
    if not isinstance(receipt, dict):
        with open(receipt, encoding="utf-8") as fh:
            receipt = json.load(fh)
    if not isinstance(receipt, dict) or receipt.get("format") != RECEIPT_FORMAT:
        raise NotReproducible(f"not a Mojolearn receipt (format {RECEIPT_FORMAT})")
    missing = [k for k in ("mojolearn_version", "numeric_mode", "estimator", "parameters", "data",
                           "fitted_state", "predictions") if k not in receipt]
    if missing:
        raise NotReproducible(f"the receipt has no {', '.join(missing)}")
    return receipt


# --------------------------------------------------------------------------
# reproduce
# --------------------------------------------------------------------------

def _build(ml, receipt, estimator, params):
    """The unfitted estimator to fit now: the one handed in, else the
    receipt's class with the receipt's parameters (and `params` over them:
    the parameters the caller's job uses today)."""
    if estimator is not None:
        return estimator
    cls = getattr(ml, receipt["estimator"], None)
    if cls is None or not callable(cls):
        raise NotReproducible(f"{receipt['estimator']} is not a class of this Mojolearn")
    built = {name: revive(value) for name, value in receipt["parameters"].items()}
    for name, value in (params or {}).items():
        # JSON has no tuple: a list given for a parameter the receipt holds as
        # a tuple is that tuple
        then = built.get(name)
        built[name] = tuple(value) if isinstance(then, tuple) and isinstance(value, list) else value
    if _has_device(cls) and "device" not in built and ml.vendor() == "cpu":
        built["device"] = "cpu"         # a CPU-only installation fits on the CPU
    try:
        return cls(**built)
    except Exception as exc:
        raise NotReproducible(f"{receipt['estimator']} cannot be built from the receipt's parameters: "
                              f"{type(exc).__name__}: {exc}"[:400]) from exc


def _compare_parameters(receipt, est):
    now, _ = parameters_of(est)
    then = receipt["parameters"]
    differing = []
    if type(est).__name__ != receipt["estimator"]:
        differing.append(dict(name="estimator class", receipt=receipt["estimator"], now=type(est).__name__))
    for name in sorted(set(then) | set(now)):
        a = _text(then[name]) if name in then else "(absent)"
        b = _text(now[name]) if name in now else "(absent)"
        if a != b:
            differing.append(dict(name=name, receipt=a, now=b))
    return dict(same=not differing, count=len(now), differing=differing)


def _compare_array(then, now):
    if then is None or now is None:
        return dict(same=then is None and now is None, differing=[] if then is now else ["presence"],
                    receipt=then, now=now)
    differing = [f for f in ("shape", "dtype") if then.get(f) != now.get(f)]
    if then.get("sha256") != now.get("sha256"):
        differing.append("bytes")
    return dict(same=not differing, differing=differing, receipt=then, now=now)


def _compare_hashed(then, now, what):
    """One recorded hash against today's. A part the receipt does not carry is
    not compared; a part the receipt carries and this installation cannot
    compute stops the run (it is never counted as the same)."""
    if not then.get("present"):
        return dict(compared=False, same=None, reason=f"the receipt has no {what}: {then.get('reason')}")
    if not now.get("present"):
        raise NotReproducible(f"the receipt carries a {what} and it cannot be computed here: {now.get('reason')}")
    part = dict(compared=True, same=then["sha256"] == now["sha256"], receipt=then["sha256"], now=now["sha256"])
    for key in ("form", "method", "attribute"):
        if key in then:
            part[key] = then[key]
    if not part["same"] and "outputs" in then and "outputs" in now:
        pairs = list(zip(then["outputs"], now["outputs"]))
        first = next((i for i, (a, b) in enumerate(pairs) if a != b), len(pairs))
        part["first_differing_output"] = first
    return part


def reproduce_with(ml, receipt, X, y=None, estimator=None, params=None):
    """Fit again and compare with the receipt. Returns the record: `ran`
    (False, with `reason`, when nothing was compared), `same`, `different`
    (the differing parts, in the order of PARTS), `first`, and one entry per
    part under `parts`."""
    record = dict(format=FORMAT, ran=False, same=None, different=[], first=None, parts={})
    try:
        receipt = load_receipt(receipt)
        record["estimator"] = receipt["estimator"]
        try:
            cb.refuse_fast(ml)
            if receipt["numeric_mode"] != "identical":
                raise cb.NotChecked(f"the receipt names numeric_mode={receipt['numeric_mode']!r}")
        except cb.NotChecked as exc:
            return dict(record, refused="fast", reason=f"{exc}. {FAST_REFUSAL}")
        parts = record["parts"]
        here = str(ml.__version__)
        parts["version"] = dict(same=here == receipt["mojolearn_version"], receipt=receipt["mojolearn_version"],
                                now=here)
        if not parts["version"]["same"]:
            # bits may change between versions: nothing after this line could
            # be read as a difference in the data or the model
            return dict(record, ran=True, same=False, different=["version"], first="version", stopped="version")
        est = _build(ml, receipt, estimator, params)
        try:
            cb.refuse_fast(ml, est)
        except cb.NotChecked as exc:
            return dict(record, refused="fast", reason=f"{exc}. {FAST_REFUSAL}")
        parts["parameters"] = _compare_parameters(receipt, est)
        now = data_fingerprint(X, y)
        dx = _compare_array(receipt["data"].get("X"), now["X"])
        dy = _compare_array(receipt["data"].get("y"), now["y"])
        parts["data"] = dict(same=dx["same"] and dy["same"], X=dx, y=dy)
        try:
            est.fit(X) if y is None else est.fit(X, y)
        except Exception as exc:
            raise NotReproducible(f"{type(est).__name__}.fit raised {type(exc).__name__}: {exc}"[:400]) from exc
        then_state, then_pred = receipt["fitted_state"], receipt["predictions"]
        parts["fitted_state"] = _compare_hashed(
            then_state, state_hash(ml, est) if then_state.get("present") else {}, "fitted-state hash")
        parts["predictions"] = _compare_hashed(
            then_pred, prediction_hash(est, X, then_pred.get("method"), then_pred.get("attribute"))
            if then_pred.get("present") else {},
            "prediction hash")
    except cb.NotChecked as exc:
        return dict(record, reason=str(exc))
    except (OSError, ValueError) as exc:        # an unreadable receipt, JSON that does not parse
        return dict(record, reason=f"{type(exc).__name__}: {exc}"[:400])
    if not (parts["fitted_state"]["compared"] or parts["predictions"]["compared"]):
        return dict(record, reason="the receipt carries neither a fitted-state hash nor a prediction hash")
    different = [name for name in PARTS if parts[name]["same"] is False]
    model_differs = any(name in different for name in ("fitted_state", "predictions"))
    record.update(ran=True, same=not different, different=different, first=different[0] if different else None,
                  # the data and the parameters are the recorded ones and the model is not
                  reproducibility_failure=model_differs and parts["parameters"]["same"] and parts["data"]["same"])
    return record


def reproduce(receipt, X, y=None, *, estimator=None):
    """Fit again on `X` (and `y`) and say whether the fit equals the one a
    receipt recorded.

    In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and
    AMD GPUs, and the CPU binaries are held to the same bits. So a fit of the
    same estimator on the same data gives the same model on any machine, and
    one hash decides it: there is no tolerance to choose. Use it as a
    regression test for a data pipeline: keep the receipt of a fit, fit again
    after the pipeline changes, and any difference means the training data
    changed.

    A receipt holds within one Mojolearn version. Bits may change between
    versions, so a receipt from another version is reported as different in
    version and nothing else is compared; fit once on the new version and
    keep the new receipt.

    Parameters
    ----------
    receipt : dict or path
        What `estimator.receipt(X, y)` returned, or the JSON file it saved.
    X : array-like
        The training rows to fit on now.
    y : array-like, optional
        The targets, for an estimator that is fitted with them.
    estimator : object, optional
        An unfitted estimator to fit, for example the one your training job
        builds. Its class and parameters are compared with the receipt's.
        Default: the receipt's class with the receipt's parameters.

    Returns
    -------
    dict
        `ran` is False, with a `reason`, when nothing was compared (the
        process or the estimator is not in `identical` mode, the receipt
        cannot be read, the fit raised). Otherwise `same` is True or False,
        `different` lists the parts that differ in this order: `'version'`,
        `'parameters'`, `'data'`, `'fitted_state'`, `'predictions'`, and
        `parts` holds the recorded and the present value of each.
        `reproducibility_failure` is True when the data and the parameters
        are the recorded ones and the model is not.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.Ridge(alpha=1.0).fit(X, X[:, 0])
    >>> receipt = model.receipt(X, X[:, 0], path="ridge-receipt.json")    # doctest: +SKIP
    >>> mojolearn.reproduce("ridge-receipt.json", X, X[:, 0])["same"]    # doctest: +SKIP
    True
    """
    import mojolearn as ml
    return reproduce_with(ml, receipt, X, y, estimator=estimator)


def receipt(estimator, X, y=None, *, method=None, path=None):
    """The receipt of a fitted estimator (`estimator.receipt` calls this)."""
    import mojolearn as ml
    return make_receipt(ml, estimator, X, y, method=method, path=path)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def _word(same):
    return "not compared" if same is None else ("same" if same else "DIFFERENT")


def _short(sha):
    return f"sha256 {sha[:16]}"


def _array_line(label, part):
    now = part["now"]
    if part["same"]:
        text = "none" if now is None else f"{tuple(now['shape'])} {now['dtype']} {_short(now['sha256'])}"
        return f"  {label:<13} same          {text}"
    if "presence" in part["differing"]:
        return (f"  {label:<13} DIFFERENT     the receipt has {'none' if part['receipt'] is None else 'one'}, "
                f"this run has {'none' if now is None else 'one'}")
    then, shown = part["receipt"], []
    for field in part["differing"]:
        if field == "bytes":
            shown.append(f"bytes: receipt {_short(then['sha256'])}, now {_short(now['sha256'])}")
        else:
            shown.append(f"{field}: receipt {then[field]}, now {now[field]}")
    return f"  {label:<13} DIFFERENT     " + "; ".join(shown)


def _hashed_line(label, part):
    if not part["compared"]:
        return f"  {label:<13} not compared  {part['reason']}"
    how = part.get("method") or part.get("attribute") or part.get("form") or ""
    if part["same"]:
        return f"  {label:<13} same          {how} {_short(part['now'])}".rstrip()
    return f"  {label:<13} DIFFERENT     {how} receipt {_short(part['receipt'])}, now {_short(part['now'])}"


def format_reproduce(record):
    """The lines a person reads: one per part, then the verdict."""
    name = record.get("estimator") or "estimator"
    if not record["ran"]:
        head = "REFUSED" if record.get("refused") else "NOT RUN"
        return f"reproduce {name}\n{head}: {record.get('reason')}\nNothing was compared: this is not a verdict."
    parts, lines = record["parts"], [f"reproduce {name}"]
    version = parts["version"]
    if not version["same"]:
        lines.append(f"  {'version':<13} DIFFERENT     receipt {version['receipt']}, this installation {version['now']}")
        lines.append("DIFFERENT: Mojolearn version. A receipt holds within one version: bits may change between "
                     f"versions, so nothing else was compared. Install mojolearn=={version['receipt']} to compare "
                     f"with this receipt, or fit once on {version['now']} and keep the new receipt.")
        return "\n".join(lines)
    lines.append(f"  {'version':<13} same          {version['now']}")
    parameters = parts["parameters"]
    if parameters["same"]:
        lines.append(f"  {'parameters':<13} same          {parameters['count']} parameters")
    else:
        shown = "; ".join(f"{d['name']}: receipt {d['receipt']}, now {d['now']}" for d in parameters["differing"][:6])
        lines.append(f"  {'parameters':<13} DIFFERENT     {shown}")
    lines.append(_array_line("data X", parts["data"]["X"]))
    lines.append(_array_line("data y", parts["data"]["y"]))
    lines.append(_hashed_line("fitted state", parts["fitted_state"]))
    lines.append(_hashed_line("predictions", parts["predictions"]))
    if record["same"]:
        lines.append("SAME: this fit equals the recorded fit, bit for bit.")
    elif record.get("reproducibility_failure"):
        lines.append("DIFFERENT: the data fingerprint and the parameters are the same and the model differs. "
                     "This is a reproducibility failure, not a change in your data.")
    else:
        names = dict(parameters="parameters", data="data fingerprint", fitted_state="fitted state",
                     predictions="predictions")
        lines.append("DIFFERENT: " + ", ".join(names[p] for p in record["different"]) + ".")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# python -m mojolearn reproduce
# --------------------------------------------------------------------------

def _load_array(path, member):
    """An array from a .npy file, or the member named `member` (else the
    first) of a .npz file: the files `verify --check` reads."""
    import numpy as np
    loaded = np.load(path, allow_pickle=False)
    if hasattr(loaded, "files"):
        names = list(loaded.files)
        if not names:
            raise ValueError(f"{path} holds no array")
        return loaded[member if member in names else names[0]]
    return loaded


def _load_xy(x_path, y_path):
    """X and y from the files named. With no --y, a .npz for --X that has a
    member `y` gives both."""
    import numpy as np
    X = _load_array(x_path, "X")
    if y_path:
        return X, _load_array(y_path, "y")
    loaded = np.load(x_path, allow_pickle=False)
    if hasattr(loaded, "files") and "y" in loaded.files and len(loaded.files) > 1:
        return X, loaded["y"]
    return X, None


def exit_code(record):
    if not record["ran"]:
        return EXIT_REFUSED_FAST if record.get("refused") else EXIT_CANNOT_RUN
    return EXIT_SAME if record["same"] else EXIT_DIFFERENT


def cmd_reproduce(args, ml=None):
    """`python -m mojolearn reproduce RECEIPT.json --X FILE [--y FILE]
    [--params JSON] [--json]`. Exit 0: the same fit. Exit 1: different, and
    the part is named. Exit 2: usage. Exit 3: refused (`fast`). Exit 4: could
    not run. With --json the whole answer is one JSON line on stdout."""
    def finish(record):
        print(json.dumps(record, sort_keys=True) if args.json else format_reproduce(record))
        return exit_code(record)

    stub = dict(format=FORMAT, ran=False, same=None, different=[], first=None, parts={})
    try:
        params = json.loads(args.params) if args.params else None
        if params is not None and not isinstance(params, dict):
            raise ValueError("not a JSON object")
    except ValueError as exc:
        print(f"USAGE: --params must be a JSON object of constructor parameters ({exc})", file=sys.stderr)
        return EXIT_USAGE
    try:
        if ml is None:
            import mojolearn as ml
    except Exception as exc:
        return finish(dict(stub, reason=f"importing mojolearn raised {type(exc).__name__}: {exc}"[:400]))
    try:
        X, y = _load_xy(args.X, args.y)
    except Exception as exc:
        return finish(dict(stub, reason=f"the data could not be read: {type(exc).__name__}: {exc}"[:400]))
    return finish(reproduce_with(ml, args.receipt, X, y, params=params))
