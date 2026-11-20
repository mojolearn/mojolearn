# SPDX-License-Identifier: Apache-2.0
"""Check your own model and data on the GPU and on the CPU.

`python -m mojolearn verify --check MODEL --estimator CLASS --inputs X.npy`,
`python -m mojolearn verify --fit-both --estimator CLASS --inputs X.npy
--fit-inputs XTRAIN.npy` (the same fit on each side) and `check(estimator, X)`
below. Both only call the shared check module
(`_check_both.py`), which `verify --cross-check` runs on the wheel's
fixtures: there is no second compare and no second report here.
"""
import json
import sys

from . import _check_both as cb

EXIT_MATCH = 0
EXIT_MISMATCH = 1
EXIT_USAGE = 2
EXIT_REFUSED_FAST = 3
EXIT_CANNOT_RUN = 4


def check(estimator, X, *, method=None, fit=None, fit_both=False, max_rows=None):
    """Run one method of a fitted estimator on the GPU and on the CPU and
    compare the two answers bit for bit.

    In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and
    AMD GPUs, and the CPU binaries are held to the same bits. This runs your
    estimator on your data both ways on this machine: the GPU answers from the
    estimator, the CPU answers from the same fitted model, saved and loaded
    with `mojolearn.host_model`.

    Parameters
    ----------
    estimator : object
        A fitted Mojolearn estimator that has `save`, of a kind
        `mojolearn.host_model` loads.
    X : array-like
        The rows to answer for.
    method : str, optional
        The method to check, for example `'predict'`, `'predict_proba'` or
        `'transform'`. Default: `predict` when the estimator has it, else
        `transform`.
    fit : tuple, optional
        Arguments for `estimator.fit`. When given, the estimator is fitted on
        the GPU first, then the fitted model is checked on both sides. Fitting
        is never done unless asked for.
    fit_both : bool, default False
        With `fit`: run the same fit once on each side, the GPU side in this
        process and the CPU side in a child process, and compare what the two
        fitted models answer. Use it for an estimator whose fitted model
        cannot be loaded on the CPU. The estimator must be unfitted, with
        constructor parameters that are plain values. A class the CPU does
        not run reports why and nothing is compared.
    max_rows : int, optional
        With `fit_both`: fit both sides on the first `max_rows` training rows
        only. The CPU fit can be slow on large inputs.

    Returns
    -------
    dict
        `ran` is False, with a `reason`, when nothing was compared: the
        process or the estimator is in `fast` mode (which makes no promise
        about bits), the installation has no GPU, or the model cannot be
        loaded on the CPU. Otherwise `sides` holds one record per side
        (device, binaries loaded, what the runtime counted while the side
        ran, wall time, and the shape, dtype, SHA-256 and first values of each
        output) and `verdict` says `MATCH`, or `MISMATCH`
        with the index and the two values of the first differing element.
        `format_check(result)` gives the same report as text.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn._verify_check import check, format_check
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.RandomForestRegressor(n_estimators=20).fit(X, X[:, 0])
    >>> print(format_check(check(model, X)))    # doctest: +SKIP
    """
    import mojolearn as ml
    return cb.check_estimator(ml, estimator, X, method=method, fit=fit, fit_both=fit_both, max_rows=max_rows)


format_check = cb.format_check


def _load_inputs(path):
    import numpy as np
    loaded = np.load(path, allow_pickle=False)
    if hasattr(loaded, "files"):
        names = list(loaded.files)
        if not names:
            raise ValueError(f"{path} holds no array")
        return loaded["X" if "X" in names else names[0]]
    return loaded


def cmd_check(args):
    """`verify --check MODEL --estimator CLASS --inputs FILE [--method NAME]`,
    and the fit-on-both-sides form `verify --fit-both --estimator CLASS
    --inputs FILE --fit-inputs FILE [--fit-targets FILE] [--params JSON]
    [--fit-rows N] [--method NAME]`."""
    json_out = getattr(args, "json", False)
    fit_both = bool(getattr(args, "fit_both", False))

    def finish(code, result):
        print(json.dumps(result, indent=1, sort_keys=True) if json_out else cb.format_check(result))
        return code

    missing = [flag for flag, value in (("--estimator", args.estimator), ("--inputs", args.inputs)) if not value]
    if fit_both and not getattr(args, "fit_inputs", None):
        missing.append("--fit-inputs")
    if missing:
        print(f"USAGE: {'--fit-both' if fit_both else '--check'} needs {' and '.join(missing)}: the class "
              "(for example RandomForestClassifier), a .npy or .npz file of rows"
              + (" and the training rows to fit on each side" if fit_both else ""), file=sys.stderr)
        return EXIT_USAGE
    stub = dict(format=cb.FORMAT, ran=False, estimator=args.estimator, method=args.method)
    try:
        import mojolearn as ml
        cb.refuse_fast(ml)
    except cb.NotChecked as exc:
        return finish(EXIT_REFUSED_FAST, dict(stub, refused="fast", reason=str(exc)))
    except Exception as exc:
        return finish(EXIT_CANNOT_RUN, dict(stub, reason=f"importing mojolearn raised {type(exc).__name__}: {exc}"))
    if ml.vendor() == "cpu":
        return finish(EXIT_CANNOT_RUN, dict(stub, reason=cb.NO_GPU))
    cls = getattr(ml, args.estimator, None)
    if cls is None or not (callable(cls) if fit_both else callable(getattr(cls, "load", None))):
        print(f"USAGE: --estimator {args.estimator} is not a mojolearn class"
              + ("" if fit_both else " with load"), file=sys.stderr)
        return EXIT_USAGE
    try:
        X = _load_inputs(args.inputs)
        if fit_both:
            params = json.loads(getattr(args, "params", None) or "{}")
            fit = (_load_inputs(args.fit_inputs),)
            if getattr(args, "fit_targets", None):
                fit += (_load_inputs(args.fit_targets),)
            result = cb.check_estimator(ml, cls(**params), X, method=args.method, fit=fit, fit_both=True,
                                        max_rows=getattr(args, "fit_rows", None))
        else:
            estimator = cls.load(args.check)
            result = cb.check_estimator(ml, estimator, X, method=args.method, saved=args.check)
    except Exception as exc:
        return finish(EXIT_CANNOT_RUN, dict(stub, reason=f"{type(exc).__name__}: {exc}"[:400]))
    if not result["ran"]:
        return finish(EXIT_REFUSED_FAST if result.get("refused") else EXIT_CANNOT_RUN, result)
    return finish(EXIT_MATCH if result["verdict"]["match"] else EXIT_MISMATCH, result)
