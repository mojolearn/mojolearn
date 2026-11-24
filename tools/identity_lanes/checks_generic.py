# SPDX-License-Identifier: Apache-2.0
# THE GENERIC `check` LANES (lane vdv-check, 2026-11-24).
#
# `estimator.check(X)` (the 196 `.check` endpoints, every class that inherits `_mode.CheckMixin`) and the function
# `mojolearn.check` are ONE mechanism (`_check_both.check_estimator`): the same fitted model answers X on the GPU and
# on the CPU in this process, through the public `device=`, and the bits are compared. So they get one lane per
# family group, driven over every class of the public registry (`mojolearn.__all__`, `model_selection.__all__`,
# `models.__all__`): no per-class code and no per-class fixture. A class gets at most what its own declared estimator
# type picks (`_estimator_type`: the classifier labels or the regression target; a member estimator of that type for a
# meta-estimator whose constructor requires one; `random_state=0` where it has one).
#
# Per class and per side (the harness calls each lane once per side, `_lane_side()`), the lane:
#   1. builds the class through the device-bound view (so it is on the running side) and fits it on the bounded
#      generic fixture (64 rows, 4 non-negative features; positive regression target);
#   2. calls `est.check(X, method=...)`: the side's estimator answers, the other device's twin answers from the same
#      fitted state, and check compares;
#   3. digests THIS side's output digest (sides[<side>]["digest"]). The verdict is enforced, not digested: a
#      MISMATCH, a NOT RUN with a reason (a class that refuses device='cpu', two sides on one device, a model that
#      cannot cross), or an exception from check fails the cell naming every such class. On an installation without
#      a GPU the CPU side is the one digested and the GPU side is "not applicable"; the CPU column is therefore the
#      same digest on a CPU-only runner and on a GPU box.
# A class the generic fixture does not apply to (its constructor requires an argument the estimator type does not
# give or refuses the generic arguments with a TypeError, it has no fit taking rows, or its own fit raises on the
# fixture with words that do not name the device) is recorded, never skipped: its part is
# "n/a:generic-fixture:<ExceptionType>:<sha of the message without device words>" (a static reason reads
# "n/a:generic-fixture: <why>"), so only a failure that is the same on both sides is the same part in both columns;
# a one-sided or different failure is a digest difference. Read at the first recorded column and listed in
# docs/plans/verify_through_device/UNMAPPED.md. A class that starts or stops running moves its part.
#
# Sizes are correctness-check sizes (AGENTS.md part 4, bounded fixtures), never product dispatch rules. Each lane is an
# independent unit: it builds its own fixture slice and its own estimators and shares nothing mutable.
#
# tools/identity_break.py executes this file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`): every lane name is literal, nothing existing is rebound, every helper is prefixed
# `_checks_generic_`.

#: (group, pattern over the defining module's last component); the first match wins; the rest is "other".
_checks_generic_GROUPS = (
    ("prep", r"prep"),
    ("decomp", r"decomp|manifold"),
    ("linear", r"linear|solver|svm|kernel|_gp|gpc|isotonic|naive|bayes|discriminant|density|calibrat"),
    ("trees", r"tree|forest|ensemble|boost|gbdt"),
    ("neighbors", r"neighbor|_ann\b|_ann_|expansion_ann"),
    ("cluster", r"cluster|hdbscan|mixture|spectral|louvain|pagerank|graph"),
    ("sequence", r"sequence|mamba|training|models|embedding|rnn|cnn|transformer"),
)
#: the public namespaces walked, top level first
_checks_generic_NAMESPACES = ("", "model_selection", "models")
#: rows and features of the generic fixture
_checks_generic_ROWS = 64
_checks_generic_FEATURES = 4


def _checks_generic_fixture(X, yc, yr):
    """The bounded generic fixture: non-negative rows (count, chi2 and NMF style classes accept them), the class
    labels, and a strictly positive regression target (Poisson, Gamma and Tweedie accept it)."""
    rows, cols = _checks_generic_ROWS, _checks_generic_FEATURES
    Xp = np.ascontiguousarray(np.abs(np.asarray(X[:rows, :cols], dtype=np.float32)))
    ycp = np.ascontiguousarray(np.asarray(yc[:rows]))
    yrp = np.ascontiguousarray(np.abs(np.asarray(yr[:rows], dtype=np.float32)) + np.float32(1.0))
    return Xp, ycp, yrp


def _checks_generic_group(cls):
    import re
    module = cls.__module__.rpartition(".")[2]
    for group, pattern in _checks_generic_GROUPS:
        if re.search(pattern, module):
            return group
    return "other"


def _checks_generic_classes(group):
    """(namespace, public name, class) of every public class with `check` whose defining module falls in `group`,
    in name order; a class exported under two names is walked once, under its first name."""
    import mojolearn
    seen, found = set(), []
    for where in _checks_generic_NAMESPACES:
        space = getattr(mojolearn, where, None) if where else mojolearn
        for name in sorted(getattr(space, "__all__", ()) if space is not None else ()):
            try:
                obj = getattr(space, name)
            except Exception:  # noqa: BLE001 -- a name that does not resolve has no check to walk
                continue
            if not isinstance(obj, type) or obj in seen or not callable(getattr(obj, "check", None)):
                continue
            seen.add(obj)
            if _checks_generic_group(obj) == group:
                found.append((where, name, obj))
    return found


def _checks_generic_required(fn):
    """The parameter names `fn` requires (no default), after `self`; None when its signature cannot be read or takes
    `*args` (what it requires is then not written down)."""
    import inspect
    try:
        params = list(inspect.signature(fn).parameters.values())[1:]
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return None
    return [p.name for p in params
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]


def _checks_generic_members(ml, kind):
    """Two member estimators of the declared type, built through the view (so on the running side)."""
    if kind == "classifier":
        return ml.LogisticRegression(), ml.DecisionTreeClassifier(max_depth=2, random_state=0)
    return ml.Ridge(alpha=1.0), ml.DecisionTreeRegressor(max_depth=2, random_state=0)


def _checks_generic_build(ml, where, name, cls):
    """(estimator built through the view, None) or (None, why the generic fixture does not apply)."""
    kind = getattr(cls, "_estimator_type", None)
    kwargs = {}
    for param in _checks_generic_required(cls.__init__) or ():
        if param in ("estimator", "base_estimator"):
            kwargs[param] = _checks_generic_members(ml, kind)[0]
        elif param == "estimators":
            first, second = _checks_generic_members(ml, kind)
            kwargs[param] = [("a", first), ("b", second)]
        else:
            return None, f"its constructor requires {param}="
    space = getattr(ml, where) if where else ml
    try:
        est = getattr(space, name)(**kwargs)
    except TypeError as exc:
        if "device" in str(exc):
            raise    # the side itself refused (or the view's "takes no device="): a failure, never n/a
        return None, _checks_generic_na(exc)
    try:
        params = est.get_params(deep=False)
    except Exception:  # noqa: BLE001 -- a class without get_params keeps its own seed
        params = {}
    if "random_state" in params and params["random_state"] is None and callable(getattr(est, "set_params", None)):
        est.set_params(random_state=0)
    return est, None


def _checks_generic_fit_args(cls, Xp, ycp, yrp):
    """The fit arguments the declared type picks, or None when fit takes no rows."""
    fit = getattr(cls, "fit", None)
    if not callable(fit):
        return None
    kind = getattr(cls, "_estimator_type", None)
    if kind == "classifier":
        return (Xp, ycp)
    if kind == "regressor":
        return (Xp, yrp)
    required = _checks_generic_required(fit)
    if required == []:
        return None
    return (Xp, ycp) if required and len(required) > 1 else (Xp,)


def _checks_generic_method(est):
    """The method check compares: predict or transform, else the refitting fit_predict / fit_transform."""
    for method in ("predict", "transform", "fit_predict", "fit_transform"):
        if callable(getattr(est, method, None)):
            return method
    return None


def _checks_generic_words(exc):
    import re
    return re.sub(r"0x[0-9a-fA-F]+", "0x?", " ".join(str(exc).split()))[:160]


def _checks_generic_na(exc):
    """The n/a part of a class whose own constructor or fit raised on the generic fixture: a pure function of the
    exception type and its message with the device words removed, so the part is the same on both sides ONLY when
    both sides fail the same way; a one-sided or a different failure is a digest difference between the columns."""
    import hashlib
    import re
    words = re.sub(r"0x[0-9a-fA-F]+", "0x?", " ".join(str(exc).split()))
    words = re.sub(r"device\s*=\s*['\"]?\w*['\"]?", "", words)
    words = re.sub(r"\b(cpu|gpu|cuda|hip|metal|host|device)\b", "", words, flags=re.I)
    words = " ".join(words.split())
    return f"n/a:generic-fixture:{type(exc).__name__}:{hashlib.sha256(words.encode()).hexdigest()[:16]}"


def _checks_generic_side_part(result, side, label, failed):
    """This side's output digest from one check result, or None with the reason appended to `failed`."""
    from mojolearn import _check_both as cb
    sides = result.get("sides") or {}
    if result.get("ran"):
        if not result["verdict"]["match"]:
            failed.append(f"{label}: {cb.verdict_line(result['verdict'])}")
            return None
    elif not (side == "cpu" and result.get("gpu_side") == cb.NOT_APPLICABLE and "cpu" in sides):
        failed.append(f"{label}: NOT RUN: {result.get('reason')}"[:400])
        return None
    return _h(np.frombuffer(bytes.fromhex(sides[side]["digest"]), dtype=np.uint8))


def _checks_generic_run(ml, X, yc, yr, group):
    """Every class of `group`: build on this side, fit on the generic fixture, `check`, digest this side."""
    import contextlib
    import io
    side = _lane_side()
    Xp, ycp, yrp = _checks_generic_fixture(X, yc, yr)
    classes = _checks_generic_classes(group)
    if not classes:
        raise AssertionError(f"check-generic-{group}: the registry walk found no class with check in this group")
    parts, failed = {}, []
    for where, name, cls in classes:  # driver-loop: over the public classes of one group, one bounded check each
        label = f"{where}.{name}" if where else name
        try:
            est, why = _checks_generic_build(ml, where, name, cls)
        except Exception as exc:  # noqa: BLE001 -- a constructor that refuses the side is a failure, never n/a
            failed.append(f"{label}({side!r} side) raised {type(exc).__name__}: {_checks_generic_words(exc)}")
            continue
        args = _checks_generic_fit_args(cls, Xp, ycp, yrp) if est is not None else None
        method = _checks_generic_method(est) if est is not None else None
        if est is not None and args is None:
            why = "its fit takes no rows"
        elif est is not None and method is None:
            why = "it has none of predict, transform, fit_predict, fit_transform"
        elif est is not None:
            try:
                est.fit(*args)
            except Exception as exc:  # noqa: BLE001 -- the class's own fit on the generic fixture
                words = _checks_generic_words(exc)
                if "device" in words:
                    # a refusal of the side (device='cpu' not connected) is a failure of the cell, never n/a
                    failed.append(f"{label}.fit refused on the {side} side: {type(exc).__name__}: {words}")
                    continue
                why = _checks_generic_na(exc)
        if why is not None:
            # a raised exception is already its n/a part; a static reason (no fit, no method, a required argument)
            # is the same on both sides by construction
            parts[label] = why if why.startswith("n/a:") else "n/a:generic-fixture: " + why
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # check prints its report; the digest is the record
                result = est.check(Xp, method=method)
        except Exception as exc:  # noqa: BLE001 -- a failure of check, collected and raised below
            failed.append(f"{label}.check raised {type(exc).__name__}: {_checks_generic_words(exc)}")
            continue
        part = _checks_generic_side_part(result, side, label, failed)
        if part is not None:
            parts[label] = part
    if failed:
        raise AssertionError(f"check-generic-{group} ({side} side): " + "; ".join(failed))
    parts["classes"] = _h(np.asarray([len(classes)], dtype=np.int64))
    return _fit(parts)


@lane("check-generic-prep")
def _(ml, X, yc, yr, Xh=None):
    return _checks_generic_run(ml, X, yc, yr, "prep")


@lane("check-generic-decomp")
def _(ml, X, yc, yr, Xh=None):
    return _checks_generic_run(ml, X, yc, yr, "decomp")


@lane("check-generic-linear")
def _(ml, X, yc, yr, Xh=None):
    return _checks_generic_run(ml, X, yc, yr, "linear")


@lane("check-generic-trees")
def _(ml, X, yc, yr, Xh=None):
    return _checks_generic_run(ml, X, yc, yr, "trees")


@lane("check-generic-neighbors")
def _(ml, X, yc, yr, Xh=None):
    return _checks_generic_run(ml, X, yc, yr, "neighbors")


@lane("check-generic-cluster")
def _(ml, X, yc, yr, Xh=None):
    return _checks_generic_run(ml, X, yc, yr, "cluster")


@lane("check-generic-sequence")
def _(ml, X, yc, yr, Xh=None):
    return _checks_generic_run(ml, X, yc, yr, "sequence")


@lane("check-generic-other")
def _(ml, X, yc, yr, Xh=None):
    return _checks_generic_run(ml, X, yc, yr, "other")


@lane("check-function")
def _(ml, X, yc, yr, Xh=None):
    """`mojolearn.check(estimator, X, fit=..., fit_both=True, max_rows=...)`: the function form, and the form the
    method does not reach: one estimator of the same class and parameters fitted on each device from scratch, on the
    first `max_rows` rows. Reached as `mojolearn.check` (a verification function: it runs both sides itself and takes
    no `device`); the estimator is built through the view, so it is on the running side."""
    import contextlib
    import io
    import mojolearn
    side = _lane_side()
    Xp, ycp, yrp = _checks_generic_fixture(X, yc, yr)
    est = ml.Ridge(alpha=1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        result = mojolearn.check(est, Xp, fit=(Xp, yrp), fit_both=True, max_rows=48)
    failed = []
    part = _checks_generic_side_part(result, side, "Ridge fit_both", failed)
    if failed:
        raise AssertionError(f"check-function ({side} side): " + "; ".join(failed))
    return _fit(dict(ridge=part, rows=_h(np.asarray([result["fit_rows"], result["fit_rows_given"]], dtype=np.int64))))


_batch_decl("n/a:verification-check (the lane runs mojolearn's own GPU-against-CPU check per class; it has no "
            "out-of-sample probe of its own, and the batch part of each class is measured on that class's own lane)",
            "check-generic-prep", "check-generic-decomp", "check-generic-linear", "check-generic-trees",
            "check-generic-neighbors", "check-generic-cluster", "check-generic-sequence", "check-generic-other",
            "check-function")
