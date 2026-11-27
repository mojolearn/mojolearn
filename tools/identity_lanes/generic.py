# SPDX-License-Identifier: Apache-2.0
# THE GENERIC MECHANISM LANES (lane vdv-cov-generic, 2026-11-24).
#
# Four mechanisms every public class shares get ONE lane each, split by family group into literal lanes so each lane
# is a light, independent unit (the verify workers run them across the box's cores). Each lane walks the public class
# registry (`mojolearn.__all__` and the `model_selection`, `models`, `training`, `tokenizer` namespaces): no per-class
# code and no per-class fixture. A class gets at most the fit recipe its own declared type picks (`_estimator_type`:
# the classifier labels or the regression target; rows only, or rows and labels, from its fit signature; a series
# when its fit takes `y` first; a member estimator of its type where its constructor requires one; `random_state=0`
# where it has one).
#
#   generic-params-<group>  get_params (shallow and deep), set_params to every parameter's own value and, for each
#                           numeric parameter, to a changed value read back and restored; the clone round trip (the
#                           class rebuilt through the view from its own get_params); every settable public property
#                           (and, where the class overrides __setattr__, every public instance attribute) set to its
#                           own value and read back; get_n_splits. Unfitted. Numeric values digest their bytes.
#   generic-attrs-<group>   after the bounded fit, every public fitted attribute (trailing `_`) and every public
#                           property, digested by name and value (and get_feature_names_out); a class the fixture does
#                           not fit has its properties read on the constructed object, as its own part.
#   generic-pickle-<group>  pickle round trip, copy.deepcopy and the clone refit of the fitted object on the side: each
#                           copy's digest_ and predict / transform answers must equal the original's (enforced), and
#                           the answers are digested.
#   generic-digest-<group>  digest_ and training_digest_ after the bounded fit; digest_ read twice must agree, and must
#                           not move across one predict / transform call (except the classes whose docstring says their
#                           digest moves as they are used).
#
# Per class and per side (the harness calls each lane once per side, `_lane_side()`), the class is built through the
# device-bound view, so it is on the running side. Every class keeps its own part (named `<namespace>.<Name>`) so a
# failure names the class. A class the generic fixture does not apply to (its constructor requires an argument the
# declared type does not give, it has no fit taking rows, or its own fit raises on the fixture with words that do not
# name the device) is recorded, never skipped: its part is "n/a:generic-fixture: <why>", read at the first recorded
# column and listed in docs/plans/verify_through_device/UNMAPPED.md. A class that starts or stops fitting moves its
# part. A refusal that names the device, or a broken round trip, fails the cell naming every such class.
#
# Names that hold timing, vendor, thread, backend, platform or file-system records are left out of the digests: they
# are not arithmetic and differ between boxes of one vendor column (AGENTS.md part 2, bits match across vendors).
#
# Sizes are correctness-check sizes (AGENTS.md part 4, bounded fixtures), never product dispatch rules. Each lane
# builds its own fixture slice and its own objects and shares nothing mutable with another lane.
#
# tools/identity_break.py executes this file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`): every lane name is literal, nothing existing is rebound, every helper is prefixed
# `_generic_`.

#: (group, pattern over the defining module's last component); the first match wins; the rest is "other". The same
#: groups as the check lanes (checks_generic.py), so one class sits in the same group in both.
_generic_GROUPS = (
    ("prep", r"prep"),
    ("decomp", r"decomp|manifold"),
    ("linear", r"linear|solver|svm|kernel|_gp|gpc|isotonic|naive|bayes|discriminant|density|calibrat"),
    ("trees", r"tree|forest|ensemble|boost|gbdt"),
    ("neighbors", r"neighbor|_ann\b|_ann_|expansion_ann"),
    ("cluster", r"cluster|hdbscan|mixture|spectral|louvain|pagerank|graph"),
    ("sequence", r"sequence|mamba|training|models|embedding|rnn|cnn|transformer"),
)
#: the public namespaces walked, top level first
_generic_NAMESPACES = ("", "model_selection", "models", "training", "tokenizer")
#: rows and features of the generic fixture
_generic_ROWS = 64
_generic_FEATURES = 4
#: read by generic-digest, left out of generic-attrs
_generic_DIGESTS = ("digest_", "training_digest_")
#: the classes whose digest_ moves as the model is used (their random draws continue from one call to the next),
#: as `_mode.CheckMixin.digest_` documents; generic-digest does not hold them to an unmoved digest across a call
_generic_DIGEST_MOVES_WHEN_USED = ("IterativeImputer", "LatentDirichletAllocation", "MinCovDet")
#: attribute and dictionary-key words that name a record which is not arithmetic (timing, the box, the files)
_generic_NOT_ARITHMETIC = r"time|elapsed|duration|seconds|vendor|thread|backend|platform|hostname|filename|filepath|directory"
#: fitted records whose name does not say they hold a time but whose value carries one: name -> how many leading
#: members are arithmetic. `fit_loop_` is `(executed iterations, loop wall time in ms)` on every family that stores
#: it (KMeans, MiniBatchKMeans, the forests and boosters, the decompositions, label propagation, the linear, recurrent
#: and convolutional fits; python/mojolearn/_digest_fields.py leaves it out of digest_ for the same reason). Its count
#: is digested, its milliseconds never: they differ on every run, so generic-attrs read the clean and the restored run
#: of one source as different (w3-fix-10-s4, sab2b-mul_add-generic-cpu "restoration changed the clean result").
_generic_TIMED_RECORDS = {"fit_loop_": 1}
_generic_MISSING = object()
#: THE TREES GROUP'S ENSEMBLE BOUND (lane slow-lanes-1, 2026-11-26). The four fitting trees lanes (generic-attrs-,
#: generic-pickle-, generic-digest-trees here and check-generic-trees in checks_generic.py) fitted every class at its
#: own default size: GradientBoosting's n_estimators=None is 1000 SymmetricTree iterations, the forests 100 trees.
#: On the L40S (nvc-ab-20261127T015224Z-34c011, wheels 01a74234110b) that made them the four slowest lanes of both
#: steps: verify units of 56-160 s (generic-pickle-trees/odd 160 s; every other lane's unit at most 23 s) and record
#: lanes of 474-666 s, which set the record's 771 s wall. These lanes hold the shared MECHANISMS (attributes, copies,
#: digests, check), and an ensemble of 8 members runs the same tree build, leaf, vote and sum kernels as one of 1000;
#: so, before the fit, a class's n_estimators (the canonical member count: every boosting alias resolves to it, see
#: _boosting_param_names.py) that is unset or above the bound is set to the bound. The class list, the parts, the
#: fixtures and the fixture's rows are unchanged; params lanes (no fit) are unchanged. What 8 no longer reaches here:
#: GradientBoosting's auto-resolved Ordered boosting (taken at 500 or more iterations, ensemble.py
#: _ORDERED_MIN_ITERATIONS) and its default leaf-estimation iterations above 200 iterations
#: (bindings/gbdt_config_run.mojo gbdt_fit_words); both stay covered by their own lanes (gbdt-ordered-weights,
#: gbdt-cat-bootstrap, the OrderedRMSE lanes and the gbdt-* families at their recorded sizes). A class whose fit
#: refuses the bound is fitted again at its own size, so the bound never turns a fitted class into an n/a part.
#: A correctness-check size (AGENTS.md part 4, bounded fixtures), never a product dispatch rule.
_generic_ENSEMBLE_BOUND = 8
_generic_ENSEMBLE_GROUPS = ("trees",)
_generic_ENSEMBLE_PARAMS = ("n_estimators",)


def _generic_bound_ensemble(est, group):
    """Set `est`'s ensemble-size parameters that are unset or above `_generic_ENSEMBLE_BOUND` to the bound (the
    `_generic_ENSEMBLE_GROUPS` only); returns {name: its own value} of those it set, to restore on a refusal."""
    if group not in _generic_ENSEMBLE_GROUPS or not callable(getattr(est, "set_params", None)):
        return {}
    try:
        params = est.get_params(deep=False)
    except Exception:  # noqa: BLE001 -- a class without get_params keeps its own size
        return {}
    own = {}
    for key in _generic_ENSEMBLE_PARAMS:  # small-loop(the ensemble-size parameter names)
        if key not in params:
            continue
        value = params[key]
        if value is None or (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                             and int(value) > _generic_ENSEMBLE_BOUND):
            own[key] = value
    if own:
        est.set_params(**{key: _generic_ENSEMBLE_BOUND for key in own})
    return own


def _generic_fixture(X, yc, yr):
    """The bounded generic fixture: non-negative rows (count, chi2 and NMF style classes accept them), the class
    labels, a strictly positive regression target (Poisson, Gamma and Tweedie accept it), the same target as a series
    for the classes whose fit takes `y` first, and four fold groups for the group splitters."""
    rows, cols = _generic_ROWS, _generic_FEATURES
    Xp = np.ascontiguousarray(np.abs(np.asarray(X[:rows, :cols], dtype=np.float32)))
    ycp = np.ascontiguousarray(np.asarray(yc[:rows]))
    yrp = np.ascontiguousarray(np.abs(np.asarray(yr[:rows], dtype=np.float32)) + np.float32(1.0))
    series = yrp.copy()
    groups = np.ascontiguousarray(np.arange(rows, dtype=np.int64) % 4)
    return Xp, ycp, yrp, series, groups


def _generic_words(exc):
    import re
    return re.sub(r"0x[0-9a-fA-F]+", "0x?", " ".join(str(exc).split()))[:160]


def _generic_tag(words):
    """`words` as uint8 bytes, the one spelling `_h` takes for text."""
    return np.frombuffer(str(words).encode("utf-8"), dtype=np.uint8)


def _generic_not_arithmetic(name):
    import re
    return re.search(_generic_NOT_ARITHMETIC, str(name).lower()) is not None


def _generic_arrays(value, out, depth=0):
    """Append to `out` the numeric arrays that spell `value`: numbers by their bytes (int64, float64 or the numpy
    scalar's own dtype), numeric arrays as they are, text as utf-8 bytes, containers element by element (dictionary
    keys sorted), a nested estimator by its class name and its digest_, anything else by its class name only (never an
    address)."""
    if depth > 3:
        out.append(_generic_tag("<deeper>"))
        return
    if value is None:
        out.append(_generic_tag("<None>"))
    elif isinstance(value, (bool, np.bool_)):
        out.append(np.asarray([bool(value)], dtype=np.bool_))
    elif isinstance(value, np.generic) and value.dtype.kind in "iufcb":
        out.append(np.asarray(value).reshape(1))
    elif isinstance(value, int):
        if -(2 ** 63) <= value < 2 ** 63:
            out.append(np.asarray([value], dtype=np.int64))
        else:
            out.append(_generic_tag(f"<int {value}>"))
    elif isinstance(value, float):
        out.append(np.asarray([value], dtype=np.float64))
    elif isinstance(value, complex):
        out.append(np.asarray([value], dtype=np.complex128))
    elif isinstance(value, str):
        out.append(_generic_tag(f"<str>{value}"))
    elif isinstance(value, (bytes, bytearray)):
        out.append(_generic_tag("<bytes>"))
        out.append(np.frombuffer(bytes(value), dtype=np.uint8))
    elif isinstance(value, np.ndarray):
        out.append(np.asarray(value.shape, dtype=np.int64))
        if value.dtype.kind in "biufc":
            out.append(value)
        elif value.dtype.kind in "US":
            for item in value.ravel():  # driver-loop: over the text entries of one bounded fitted array
                out.append(_generic_tag(item))
        else:
            for item in value.ravel():  # driver-loop: over the objects of one bounded fitted array
                _generic_arrays(item, out, depth + 1)
    elif isinstance(value, dict):
        out.append(_generic_tag(f"<dict {len(value)}>"))
        for key in sorted(value, key=str):  # driver-loop: over the keys of one fitted record
            if _generic_not_arithmetic(key):
                continue
            out.append(_generic_tag(key))
            _generic_arrays(value[key], out, depth + 1)
    elif isinstance(value, (list, tuple)):
        out.append(_generic_tag(f"<{type(value).__name__} {len(value)}>"))
        for item in value:  # driver-loop: over the members of one fitted list
            _generic_arrays(item, out, depth + 1)
    elif isinstance(value, (set, frozenset)):
        out.append(_generic_tag(f"<set {len(value)}>"))
        for item in sorted(value, key=str):  # driver-loop: over the members of one fitted set
            _generic_arrays(item, out, depth + 1)
    elif isinstance(value, type):
        out.append(_generic_tag(f"<class {value.__name__}>"))
    elif callable(getattr(value, "get_params", None)) or hasattr(type(value), "digest_"):
        out.append(_generic_tag(f"<{type(value).__name__}>"))
        out.append(_generic_digest_bytes(value, "digest_"))
    else:
        try:
            arr = np.asarray(value)
        except Exception:  # noqa: BLE001 -- an object with no array view is spelled by its class name
            arr = None
        if arr is not None and arr.dtype.kind in "biufc":
            out.append(np.asarray(arr.shape, dtype=np.int64))
            out.append(arr)
        else:
            out.append(_generic_tag(f"<{type(value).__name__}>"))


def _generic_value_hash(value):
    out = []
    _generic_arrays(value, out)
    return _h(*out)


def _generic_digest_bytes(est, name):
    """The hex digest `name` of `est` as bytes, or a tag naming the exception it raised."""
    try:
        text = getattr(est, name)
    except Exception as exc:  # noqa: BLE001 -- the raise is the record (unfitted, unhashable)
        return _generic_tag(f"<{name} raises {type(exc).__name__}>")
    try:
        return np.frombuffer(bytes.fromhex(str(text)), dtype=np.uint8)
    except ValueError:
        return _generic_tag(f"<{name} not hex>{text}")


def _generic_same(a, b):
    """True when two parameter or attribute values are the same value (the same object, equal numbers or text, equal
    arrays elementwise, NaN equal to NaN)."""
    if a is b:
        return True
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=np.asarray(a).dtype.kind in "fc")
        if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
            return True
        same = a == b
        if isinstance(same, np.ndarray):
            return bool(same.all())
        return bool(same)
    except Exception:  # noqa: BLE001 -- values that refuse comparison are the same only by identity
        return False


def _generic_expect_same(before, after, what):
    """Raise naming the first parameter whose value `after` does not hold as `before` did."""
    if sorted(before) != sorted(after):
        raise AssertionError(f"{what}: get_params keys {sorted(set(before) ^ set(after))} differ")
    for key in sorted(before):  # driver-loop: over one class's parameters
        if not _generic_same(before[key], after[key]):
            raise AssertionError(f"{what}: {key} reads back {type(after[key]).__name__} "
                                 f"where {type(before[key]).__name__} was set")


def _generic_changed(key, value):
    """A changed value of the same numeric type, or None where the type gives no generic change."""
    if key == "device" or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return type(value)(int(value) + 1)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return type(value)(value * 2.0 if value != 0 else 1.0)
    return None


def _generic_group(cls):
    import re
    module = cls.__module__.rpartition(".")[2]
    for group, pattern in _generic_GROUPS:
        if re.search(pattern, module):
            return group
    return "other"


def _generic_classes(group, want):
    """(namespace, public name, class) of every public class whose defining module falls in `group` and that `want`
    accepts, in name order; a class exported under two names is walked once, under its first name; exception classes
    are not walked (they compute nothing)."""
    import importlib
    seen, found = set(), []
    for where in _generic_NAMESPACES:
        space = importlib.import_module(f"mojolearn.{where}" if where else "mojolearn")
        for name in sorted(getattr(space, "__all__", ()) if space is not None else ()):
            try:
                obj = getattr(space, name)
            except Exception:  # noqa: BLE001 -- a name that does not resolve has nothing to walk
                continue
            if not isinstance(obj, type) or issubclass(obj, BaseException) or obj in seen:
                continue
            seen.add(obj)
            if _generic_group(obj) == group and want(obj):
                found.append((where, name, obj))
    return found


def _generic_required(fn):
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


def _generic_members(ml, kind):
    """Two member estimators of the declared type, built through the view (so on the running side)."""
    if kind == "classifier":
        return ml.LogisticRegression(), ml.DecisionTreeClassifier(max_depth=2, random_state=0)
    return ml.Ridge(alpha=1.0), ml.DecisionTreeRegressor(max_depth=2, random_state=0)


def _generic_make(ml, where, name, kwargs):
    """`<namespace>.<name>(**kwargs)` through the view."""
    space = getattr(ml, where) if where else ml
    return getattr(space, name)(**kwargs)


def _generic_build(ml, where, name, cls):
    """(object built through the view, None) or (None, why the generic fixture does not apply). A refusal whose words
    name the device raises: it is a failure of the side, never n/a."""
    import re
    kind = getattr(cls, "_estimator_type", None)
    kwargs = {}
    for param in _generic_required(cls.__init__) or ():
        if param in ("estimator", "base_estimator"):
            kwargs[param] = _generic_members(ml, kind)[0]
        elif param == "estimators":
            first, second = _generic_members(ml, kind)
            kwargs[param] = [("a", first), ("b", second)]
        else:
            return None, f"its constructor requires {param}="
    try:
        est = _generic_make(ml, where, name, kwargs)
    except Exception as exc:  # noqa: BLE001 -- the class's own refusal of the generic arguments is its record
        words = _generic_words(exc)
        # the side's own word, not `devices=` (a builder's GPU list): DistributedIVFIndex's bare constructor
        # refuses with "use DistributedIVFIndex.from_index(index, devices=...)", the generic fixture not applying
        # (w3-fix-12, gate 9 row 1b); a refusal naming `device` (the view's NoDevice, a side the product refuses)
        # still fails the cell
        if re.search(r"\bdevice\b", words):
            raise
        return None, f"its constructor raised {type(exc).__name__}: {words}"
    try:
        params = est.get_params(deep=False)
    except Exception:  # noqa: BLE001 -- a class without get_params keeps its own seed
        params = {}
    if "random_state" in params and params["random_state"] is None and callable(getattr(est, "set_params", None)):
        est.set_params(random_state=0)
    return est, None


def _generic_fit_args(cls, fixture):
    """(the fit arguments the declared type and the fit signature pick, None) or (None, why there are none)."""
    Xp, ycp, yrp, series, _groups = fixture
    fit = getattr(cls, "fit", None)
    if not callable(fit):
        return None, "it has no fit"
    required = _generic_required(fit)
    if required and required[0] == "y":
        if len(required) > 1:
            return None, f"its fit requires {', '.join(required)}"
        return (series,), None
    kind = getattr(cls, "_estimator_type", None)
    if kind == "classifier":
        return (Xp, ycp), None
    if kind == "regressor":
        return (Xp, yrp), None
    if required == []:
        return None, "its fit takes no rows"
    if required is None:
        return (Xp,), None
    if required[0] not in ("X", "A", "x"):
        return None, f"its fit requires {required[0]}= first, not rows"
    if len(required) > 2:
        return None, f"its fit requires {', '.join(required)}"
    return ((Xp, ycp) if len(required) == 2 else (Xp,)), None


def _generic_prepare(ml, fixture, where, name, cls, fit, label, side, failed, group=None):
    """(est, fit args, why). est None and why None: a failure was appended to `failed`. est None with a why: the
    class could not be built. est set with a why: built, not fitted (the why says why). args None with no why: built
    and not asked to fit. In `_generic_ENSEMBLE_GROUPS` the fit runs at the ensemble bound; a fit that refuses the
    bound runs again at the class's own size."""
    try:
        est, why = _generic_build(ml, where, name, cls)
    except Exception as exc:  # noqa: BLE001 -- a constructor that refuses the side is a failure, never n/a
        failed.append(f"{label}({side!r} side) raised {type(exc).__name__}: {_generic_words(exc)}")
        return None, None, None
    if est is None or not fit:
        return est, None, why
    args, why = _generic_fit_args(cls, fixture)
    if args is None:
        return est, None, why
    own = _generic_bound_ensemble(est, group)
    try:
        try:
            est.fit(*args)
        except Exception as exc:  # noqa: BLE001 -- the bounded fit refused: the class's own size, never an n/a
            if not own or "device" in _generic_words(exc):
                raise
            est.set_params(**own)
            est.fit(*args)
    except Exception as exc:  # noqa: BLE001 -- the class's own fit on the generic fixture
        words = _generic_words(exc)
        if "device" in words:
            failed.append(f"{label}.fit refused on the {side} side: {type(exc).__name__}: {words}")
            return None, None, None
        return None, None, f"its fit raised {type(exc).__name__}: {words}"
    return est, args, None


def _generic_static(cls, name):
    import inspect
    return inspect.getattr_static(cls, name, _generic_MISSING)


def _generic_is_method(value):
    import inspect
    return (inspect.isroutine(value) or isinstance(value, (staticmethod, classmethod))
            or inspect.ismethoddescriptor(value) and not inspect.isdatadescriptor(value))


def _generic_properties(cls, settable):
    """The public property names of `cls` (setter required when `settable`), sorted, digests and records that are not
    arithmetic left out."""
    names = []
    for name in sorted(dir(cls)):  # driver-loop: over one class's public names
        if name.startswith("_") or name in _generic_DIGESTS or _generic_not_arithmetic(name):
            continue
        static = _generic_static(cls, name)
        if isinstance(static, property) and (not settable or static.fset is not None):
            names.append(name)
    return names


def _generic_attribute_names(est, cls):
    """Every public fitted attribute (a trailing `_`, not a method) and every public property of `est`, sorted."""
    names = set(_generic_properties(cls, settable=False))
    for name in dir(est):  # driver-loop: over one fitted object's public names
        if (name.startswith("_") or not name.endswith("_") or name in _generic_DIGESTS
                or _generic_not_arithmetic(name)):
            continue
        static = _generic_static(cls, name)
        if static is not _generic_MISSING and _generic_is_method(static):
            continue
        names.add(name)
    return sorted(names)


def _generic_set_round_trip(est, name, out):
    """Read public `name`, set it to its own value, read it back (it must be the same), digest it into `out`."""
    try:
        value = getattr(est, name)
    except Exception as exc:  # noqa: BLE001 -- an unfitted read raises; the raise is the record
        out.append(_generic_tag(f"{name}: read raises {type(exc).__name__}"))
        return
    try:
        setattr(est, name, value)
    except Exception as exc:  # noqa: BLE001 -- a refused assignment is the record, unless it names the device
        words = _generic_words(exc)
        if "device" in words:
            raise
        out.append(_generic_tag(f"{name}: set raises {type(exc).__name__}"))
        return
    again = getattr(est, name)
    if not _generic_same(value, again):
        raise AssertionError(f"{name} set to its own value reads back a different {type(again).__name__}")
    out.append(_generic_tag(name))
    _generic_arrays(again, out)


def _generic_params_part(ml, where, name, cls, est, args, fixture):
    """generic-params: the class's parameter mechanisms on the unfitted object."""
    Xp, ycp, _yrp, _series, groups = fixture
    out = []
    if callable(getattr(est, "get_params", None)):
        shallow = est.get_params(deep=False)
        deep = est.get_params(deep=True)
        for label, params in (("shallow", shallow), ("deep", deep)):
            out.append(_generic_tag(f"<{label} {len(params)}>"))
            for key in sorted(params):  # driver-loop: over one class's parameters
                out.append(_generic_tag(key))
                _generic_arrays(params[key], out)
        if callable(getattr(est, "set_params", None)):
            est.set_params(**shallow)
            _generic_expect_same(shallow, est.get_params(deep=False), "set_params to its own values")
            for key in sorted(shallow):  # driver-loop: over one class's numeric parameters
                changed = _generic_changed(key, shallow[key])
                if changed is None:
                    continue
                try:
                    est.set_params(**{key: changed})
                except Exception as exc:  # noqa: BLE001 -- a validated refusal is the record
                    words = _generic_words(exc)
                    if "device" in words:
                        raise
                    out.append(_generic_tag(f"{key}: set_params refuses {type(exc).__name__}"))
                    continue
                got = est.get_params(deep=False).get(key, _generic_MISSING)
                if got is _generic_MISSING or not _generic_same(got, changed):
                    raise AssertionError(f"set_params({key}=<changed {type(changed).__name__}>) does not read back")
                out.append(_generic_tag(f"{key}:changed"))
                _generic_arrays(got, out)
                est.set_params(**{key: shallow[key]})
            _generic_expect_same(shallow, est.get_params(deep=False), "set_params back to its own values")
        try:
            twin = _generic_make(ml, where, name, dict(shallow))
        except Exception as exc:  # noqa: BLE001 -- a class whose constructor refuses its own get_params
            words = _generic_words(exc)
            if "device" in words:
                raise
            out.append(_generic_tag(f"clone: constructor raises {type(exc).__name__}"))
        else:
            _generic_expect_same(shallow, twin.get_params(deep=False), "clone (rebuilt from get_params)")
            out.append(_generic_tag("clone"))
    for prop in _generic_properties(cls, settable=True):  # driver-loop: over one class's settable properties
        _generic_set_round_trip(est, prop, out)
    if _generic_static(cls, "__setattr__") is not object.__setattr__:
        for attr in sorted(k for k in getattr(est, "__dict__", {}) if not k.startswith("_")):
            _generic_set_round_trip(est, attr, out)  # driver-loop: over one object's public instance attributes
    if callable(getattr(est, "get_n_splits", None)):
        try:
            n = est.get_n_splits(Xp, ycp, groups)
        except Exception as exc:  # noqa: BLE001 -- a splitter that needs other inputs records its refusal
            words = _generic_words(exc)
            if "device" in words:
                raise
            out.append(_generic_tag(f"get_n_splits raises {type(exc).__name__}"))
        else:
            out.append(np.asarray([int(n)], dtype=np.int64))
    if not out:
        out.append(_generic_tag("<no parameter mechanism>"))
    return _h(*out)


def _generic_attrs_part(ml, where, name, cls, est, args, fixture):
    """generic-attrs: every public fitted attribute and property, by name and value."""
    out = []
    for attr in _generic_attribute_names(est, cls):  # driver-loop: over one fitted object's public attributes
        try:
            value = getattr(est, attr)
        except Exception as exc:  # noqa: BLE001 -- an attribute that does not apply raises; the raise is the record
            out.append(_generic_tag(f"{attr}: raises {type(exc).__name__}"))
            continue
        if _generic_is_method(value):
            continue
        keep = _generic_TIMED_RECORDS.get(attr)
        if keep is not None and isinstance(value, (tuple, list)):
            value = tuple(value[:keep])  # the arithmetic members only; the wall time is left out
        out.append(_generic_tag(attr))
        _generic_arrays(value, out)
    if args is not None and callable(getattr(est, "get_feature_names_out", None)):
        try:
            names = est.get_feature_names_out()
        except Exception as exc:  # noqa: BLE001 -- the raise is the record
            out.append(_generic_tag(f"get_feature_names_out raises {type(exc).__name__}"))
        else:
            out.append(_generic_tag("get_feature_names_out"))
            _generic_arrays(np.asarray(names), out)
    if not out:
        out.append(_generic_tag("<no public attribute>"))
    return _h(*out)


def _generic_answer_method(est, args, Xp):
    """predict or transform where the object was fitted on rows; None otherwise."""
    if args is None or args[0] is not Xp:
        return None
    for method in ("predict", "transform"):
        if callable(getattr(est, method, None)):
            return method
    return None


def _generic_answers(est, args, Xp):
    """digest_ and the predict / transform answers (fitted on rows), or the shallow parameters (not fitted), as one
    list of arrays."""
    out = [_generic_digest_bytes(est, "digest_")]
    method = _generic_answer_method(est, args, Xp)
    if method is not None:
        try:
            answer = getattr(est, method)(Xp)
        except Exception as exc:  # noqa: BLE001 -- the raise is the record, and must be the copy's too
            out.append(_generic_tag(f"{method} raises {type(exc).__name__}"))
        else:
            out.append(_generic_tag(method))
            _generic_arrays(answer, out)
    elif callable(getattr(est, "get_params", None)):
        params = est.get_params(deep=False)
        for key in sorted(params):  # driver-loop: over one class's parameters
            out.append(_generic_tag(key))
            _generic_arrays(params[key], out)
    return out


def _generic_pickle_part(ml, where, name, cls, est, args, fixture):
    """generic-pickle: pickle, deepcopy and clone-refit copies answer as the original does."""
    import copy
    import pickle
    Xp = fixture[0]
    twins, out = [], []
    # Every copy is taken from the freshly fitted state BEFORE the original answers: a model whose state moves as it
    # answers (its random draws continue) is then compared from the same state.
    for how in ("pickle", "deepcopy", "clone"):  # driver-loop: over the three copy mechanisms
        if how == "clone" and not callable(getattr(est, "get_params", None)):
            out.append(_generic_tag("clone: no get_params"))
            continue
        try:
            if how == "pickle":
                twin = pickle.loads(pickle.dumps(est, protocol=pickle.HIGHEST_PROTOCOL))
            elif how == "deepcopy":
                twin = copy.deepcopy(est)
            else:
                twin = _generic_make(ml, where, name, dict(est.get_params(deep=False)))
        except Exception as exc:  # noqa: BLE001 -- a class that does not copy records it, unless it names the device
            words = _generic_words(exc)
            if "device" in words:
                raise
            out.append(_generic_tag(f"{how}: raises {type(exc).__name__}"))
            continue
        if how == "clone" and args is not None:
            twin.fit(*args)   # the original fitted on these arguments: a clone that cannot is a failure, never n/a
        twins.append((how, twin))
    base = _generic_answers(est, args, Xp)
    want = _h(*base)
    out.extend(base)
    for how, twin in twins:  # driver-loop: over the copies taken
        if _h(*_generic_answers(twin, args, Xp)) != want:
            raise AssertionError(f"its {how} copy answers differently from the original (digest_ or "
                                 f"{_generic_answer_method(est, args, Xp) or 'parameters'})")
        out.append(_generic_tag(how))
    return _h(*out)


def _generic_digest_part(ml, where, name, cls, est, args, fixture):
    """generic-digest: digest_ and training_digest_ of the fitted object, digest_ stable across reads and calls."""
    Xp = fixture[0]
    first = _generic_digest_bytes(est, "digest_")
    second = _generic_digest_bytes(est, "digest_")
    if _h(first) != _h(second):
        raise AssertionError("digest_ read twice gives two values")
    training = _generic_digest_bytes(est, "training_digest_")
    method = _generic_answer_method(est, args, Xp)
    out = [_generic_tag("digest_"), first, _generic_tag("training_digest_"), training]
    if method is not None and cls.__name__ not in _generic_DIGEST_MOVES_WHEN_USED:
        try:
            getattr(est, method)(Xp)
        except Exception as exc:  # noqa: BLE001 -- the raise is the record
            out.append(_generic_tag(f"{method} raises {type(exc).__name__}"))
        else:
            if _h(_generic_digest_bytes(est, "digest_")) != _h(first):
                raise AssertionError(f"digest_ moves across one {method} call")
            out.append(_generic_tag(f"unmoved by {method}"))
    return _h(*out)


def _generic_has_params(cls):
    return (callable(getattr(cls, "get_params", None)) or callable(getattr(cls, "get_n_splits", None))
            or bool(_generic_properties(cls, settable=True))
            or _generic_static(cls, "__setattr__") is not object.__setattr__)


def _generic_has_digest(cls):
    return _generic_static(cls, "digest_") is not _generic_MISSING


def _generic_any(cls):
    return True


def _generic_run(ml, X, yc, yr, mechanism, group):
    """Every class of `group` that the mechanism applies to: build on this side, fit on the generic fixture where the
    mechanism reads a fit, and digest the mechanism's part per class."""
    lane_name = f"generic-{mechanism}-{group}"
    want, part_of, fit, unfitted_too = {
        "params": (_generic_has_params, _generic_params_part, False, False),
        "attrs": (_generic_any, _generic_attrs_part, True, True),
        "pickle": (_generic_any, _generic_pickle_part, True, True),
        "digest": (_generic_has_digest, _generic_digest_part, True, False),
    }[mechanism]
    side = _lane_side()
    fixture = _generic_fixture(X, yc, yr)
    classes = _generic_classes(group, want)
    if not classes:
        raise AssertionError(f"{lane_name}: the registry walk found no class in this group")
    parts, failed = {}, []
    for where, name, cls in classes:  # driver-loop: over the public classes of one group, one bounded unit each
        label = f"{where}.{name}" if where else name
        est, args, why = _generic_prepare(ml, fixture, where, name, cls, fit, label, side, failed, group)
        if est is None and why is None:
            continue
        if why is not None:
            parts[label] = "n/a:generic-fixture: " + why
            if est is None or not unfitted_too:
                continue
            label = label + ".unfitted"
        try:
            parts[label] = part_of(ml, where, name, cls, est, args, fixture)
        except Exception as exc:  # noqa: BLE001 -- a broken round trip or a refusal of the side, collected below
            failed.append(f"{label}: {type(exc).__name__}: {_generic_words(exc)}")
    if failed:
        raise AssertionError(f"{lane_name} ({side} side): " + "; ".join(failed))
    parts["classes"] = _h(np.asarray([len(classes)], dtype=np.int64))
    return _fit(parts)


@lane("generic-params-prep")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "params", "prep")


@lane("generic-params-decomp")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "params", "decomp")


@lane("generic-params-linear")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "params", "linear")


@lane("generic-params-trees")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "params", "trees")


@lane("generic-params-neighbors")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "params", "neighbors")


@lane("generic-params-cluster")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "params", "cluster")


@lane("generic-params-sequence")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "params", "sequence")


@lane("generic-params-other")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "params", "other")


@lane("generic-attrs-prep")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "attrs", "prep")


@lane("generic-attrs-decomp")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "attrs", "decomp")


@lane("generic-attrs-linear")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "attrs", "linear")


@lane("generic-attrs-trees")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "attrs", "trees")


@lane("generic-attrs-neighbors")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "attrs", "neighbors")


@lane("generic-attrs-cluster")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "attrs", "cluster")


@lane("generic-attrs-sequence")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "attrs", "sequence")


@lane("generic-attrs-other")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "attrs", "other")


@lane("generic-pickle-prep")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "pickle", "prep")


@lane("generic-pickle-decomp")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "pickle", "decomp")


@lane("generic-pickle-linear")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "pickle", "linear")


@lane("generic-pickle-trees")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "pickle", "trees")


@lane("generic-pickle-neighbors")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "pickle", "neighbors")


@lane("generic-pickle-cluster")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "pickle", "cluster")


@lane("generic-pickle-sequence")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "pickle", "sequence")


@lane("generic-pickle-other")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "pickle", "other")


@lane("generic-digest-prep")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "digest", "prep")


@lane("generic-digest-decomp")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "digest", "decomp")


@lane("generic-digest-linear")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "digest", "linear")


@lane("generic-digest-trees")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "digest", "trees")


@lane("generic-digest-neighbors")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "digest", "neighbors")


@lane("generic-digest-cluster")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "digest", "cluster")


@lane("generic-digest-sequence")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "digest", "sequence")


@lane("generic-digest-other")
def _(ml, X, yc, yr, Xh=None):
    return _generic_run(ml, X, yc, yr, "digest", "other")


_batch_decl("n/a:generic-mechanism (the lane walks every public class of one group through one shared mechanism: "
            "parameters, attributes, copies or digests; it has no out-of-sample probe of its own, and the batch part of "
            "each class is measured on that class's own lane)",
            "generic-params-prep", "generic-params-decomp", "generic-params-linear", "generic-params-trees",
            "generic-params-neighbors", "generic-params-cluster", "generic-params-sequence", "generic-params-other",
            "generic-attrs-prep", "generic-attrs-decomp", "generic-attrs-linear", "generic-attrs-trees",
            "generic-attrs-neighbors", "generic-attrs-cluster", "generic-attrs-sequence", "generic-attrs-other",
            "generic-pickle-prep", "generic-pickle-decomp", "generic-pickle-linear", "generic-pickle-trees",
            "generic-pickle-neighbors", "generic-pickle-cluster", "generic-pickle-sequence", "generic-pickle-other",
            "generic-digest-prep", "generic-digest-decomp", "generic-digest-linear", "generic-digest-trees",
            "generic-digest-neighbors", "generic-digest-cluster", "generic-digest-sequence", "generic-digest-other")
