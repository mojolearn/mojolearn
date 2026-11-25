# SPDX-License-Identifier: Apache-2.0
"""The device-bound view of the package that verify runs each lane through.

Verification tooling (test control, never product execution). Verify checks
both sides of every endpoint in ONE process: the CPU side with
``device="cpu"`` and the GPU side with ``device="gpu"``, through the public
parameter, never through ``MOJOLEARN_VENDOR`` or a second process
(docs/plans/VERIFY_THROUGH_DEVICE.md, item 2).

A lane receives ``view(mojolearn, side)`` in place of the package. Every
public class and computing function it reaches as ``ml.<Name>`` or
``ml.<sub>.<Name>`` is called with ``device=<side>`` by keyword, unless the
lane passes ``device`` itself. Objects a lane builds through the view
(pipelines, meta-estimators, optimizers, schedules) therefore carry the side
by construction. Each name's signature is resolved once and cached; a call
through the view costs one dict lookup and one keyword.

A public callable that takes no ``device`` is a cell error
("<name> takes no device="), unless it is in ``NON_COMPUTING`` (containers,
config and file readers, the CPU-inference classes, the process readers) or
the lane is declared one-sided (the view is then built ``strict=False``).
"""
import functools
import inspect
import types
from contextlib import contextmanager

SIDES = ("cpu", "gpu")

#: The value a GPU side reads on an installation without a GPU.
NO_GPU = "n/a:not applicable (no GPU in this installation)"

#: Public names that take no ``device`` by design (the plan's 20 non-computing
#: names: settings and config readers, file and data containers, the
#: CPU-inference classes and the verify tools), plus the process readers the
#: lanes use to describe the run. Short names: a name is matched on its last
#: dotted part. Exception classes always pass through (``except`` needs the
#: real class).
NON_COMPUTING = frozenset((
    "Array", "HostGBDT", "LanguageModelConfig", "LanguageModelInference",
    "check", "recipe", "host_model", "host_predict", "host_predict_proba",
    "Checkpoint", "HFConfig", "ModelPlan", "ParallelCausalLM", "SafetensorsFile", "Tokenizer",
    "plan_for", "numeric_profile", "numeric_profile_measured", "numeric_profiles",
    "set_numeric_profile",
    # process and format readers: they compute nothing on a side
    "vendor", "numeric_mode", "runtime_report", "format_of", "is_packed",
    # scorer factories and readers (vd-verify, 2026-11-24): they hold a metric,
    # they compute nothing on a side; the metric handed in reaches them unwrapped
    "get_scorer", "make_scorer", "get_scorer_names",
    # config and caller-owned state containers (vdv-gaps, 2026-11-24): their
    # constructors admit and hold the caller's buffers or sizes
    # (mamba_native_call and transformer_public_call "state_init", samba
    # "state_construct", causal_state_init, samba_config_call "initialize");
    # the block or stack that reads them carries the side
    "SambaConfig", "Mamba1State", "Mamba2State", "Mamba3State", "TransformerState", "SambaState",
    "CausalLMState",
    # specification and container objects (w3-integrate-8, 2026-11-24): the
    # Gaussian-process kernel specs (hyperparameters and their tree; the
    # GaussianProcess* estimator that reads them carries the side), a trained
    # vocabulary's tokens and merges, the byte LM's settings, a safetensors
    # header entry and the parameter-grid sampler (it draws settings, not data)
    "Kernel", "ConstantKernel", "WhiteKernel", "RBF", "Matern",
    "TrainedBpeVocabulary", "ByteLanguageModelConfig", "TensorInfo", "ParameterSampler", "ParameterGrid",
    # a predicate over two integers (whether a microbatch split keeps the bits)
    "accumulation_is_aligned",
    # tokenizer and corpus readers (w3-integrate-9, 2026-11-25): the name of a
    # known pattern spelling, the vocabulary match check and the tokenizer it
    # returns, and the readers of a corpus already tokenized (cached ids and
    # their vocabulary); pretokenize, prepare and the trainer carry the side
    "pattern_name", "require_vocabulary", "tokenizer_for", "TokenizedCorpus", "TokenBatches",
    # the rows docs/plans/verify_through_device/functions_without_device.tsv
    # (vd-functions sweep) classes "no device" that this list lacked
    # (w3-integrate-9: the self-test's endpoint part read NoDevice on them):
    # CPU-inference classes, settings readers and writers, result and program
    # containers, a closed-form scalar of two integers
    "MLPInference", "SambaInference", "catboost_auto_learning_rate", "gpu_arch", "gpu_arch_how",
    "ImportedProgram", "profile", "require_identical", "Program", "BootstrapResult", "MonteCarloResult",
    "PermutationTestResult", "set_numeric_mode", "numeric_mode_used", "vendor_used",
))


class NoDevice(TypeError):
    """A public callable reached through the view takes no ``device``."""


_SIDE = [None]


def current_side():
    """"cpu" or "gpu" while a lane runs through `side()`, else None."""
    return _SIDE[0]


@contextmanager
def side(name):
    """Run the body with `name` as the current side."""
    if name not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, not {name!r}")
    previous = _SIDE[0]
    _SIDE[0] = name
    try:
        yield name
    finally:
        _SIDE[0] = previous


def gpu_available(ml):
    """True when this installation has a GPU side. An installation fact read
    once per call site (the process's loaded binding set), never a side choice."""
    return str(ml.vendor()) != "cpu"


#: id(callable) -> (callable, mode). The callable is held so its id cannot be
#: reused by another object while the entry lives.
_MODES = {}


def device_mode(fn):
    """How `fn` takes the side, resolved once: "device" (a `device`
    parameter), "kwargs" (a **kwargs that forwards it), "unknown" (no readable
    signature: a native callable; the keyword is passed and the product
    refuses it by name if it does not take it), "devices" (a GPU list only:
    the `parallel_*` drivers) or "none"."""
    hit = _MODES.get(id(fn))
    if hit is not None and hit[0] is fn:
        return hit[1]
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        mode = "unknown"
    else:
        if "device" in params:
            mode = "device"
        elif any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            mode = "kwargs"
        elif "devices" in params:
            mode = "devices"
        else:
            mode = "none"
    _MODES[id(fn)] = (fn, mode)
    return mode


def _short(name):
    return name.rsplit(".", 1)[-1]


def _non_computing(name):
    """`name` (dotted) is a NON_COMPUTING name or a method of one (the
    endpoint checks resolve e.g. ``mojolearn.ConstantKernel.__add__``)."""
    parts = name.rsplit(".", 2)
    return parts[-1] in NON_COMPUTING or (len(parts) > 1 and parts[-2] in NON_COMPUTING)


# THE CALLER THE PRODUCT SEES. `_mode._caller_outside` reads the module
# globals of the frame that calls a public method or function: a user's code
# gets the widened public result, the package's own code (and the identity
# harness, marked `__mojolearn_internal__`) the native one. The wrappers below
# are that frame now, so they are compiled into a namespace carrying the SAME
# marker as the code that reached the view: "internal" for the harness and the
# tape checks, "user" for the public endpoint checks (`_verify_api*`, marked
# `__mojolearn_user_boundary__`). A wrapper therefore never changes which
# result a lane hashes.
# ARGUMENTS REACH THE PRODUCT AS THE PRODUCT'S OWN OBJECTS. A view-wrapped
# function or a BoundClass handed to the product (a metric into make_scorer, a
# class into a meta-estimator) is replaced by the real object it wraps
# (`unwrap`); the side reaches those through the objects built with it.
_WRAPPER_SOURCE = """
def _args(unwrap, args, kwargs):
    return (tuple(unwrap(a) for a in args),
            {k: unwrap(v) for k, v in kwargs.items()})


def inject(fn, side_name, unwrap):
    def call(*args, **kwargs):
        args, kwargs = _args(unwrap, args, kwargs)
        if "device" not in kwargs:
            kwargs["device"] = side_name
        return fn(*args, **kwargs)
    return call


def passthrough(fn, unwrap):
    def call(*args, **kwargs):
        args, kwargs = _args(unwrap, args, kwargs)
        return fn(*args, **kwargs)
    return call


def members(fn, side_name, message, error, unwrap):
    # a `devices=` driver (a GPU list, no `device`): it runs on the side its
    # member estimators carry. Pass the call through only when a member
    # argument carries this side (its `device` parameter); otherwise the side
    # is not selected through the public parameter and the call is refused.
    def call(*args, **kwargs):
        args, kwargs = _args(unwrap, args, kwargs)
        for a in tuple(args) + tuple(kwargs.values()):
            if getattr(a, "device", None) == side_name:
                return fn(*args, **kwargs)
        raise error(message)
    return call


def refuse(fn, message, error, unwrap):
    def call(*args, **kwargs):
        if "device" in kwargs:
            args, kwargs = _args(unwrap, args, kwargs)
            return fn(*args, **kwargs)
        raise error(message)
    return call
"""


def unwrap(value):
    """The product object behind a view wrapper or BoundClass; else `value`."""
    if type(value) is BoundClass:
        return value._cls
    real = getattr(value, "_verify_device_real", None) if callable(value) else None
    return value if real is None else real
BOUNDARIES = ("internal", "user")
_BUILDERS = {}


def _builders(boundary):
    hit = _BUILDERS.get(boundary)
    if hit is None:
        if boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}, not {boundary!r}")
        marker = "__mojolearn_internal__" if boundary == "internal" else "__mojolearn_user_boundary__"
        hit = {"__name__": f"{__name__}.{boundary}", "__builtins__": __builtins__, marker: True}
        exec(compile(_WRAPPER_SOURCE, __file__, "exec"), hit)  # noqa: S102 -- this module's own source
        _BUILDERS[boundary] = hit
    return hit


def _caller(fn, name, side_name, strict, boundary="internal"):
    """`fn` wrapped so a call carries `device=side_name`. Resolved once."""
    mode = device_mode(fn)
    build = _builders(boundary)
    if mode in ("device", "kwargs", "unknown") and not _non_computing(name):
        call = build["inject"](fn, side_name, unwrap)
    elif _non_computing(name) or not strict:
        call = build["passthrough"](fn, unwrap)
    elif mode == "devices":
        call = build["members"](fn, side_name, f"{name} takes devices= and no device=, and no member argument "
                                f"carries device={side_name!r}", NoDevice, unwrap)
    else:
        call = build["refuse"](fn, f"{name} takes no device= (the {side_name.upper()} side cannot be "
                                   "selected through the public parameter)", NoDevice, unwrap)
    try:
        functools.update_wrapper(call, fn)
    except (AttributeError, TypeError):
        pass
    call._verify_device_real = fn
    return call


class BoundClass:
    """A public class whose construction carries the side. Attribute access
    reaches the class; a class attribute that takes `device` (a classmethod
    loader such as `from_checkpoint`) carries the side too. `isinstance` and
    `issubclass` read the real class."""
    __slots__ = ("_cls", "_name", "_side", "_new", "_attrs", "_boundary")

    def __init__(self, cls, name, side_name, strict, boundary="internal"):
        self._cls, self._name, self._side, self._boundary = cls, name, side_name, boundary
        self._new = _caller(cls, name, side_name, strict, boundary)
        self._attrs = {}

    def __call__(self, *args, **kwargs):
        return self._new(*args, **kwargs)

    def __getattr__(self, attr):
        attrs = self._attrs
        if attr in attrs:
            return attrs[attr]
        value = getattr(self._cls, attr)
        if (not attr.startswith("_") and callable(value) and not isinstance(value, type)
                and device_mode(value) == "device"):
            value = _caller(value, f"{self._name}.{attr}", self._side, False, self._boundary)
        attrs[attr] = value
        return value

    @property
    def __signature__(self):
        # `inspect.signature(bound)` reads the class's own parameters
        return inspect.signature(self._cls)

    @property
    def __name__(self):
        return self._cls.__name__

    def __instancecheck__(self, obj):
        return isinstance(obj, self._cls)

    def __subclasscheck__(self, sub):
        return issubclass(sub, self._cls)

    def __repr__(self):
        return f"<{self._name} on the {self._side} side>"


class DeviceView:
    """`module` with every public class and computing function bound to one
    side. Submodules of the package are views too. Private names (`_x`),
    non-callables and modules outside the package pass through unchanged."""

    def __init__(self, module, side_name, strict=True, prefix=None, boundary="internal"):
        if side_name not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, not {side_name!r}")
        object.__setattr__(self, "_boundary", boundary)
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_side", side_name)
        object.__setattr__(self, "_strict", strict)
        object.__setattr__(self, "_prefix", prefix or module.__name__)
        object.__setattr__(self, "_names", {})

    def __getattr__(self, name):
        names = self._names
        if name in names:
            return names[name]
        value = self._wrap(name, getattr(self._module, name))
        names[name] = value
        return value

    def __setattr__(self, name, value):
        raise AttributeError("a device view is read only")

    def __dir__(self):
        return dir(self._module)

    def __repr__(self):
        return f"<{self._prefix} on the {self._side} side>"

    def _wrap(self, name, value):
        qualified = f"{self._prefix}.{name}"
        if name.startswith("_"):
            return value
        if isinstance(value, types.ModuleType):
            root = self._module.__name__.split(".", 1)[0]
            if value.__name__.split(".", 1)[0] != root:
                return value
            return view(value, self._side, self._strict, self._boundary)
        if isinstance(value, type):
            if issubclass(value, BaseException) or name in NON_COMPUTING:
                return value
            return BoundClass(value, qualified, self._side, self._strict, self._boundary)
        if callable(value):
            return _caller(value, qualified, self._side, self._strict, self._boundary)
        return value


#: (id(module), side, strict, boundary) -> (module, view): one view per side
#: per process.
_VIEWS = {}


def view(module, side_name, strict=True, boundary="internal"):
    """The device-bound view of `module` for `side_name`, built once.
    `boundary` is the caller the product sees (`_builders`)."""
    key = (id(module), side_name, bool(strict), boundary)
    hit = _VIEWS.get(key)
    if hit is not None and hit[0] is module:
        return hit[1]
    bound = DeviceView(module, side_name, strict, module.__name__, boundary)
    _VIEWS[key] = (module, bound)
    return bound


def bind(value, name, side_name, strict=True, boundary="internal"):
    """One resolved public object bound to `side_name` the way the view binds
    it (`_verify_api.resolve` uses this for a dotted endpoint name)."""
    if isinstance(value, types.ModuleType):
        return view(value, side_name, strict, boundary)
    if isinstance(value, type):
        if issubclass(value, BaseException) or _short(name) in NON_COMPUTING:
            return value
        return BoundClass(value, name, side_name, strict, boundary)
    if callable(value):
        return _caller(value, name, side_name, strict, boundary)
    return value


def side_call(fn, *args, **kwargs):
    """Call `fn` (a loader such as `type(est).load`) on the current side when
    it takes `device`; unchanged otherwise or when no side is set."""
    current = _SIDE[0]
    if current is not None and "device" not in kwargs and device_mode(fn) == "device":
        kwargs["device"] = current
    return fn(*args, **kwargs)
