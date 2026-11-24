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


def _caller(fn, name, side_name, strict):
    """`fn` wrapped so a call carries `device=side_name`. Resolved once."""
    mode = device_mode(fn)
    if mode in ("device", "kwargs", "unknown"):
        def call(*args, **kwargs):
            if "device" not in kwargs:
                kwargs["device"] = side_name
            return fn(*args, **kwargs)
    elif _short(name) in NON_COMPUTING or not strict:
        return fn
    else:
        def call(*args, **kwargs):
            if "device" in kwargs:
                return fn(*args, **kwargs)
            raise NoDevice(f"{name} takes no device= (the {side_name.upper()} side cannot be "
                           "selected through the public parameter)")
    try:
        functools.update_wrapper(call, fn)
    except (AttributeError, TypeError):
        pass
    return call


class BoundClass:
    """A public class whose construction carries the side. Attribute access
    reaches the class; a class attribute that takes `device` (a classmethod
    loader such as `from_checkpoint`) carries the side too. `isinstance` and
    `issubclass` read the real class."""
    __slots__ = ("_cls", "_name", "_side", "_new", "_attrs")

    def __init__(self, cls, name, side_name, strict):
        self._cls, self._name, self._side = cls, name, side_name
        self._new = _caller(cls, name, side_name, strict)
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
            value = _caller(value, f"{self._name}.{attr}", self._side, False)
        attrs[attr] = value
        return value

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

    def __init__(self, module, side_name, strict=True, prefix=None):
        if side_name not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, not {side_name!r}")
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_side", side_name)
        object.__setattr__(self, "_strict", strict)
        object.__setattr__(self, "_prefix", prefix or module.__name__)
        object.__setattr__(self, "_names", {})

    @property
    def side(self):
        return self._side

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
            return DeviceView(value, self._side, self._strict, value.__name__)
        if isinstance(value, type):
            if issubclass(value, BaseException) or name in NON_COMPUTING:
                return value
            return BoundClass(value, qualified, self._side, self._strict)
        if callable(value):
            return _caller(value, qualified, self._side, self._strict)
        return value


#: (id(module), side, strict) -> (module, view): one view per side per process.
_VIEWS = {}


def view(module, side_name, strict=True):
    """The device-bound view of `module` for `side_name`, built once."""
    key = (id(module), side_name, bool(strict))
    hit = _VIEWS.get(key)
    if hit is not None and hit[0] is module:
        return hit[1]
    bound = DeviceView(module, side_name, strict)
    _VIEWS[key] = (module, bound)
    return bound


def bind(value, name, side_name, strict=True):
    """One resolved public object bound to `side_name` the way the view binds
    it (`_verify_api.resolve` uses this for a dotted endpoint name)."""
    if isinstance(value, types.ModuleType):
        return view(value, side_name, strict)
    if any(part.startswith("_") for part in name.split(".")[1:-1]) and not isinstance(value, type):
        return value
    if isinstance(value, type):
        if issubclass(value, BaseException) or _short(name) in NON_COMPUTING:
            return value
        return BoundClass(value, name, side_name, strict)
    if callable(value):
        return _caller(value, name, side_name, strict)
    return value


def side_call(fn, *args, **kwargs):
    """Call `fn` (a loader such as `type(est).load`) on the current side when
    it takes `device`; unchanged otherwise or when no side is set."""
    current = _SIDE[0]
    if current is not None and "device" not in kwargs and device_mode(fn) == "device":
        kwargs["device"] = current
    return fn(*args, **kwargs)
