# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE GEMM PROFILE SELECTOR: which arithmetic a model's matrix products run.

    mojolearn.gemm_profile()                         # "fp32.v1"
    mojolearn.set_gemm_profile("fp32.v1")            # process default
    mojolearn.models.CausalLM.load(path, gemm_profile="fp32.v1")
    mojolearn.gemm_profiles()                        # every registered row

OPT IN, ALWAYS. The default is `fp32.v1`, the arithmetic every identity
record of this package was produced under, and it does not move. Another
profile runs only where the caller names it (the keyword, the process
default, or `MOJOLEARN_GEMM_PROFILE` for the starting value).

A PROFILE IS A NAME, NEVER A SWITCH INSIDE ANOTHER PROFILE.
`gemm/IDENTICAL_FP32_CONTRACT.md` forbids a flag inside `fp32.v1` that
changes the operand type or the accumulator, and names later profiles as
the way to do it. Each row below is one such name. The short name
(`fp32.v1`) and the full one (`mojolearn.identical.gemm.fp32.v1`) are the
same profile.

`weight_format` IS STORAGE, THIS IS ARITHMETIC. `weight_format="int8"`
with `gemm_profile="fp32.v1"` is what the package has done since
2026-09-17: the weights are stored as codes, materialized exactly, and the
products run in fp32. `gemm_profile="int8i32.v1"` would run the products
on the codes.

QUICK ON ONE VENDOR, SLOW ON ANOTHER. A profile is ONE arithmetic on every
vendor, so it cannot be switched per box: a model that ran `fp32.v1` on
NVIDIA and another profile on Apple would not be the same model. What
differs per box is the time, and `MEASURED` records it per vendor. Choosing
a profile that was measured slower than `fp32.v1` on the vendor in use
raises a `GemmProfileSpeedWarning` once and then runs, because the bits are
what was asked for. WITHIN a profile every plan gives the same bits (the
integer unit, the flat kernel, Apple's exact chunks), so which plan runs is
the dispatcher's choice per box and shape and is no flag at all.

REFUSES, NEVER WIDENS. A registered profile that no model class computes
under yet is refused BY NAME at the line that asked for it. It never falls
back to `fp32.v1`: a caller who asked for one arithmetic and silently got
another would hold results under the wrong name.

A CHECKPOINT CARRIES ITS PROFILE. `state_field` is what a writer stores and
`check_saved` is what a reader calls. A state with no field was written
under `fp32.v1` (every state before this module), and a writer under the
default stores no field, so the bytes of a default checkpoint do not change.

WHAT A ROW MUST SAY BEFORE IT IS AVAILABLE TO MODELS (docs/lanes/
LOWBIT_UNITS_PLAN.md, the three gates): identity on three vendors with a
sabotage arm seen failing, the measured relative change in held-out
perplexity against `fp32.v1` (`quality`, under 1 percent), and an
end-to-end time with the conversion costs counted. `quality` is None until
it is measured; a row never carries an estimate.
"""
import os

__all__ = ["DEFAULT", "PREFIX", "PROFILES", "MEASURED", "GemmProfileSpeedWarning",
           "canonical", "default_profile", "set_default_profile", "resolve",
           "profiles", "measured", "state_field", "check_saved"]

PREFIX = "mojolearn.identical.gemm."
DEFAULT = "fp32.v1"
ENV = "MOJOLEARN_GEMM_PROFILE"

#: name -> row. `models` is True when the model classes compute under the
#: profile; `gemm` is True when the GEMM itself exists and is gated
#: (`mojolearn.linalg`). `quality` is the measured relative change in
#: held-out perplexity against fp32.v1 as a fraction, or None (not measured).
PROFILES = {
    "fp32.v1": {
        "operands": "float32 x float32, float32 accumulator",
        "contract": "gemm/IDENTICAL_FP32_CONTRACT.md",
        "gemm": True, "models": True, "quality": 0.0,
        "note": "the default; every identity record of the package",
    },
    "bf16f32.v1": {
        "operands": "bfloat16 widened exactly, fp32.v1 arithmetic",
        "contract": "gemm/IDENTICAL_LOWBIT_CONTRACT.md",
        "gemm": True, "models": False, "quality": None,
        "note": "gated at the GEMM; no model class computes under it yet",
    },
    "int8i32.v1": {
        "operands": "int8 codes x int8 codes, Int32 accumulator, one exact scale per row",
        "contract": "gemm/IDENTICAL_LOWBIT_CONTRACT.md",
        "gemm": True, "models": False, "quality": None,
        "note": "gated at the GEMM, on the integer matrix units of NVIDIA and AMD; "
                "no model class computes under it yet",
    },
}

#: MEASURED TIME, because a profile is quick on one vendor and slow on
#: another. profile -> vendor (what `mojolearn.vendor()` answers) -> a row:
#: `over` is the range, over the shapes timed, of the profile's time over
#: fp32.v1's at the same shape on the same box (above 1 it took LONGER),
#: `what` says which operation was timed and `source` where the run is.
#: ONLY WHAT WAS MEASURED: a vendor with no row reads "not measured", never
#: a number borrowed from another box. The profile is the same bits on every
#: vendor whatever these say; they are why the choice is the caller's.
MEASURED = {
    "int8i32.v1": {
        "cuda": {"over": (2.06, 4.63), "box": "NVIDIA H100",
                 "what": "GEMM on the integer unit with activations quantized per call, training rows",
                 "source": "bench/results/lowbit_units/2026-09-29/TABLE.md"},
        "metal": {"over": (0.51, 0.68), "box": "Apple M2 Pro",
                  "what": "GEMM on the float unit in exact chunks with activations quantized per call, "
                          "training rows; a probe, not dispatched",
                  "source": "bench/results/lowbit_units/2026-09-29/TABLE.md"},
    },
}


class GemmProfileSpeedWarning(UserWarning):
    """The selected profile was measured to take longer than fp32.v1 on this
    vendor. The bits are the profile's either way."""


_warned = set()
_default = None


def measured(profile, vendor=None):
    """The measured rows of `profile`: every vendor's, or one vendor's row
    (None when that vendor was not measured)."""
    rows = MEASURED.get(canonical(profile), {})
    return dict(rows) if vendor is None else rows.get(vendor)


def _this_vendor():
    try:
        from . import _backend
        return _backend.vendor()
    except Exception:  # noqa: BLE001  (loaded by path, or no binding on this box)
        return None


def _warn_if_slow_here(short, vendor=None):
    """Once per profile and vendor: say so when the profile was measured to
    take longer than fp32.v1 on the vendor this process runs on."""
    vendor = _this_vendor() if vendor is None else vendor
    row = MEASURED.get(short, {}).get(vendor) if vendor else None
    if row is None or row["over"][0] <= 1.0 or (short, vendor) in _warned:
        return None
    _warned.add((short, vendor))
    import warnings
    lo, hi = row["over"]
    msg = (f"mojolearn: GEMM profile {short!r} was measured on {row['box']} to take {lo:g} to {hi:g} "
           f"times fp32.v1's time ({row['what']}; {row['source']}). Its bits are the same on every "
           f"vendor; on this one it costs time.")
    warnings.warn(msg, GemmProfileSpeedWarning, stacklevel=3)
    return msg


def canonical(name, what="gemm_profile"):
    """The short name of a registered profile, from its short or full name.
    Anything else is refused with the registered names."""
    if not isinstance(name, str):
        raise TypeError(f"mojolearn: {what} must be a profile name (a str), got {type(name).__name__}")
    short = name.strip()
    if short.startswith(PREFIX):
        short = short[len(PREFIX):]
    if short not in PROFILES:
        raise ValueError(f"mojolearn: {what}={name!r} is not a registered GEMM profile; "
                         f"registered: {', '.join(PROFILES)}")
    return short


def _refuse_unavailable(short, what):
    row = PROFILES[short]
    if not row["models"]:
        raise NotImplementedError(
            f"mojolearn: {what}={short!r} is registered ({row['note']}) and is refused here; "
            f"it is never replaced by {DEFAULT!r} silently. Use gemm_profile={DEFAULT!r}, "
            f"or mojolearn.linalg for the product itself.")
    return short


def default_profile():
    """The profile a call that names none runs under. Starts at
    `MOJOLEARN_GEMM_PROFILE` when that is set, else `fp32.v1`."""
    global _default
    if _default is None:
        env = os.environ.get(ENV, "").strip()
        _default = _refuse_unavailable(canonical(env, ENV), ENV) if env else DEFAULT
    return _default


def set_default_profile(name):
    """Choose the process default IN CODE. Returns the previous value. A
    name that cannot be honored fails HERE, not at a later forward."""
    global _default
    prev = default_profile()
    _default = _refuse_unavailable(canonical(name, "set_gemm_profile"), "set_gemm_profile")
    return prev


def resolve(name=None, what="gemm_profile"):
    """What a model class calls with its keyword: None is the process
    default, a name is validated and refused by name when unavailable."""
    short = default_profile() if name is None else _refuse_unavailable(canonical(name, what), what)
    _warn_if_slow_here(short)
    return short


def profiles():
    """Every registered row, the default first, as plain dicts, each with
    its measured times per vendor (`measured`, empty when none)."""
    return tuple({"name": k, "full_name": PREFIX + k, "default": k == DEFAULT, **v,
                  "measured": dict(MEASURED.get(k, {}))}
                 for k, v in PROFILES.items())


def state_field(profile):
    """What a checkpoint writer merges into its state: nothing under the
    default (so a default checkpoint's bytes do not change), the full name
    otherwise."""
    short = canonical(profile, "gemm_profile")
    return {} if short == DEFAULT else {"gemm_profile": PREFIX + short}


def check_saved(state, profile, what="state"):
    """Refuse by name a state written under another profile than the one
    this object computes under. A state with no field is `fp32.v1`."""
    saved = state.get("gemm_profile") if hasattr(state, "get") else None
    saved = DEFAULT if saved is None else canonical(saved, f"{what} gemm_profile")
    mine = canonical(profile, "gemm_profile")
    if saved != mine:
        raise ValueError(f"mojolearn: {what} was written under GEMM profile {saved!r} and this "
                         f"object computes under {mine!r}; a run keeps one arithmetic, "
                         f"so build the object with gemm_profile={saved!r}")
    return saved
