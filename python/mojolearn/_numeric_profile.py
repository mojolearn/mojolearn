# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`numeric_profile=`: which arithmetic a model's matrix products run.

    mojolearn.numeric_profile()                          # "fp32_v1"
    mojolearn.set_numeric_profile("fp32_v1")             # process default
    mojolearn.models.CausalLM.load(path, numeric_profile="fp32_v1")
    mojolearn.numeric_profiles()                         # every registered row

    numeric_profile="fp32_v1"       the default
    numeric_profile="fixed15_v1"    experimental

ITS OWN PARAMETER, NOT A VALUE OF `numeric_mode`. `numeric_mode` is the
PROMISE (identical, deterministic, fast). `numeric_profile` is the NUMBER
FORMAT the matrix products compute in. They are independent: a profile is
bitwise identical across vendors under `numeric_mode="identical"` exactly
as `fp32_v1` is, and picking one never changes the mode.

OPT IN, ALWAYS. The default is `fp32_v1`, the arithmetic every identity
record of this package was produced under, and it does not move. Another
profile runs only where the caller names it (the keyword, the process
default, or `MOJOLEARN_NUMERIC_PROFILE` for the starting value).

A NAME IS ONE ARITHMETIC, FOREVER. Every name carries its version. A changed
arithmetic is a new version (`_v2`), never a new meaning of an old name, and
there is no unversioned alias: "fixed15" meaning one thing in one release
and another in the next would change a model's bits without anyone asking.

WHAT IS OFFERED IS WHAT PASSED. A profile is registered only if its
arithmetic kept quality when it was measured. One that was measured and
FAILED is not offered at all; it sits in `REJECTED` with the number, so that
asking for it answers with the reason instead of "unknown name". int8 codes
on both operands is the first entry: perplexity rose by about a third. The
int8 GEMM itself stays in `mojolearn.linalg`, and it is the piece product
`fixed15_v1` is built from; what is not offered is int8 as a model's
arithmetic.

A PROFILE SAYS WHICH GEMM EACH PRODUCT FAMILY RUNS (`products`). The GEMM
profiles are the contracts' (`gemm/IDENTICAL_FP32_CONTRACT.md`,
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`); `gemm/IDENTICAL_FP32_CONTRACT.md`
forbids a flag inside `fp32.v1` that changes the operand type or the
accumulator, and names later profiles as the way to do it.

`weight_format` IS STORAGE, THIS IS ARITHMETIC. `weight_format="int8"` with
the default profile is what the package has done since 2026-09-17: the
weights are stored as codes, materialized exactly, and the products run in
fp32.

QUICK ON ONE VENDOR, SLOW ON ANOTHER. A profile is ONE arithmetic on every
vendor, so it cannot be switched per box: a model that ran `fp32_v1` on
NVIDIA and another profile on Apple would not be the same model. What
differs per box is the time, and `MEASURED` records it per vendor. Choosing
a profile that was measured slower than `fp32_v1` on the vendor in use
raises a `NumericProfileSpeedWarning` once and then runs, because the bits
are what was asked for. WITHIN a profile every plan gives the same bits (the
integer unit, the flat kernel, Apple's exact chunks), so which plan runs is
the dispatcher's choice per box and shape and is no flag at all.

REFUSES, NEVER WIDENS. A registered profile that no model class computes
under yet is refused BY NAME at the line that asked for it. It never falls
back to `fp32_v1`: a caller who asked for one arithmetic and silently got
another would hold results under the wrong name.

A CHECKPOINT CARRIES ITS PROFILE. `state_field` is what a writer stores and
`check_saved` is what a reader calls. A state with no field was written
under `fp32_v1` (every state before this module), and a writer under the
default stores no field, so the bytes of a default checkpoint do not change.

WHAT A ROW MUST SHOW BEFORE `models` IS TRUE (docs/lanes/
LOWBIT_UNITS_PLAN.md, the three gates): identity on three vendors with a
sabotage arm seen failing; the relative change in held-out perplexity
against `fp32_v1`, measured as ONE complete configuration on two texts,
with the change and the upper end of its interval both under 1 percent; and
an end-to-end time with the conversion costs counted. `quality` holds only
what was measured; a row never carries an estimate.
"""
import os

__all__ = ["DEFAULT", "ENV", "PROFILES", "REJECTED", "MEASURED",
           "NumericProfileSpeedWarning", "canonical", "default_profile",
           "set_default_profile", "resolve", "profiles", "measured",
           "state_field", "check_saved", "STATE_KEY"]

DEFAULT = "fp32_v1"
ENV = "MOJOLEARN_NUMERIC_PROFILE"
STATE_KEY = "numeric_profile"

#: name -> row.
#:   products  product family -> the GEMM profile it runs (the contract name)
#:   models    True when the model classes compute under the profile
#:   quality   text -> the measured relative change in held-out perplexity
#:             against fp32_v1, as a fraction, the WHOLE configuration at
#:             once; empty when not measured
PROFILES = {
    "fp32_v1": {
        "status": "default",
        "products": {"projections": "mojolearn.identical.gemm.fp32.v1",
                     "attention": "mojolearn.identical.gemm.fp32.v1"},
        "models": True,
        "quality": {},
        "quality_note": "the baseline every other profile is measured against",
        "note": "the default; every identity record of the package",
    },
    "fixed15_v1": {
        "status": "experimental",
        "products": {"projections": "mojolearn.identical.gemm.int15i64.v1",
                     "attention": "mojolearn.identical.gemm.int15i64.v1"},
        "models": False,
        "quality": {"enwik8, last 1 MB": -0.000015, "pile_github": 0.000055},
        "quality_note": "SmolLM2-360M, inference, 2026-09-29 (lane/lowbit-quality); "
                        "training and a task evaluation are owed",
        "note": "15-bit integer codes with one power-of-two scale per row, exact integer "
                "sums; its GEMM is being built and no model class computes under it yet",
    },
}

#: NOT OFFERED. name -> why, and each reason says whether the arithmetic was
#: measured. Asking for one of these is refused with the reason. An
#: arithmetic leaves this table only by being measured under a NEW name.
REJECTED = {
    "int8_v1": "MEASURED: int8 codes on both operands of every product (`int8i32.v1`) raised "
               "held-out perplexity by 32.2 percent on enwik8 and 28.9 percent on pile_github "
               "(SmolLM2-360M, 2026-09-29), against a bar of 1 percent; the int8 activation "
               "codes cause nearly all of it",
    "fixed15_int8_attention_v1": "DROPPED BEFORE ITS WHOLE CONFIGURATION WAS MEASURED (Andrew, "
                                 "2026-09-29): int8 codes on the attention products alone read "
                                 "+0.6 percent on enwik8, which leaves little of a 1 percent bar, "
                                 "and int8 is not offered anywhere else",
}

#: MEASURED TIME, because a profile is quick on one vendor and slow on
#: another. profile -> vendor (what `mojolearn.vendor()` answers) -> a row:
#: `over` is the range, over the shapes timed, of the profile's time over
#: fp32_v1's at the same shape on the same box (above 1 it took LONGER),
#: `what` says which operation was timed and `source` where the run is.
#: ONLY WHAT WAS MEASURED: a vendor with no row reads "not measured", never
#: a number borrowed from another box or from another profile. Empty today:
#: no registered profile's complete operation has been timed yet.
MEASURED = {}


class NumericProfileSpeedWarning(UserWarning):
    """The selected profile was measured to take longer than fp32_v1 on this
    vendor. The bits are the profile's either way."""


_warned = set()
_default = None


def canonical(name, what="numeric_profile"):
    """The name of a registered profile. A rejected arithmetic is refused
    with its measured reason, anything else with the registered names."""
    if not isinstance(name, str):
        raise TypeError(f"mojolearn: {what} must be a profile name (a str), got {type(name).__name__}")
    key = name.strip()
    if key in PROFILES:
        return key
    if key in REJECTED:
        raise ValueError(f"mojolearn: {what}={name!r} is not offered: {REJECTED[key]}. "
                         f"Registered: {', '.join(PROFILES)}")
    raise ValueError(f"mojolearn: {what}={name!r} is not a registered numeric profile; "
                     f"registered: {', '.join(PROFILES)}")


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


def _warn_if_slow_here(key, vendor=None):
    """Once per profile and vendor: say so when the profile was measured to
    take longer than fp32_v1 on the vendor this process runs on."""
    vendor = _this_vendor() if vendor is None else vendor
    row = MEASURED.get(key, {}).get(vendor) if vendor else None
    if row is None or row["over"][0] <= 1.0 or (key, vendor) in _warned:
        return None
    _warned.add((key, vendor))
    import warnings
    lo, hi = row["over"]
    msg = (f"mojolearn: numeric profile {key!r} was measured on {row['box']} to take {lo:g} to {hi:g} "
           f"times {DEFAULT}'s time ({row['what']}; {row['source']}). Its bits are the same on every "
           f"vendor; on this one it costs time.")
    warnings.warn(msg, NumericProfileSpeedWarning, stacklevel=3)
    return msg


def _refuse_unavailable(key, what):
    row = PROFILES[key]
    if not row["models"]:
        raise NotImplementedError(
            f"mojolearn: {what}={key!r} is registered ({row['note']}) and is refused here; "
            f"it is never replaced by {DEFAULT!r} silently. Use numeric_profile={DEFAULT!r}.")
    return key


def default_profile():
    """The profile a call that names none runs under. Starts at
    `MOJOLEARN_NUMERIC_PROFILE` when that is set, else `fp32_v1`."""
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
    _default = _refuse_unavailable(canonical(name, "set_numeric_profile"), "set_numeric_profile")
    return prev


def resolve(name=None, what="numeric_profile"):
    """What a model class calls with its keyword: None is the process
    default, a name is validated and refused by name when unavailable."""
    key = default_profile() if name is None else _refuse_unavailable(canonical(name, what), what)
    _warn_if_slow_here(key)
    return key


def profiles():
    """Every registered row, the default first, as plain dicts, each with
    its measured times per vendor (`measured`, empty when none)."""
    return tuple({"name": k, "default": k == DEFAULT, **v, "products": dict(v["products"]),
                  "quality": dict(v["quality"]), "measured": dict(MEASURED.get(k, {}))}
                 for k, v in PROFILES.items())


def state_field(profile):
    """What a checkpoint writer merges into its state: nothing under the
    default (so a default checkpoint's bytes do not change), the name
    otherwise."""
    key = canonical(profile)
    return {} if key == DEFAULT else {STATE_KEY: key}


def check_saved(state, profile, what="state"):
    """Refuse by name a state written under another profile than the one
    this object computes under. A state with no field is `fp32_v1`."""
    saved = state.get(STATE_KEY) if hasattr(state, "get") else None
    saved = DEFAULT if saved is None else canonical(saved, f"{what} {STATE_KEY}")
    mine = canonical(profile)
    if saved != mine:
        raise ValueError(f"mojolearn: {what} was written under numeric profile {saved!r} and this "
                         f"object computes under {mine!r}; a run keeps one arithmetic, "
                         f"so build the object with numeric_profile={saved!r}")
    return saved
