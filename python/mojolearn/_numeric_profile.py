# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`numeric_profile=`: which arithmetic a model's matrix products run.

    mojolearn.numeric_profile()                          # "fixed15_v1"
    mojolearn.set_numeric_profile("fp32_v1")             # process default: the old behaviour
    mojolearn.models.CausalLM.load(path, numeric_profile="fp32_v1")
    mojolearn.numeric_profiles()                         # every registered row

    numeric_profile="fixed15_v1"    the inference default of the transformer models
    numeric_profile="fp32_v1"       the baseline: every other family, every trainer

ITS OWN PARAMETER, NOT A VALUE OF `numeric_mode`. `numeric_mode` is the
PROMISE (identical, deterministic, fast). `numeric_profile` is the NUMBER
FORMAT the matrix products compute in. They are independent: a profile is
bitwise identical across vendors under `numeric_mode="identical"` exactly
as `fp32_v1` is, and picking one never changes the mode.

THE DEFAULT, PER USE AND PER FAMILY (Andrew, 2026-09-29: "make this new
change the default even if apple is slower"). A call that names no profile
gets:
  - INFERENCE, in a family the default reaches (`default_for`: the
    transformer models of `mojolearn.models`, `CausalLM` and
    `ParallelCausalLM`): `DEFAULT`, `fixed15_v1`.
  - INFERENCE, in any other family (the Mamba models, a bare
    `TransformerBlock`, which is also the trainers' building block):
    `BASELINE`, `fp32_v1`, exactly as before, and the object's
    `numeric_profile` attribute says `fp32_v1`. Never refused, never
    relabelled.
  - TRAINING: `TRAINING_DEFAULT`, `fp32_v1`, whatever the inference
    default is.
A caller who NAMES a profile gets that profile or a refusal by name: a
family the profile does not compute (`computes`) and a use the row has not
passed are refused, never widened. `fp32_v1` is the arithmetic every
identity record before 2026-09-29 was produced under; the old behaviour
comes back bit for bit with `numeric_profile="fp32_v1"`,
`set_numeric_profile("fp32_v1")` or `MOJOLEARN_NUMERIC_PROFILE=fp32_v1`
(which set the INFERENCE default; the training default is `fp32_v1`
already).

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

A CHECKPOINT CARRIES ITS PROFILE. `state_field` is what a writer stores;
`adopt_saved` and `check_saved` are what a reader calls. A state with no
field was written under `fp32_v1` (every state before this module), and a
writer under `fp32_v1` still stores no field, so the bytes of an `fp32_v1`
checkpoint do not change; a state written under `fixed15_v1` (the inference
default) carries the field. A reader whose caller named no profile ADOPTS
the state's own (`adopt_saved`); one that named another is refused by name.
A model's weight file (a Hugging Face checkpoint) is not such a state: it
holds weights, not the results of an arithmetic, and carries no field.

INFERENCE AND TRAINING ARE SEPARATE GATES. A row says `inference` and
`training` apart, and each use has its own default. Every trainer resolves
the TRAINING default, so the inference default never reaches a trainer; a
trainer asked by name for a profile whose `training` is False refuses it.

WHAT A ROW MUST SHOW BEFORE `inference` IS TRUE (docs/lanes/
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
           "state_field", "check_saved", "adopt_saved", "STATE_KEY", "USES",
           "require_training", "BASELINE", "TRAINING_DEFAULT", "FAMILIES",
           "computes", "default_reaches"]

#: The arithmetic every family computes under and every identity record
#: before 2026-09-29 was produced under.
BASELINE = "fp32_v1"
#: The INFERENCE default (2026-09-29; `fp32_v1` before). It reaches only the
#: families its row names in `default_for`.
DEFAULT = "fixed15_v1"
#: The TRAINING default. No other profile has passed its training gates.
TRAINING_DEFAULT = "fp32_v1"
#: The families a caller passes to `resolve`:
#:   transformer        a `mojolearn.models` transformer model (CausalLM, ParallelCausalLM)
#:   mamba1, mamba2     a `mojolearn.models` Mamba model
#:   transformer_block  a bare `TransformerBlock` (also the trainers' building block)
FAMILIES = ("transformer", "mamba1", "mamba2", "transformer_block")
ENV = "MOJOLEARN_NUMERIC_PROFILE"
STATE_KEY = "numeric_profile"

#: name -> row.
#:   products   product family -> the GEMM profile it runs (the contract name)
#:   inference  True when the inference classes compute under the profile
#:   training   True when the trainers do. A SEPARATE gate: a profile that
#:              passed for inference is refused by every trainer until its
#:              training gates pass too (backward products, a training
#:              step's identity, quality over seeds)
#:   quality   text -> the measured relative change in held-out perplexity
#:             against fp32_v1, as a fraction, the WHOLE configuration at
#:             once; empty when not measured
#:   computes     the families (`FAMILIES`) that compute under the profile;
#:                None is every family. A family outside it asked BY NAME is
#:                refused by name
#:   default_for  the families a call naming no profile gets this one in,
#:                when it is the inference default; the rest get BASELINE
PROFILES = {
    "fp32_v1": {
        "status": "baseline",
        "products": {"projections": "mojolearn.identical.gemm.fp32.v1",
                     "attention": "mojolearn.identical.gemm.fp32.v1"},
        "inference": True, "training": True,
        "computes": None, "default_for": None,
        "quality": {},
        "quality_note": "the baseline every other profile is measured against",
        "note": "the baseline: every family, every trainer; the default before 2026-09-29 "
                "and every identity record made before then",
    },
    "fixed15_v1": {
        "status": "default",
        "products": {"projections": "mojolearn.identical.gemm.int15i64.v1",
                     "attention": "mojolearn.identical.gemm.int15i64.v1"},
        "inference": True, "training": False,
        "computes": ("transformer", "transformer_block"),
        "default_for": ("transformer",),
        "quality": {"enwik8, last 1 MB": 0.000006, "pile_github": 0.000033},
        "quality_note": "SmolLM2-360M through CausalLM, the whole model computing under the "
                        "profile against the same model under fp32_v1, 2026-09-29 "
                        "(lane/lowbit-blocks (e), nvc2-0019; intervals -0.0044% to +0.0055% "
                        "and -0.0010% to +0.0076%); training and a task evaluation are owed",
        "note": "15-bit integer codes with one power-of-two scale per row, exact integer "
                "sums (the projections, the head and Q.K^T; P.V stays on fp32.v1); the "
                "inference default of the transformer models since 2026-09-29; the same bits "
                "on NVIDIA, AMD, Apple and the CPU",
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
#: a number borrowed from another box or from another profile.
MEASURED = {
    "fixed15_v1": {
        "cuda": {"over": (0.34, 0.68), "box": "RTX 4090 (0.34 to 0.40) and H100 NVL (0.64 to 0.68)",
                 "what": "SmolLM2-360M, CausalLM.forward, prefill of 512 tokens at B=1, the whole "
                         "model (decode per token 0.30 to 0.48)",
                 "source": "lane/lowbit-blocks (f), jobs nvc2-0020 and nvc3-0038, "
                           "docs/lanes/progress/lowbit-blocks.md"},
        "hip": {"over": (0.85, 0.89), "box": "MI325X",
                "what": "SmolLM2-360M, CausalLM.forward, prefill of 512 tokens at B=1, the whole "
                        "model (decode per token 0.63 to 0.65)",
                "source": "lane/lowbit-blocks (f), steward job 1790660933293, "
                          "docs/lanes/progress/lowbit-blocks.md"},
        "metal": {"over": (3.40, 3.65), "box": "M3 Ultra (3.40 to 3.65) and M2 Pro (3.44 to 3.57)",
                  "what": "the complete 15-bit inference GEMM call at SmolLM2-360M's four "
                          "512-token rows (qkv, mlp_up, mlp_down, lm_head), the plan this "
                          "release dispatches on Apple; the whole model was not timed on Apple",
                  "source": "lane/lowbit-apple-tuned job 1 (19ad540d3), "
                            "bench/results/lowbit_apple_tuned/job1_19ad540d3"},
    },
}


class NumericProfileSpeedWarning(UserWarning):
    """The selected profile was measured to take longer than fp32_v1 on this
    vendor. The bits are the profile's either way."""


_warned = set()


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
           f"times {BASELINE}'s time ({row['what']}; {row['source']}). Its bits are the same on every "
           f"vendor; on this one it costs time. numeric_profile={BASELINE!r} (or "
           f"{ENV}={BASELINE}) gives the previous arithmetic.")
    warnings.warn(msg, NumericProfileSpeedWarning, stacklevel=3)
    return msg


USES = ("inference", "training")


def _refuse_unavailable(key, what, use="inference"):
    if use not in USES:
        raise ValueError(f"mojolearn: use must be one of {USES}, got {use!r}")
    row = PROFILES[key]
    if not row[use]:
        other = "inference" if use == "training" else "training"
        also = f" It is available for {other}." if row[other] else ""
        raise NotImplementedError(
            f"mojolearn: {what}={key!r} is registered ({row['note']}) and is refused for {use}; "
            f"it is never replaced by {BASELINE!r} silently.{also} Use numeric_profile={BASELINE!r}.")
    return key


def _family(family):
    if family is not None and family not in FAMILIES:
        raise ValueError(f"mojolearn: family must be one of {FAMILIES} or None, got {family!r}")
    return family


def computes(profile, family):
    """True when `family` computes under `profile`."""
    fams = PROFILES[canonical(profile)]["computes"]
    return fams is None or _family(family) in fams


def default_reaches(profile, family):
    """True when a call in `family` that names no profile gets `profile`
    while it is the inference default."""
    fams = PROFILES[canonical(profile)]["default_for"]
    return fams is None or _family(family) in fams


_defaults = {}


def default_profile(use="inference"):
    """The profile a call that names none runs under, for `use`. The
    inference default starts at `MOJOLEARN_NUMERIC_PROFILE` when that is
    set, else `DEFAULT`; the training default is `TRAINING_DEFAULT`. A
    family the inference default does not reach gets `BASELINE`
    (`resolve`)."""
    if use not in USES:
        raise ValueError(f"mojolearn: use must be one of {USES}, got {use!r}")
    if use not in _defaults:
        if use == "inference":
            env = os.environ.get(ENV, "").strip()
            _defaults[use] = _refuse_unavailable(canonical(env, ENV), ENV) if env else DEFAULT
        else:
            _defaults[use] = TRAINING_DEFAULT
    return _defaults[use]


def set_default_profile(name, use="inference"):
    """Choose the process default for `use` IN CODE. Returns the previous
    value. A name that cannot be honored for `use` fails HERE, not at a
    later forward. `set_default_profile("fp32_v1")` is the arithmetic every
    call made before 2026-09-29."""
    prev = default_profile(use)
    _defaults[use] = _refuse_unavailable(canonical(name, "set_numeric_profile"),
                                         "set_numeric_profile", use)
    return prev


def resolve(name=None, what="numeric_profile", use="inference", family=None):
    """What a class calls with its keyword. None is the process default for
    `use`, and in a `family` the default does not reach it is `BASELINE`
    (the object reports it: its `numeric_profile` attribute is what this
    returns). A NAME is validated, and refused by name when the profile is
    not available for `use` or does not compute `family`; it is never
    replaced by another profile."""
    _family(family)
    if name is None:
        key = default_profile(use)
        if family is not None and not (computes(key, family) and default_reaches(key, family)):
            key = BASELINE
    else:
        key = canonical(name, what)
        if family is not None and not computes(key, family):
            raise NotImplementedError(
                f"mojolearn: {what}={key!r} does not compute the {family!r} family (it computes "
                f"{', '.join(PROFILES[key]['computes'])}); it is never replaced by {BASELINE!r} "
                f"silently. Use numeric_profile={BASELINE!r}.")
    _refuse_unavailable(key, what, use)
    if key != BASELINE:
        _warn_if_slow_here(key)
    return key


def require_training(what, name=None):
    """One line for a trainer's constructor: the TRAINING default when the
    caller named nothing (the inference default never reaches a trainer),
    else the named profile, refused by name unless it has passed for
    training."""
    return resolve(name, what, use="training")


def profiles():
    """Every registered row, the baseline first, as plain dicts, each with
    its measured times per vendor (`measured`, empty when none). `default`
    marks the inference default, `training_default` the training one."""
    return tuple({"name": k, "default": k == DEFAULT, "training_default": k == TRAINING_DEFAULT,
                  **v, "products": dict(v["products"]),
                  "computes": None if v["computes"] is None else tuple(v["computes"]),
                  "default_for": None if v["default_for"] is None else tuple(v["default_for"]),
                  "quality": dict(v["quality"]), "measured": dict(MEASURED.get(k, {}))}
                 for k, v in PROFILES.items())


def state_field(profile):
    """What a checkpoint writer merges into its state: nothing under
    `fp32_v1` (so an fp32_v1 checkpoint's bytes do not change, and a state
    with no field reads as fp32_v1), the name otherwise; a state written
    under the inference default `fixed15_v1` carries it."""
    key = canonical(profile)
    return {} if key == BASELINE else {STATE_KEY: key}


def _saved(state, what):
    saved = state.get(STATE_KEY) if hasattr(state, "get") else None
    return BASELINE if saved is None else canonical(saved, f"{what} {STATE_KEY}")


def check_saved(state, profile, what="state"):
    """Refuse by name a state written under another profile than the one
    this object computes under. A state with no field is `fp32_v1`."""
    saved = _saved(state, what)
    mine = canonical(profile)
    if saved != mine:
        raise ValueError(f"mojolearn: {what} was written under numeric profile {saved!r} and this "
                         f"object computes under {mine!r}; a run keeps one arithmetic, "
                         f"so build the object with numeric_profile={saved!r}")
    return saved


def adopt_saved(state, name=None, what="state", use="inference"):
    """What a reader calls when the object it builds takes its profile FROM
    the state: the caller named nothing, so the state's own profile (a state
    with no field is `fp32_v1`, whatever the process default is now); a
    caller who named a different one is refused by name. The adopted
    profile must still be available for `use`."""
    saved = _saved(state, what)
    if name is not None:
        check_saved(state, name, what)
    return _refuse_unavailable(saved, f"{what} {STATE_KEY}", use)
