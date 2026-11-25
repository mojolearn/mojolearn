# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`numeric_profile=`: which arithmetic a model's matrix products run.

    mojolearn.numeric_profile()                          # "fixed15_v1"
    mojolearn.set_numeric_profile("fp32_v1")             # the inference opt-out
    mojolearn.models.CausalLM.load(path, numeric_profile="fp32_v1")
    mojolearn.models.CausalLM.load(path, numeric_profile={"preset": "fixed15_v1", "lm_head": "fp32_v1"})
    mojolearn.numeric_profiles()                         # every registered row

    numeric_profile="fixed15_v1"    the transformer models' inference default
    numeric_profile="fp32_v1"       the opt-out; every other family, every trainer

ITS OWN PARAMETER, NOT A VALUE OF `numeric_mode`. `numeric_mode` is the
PROMISE (identical, fast). `numeric_profile` is the NUMBER
FORMAT the matrix products compute in. They are independent: a profile is
bitwise identical across vendors under `numeric_mode="identical"` exactly
as `fp32_v1` is, and picking one never changes the mode.

THE DEFAULT (Andrew, 2026-11-25: fixed15_v1 is the default inference
arithmetic of the transformer models on every vendor). Decided on the
measurements in the `fixed15_v1` row: held-out perplexity of SmolLM2-360M
changed by +0.0006 % (enwik8) and +0.0033 % (pile_github) against
`fp32_v1`, and `MEASURED` times it at 0.34 to 0.68 of `fp32_v1`'s on NVIDIA
and 0.85 to 0.89 on AMD. On Apple it takes 3.40 to 3.65 times as long;
that cost is accepted, and the `NumericProfileSpeedWarning` still says so
there. A call that names no profile gets:
  - INFERENCE, in a family the default reaches (`default_for`: the
    transformer models of `mojolearn.models`, `CausalLM` and
    `ParallelCausalLM`): the process inference default, `fixed15_v1`
    unless `set_numeric_profile` or `MOJOLEARN_NUMERIC_PROFILE` set
    another.
  - INFERENCE, in any other family (the Mamba models, a bare
    `TransformerBlock`, which is also the trainers' building block):
    `BASELINE`, `fp32_v1`, and the object's `numeric_profile` attribute
    says `fp32_v1`. Never refused, never relabelled.
  - TRAINING: `TRAINING_DEFAULT`, `fp32_v1`, whatever the inference
    default is.
A caller who NAMES a profile gets that profile or a refusal by name: a
family the profile does not compute (`computes`) and a use the row has not
passed are refused, never widened. `fp32_v1` is the arithmetic every
identity record before 2026-09-29 was produced under; it comes back bit
for bit with `numeric_profile="fp32_v1"`, `set_numeric_profile("fp32_v1")`
or `MOJOLEARN_NUMERIC_PROFILE=fp32_v1` (which set the INFERENCE default;
the training default is `fp32_v1` already).

EACH PART CONFIGURABLE (Andrew, 2026-11-25; docs/plans/NUMERIC_PROFILE_PARTS.md).
A spec is a profile name (a PRESET) or an exact dict `{"preset": name,
part: name, ...}` that overrides single products. `PARTS` names the ten
products in the order of `core/numeric_profile_parts.mojo`; `PART_FORMATS`
says which profiles each part can compute under (`attn_context` is
`fp32_v1` only); `FAMILY_PARTS` says which parts a family has (a bare
`TransformerBlock` has no `lm_head`). An unknown part, a part the family
does not have, a format the part cannot use, an unregistered name, a
missing "preset" or a non-str key or value is refused by name; a trainer is
refused a dict. `resolve` returns the CANONICAL form: overrides equal to
the preset's own format are dropped, and with none left the preset name.
The canonical form is what a model reports and what a checkpoint state
stores; the native plan keeps only the parts mask it resolves to, once.

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
checkpoint do not change; a state written under `fixed15_v1` or a
per-part dict carries the field (the canonical form). A reader whose caller named no profile ADOPTS
the state's own (`adopt_saved`); one that named another is refused by name.
A model's weight file (a Hugging Face checkpoint) is not such a state: it
holds weights, not the results of an arithmetic, and carries no field.

INFERENCE AND TRAINING ARE SEPARATE GATES. A row says `inference` and
`training` apart, and each use has its own default. Every trainer resolves
the TRAINING default, so the inference default never reaches a trainer; a
trainer asked by name for a profile whose `training` is False refuses it.

WHAT A ROW MUST SHOW BEFORE `inference` IS TRUE (the three gates): identity on three vendors with a
sabotage arm seen failing; the relative change in held-out perplexity
against `fp32_v1`, measured as ONE complete configuration on two texts,
with the change and the upper end of its interval both under 1 percent; and
an end-to-end time with the conversion costs counted. `quality` holds only
what was measured; a row never carries an estimate.
"""
__all__ = ["DEFAULT", "ENV", "PROFILES", "REJECTED", "MEASURED",
           "NumericProfileSpeedWarning", "canonical", "default_profile",
           "set_default_profile", "resolve", "profiles", "measured",
           "state_field", "check_saved", "adopt_saved", "STATE_KEY", "USES",
           "require_training", "BASELINE", "TRAINING_DEFAULT", "FAMILIES",
           "computes", "default_reaches", "PARTS", "PART_FORMATS",
           "FAMILY_PARTS", "parts"]

#: The arithmetic every family computes under and every identity record
#: before 2026-09-29 was produced under.
BASELINE = "fp32_v1"
#: The INFERENCE default (Andrew, 2026-11-25; module docstring, THE DEFAULT).
#: It reaches only the families its row names in `default_for`; `fp32_v1` is
#: the opt-out.
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

#: The parts of a spec, the products of a Llama-shaped model, in the bit
#: order of `core/numeric_profile_parts.mojo` (NP_PART_*).
PARTS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
         "lm_head", "attn_scores", "attn_context")
#: part -> the profiles it can compute under. attn_context (P.V) is fp32_v1
#: only: under fixed15_v1's per-row rule its scales would depend on the key
#: span, so decode would not equal prefill.
PART_FORMATS = {
    "q_proj": ("fp32_v1", "fixed15_v1"), "k_proj": ("fp32_v1", "fixed15_v1"),
    "v_proj": ("fp32_v1", "fixed15_v1"), "o_proj": ("fp32_v1", "fixed15_v1"),
    "gate_proj": ("fp32_v1", "fixed15_v1"), "up_proj": ("fp32_v1", "fixed15_v1"),
    "down_proj": ("fp32_v1", "fixed15_v1"), "lm_head": ("fp32_v1", "fixed15_v1"),
    "attn_scores": ("fp32_v1", "fixed15_v1"), "attn_context": ("fp32_v1",),
}
#: family -> the parts it has. A family with none takes no part overrides.
FAMILY_PARTS = {
    "transformer": PARTS,
    "transformer_block": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                          "down_proj", "attn_scores", "attn_context"),
    "mamba1": (),
    "mamba2": (),
}

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
        "status": "experimental",
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
                "default inference profile of the transformer models; the same bits "
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
    "fixed15_int8_attention_v1": "DROPPED BEFORE ITS WHOLE CONFIGURATION WAS MEASURED "
                                 "(2026-09-29): int8 codes on the attention products alone read "
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
                 "source": "lane/lowbit-blocks (f), jobs nvc2-0020 and nvc3-0038"},
        "hip": {"over": (0.85, 0.89), "box": "MI325X",
                "what": "SmolLM2-360M, CausalLM.forward, prefill of 512 tokens at B=1, the whole "
                        "model (decode per token 0.63 to 0.65)",
                "source": "lane/lowbit-blocks (f), steward job 1790660933293"},
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

# The compiled numeric profile operations (`numeric_profile_native_door.mojo`),
# bound by `_native_numeric_profile.connect` at package import. Every function
# below is one call into it; validation, refusals, defaults and the speed
# warning run in Mojo.
_CALL = None
_CONTEXT = None


def canonical(name, what="numeric_profile"):
    """The canonical form of a spec: the name of a registered profile, or
    for a per-part dict the preset name when no override differs from the
    preset's own format, else a dict with "preset" first and the differing
    overrides in `PARTS` order. A rejected arithmetic is refused with its
    measured reason, anything else with the registered names."""
    return _CALL("canonical", (name, what), _CONTEXT)


def measured(profile, vendor=None):
    """Measured speed of a numeric profile relative to ``fp32_v1``.

    Parameters
    ----------
    profile : str
        A registered profile name (see :func:`mojolearn.numeric_profiles`).
    vendor : {"metal", "cuda", "hip"} or None, default None
        One vendor, as :func:`mojolearn.vendor` names it; None returns every
        measured vendor.

    Returns
    -------
    dict or None
        With ``vendor=None``, a dict from vendor to its row. With a vendor, that
        vendor's row, or None when it was not measured. A row's ``over`` is the
        range of the profile's time divided by ``fp32_v1``'s on the same shapes
        (above 1 means slower); ``what`` says which operation was timed.

    Raises
    ------
    ValueError
        If ``profile`` is not a registered profile.

    Examples
    --------
    >>> import mojolearn
    >>> rows = mojolearn.numeric_profile_measured("fixed15_v1")
    """
    return _CALL("measured", (profile, vendor), _CONTEXT)


def _warn_if_slow_here(key, vendor=None):
    """Once per profile and vendor: say so when the profile was measured to
    take longer than fp32_v1 on `vendor` (None: the vendor this process runs
    on). Returns the message, or None."""
    return _CALL("warn_if_slow", (key, vendor), _CONTEXT)


USES = ("inference", "training")


def computes(profile, family):
    """True when `family` computes under `profile`."""
    return _CALL("computes", (profile, family), _CONTEXT)


def default_reaches(profile, family):
    """True when a call in `family` that names no profile gets `profile`
    while it is the inference default."""
    return _CALL("default_reaches", (profile, family), _CONTEXT)


_defaults = {}


def default_profile(use="inference"):
    """The numeric profile used when a call names none.

    A numeric profile is the number format the matrix products of the neural
    models compute in. It is independent of the numeric mode: under
    ``numeric_mode="identical"`` every profile gives the same bits across
    vendors. The inference default is ``"fixed15_v1"`` and the training default
    is ``"fp32_v1"``. The ``MOJOLEARN_NUMERIC_PROFILE`` environment variable or
    :func:`mojolearn.set_numeric_profile` sets another inference default, such
    as ``"fp32_v1"``. The inference default applies only to the families it
    covers (the transformer models of ``mojolearn.models``); other families use
    ``"fp32_v1"``.

    Parameters
    ----------
    use : {"inference", "training"}, default "inference"
        Which default to return.

    Returns
    -------
    str
        The profile name.

    Examples
    --------
    >>> import mojolearn
    >>> mojolearn.numeric_profile()
    'fixed15_v1'
    >>> mojolearn.numeric_profile("training")
    'fp32_v1'
    """
    return _CALL("default", (use,), _CONTEXT)


def set_default_profile(name, use="inference"):
    """Set the process-wide default numeric profile.

    The name is checked immediately: a profile that is not registered, or not
    available for ``use``, raises here rather than at a later call. Only
    ``"fp32_v1"`` is available for training.

    Parameters
    ----------
    name : str
        A registered profile name, such as ``"fp32_v1"`` or ``"fixed15_v1"``
        (see :func:`mojolearn.numeric_profiles`).
    use : {"inference", "training"}, default "inference"
        Which default to set.

    Returns
    -------
    str
        The previous default.

    Raises
    ------
    ValueError
        If the name is not a registered profile.

    Examples
    --------
    >>> import mojolearn
    >>> previous = mojolearn.set_numeric_profile("fp32_v1")
    """
    return _CALL("set_default", (name, use), _CONTEXT)


def resolve(name=None, what="numeric_profile", use="inference", family=None):
    """What a class calls with its keyword. None is the process default for
    `use`, and in a `family` the default does not reach it is `BASELINE`
    (the object reports it: its `numeric_profile` attribute is what this
    returns). A NAME or a per-part dict is validated and returned in its
    canonical form (`canonical`); it is refused by name when a profile it
    names is not available for `use` or does not compute `family`, when a
    part is unknown, missing from `family` (`FAMILY_PARTS`) or given a
    format it cannot use (`PART_FORMATS`); a dict is refused for training.
    It is never replaced by another profile."""
    return _CALL("resolve", (name, what, use, family), _CONTEXT)


def parts(spec, family=None):
    """The whole parts table of a spec: a dict from every part in `PARTS`
    to the profile it computes under (a preset gives each part its own
    format; `attn_context` stays `fp32_v1`). Validated as `resolve` does,
    against `family` when one is given."""
    return _CALL("parts", (spec, family), _CONTEXT)


def require_training(what, name=None):
    """One line for a trainer's constructor: the TRAINING default when the
    caller named nothing (the inference default never reaches a trainer),
    else the named profile, refused by name unless it has passed for
    training."""
    return _CALL("resolve", (name, what, "training", None), _CONTEXT)


def profiles():
    """Every registered numeric profile, the baseline first.

    Returns
    -------
    tuple of dict
        One record per profile with its ``name``; ``default`` and
        ``training_default`` flags; ``inference`` and ``training``
        availability; ``products`` (the arithmetic of each product family);
        ``computes`` (the model families it applies to, None for all);
        ``default_for``; ``quality`` (measured relative change in held-out
        perplexity against ``fp32_v1``); a ``note``; and ``measured`` speed
        rows per vendor (see :func:`mojolearn.numeric_profile_measured`).

    Examples
    --------
    >>> import mojolearn
    >>> [p["name"] for p in mojolearn.numeric_profiles()]
    ['fp32_v1', 'fixed15_v1']
    """
    return _CALL("profiles", (), _CONTEXT)


def state_field(profile):
    """What a checkpoint writer merges into its state: nothing under
    `fp32_v1` (so an fp32_v1 checkpoint's bytes do not change, and a state
    with no field reads as fp32_v1), the name otherwise; a state written
    under `fixed15_v1`, or under a per-part dict, carries its canonical form."""
    return _CALL("state_field", (profile,), _CONTEXT)


def check_saved(state, profile, what="state"):
    """Refuse by name a state written under another profile than the one
    this object computes under (a name or a per-part dict; canonical forms
    are compared). A state with no field is `fp32_v1`."""
    return _CALL("check_saved", (state, profile, what), _CONTEXT)


def adopt_saved(state, name=None, what="state", use="inference"):
    """What a reader calls when the object it builds takes its profile FROM
    the state: the caller named nothing, so the state's own profile (a state
    with no field is `fp32_v1`, whatever the process default is now); a
    caller who named a different one is refused by name. The adopted
    profile must still be available for `use`, and a saved per-part dict is
    adopted whole (its canonical form)."""
    return _CALL("adopt_saved", (state, name, what, use), _CONTEXT)
