# SPDX-License-Identifier: Apache-2.0
"""What `model.digest_` hashes, per class, and its one native connection.

`digest_` is the hash of a fitted model's state (bindings/fitted_digest_door.mojo
has the walk, core/fitted_digest.mojo the format). The compiled walk reads the
tables below; `tools/gen_digest_fields.py` writes docs/api/DIGEST_FIELDS.md
from the same tables and the save schemas, so the table and the code cannot
disagree.

The field source of a class, first match in its ancestry:

1. `REFUSED`: no digest yet; the read raises AttributeError with the reason.
2. A connected save schema (`_native_archive_bridge.register`): the members
   `save` writes.
3. Otherwise: every public fitted attribute (an instance field whose name ends
   in "_" and does not start with "_", value not None), plus the private
   fields in `EXTRA`.

This module holds declarations and import-time glue only: nothing here runs
when a model is fitted, and the read itself is one native call.
"""
#: class name -> the private instance fields that are fitted state (they
#: decide the model's answers and no public attribute carries them).
EXTRA_BY_NAME = {}

#: class name -> why it has no digest yet, in words a user reads.
REFUSED_BY_NAME = {
    "_BaggingBase": "its fitted members are held as one forest by the trees owner, which the digest does not read yet",
    "_RecurrentBase": "its fitted weights are held by the recurrent owner (state_dict), which the digest does not read yet",
    "Sequential": "its layers hold the fitted weights, and the digest does not walk layers yet",
    "CNNClassifier": "its layers hold the fitted weights, and the digest does not walk layers yet",
}

#: The tables the compiled walk reads, keyed by class (filled by `declare`).
EXTRA = {}
REFUSED = {}

#: The one connection of the `digest_` property: the property object, the
#: compiled entry and its context. Filled in place by `connect`; `_mode` puts
#: this dictionary in its metadata context, where
#: bindings/fitted_digest_property_pack.mojo reads it.
CONNECTION = {"property": None, "entry": None, "context": None}


def declare(cls):
    """Enter `cls` in the class-keyed tables when a table above names it
    (called once per class, at its definition, by the shared estimator
    base)."""
    name = cls.__name__
    if name in EXTRA_BY_NAME:
        EXTRA[cls] = EXTRA_BY_NAME[name]
    if name in REFUSED_BY_NAME:
        REFUSED[cls] = REFUSED_BY_NAME[name]


def connect(base):
    """Connect the compiled digest entry once: `(entry, context)`. `base` is
    the shared estimator base whose `digest_` property this serves. On an
    installation without the saved-model owner the entry raises
    ModuleNotFoundError in words, as `save` does there."""
    if CONNECTION["context"] is not None:
        return CONNECTION["entry"], CONNECTION["context"]
    from ._native_archive_bridge import connect as connect_archive
    entry, context = connect_archive()
    if "archive_unavailable" not in context:
        context.update(digest_base=base, digest_extra=EXTRA, digest_refused=REFUSED)
        CONNECTION["entry"] = entry
    CONNECTION["context"] = context
    return entry, context


def read(model):
    """`model.digest_`: one native call."""
    entry = CONNECTION["entry"]
    if entry is None:
        from ._mode import CheckMixin
        entry = connect(CheckMixin)[0]
    return entry(model, "fitted_digest", (), CONNECTION["context"])

