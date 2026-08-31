# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`numeric_mode=` as a PARAMETER, on the estimator, in your code.

WHAT THIS REPLACES. The mode used to be reachable only as
`MOJOLEARN_NUMERIC_MODE`, an environment variable read ONCE by
`_backend.select()` before the first estimator was imported. One install, but
a global set outside the program: you could not change it after import, and
you could not give two estimators in one script two different modes. The
library was shipping a choice you had to make from the shell.

It is now three things, and only the first is unchanged:

    ONE install                    the wheel carries every tier's binaries
    a default you can set in code  mojolearn.set_numeric_mode("deterministic")
    a per-estimator parameter      RandomForestClassifier(numeric_mode=...)

The environment variable still works and still sets the STARTING default, so
nothing written against the old spelling breaks.

WHY THIS IS AN ATTRIBUTE LOOKUP AND NOT A REBIND. **The three tiers are one
source under one flag, not three implementations.** `GLOBAL_NUMERIC_MODE` in
`mojo_only/numerics.mojo` is that flag; `PIN_DETERMINISM` and
`PIN_CROSS_VENDOR` derive from it and every kernel reads those. But the flag
is COMPTIME, which is what lets the fast build carry none of the pinning code
at all rather than branching past it at run time, so the three settings are
compiled and shipped side by side and a Python parameter selects among them
rather than flipping anything inside one of them. That is
`_backend.load_set`, and every call site therefore has to ask for its binding
at CALL TIME rather than binding a module-level name at import.

THAT THE TIERS COEXIST IS MEASURED (2026-08-29). Each `.so` carries its own
Mojo runtime and opens its own device context, so "two of them in one process
will fight" was the real risk. All three were loaded together and then called
INTERLEAVED -- fast, deterministic, identical, fast -- twice, on one
256x4096 @ 4096x128 product on an Apple M4. Each returned its own arithmetic
every time, and a call made after the identical set did not inherit its
answer. See `_backend.load_set`'s header.
"""

import functools

from . import _backend


class NumericModeMixin:
    """Gives an estimator `self._bind(name)`.

    `numeric_mode` is read off the instance at every call rather than
    resolved in `__init__`, so an estimator that is unpickled from an older
    version (no such attribute) falls back to the process default instead of
    raising -- and so that setting the attribute after construction works the
    way a caller would expect it to.
    """

    def __init_subclass__(cls, **kw):
        """Give every estimator a `numeric_mode=` keyword without editing
        eleven constructor signatures.

        WHY A WRAPPER AND NOT ELEVEN EDITS. The parameter is identical in
        every class and is not part of any estimator's own contract -- it
        selects which BINARY answers, which is a property of the library, not
        of k-means. Eleven hand-written copies is eleven chances for one of
        them to drift, and the drift would be silent: an estimator that
        quietly ignored the keyword would run on the process default and
        report a tier it was not using.

        The assignment happens AFTER the wrapped `__init__`, because these
        classes inherit (`ExtraTreesClassifier` -> `_ExtraTreesBase`,
        `KNeighborsClassifier` -> `NearestNeighbors`) and BOTH ends get
        wrapped. The base runs first and would otherwise write its own
        `None` over the subclass's real answer.
        """
        super().__init_subclass__(**kw)
        orig = cls.__dict__.get("__init__")
        if orig is None:
            return

        @functools.wraps(orig)
        def __init__(self, *args, numeric_mode=None, **kwargs):
            orig(self, *args, **kwargs)
            if numeric_mode is not None or not hasattr(self, "numeric_mode"):
                self.numeric_mode = numeric_mode

        cls.__init__ = __init__

    #: The binding this estimator family talks to. Subclasses that use more
    #: than one pass the name explicitly.
    _BINDING = "_mojolearn"

    def _bind(self, name=None):
        return _backend.binding(
            name or self._BINDING, getattr(self, "numeric_mode", None)
        )

    def numeric_mode_used(self):
        """The tier THIS estimator will run on, resolved and read back from
        the binary it actually holds -- not the string that was passed in."""
        return self._bind().__name__.split(".")[-2]

    def vendor_used(self):
        """'metal', 'cuda' or 'hip': the accelerator API of the binary THIS
        estimator will call, read back from that binary's own compile-time
        constant (`<prefix>_vendor()`, `mojo_only/vendor.mojo`), not from
        the directory it was loaded from and not from the platform. None
        for a binary built before the read-back existed."""
        return _backend.read_vendor(self._bind())
