#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DOES cuML's JSON DUMP HAVE THE SHAPE OUR FIT-EQUIVALENCE READER WALKS?

    python3 tools/cuml_forest_json_probe.py

ONE MINUTE, ON ANY BOX WITH cuML. No dataset, no lease, no benchmark: it fits
a 100-row, 2-tree, depth-3 forest on random numbers, which is large enough to
have internal nodes and leaves and small enough that the whole dump can be
printed and read by a person.

WHY IT EXISTS. `speed_gbdt_arm.py::_shape_cuml` is the one fit-shape reader in
that file whose ACCESSOR AND SCHEMA are both unverified. Every other arm was
either checked against fixtures that mirror a documented surface
(`estimators_[i].tree_`, `booster.get_dump`, `dump_model()['tree_info']`) or
reads our own model text. cuML's Python forest exposes no node or leaf
accessor at all; `get_json()` / `dump_as_json()` is the only door, it is not
on every build, and its schema is not something this repository pins.

THE FAILURE MODE THIS SETTLES, and it is not the obvious one. If the dump is
ABSENT the reader says UNAVAILABLE, the cell reads `verdict=UNKNOWN`, and
nothing is claimed -- that is safe because it is loud. The dangerous case is a
dump that PARSES into a shape the walk misreads: the walk treats a node as a
leaf when its `children` key is missing or empty, so a flat per-tree node list
(or a different spelling) makes every node look like a leaf and yields a
plausible WRONG leaf count, which then flows into FSPEED-FIT-VERDICT as though
it had been measured. A wrong number dressed as a measurement is worse than no
number, which is the whole argument of the lane that added the reader.

The reader carries an invariant (`nodes >= leaves >= trees >= 1`) that
degrades a misread parse to UNAVAILABLE. That catches a misread which produces
impossible counts. It CANNOT catch a schema that is wrong but self-consistent,
so this probe exists to settle the question by looking.

WHAT TO DO WITH THE ANSWER. `SCHEMA OK` means `_shape_cuml` may be trusted on
this cuML version and the rf/et cells can stop reading UNKNOWN. Anything else
is a reader bug to fix against the dump this prints -- never a reason to
loosen the invariant.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def say(*a):
    print(*a, flush=True)


def main():
    try:
        import cuml
        from cuml.ensemble import RandomForestClassifier
    except Exception as exc:                       # noqa: BLE001
        say("PROBE-REFUSED cuML does not import here: %r" % (exc,))
        say("PROBE-REFUSED this probe is for a box with cuML; nothing else "
            "about the harness depends on running it.")
        return 2
    # The version rides on every verdict line below, not only here: a reader
    # finding one of those lines quoted in a log or a note needs to know which
    # build the answer is about, and cuML's dump schema is exactly the kind of
    # thing that changes between them.
    ver = getattr(cuml, "__version__", "unknown")
    say("PROBE-VERSION cuml=%s" % ver)

    rng = np.random.default_rng(7)
    x = rng.random((100, 4), dtype=np.float32)
    y = (x[:, 0] > 0.5).astype(np.int32)
    est = RandomForestClassifier(n_estimators=2, max_depth=3, n_bins=16,
                                 random_state=7).fit(x, y)

    dumper = None
    for name in ("get_json", "dump_as_json"):
        if callable(getattr(est, name, None)):
            dumper = getattr(est, name)
            say("PROBE-ACCESSOR %s is present" % name)
            break
        say("PROBE-ACCESSOR %s is ABSENT" % name)
    if dumper is None:
        say("PROBE-RESULT NO ACCESSOR: _shape_cuml will report UNAVAILABLE and "
            "every rf/et cell will read verdict=UNKNOWN on this build. That is "
            "the safe outcome, not a bug to work around. [cuml=%s]" % ver)
        return 1

    raw = dumper()
    say("PROBE-DUMP type=%s bytes=%d" % (type(raw).__name__, len(str(raw))))
    try:
        trees = json.loads(raw)
    except Exception as exc:                       # noqa: BLE001
        say("PROBE-RESULT THE DUMP DID NOT PARSE: %r [cuml=%s]" % (exc, ver))
        return 1
    if not isinstance(trees, list):
        say("PROBE-SHAPE top level is %s, NOT a list of trees"
            % type(trees).__name__)
        trees = [trees]
    else:
        say("PROBE-SHAPE top level is a list of %d entries (asked for 2 trees)"
            % len(trees))

    first = trees[0]
    if isinstance(first, dict):
        say("PROBE-KEYS first tree's top-level keys: %s"
            % sorted(first.keys()))
    say("PROBE-SAMPLE %s" % json.dumps(first)[:600])

    # The exact walk `_shape_cuml` performs, repeated here rather than
    # imported, so this probe still answers if that function is changed.
    nodes = leaves = depth = 0
    for tree in trees:
        stack = [(tree, 0)]
        while stack:
            node, d = stack.pop()
            if not isinstance(node, dict):
                continue
            nodes += 1
            kids = node.get("children")
            if not kids:
                leaves += 1
                depth = max(depth, d)
                continue
            for kid in kids:
                stack.append((kid, d + 1))
    say("PROBE-WALK trees=%d nodes=%d leaves=%d depth_max=%d"
        % (len(trees), nodes, leaves, depth))

    ok = nodes >= leaves >= len(trees) >= 1
    say("PROBE-INVARIANT nodes >= leaves >= trees >= 1 : %s" % ok)
    if not ok:
        say("PROBE-RESULT SCHEMA MISMATCH (impossible counts). _shape_cuml "
            "degrades to UNAVAILABLE, which is correct and safe. Fix the "
            "reader against the sample above; do not loosen the invariant. "
            "[cuml=%s]" % ver)
        return 1
    if leaves == nodes:
        say("PROBE-RESULT SUSPICIOUS: every node counted as a leaf, which is "
            "what a FLAT node list looks like to this walk. The invariant "
            "cannot catch this because the counts stay self-consistent. Read "
            "the sample above and fix the reader before trusting any cuML "
            "leaf count. [cuml=%s]" % ver)
        return 1
    say("PROBE-RESULT SCHEMA OK: internal nodes and leaves are distinguished "
        "(%d internal, %d leaves over %d trees). _shape_cuml may be trusted "
        "on this cuML version. [cuml=%s]"
        % (nodes - leaves, leaves, len(trees), ver))
    return 0


if __name__ == "__main__":
    sys.exit(main())
