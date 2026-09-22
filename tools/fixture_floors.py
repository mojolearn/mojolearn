#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FIXTURE FLOORS: the smallest a lane's fixture may get, enforced in the gate.

    python3 tools/fixture_floors.py              # fail on any violation
    python3 tools/fixture_floors.py --list       # the floor table, derived
    python3 tools/fixture_floors.py --self-test  # prove the check REFUSES

WHY THIS EXISTS. On 2026-09-16 an audit found two shrunken identity cells that
can no longer FAIL. One of them, `samba-untied-dropout-accum`, had a floor
WRITTEN DOWN before it was cut: a design note titled "LEFT BIG" said three
steps is the floor because step 3 is the first that evaluates the cosine arm
of the schedule. It was cut to one step anyway, a later scoping note carried
forward only the rows half of that reasoning, and a third note then recorded
as fact that the third step was kept. Nobody lied. A floor in prose
is not a floor, because the reason and the number live in a different file
from the fixture and a change never has to walk past them.

So a floor lives in the SOURCE, on the lane, with its reason attached:

    @lane("samba-untied-dropout-accum")
    @floor(steps=(3, "step 3 is the first that evaluates the cosine arm ..."))
    def _(ml, X, yc, yr, Xh=None):
        steps = 3                      # <- the floored local, read at the site
        ...
        for k in range(steps): ...     # <- the site

and this file is the thing that refuses a change that walks past it.

THE FOUR RULES, all read from the source with `ast`, no numpy, no bindings, so
the cheapest push gate can run it.

  1. A REVISION THAT NAMES A SIZE MUST CARRY A FLOOR ON IT, and that is
     DERIVED, not a hand-kept list. `LANE_REVISIONS` is where a fixture change
     is recorded (without an entry every committed column reads DIVERGENT,
     which is loud), and a key that changed a SIZE already says so in its first
     token: `rows-1500-1`, `obs-128-1`, `steps-3-1`, `seqlen-8-1`. Those lanes
     must floor that dimension, AND the number in the key must be the number
     the lane runs at, so the key, the floor and the body cannot drift apart.
     Any other key (`transform-row-separable-1` is arithmetic,
     `synthetic-vocab-1` is a vocabulary) must say what it changed in
     `NON_SIZE_REVISIONS`, and that entry is refused if the key DOES name a
     size, so the list cannot grow into an excuse.
  2. THE FLOOR'S VALUE IS READ FROM THE LANE BODY, not from the decorator. The
     lane binds a local named for the dimension to an integer literal, and
     that literal is what the lane actually runs at. Below the floor is a
     refusal that PRINTS THE REASON.
  3. THE SITE MUST READ THE FLOORED LOCAL. `steps = 3` next to `range(1)` is
     refused by name, so the number cannot be bypassed while the floor is left
     looking satisfied.
  4. THE REASON MUST BE TRACEABLE. A `why` shorter than 60 characters, or one
     that cites no date, `docs/` path or `lane/` name, is refused. The whole
     failure above was a reason nobody could follow back to its measurement.

WHAT THIS DOES NOT DO. It does not judge whether a floor is the RIGHT number;
only a measurement does that, which is why rule 4 makes the reason cite one.
It cannot floor what is not a size: the `tokenizer` lane's vocabulary and
`umap`'s row-separable transform moved those lanes' bytes without moving a
number, which is what `NON_SIZE_REVISIONS` is for.
"""
import argparse
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "tools", "identity_break.py")

#: A dimension a floor may protect, and the SITE RULE that finds it in a lane
#: body. Dimensions sharing a `group` are alternative names for one axis, so a
#: lane triggers the group and satisfies it with either name.
DIMENSIONS = {
    "steps": ("steps", "a `range(...)` loop or comprehension whose body calls `train_step`"),
    "rows": ("row_axis", "a `X[:n]` or `X[:n, ...]` slice of the fixture"),
    "observations": ("row_axis", "a `X[:n]` or `X[:n, ...]` slice of the fixture"),
    "batch": ("batch", "the row-count argument of `_ids(X, n, l)`"),
    "seqlen": ("seqlen", "the length argument of `_seq(X, b, l, ...)`"),
}
GROUPS = {}
for _name, (_group, _rule) in DIMENSIONS.items():
    GROUPS.setdefault(_group, [_rule, []])[1].append(_name)

#: A REVISION KEY THAT NAMES A SIZE. `LANE_REVISIONS` records every lane whose
#: input moved, which is broader than "was shrunk": `transform-row-separable-1`
#: is an arithmetic change and `synthetic-vocab-1` is a vocabulary swap, and
#: neither is a size a floor can hold. The keys that DO name a size already say
#: so in their first token, `rows-1500-1`, `obs-128-1`, `steps-3-1`,
#: `seqlen-8-1`. Those lanes must floor that dimension, and the number in the
#: key must be the number the lane runs at, so the key, the floor and the body
#: cannot drift apart. Every other key must be declared in
#: `NON_SIZE_REVISIONS`, and that entry is refused if the key does name a size.
REVISION_DIMENSION = {"rows": "rows", "obs": "observations", "steps": "steps",
                      "seqlen": "seqlen", "batch": "batch"}


def revision_size(rev):
    """(dimension, value) if a revision key names a size, else (None, None)."""
    parts = (rev or "").split("-")
    if len(parts) >= 2 and parts[0] in REVISION_DIMENSION and parts[1].isdigit():
        return REVISION_DIMENSION[parts[0]], int(parts[1])
    return None, None


#: A `why` must be traceable back to the measurement that set the number.
TRACEABLE = re.compile(r"20\d\d-\d\d-\d\d|docs/|lane/")
WHY_MIN_CHARS = 60


# --------------------------------------------------------------- source reading
def _const(node, kind):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, kind) else None


def _dict_of_str(tree, name):
    """A module-level `NAME = {"a": "b", ...}` read as a dict of str to str."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            if not isinstance(node.value, ast.Dict):
                raise AssertionError(f"{name} is not a dict literal")
            return {_const(k, str): _const(v, str) for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError(f"no {name} in {HARNESS}")


def _lane_functions(tree):
    """Every `@lane("name")` function, as name -> FunctionDef, in source order."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "lane" and dec.args:
                name = _const(dec.args[0], str)
                if name is not None:
                    out[name] = node
    return out


def _declared_floors(fn):
    """The `@floor(dim=(minimum, why))` entries on one lane, with the problems
    found reading them. Returns (floors, problems)."""
    floors, problems = {}, []
    for dec in fn.decorator_list:
        if not (isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "floor"):
            continue
        if dec.args:
            problems.append("floor() takes only keyword arguments, one per dimension")
        for kw in dec.keywords:
            dim = kw.arg
            if dim not in DIMENSIONS:
                problems.append(f"floor({dim}=...) names no known dimension; known: {', '.join(sorted(DIMENSIONS))}")
                continue
            if not (isinstance(kw.value, ast.Tuple) and len(kw.value.elts) == 2):
                problems.append(f"floor({dim}=...) must be a (minimum, why) pair written out in full")
                continue
            lo, why = _const(kw.value.elts[0], int), _const(kw.value.elts[1], str)
            if lo is None or lo < 1:
                problems.append(f"floor({dim}=...): the minimum must be an integer literal of at least 1")
                continue
            if why is None:
                problems.append(f"floor({dim}=...): the reason must be a string literal")
                continue
            if len(why) < WHY_MIN_CHARS or not TRACEABLE.search(why):
                problems.append(
                    f"floor({dim}=...): the reason is not traceable. It must be at least {WHY_MIN_CHARS} "
                    f"characters and cite a date, a docs/ path or a lane/ name, so the next person can "
                    f"reach the measurement that set {lo}. Got: {why!r}")
                continue
            if dim in floors:
                problems.append(f"floor({dim}=...) is declared twice")
                continue
            floors[dim] = (lo, why)
    return floors, problems


def _int_locals(fn):
    """`name = <int literal>` and `a, b = <int>, <int>` bindings in the body,
    as name -> (value, count). A name bound more than once is not a floor the
    site can be held to, so the count travels with the value."""
    out = {}

    def bind(target, value):
        if isinstance(target, ast.Name) and _const(value, int) is not None:
            v, n = out.get(target.id, (None, 0))
            out[target.id] = (_const(value, int), n + 1)

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t, v = node.targets[0], node.value
            if isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple) and len(t.elts) == len(v.elts):
                for a, b in zip(t.elts, v.elts):
                    bind(a, b)
            else:
                bind(t, v)
    return out


def _names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _sites(fn):
    """Every size SITE in a lane body, as group -> list of expressions that
    carry the size. This is the derivation: what the lane actually runs at."""
    found = {g: [] for g in GROUPS}

    def calls_train_step(node):
        return any(isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "train_step"
                   for c in ast.walk(node))

    for node in ast.walk(fn):
        # steps: the loop or comprehension that drives training
        iters = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iters = [(node.iter, node.body)]
        elif isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            iters = [(g.iter, [node.elt]) for g in node.generators]
        for it, body in iters:
            if (isinstance(it, ast.Call) and getattr(it.func, "id", None) == "range" and len(it.args) == 1
                    and any(calls_train_step(b) for b in body)):
                found["steps"].append(it.args[0])
        # row_axis: a slice of the fixture matrix by name
        if isinstance(node, ast.Subscript) and getattr(node.value, "id", None) == "X":
            sl = node.slice
            if isinstance(sl, ast.Tuple) and sl.elts:
                sl = sl.elts[0]
            if isinstance(sl, ast.Slice) and sl.upper is not None and sl.lower is None:
                found["row_axis"].append(sl.upper)
        # batch: the token stream's row count
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_ids"
                and len(node.args) >= 2 and getattr(node.args[0], "id", None) == "X"):
            found["batch"].append(node.args[1])
        # seqlen: the activation slab's length axis
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_seq"
                and len(node.args) >= 3 and getattr(node.args[0], "id", None) == "X"):
            found["seqlen"].append(node.args[2])
    return found


def _resolve(expr, local):
    """The integer a site expression comes to, using only literals and the
    lane's own integer locals. None when the site is not a fixture size this
    file can read (`steps * shape.batch` is a model config, not a row count);
    an unreadable site requires no floor, and `--list` prints it so the gap is
    on the record rather than silent."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
        return expr.value
    if isinstance(expr, ast.Name):
        v, n = local.get(expr.id, (None, 0))
        return v if n == 1 else None
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Mult, ast.Add, ast.Sub)):
        a, b = _resolve(expr.left, local), _resolve(expr.right, local)
        if a is None or b is None:
            return None
        return a * b if isinstance(expr.op, ast.Mult) else (a + b if isinstance(expr.op, ast.Add) else a - b)
    return None


# ------------------------------------------------------------------- the check
def check(src=None, path=HARNESS):
    """Every floor violation in the harness source, as a list of strings. An
    empty list is the only pass."""
    src = open(path).read() if src is None else src
    tree = ast.parse(src)
    revisions = _dict_of_str(tree, "LANE_REVISIONS")
    unfloored = _dict_of_str(tree, "NON_SIZE_REVISIONS")
    lanes = _lane_functions(tree)
    bad = []

    # A @floor that is not on a lane enforces nothing and looks like it does.
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        has_floor = any(isinstance(d, ast.Call) and getattr(d.func, "id", None) == "floor"
                        for d in node.decorator_list)
        has_lane = any(isinstance(d, ast.Call) and getattr(d.func, "id", None) == "lane"
                       for d in node.decorator_list)
        if has_floor and not has_lane:
            bad.append(f"line {node.lineno}: a @floor() that is not on an @lane() function. Nothing "
                       f"enforces it, and it reads as though something does.")

    seen_why = {}
    for name, fn in sorted(lanes.items()):
        floors, problems = _declared_floors(fn)
        bad += [f"{name}: {p}" for p in problems]
        sites = _sites(fn)
        local = _int_locals(fn)

        # RULE 1, the derived half: a revision key that NAMES a size must have
        # a floor on that dimension, and the key's number must be the number
        # the lane runs at.
        if name in revisions:
            rdim, rvalue = revision_size(revisions[name])
            group = DIMENSIONS[rdim][0] if rdim else None
            if rdim and not (set(GROUPS[group][1]) & set(floors)):
                bad.append(
                    f"{name}: LANE_REVISIONS[{name!r}] = {revisions[name]!r} names a SIZE "
                    f"({rdim} {rvalue}), but the lane declares no floor for it. Add "
                    f"@floor({rdim}=(<minimum>, \"why, with the measurement that set it\")) under "
                    f"@lane({name!r}), or this lane can be cut again with nothing to walk past.")
            elif rdim:
                have = next(d for d in GROUPS[group][1] if d in floors)
                ran = local.get(have, (None, 0))[0]
                if ran is not None and ran != rvalue:
                    bad.append(
                        f"{name}: LANE_REVISIONS[{name!r}] = {revisions[name]!r} says {rdim} {rvalue}, "
                        f"but the lane runs at {have} = {ran}. A revision key that disagrees with the "
                        f"body is how a record ends up describing bytes nobody made.")

        for dim, (lo, why) in sorted(floors.items()):
            group, rule = DIMENSIONS[dim][0], DIMENSIONS[dim][1]
            if not sites[group]:
                bad.append(f"{name}: floors {dim} but its body has no {rule}, so the floor is unreadable "
                           f"prose again. Put the size at a site this file can find, or drop the floor.")
                continue
            # RULE 2: the value is what the lane RUNS at, read from the body.
            value, times = local.get(dim, (None, 0))
            if value is None:
                bad.append(f"{name}: floors {dim} but binds no local `{dim} = <integer>` in the body. The "
                           f"floored number has to be in the lane, next to its reason, not only in the "
                           f"decorator.")
                continue
            if times != 1:
                bad.append(f"{name}: binds `{dim}` {times} times, so there is no single number to hold to "
                           f"the floor.")
                continue
            # RULE 3: the site must READ that local, or the number is bypassed.
            for expr in sites[group]:
                if dim not in _names_in(expr):
                    bad.append(
                        f"{name}: {rule} does not read the floored local `{dim}`; it is "
                        f"`{ast.unparse(expr)}`. A literal at the site means the fixture can be cut without "
                        f"touching `{dim} = {value}` or the reason above it.")
            # A reason copied from another floor is a reason about another
            # lane, which is how the number and the measurement come apart.
            if why in seen_why:
                bad.append(f"{name}: the reason for {dim} is copied word for word from "
                           f"{seen_why[why]}. A floor's reason is about THIS lane's fixture; if the "
                           f"measurement really is shared, say which lane it was taken on.")
            seen_why[why] = f"{name}'s {dim} floor"
            if value < lo:
                bad.append(
                    f"{name}: REFUSED, {dim} = {value} is BELOW the floor of {lo}.\n"
                    f"      WHY THE FLOOR IS THERE: {why}\n"
                    f"      If the floor is wrong, change it HERE with the measurement that says so; do not "
                    f"cut past it.")

    # RULE 1, the fail-closed half: every revision is either a size, and
    # floored, or declared not to be one, and the declaration is checked.
    for name, why in sorted(unfloored.items()):
        if name not in revisions:
            bad.append(f"NON_SIZE_REVISIONS[{name!r}]: no LANE_REVISIONS entry, so there is nothing to "
                       f"declare about; remove it.")
            continue
        if name not in lanes:
            bad.append(f"NON_SIZE_REVISIONS[{name!r}]: no such lane.")
            continue
        rdim, rvalue = revision_size(revisions[name])
        if rdim:
            bad.append(f"NON_SIZE_REVISIONS[{name!r}]: REFUSED, its revision {revisions[name]!r} DOES "
                       f"name a size ({rdim} {rvalue}). Declare the floor instead of the exemption.")
        if not why or len(why) < WHY_MIN_CHARS or not TRACEABLE.search(why):
            bad.append(f"NON_SIZE_REVISIONS[{name!r}]: the reason must be at least {WHY_MIN_CHARS} "
                       f"characters and cite a date, a docs/ path or a lane/ name, so the next person can "
                       f"see WHAT changed if it was not a size.")
    for name, rev in sorted(revisions.items()):
        if name in lanes and revision_size(rev)[0] is None and name not in unfloored:
            bad.append(f"{name}: LANE_REVISIONS[{name!r}] = {rev!r} names no size this file knows "
                       f"({', '.join(sorted(REVISION_DIMENSION))}). Either name one, so the floor and the "
                       f"record agree, or say in NON_SIZE_REVISIONS what changed instead.")
    return bad


def table(src=None, path=HARNESS):
    """The floor table, derived from the source: lane, dimension, value the
    lane runs at, floor, and the reason."""
    src = open(path).read() if src is None else src
    tree = ast.parse(src)
    rows = []
    for name, fn in sorted(_lane_functions(tree).items()):
        floors, _ = _declared_floors(fn)
        local = _int_locals(fn)
        for dim, (lo, why) in sorted(floors.items()):
            rows.append((name, dim, local.get(dim, (None, 0))[0], lo, why))
    return rows


def unreadable_sites(src=None, path=HARNESS):
    """Sites in a SHRUNK lane that carry a size this file cannot resolve to an
    integer, so they require no floor. This is the mechanism's hole, printed
    rather than left implicit: a future lane could dodge a floor by making its
    size expression unreadable, and the list is where that would show up."""
    src = open(path).read() if src is None else src
    tree = ast.parse(src)
    revisions = _dict_of_str(tree, "LANE_REVISIONS")
    out = []
    for name, fn in sorted(_lane_functions(tree).items()):
        if name not in revisions:
            continue
        local = _int_locals(fn)
        for group, exprs in sorted(_sites(fn).items()):
            for e in exprs:
                if _resolve(e, local) is None:
                    out.append((name, group, ast.unparse(e)))
    return out


# ---------------------------------------------------------------- the self-test
#: A check that has not been seen to refuse anything is the prose floor it
#: replaces. Each entry mutates the REAL harness source one way and must be
#: refused with a message naming the lane; the near miss beside it must NOT be.
MUTATIONS = (
    ("a floored lane is cut below its floor",
     "    steps, batch = 3, 32", "    steps, batch = 1, 32", "samba-untied-dropout-accum"),
    ("the floor decorator is deleted",
     "@floor(steps=(2,", "@_gone(steps=(2,", "byte-lm"),
    ("the site stops reading the floored local, leaving the floor looking satisfied",
     "_seq(X, 2, seqlen, dm), _seq(", "_seq(X, 2, 4, dm), _seq(", "mamba2-dtlimit"),
    ("a reason nothing can be traced back to",
     '@lane("mamba2-dtlimit")\n@floor(seqlen=(8, "L=8 keeps',
     '@lane("mamba2-dtlimit")\n@floor(seqlen=(8, "measured, it is fine"))\n@floor(seqlen=(8, "L=8 keeps',
     "mamba2-dtlimit"),
    ("the revision key says one size and the body runs another",
     '"hdbscan": "rows-4000-1"', '"hdbscan": "rows-3000-1"', "hdbscan"),
    ("a revision key that names no size at all, undeclared",
     '"spectral": "rows-512-1"', '"spectral": "made-it-smaller-1"', "spectral"),
    ("the non-size list used on a revision that DOES name a size",
     '    "tokenizer": (', '    "holtwinters": "an exemption this check must refuse",\n    "tokenizer": (',
     "holtwinters"),
)


#: Two rules cannot be reached by mutating the real harness one line at a time,
#: so they are watched on a source written to break exactly them. Each entry is
#: (label, source, a fragment the refusal must contain).
SYNTHETIC = (
    ("a @floor() that is not on a lane function",
     '''
LANE_REVISIONS = {}
NON_SIZE_REVISIONS = {}

@floor(rows=(10, "a perfectly good reason citing docs/lanes/SOMETHING.md and the date 2026-09-16"))
def not_a_lane(ml, X, yc, yr, Xh=None):
    rows = 10
    return X[:rows]
''', "not on an @lane() function"),
    ("a reason copied word for word from another floor",
     '''
LANE_REVISIONS = {}
NON_SIZE_REVISIONS = {}

@lane("a")
@floor(rows=(10, "measured on 2026-09-16, see docs/lanes/SOMETHING.md, this is the shared sentence"))
def _(ml, X, yc, yr, Xh=None):
    rows = 10
    return X[:rows]

@lane("b")
@floor(rows=(10, "measured on 2026-09-16, see docs/lanes/SOMETHING.md, this is the shared sentence"))
def _(ml, X, yc, yr, Xh=None):
    rows = 10
    return X[:rows]
''', "copied word for word"),
)


def self_test(path=HARNESS):
    src = open(path).read()
    ok = True
    clean = check(src, path)
    print(f"# unmutated {os.path.relpath(path, ROOT)}: {len(clean)} violation(s)")
    for line in clean:
        print("    " + line.replace("\n", "\n    "))
    ok &= not clean
    for label, old, new, lane in MUTATIONS:
        if src.count(old) < 1:
            print(f"REFUSE-TEST BROKEN: {label}: the anchor {old!r} is not in the source, so this mutation "
                  f"changes nothing and the test below cannot fail")
            ok = False
            continue
        got = check(src.replace(old, new, 1), path)
        hit = [g for g in got if g.split(":")[0].strip().strip("\"'[]") == lane or lane in g.split("\n")[0]]
        print(f"\n# MUTATION: {label}")
        if hit:
            for line in hit:
                print("    REFUSED " + line.replace("\n", "\n    "))
        else:
            print(f"    NOT REFUSED. This check cannot see {label}; every other verdict it gives is "
                  f"worth less for it. All violations seen: {got}")
            ok = False
    for label, source, fragment in SYNTHETIC:
        got = check(source, path)
        hit = [g for g in got if fragment in g]
        print(f"\n# SYNTHETIC: {label}")
        if hit:
            for line in hit:
                print("    REFUSED " + line.replace("\n", "\n    "))
        else:
            print(f"    NOT REFUSED. All violations seen: {got}")
            ok = False
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="print the derived floor table")
    p.add_argument("--self-test", action="store_true", help="prove the check refuses a violating shrink")
    p.add_argument("--harness", default=HARNESS)
    a = p.parse_args(argv)
    if a.list:
        for name, dim, value, lo, why in table(path=a.harness):
            print(f"{name:34s} {dim:13s} runs at {value:<6} floor {lo:<6} {why}")
        gaps = unreadable_sites(path=a.harness)
        print(f"\n# SIZE SITES IN A SHRUNK LANE THIS FILE CANNOT READ, so no floor is required for them.")
        print(f"# They are the mechanism's hole; see the module docstring.")
        for name, group, expr in gaps:
            print(f"#   {name:32s} {group:10s} {expr}")
        if not gaps:
            print("#   (none)")
        return 0
    if a.self_test:
        good = self_test(a.harness)
        print("\nok: the floor check refuses every violation above" if good
              else "\nFAILED: see above")
        return 0 if good else 1
    bad = check(path=a.harness)
    for line in bad:
        print("FIXTURE FLOOR: " + line)
    print(f"ok: {len(table(path=a.harness))} fixture floor(s), none violated" if not bad
          else f"{len(bad)} fixture floor violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
