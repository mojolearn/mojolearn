"""Every identity_break lane whose IN-CELL ORACLE is a bare raise.

A lane that calls `_same_bytes` in its TRAIN body has an oracle that, when it
fires, raises before the lane can hash anything: the cell reads REFUSED and a
negative control that WORKED is recorded as a crash. `_mismatch_bytes` +
`NumericalMismatch(msg, parts)` is the remedy (par-scaler, 2026-09-17).

A call inside the INFER lambda is a weaker case: it costs the infer/model/batch
columns, not the cell, and the train hashes survive.

Prints one row per lane, and a self-check first: a synthetic module with one
known-bare and one known-fixed lane must classify both correctly.
"""
import ast
import sys


def _calls(node, name):
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
            yield n


def lane_names(dec):
    out = []
    for d in dec:
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "lane":
            for a in d.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    out.append(a.value)
    return out


def classify(tree):
    """[(lanes, bare_in_body, bare_in_lambda, uses_numerical_mismatch)]"""
    rows = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        names = lane_names(node.decorator_list)
        if not names:
            continue
        lambdas = [n for n in ast.walk(node) if isinstance(n, ast.Lambda)]
        in_lambda = set()
        for lam in lambdas:
            for c in _calls(lam, "_same_bytes"):
                in_lambda.add(id(c))
        body_hits = [c for c in _calls(node, "_same_bytes") if id(c) not in in_lambda]
        lam_hits = [c for c in _calls(node, "_same_bytes") if id(c) in in_lambda]
        fixed = bool(list(_calls(node, "_mismatch_bytes")) or list(_calls(node, "_oracle_mismatch"))
                     or [n for n in ast.walk(node)
                         if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                         and isinstance(n.exc.func, ast.Name) and n.exc.func.id == "NumericalMismatch"])
        rows.append((names, len(body_hits), len(lam_hits), fixed))
    return rows


SELFCHECK = '''
@lane("bare-one")
def _(ml, X, yc, yr, Xh=None):
    _same_bytes("a", a, "b", b)
    return _fit(dict(x=_h(a)), m, lambda e: (e.predict(Xh),))

@lane("fixed-one")
def _(ml, X, yc, yr, Xh=None):
    mismatch = _mismatch_bytes("a", a, "b", b)
    parts = dict(x=_h(a))
    if mismatch:
        raise NumericalMismatch(mismatch, parts)
    return _fit(parts, m, lambda e: _same_bytes("c", c, "d", d))
'''


def selfcheck():
    rows = classify(ast.parse(SELFCHECK))
    got = {r[0][0]: (r[1], r[2], r[3]) for r in rows}
    assert got["bare-one"] == (1, 0, False), got
    assert got["fixed-one"] == (0, 1, True), got
    print("selfcheck: the audit separates a bare body oracle from a fixed one\n")


def main():
    selfcheck()
    src = open(sys.argv[1]).read()
    rows = classify(ast.parse(src))
    bare_body, only_lambda = [], []
    for names, body, lam, fixed in rows:
        if body:
            bare_body.append((names, body, lam, fixed))
        elif lam:
            only_lambda.append((names, lam, fixed))
    print("A. BARE `_same_bytes` IN THE LANE BODY -- the cell reads REFUSED when it fires")
    print("   %d lane declarations\n" % sum(len(n) for n, _, _, _ in bare_body))
    for names, body, lam, fixed in sorted(bare_body):
        par = "par-*" if all(n.startswith("par-") for n in names) else "     "
        print("   %-5s %-34s body=%d lambda=%d also_uses_NumericalMismatch=%s"
              % (par, ",".join(names), body, lam, fixed))
    print("\nB. `_same_bytes` ONLY IN THE INFER LAMBDA -- costs infer/model/batch, not the cell")
    print("   %d lane declarations\n" % sum(len(n) for n, _, _ in only_lambda))
    for names, lam, fixed in sorted(only_lambda):
        print("   %-34s lambda=%d" % (",".join(names), lam))
    print("\ntotals: %d lanes with a bare body oracle, %d with a lambda-only one, %d lanes scanned"
          % (sum(len(n) for n, _, _, _ in bare_body),
             sum(len(n) for n, _, _ in only_lambda),
             sum(len(r[0]) for r in rows)))


main()
