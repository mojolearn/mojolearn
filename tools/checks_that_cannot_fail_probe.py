# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""THREE VERIFICATION CHECKS THAT CANNOT PRODUCE THEIR NEGATIVE OUTCOME.

Run it: `python3 tools/checks_that_cannot_fail_probe.py`. No native binding,
no GPU, under a second. Every section builds the input that SHOULD make the
check fail and prints what the shipped code does with it.

This file exists because all three were found by stumbling over them, and a
finding nobody can re-run is a finding that dies with the transcript. NONE of
them is fixed here; each needs a decision that is not this lane's to make.

  A  `_verify_all._collapse` decides MOVED by `len(set(values))`. `verify
     --all` and `verify --par` default to `--repeats 1`, so `values` always
     holds one element, the set always holds one element, and MOVED is
     unreachable. Every consumer of MOVED goes with it:
     `_verify_reference.judge`'s `value == "MOVED"` arm, and `ONE-COLUMN`'s
     sibling in `_verify_par.GATING`. BLOCKED on Andrew's standing
     `--repeats 1` instruction; `tools/identity_break.py` defaults to 2.

  B  `_verify_all.commitment_state` and `challenge_state` guard their
     self-consistency catch with `isinstance(stored, str)`. A tampered
     document whose carried commitment is the right value in the wrong type
     -- a one-element list, the shape this repo already has a rule about --
     skips the catch and reads `self-declared`, which is in `_COMMITMENT_OK`.
     FIXED on main by c5d483301 (it now reads MALFORMED); section B below is
     kept as a regression check that fails if the finding returns.

  C  `_identity.py` drops `--require-columns 4` when `--fixtures` narrows,
     and `_identity._judge`'s regexes match only the `infer` and `model`
     columns of the diff's second table. The `batch`, `rlpair` and
     `stepfull` rows are invisible to both gates at once, while the command
     still prints `read IDENTICAL x4`.
"""
import ast
import contextlib
import io
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

RULE = "=" * 74


def _lift(path, names):
    """Top-level functions and CONSTANTS out of a module's source text.

    IMPORTING IS NOT AN OPTION: `mojolearn._verify_all` pulls the native
    binding in, and a probe that needs a built wheel is a probe nobody runs.
    Lifting reads the shipped source every time, so the probe cannot drift
    away from the file it is about."""
    text = open(path, encoding="utf-8").read()
    picked = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            picked.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
            picked.append(node)
    ns = {"json": json, "re": re, "os": os, "sys": sys,
          "hashlib": __import__("hashlib")}
    exec(compile(ast.Module(body=picked, type_ignores=[]), path, "exec"), ns)
    missing = sorted(n for n in names if n not in ns)
    if missing:
        raise SystemExit(f"{path}: could not lift {missing}; the probe is stale")
    return ns


# --------------------------------------------------------------------------
# A. MOVED cannot be reached at --repeats 1
# --------------------------------------------------------------------------

def probe_moved():
    src = os.path.join(ROOT, "python", "mojolearn", "_verify_all.py")
    collapse = _lift(src, {"_collapse"})["_collapse"]
    print(RULE)
    print("A. `verify --all` / `verify --par` cannot report MOVED at their default")
    print("   _verify_all._collapse, the last line:")
    print("     return (values[0], None) if len(set(values)) == 1 else ('MOVED', None)")
    print()
    print("   THE INPUT THAT SHOULD SAY MOVED -- a box that hashed two ways:")
    two = collapse(["a" * 16, "b" * 16], [])
    print(f"     repeats=2  -> {two!r}")
    assert two == ("MOVED", None), "the control arm is broken; this probe proves nothing"
    print()
    print("   THE SAME BOX, THE SAME NON-DETERMINISM, AT --repeats 1. The runner")
    print("   collects one hash, so only one of those two is ever in the list:")
    hits = 0
    tried = ["a" * 16, "b" * 16, "0" * 16, "n/a:no-sampler-trainer-pair", "", "0", 0, 1]
    for v in tried:
        got = collapse([v], [])
        hits += got[0] == "MOVED"
    for v in tried[:2]:
        print(f"     repeats=1  values=[{v!r}] -> {collapse([v], [])!r}")
    print(f"     one-element inputs tried: {len(tried)};  MOVED verdicts: {hits}")
    assert hits == 0, "MOVED turned out to be reachable; re-read this probe"
    print()
    print("   STILL ALIVE at one repeat, because the harness decides them inside")
    print("   ONE fit rather than across repeats:")
    for v in ("BATCH_MOVED:forward L=64:row 3", "RELOAD-MOVED", "RLPAIR_MOVED:x"):
        print(f"     {v[:34]:<34} -> {collapse([v], [])[0]!r}")
    print()
    print("   So batch, reload and rlpair keep their teeth; the plain same-box")
    print("   MOVED on train/infer/model does not. `tools/identity_break.py`")
    print("   defaults to --repeats 2 and is unaffected.")


# --------------------------------------------------------------------------
# B. a commitment in the wrong type skips the self-consistency catch
# --------------------------------------------------------------------------

def probe_commitment():
    src = os.path.join(ROOT, "python", "mojolearn", "_verify_all.py")
    ns = _lift(src, {"commitment_state", "commitment_report", "seal_document",
                     "_without_reveal", "commitment_preimage",
                     "read_published_commitment", "commitment_digest"})
    commitment_state, seal_document = ns["commitment_state"], ns["seal_document"]
    commitment_report = ns["commitment_report"]
    commitment_digest, REVEAL = ns["commitment_digest"], ns["REVEAL_KEY"]
    ok_states = ns["_COMMITMENT_OK"]

    doc = {"format": "mojolearn.verify.v1", "verdict": "VERIFIED",
           "cells": [{"lane": "rf-clf", "value": "0" * 16}],
           "device": {"vendor": "cpu"}, "verification_contract": {"clause": 1}}
    sealed = dict(doc)
    published = seal_document(sealed)

    print()
    print(RULE)
    print("B. a carried commitment in the wrong TYPE skips the self-consistency catch")
    print(f"   _COMMITMENT_OK = {ok_states}")
    print()
    print("   CONTROL, untouched:        ", commitment_state(sealed, published, "doc")["state"])
    tampered = json.loads(json.dumps(sealed))
    tampered["verdict"] = "MISMATCH"
    st = commitment_state(tampered, None, "doc")
    print("   CONTROL, edited after seal:", st["state"],
          "  <- the catch CAN fire, so this is a real check")
    assert st["state"] == "SELF-INCONSISTENT", "the control arm is broken"

    print()
    print("   THE SAME EDITED DOCUMENT, carried commitment in a different type:")
    for label, wrap in (("a one-element list", lambda s: [s]),
                        ("a dict", lambda s: {"value": s}),
                        ("an int", lambda s: int(s[:8], 16)),
                        ("null (a stripped field)", lambda s: None)):
        t = json.loads(json.dumps(tampered))
        t[REVEAL]["commitment"] = wrap(t[REVEAL]["commitment"])
        state = commitment_state(t, None, "doc")["state"]
        print(f"     {label:<26} -> {state!r}"
              + ("   PASSES" if state in ok_states else "   gates"))

    print()
    print("   FIXED ON MAIN (c5d483301, 2026-09-20): a carried commitment in the")
    print("   wrong type now reads MALFORMED, which outranks every other state.")
    for label, wrap in (("a one-element list", lambda s: [s]),
                        ("null (a stripped field)", lambda s: None)):
        t = json.loads(json.dumps(tampered))
        t[REVEAL]["commitment"] = wrap(t[REVEAL]["commitment"])
        state = commitment_state(t, None, "doc")["state"]
        assert state not in ok_states, f"{label} passes again; finding B has regressed"
    t = json.loads(json.dumps(tampered))
    t[REVEAL]["commitment"] = [t[REVEAL]["commitment"]]
    honest = commitment_digest(sealed, sealed[REVEAL]["nonce"])
    st = commitment_state(t, honest, "doc")
    print(f"     with the original published commitment -> {st['state']!r}")
    assert st["state"] not in ok_states, "the published check passes a wrong-type commitment"

    other = dict(doc)
    seal_document(other)
    rep_str = commitment_report(tampered, other, "A", "B", None, None)
    rep_list = commitment_report(t, other, "A", "B", None, None)
    print(f"     carried commitment a STRING -> broken={rep_str['broken']}")
    print(f"     carried commitment a LIST   -> a.state {rep_list['a']['state']!r}, "
          f"broken={rep_list['broken']}")
    assert rep_str["broken"] is True, "the control arm is broken"
    assert rep_list["a"]["state"] not in ok_states, "finding B has regressed"
    print("   B is a regression check now: every assertion here fails if it returns.")


# --------------------------------------------------------------------------
# C. the narrowing flag and the row regexes turn both gates off at once
# --------------------------------------------------------------------------

_HASH = "0123456789abcdef"
_BATCH = "beefbeefbeefbeef"
_KEY = "knn/base"

# lifted verbatim from python/mojolearn/_identity.py:204-205
_ROW = re.compile(r"^\| (?P<key>\S+)\s+\| (?P<verdict>[A-Z][A-Za-z0-9 x-]*?)\s+\|")
_ROW2 = re.compile(r"^\| (?P<key>\S+)\s+\| (?P<col>infer|model)\s+\| "
                   r"(?P<verdict>[A-Z][A-Za-z0-9/ x-]*?)\s+\|")


def _cell(batch=None):
    c = dict(verdict="STABLE", hashes=[_HASH, _HASH], parts=[{}],
             infer=[_HASH, _HASH], infer_verdict="STABLE",
             model=[_HASH, _HASH], model_verdict="STABLE")
    if batch is not None:
        c.update(batch=[batch, batch], batch_verdict="STABLE")
    return c


def _column(vendor, batch):
    return dict(vendor=vendor, mode="identical", commit="c" * 40, package={},
                cells={_KEY: _cell(batch)})


def _run_diff(ib, columns, **kw):
    tmp = tempfile.mkdtemp(prefix="cannot-fail-probe-")
    paths = []
    for i, col in enumerate(columns):
        p = os.path.join(tmp, f"col{i}.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(col, fh)
        paths.append(p)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ib.diff(paths, **kw)
    return rc, buf.getvalue()


def _batch_rows(text):
    return [l for l in text.splitlines() if l.startswith("| " + _KEY) and "| batch" in l]


def probe_narrowing():
    import identity_break as ib
    print()
    print(RULE)
    print("C. --fixtures drops --require-columns, and _judge cannot see the batch row")
    print()
    three = [_column(f"rec{i}", _BATCH) for i in range(3)]

    print("   CONTROL: four columns, all carrying the batch part.")
    rc, text = _run_diff(ib, three + [_column("local", _BATCH)])
    print(f"     diff exit {rc}   {_batch_rows(text)[0][:78]}")

    print()
    print("   THE INPUT THAT SHOULD FAIL: the LOCAL column's batch part is absent,")
    print("   so the batch cell rests on three hashes, not four.")
    short = three + [_column("local", None)]
    rc, text = _run_diff(ib, short)
    print(f"     no --require-columns   -> diff exit {rc}   {_batch_rows(text)[0][:60]}")
    rc4, text4 = _run_diff(ib, short, require_columns=4, require_lanes=["knn"])
    print(f"     --require-columns 4    -> diff exit {rc4}")
    for l in text4.splitlines():
        if l.startswith("REQUIRE FAIL"):
            print("       " + l[:100])
    assert rc == 0 and rc4 == 1, "the two arms did not differ; re-read this probe"
    print("     `_identity.py:371-375` adds `--require-columns 4` ONLY when the")
    print("     fixture set is complete, so `--fixtures base` takes the first row.")

    print()
    print("   AND `_judge` CANNOT SEE THE ROW EITHER. Every row of a real diff,")
    print("   against the two regexes `_identity._judge` matches with:")
    for line in text.splitlines():
        if not line.startswith("| " + _KEY):
            continue
        m2, m = _ROW2.match(line), _ROW.match(line)
        if m2:
            seen = f"_ROW2 -> {m2.group('col')}/{m2.group('verdict').strip()}"
        elif m:
            seen = f"_ROW  -> train/{m.group('verdict').strip()}"
        else:
            seen = "INVISIBLE TO _judge"
        print(f"     {line[:52]:<52}  {seen}")
    assert not any(_ROW.match(l) or _ROW2.match(l) for l in _batch_rows(text)), \
        "_judge can see the batch row after all; this finding is stale"
    print()
    print("     `_ROW2` requires the literal `infer|model`, and `_ROW` requires the")
    print("     second field to start with a capital. `batch`, `rlpair` and the")
    print("     EXTRA_PARTS rows match neither, so they are never in `bad` and")
    print("     never in `seen` -- while the pass prints `read IDENTICAL x4`.")

    print()
    print("   WORSE: a part only the LOCAL column carries reads NOT-COMPARED, and")
    print("   `_real_count` returns None for it, so --require-columns lets it by:")
    one_side = [_column(f"rec{i}", None) for i in range(3)] + [_column("local", _BATCH)]
    rc, text = _run_diff(ib, one_side)
    rc4, _ = _run_diff(ib, one_side, require_columns=4, require_lanes=["knn"])
    print(f"     {_batch_rows(text)[0][:72]}")
    print(f"     diff exit {rc} without the gate, {rc4} WITH --require-columns 4")
    assert rc == 0 and rc4 == 0, "NOT-COMPARED gates now; this finding is stale"


def main():
    print(__doc__.split("\n\n", 1)[0])
    print()
    probe_moved()
    probe_commitment()
    probe_narrowing()
    print()
    print(RULE)
    print("Every assertion above is a CONTROL: it fails loudly if the shipped code")
    print("has been changed so that the finding no longer holds. A green run means")
    print("A and C are still live and B stays fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
