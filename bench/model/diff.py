#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The verdicts over bench/model records, in the style of
tools/identity_break.py --diff.

    python3 bench/model/diff.py --diff a.json b.json [c.json ...] [--require-columns N] [--arms f1,f2]
    python3 bench/model/diff.py --ratio ours.json torch.json [--arms f1,f2]

--diff: per weight format and per prompt, IDENTICAL xN when every column's
generated-id hash AND first-step-logits hash agree (N real hashes),
DIVERGENT when two columns disagree (and which of the two hashes differs
is named), MOVED when a column disagreed with itself run to run, ONE-COLUMN
when only one column hashed the cell, REFUSED when no column did. A
`summary:` line counts the verdicts, one `summary (<format>):` line per
format follows, and --require-columns N prints REQUIRE FAIL for any cell
resting on fewer than N real hashes. Exit 1 on any DIVERGENT, MOVED or
REQUIRE FAIL. Only `kind: ours` records are diffed: the incumbent is
never compared for bits, its number is a time (--ratio). Records of two
different models or two different prompt files are refused: their cells are
different questions.

--ratio: per format of ours and per arm of the incumbent, the prompt-wise
ratio ours / incumbent of the per-token milliseconds, for the prefill and
for the decode, as a median and a range over the prompts both hashed
STABLE, printed as

    identical mode (<format>) takes X.XX times the incumbent's time (<arm>) for prefill on <box>

which is the ONLY comparison this repository reports (bench/OPPONENT_REFERENCE.md,
CONTRIBUTING.md): our identical mode against the incumbent's fast
default, ours over theirs, where above 1.0 ours takes longer. The
incumbent's own determinism arm, when present, is reported the same way
against its fast arm, so the record shows what that setting costs it.
"""
import argparse
import json
import os
import statistics
import sys

REAL_BLOCKERS = ("MOVED", "REFUSED", "(not run)")


def load(paths):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        if j.get("schema") != "mojolearn.model_leg.v1":
            raise SystemExit(f"{p}: not a mojolearn.model_leg.v1 record")
        out.append((j.get("column") or os.path.basename(p), j))
    return out


def _cell_shown(cell):
    if cell is None:
        return "(not run)", None
    v = cell.get("verdict")
    if v == "REFUSED":
        return "REFUSED", None
    if v == "MOVED":
        return "MOVED", None
    ids, lg = cell.get("ids_sha256") or "", cell.get("first_logits_sha256") or ""
    if lg.startswith("MOVED:"):
        return "MOVED", None
    return f"{ids[:8]}/{lg[:8]}", (ids, lg)


def diff(paths, require_columns=0, arms=None):
    cols = load(paths)
    for name, j in cols:
        if j.get("kind") != "ours":
            raise SystemExit(f"REFUSING TO DIFF: {name} is a {j.get('kind')!r} record; the incumbent is "
                             "compared by --ratio, never for bits")
    models = set((j.get("model") or {}).get("config_sha256") for _, j in cols)
    if len(models) > 1:
        raise SystemExit(f"REFUSING TO DIFF: the columns ran different models (config_sha256 {sorted(map(str, models))})")
    weights = set((j.get("model") or {}).get("weights_sha256") for _, j in cols)
    if len(weights) > 1:
        print(f"WEIGHTS MISMATCH: the columns loaded different weight bytes ({sorted(map(str, weights))}); "
              "a DIVERGENT cell here is the checkpoint's divergence, not the library's")
    prompts = set((j.get("protocol") or {}).get("prompts_sha256") for _, j in cols)
    if len(prompts) > 1:
        raise SystemExit("REFUSING TO DIFF: the columns were not handed the same prompt file")
    maxnew = set((j.get("protocol") or {}).get("max_new") for _, j in cols)
    if len(maxnew) > 1:
        print(f"NOTE: the columns generated different lengths (max_new {sorted(map(str, maxnew))}); "
              "the id hashes cannot agree past the shorter one")
    names = [n for n, _ in cols]
    for n, j in cols:
        lib = j.get("library") or {}
        print(f"NOTE: column {n}: numeric_mode={lib.get('numeric_mode')} vendor={lib.get('vendor')} "
              f"commit={j.get('commit')} box={(j.get('box') or {}).get('gpu') or (j.get('box') or {}).get('cpu_model')}"
              + ("" if j.get("complete", True) else " INCOMPLETE (the run was cut; absent cells are absent, not clean)"))
        if lib.get("numeric_mode") not in (None, "identical"):
            print(f"NOTE: column {n} READ BACK numeric_mode={lib.get('numeric_mode')!r}; it is not an identical column")
    keys = sorted(set((a, p) for _, j in cols for a, arm in (j.get("arms") or {}).items()
                      for p in (arm.get("prompts") or {})))
    if arms:
        keys = [k for k in keys if k[0] in set(arms)]
    if require_columns and require_columns > len(cols):
        print(f"REQUIRE FAIL: --require-columns {require_columns} with {len(cols)} JSONs given")
    print(f"| {'format/prompt':<20} | {'verdict':<13} | " + " | ".join(f"{n:<18}" for n in names) + " |")
    print(f"|{'-'*22}|{'-'*15}|" + "|".join("-" * 20 for _ in names) + "|")
    bad, short, counts, per_arm = 0, [], {}, {}
    for a, p in keys:
        shown, real, refused = [], [], False
        for _, j in cols:
            cell = ((j.get("arms") or {}).get(a) or {}).get("prompts", {}).get(p)
            s, pair = _cell_shown(cell)
            shown.append(s)
            if s == "REFUSED":
                refused = True
            if pair is not None:
                real.append(pair)
        if "MOVED" in shown:
            verdict = "MOVED"
        elif len(real) >= 2:
            if len(set(real)) == 1:
                verdict = f"IDENTICAL x{len(real)}"
            else:
                verdict = "DIVERGENT"
                ids_same = len(set(r[0] for r in real)) == 1
                lg_same = len(set(r[1] for r in real)) == 1
                which = "first logits differ, ids agree" if ids_same and not lg_same else (
                    "ids differ, first logits agree" if lg_same and not ids_same else "ids and first logits differ")
                shown[0] = which
        elif len(real) == 1:
            verdict = "ONE-COLUMN"
        else:
            verdict = "REFUSED" if refused else "NOT-RUN"
        if verdict in ("MOVED", "DIVERGENT"):
            bad += 1
        n_real = len(real) if verdict.startswith(("IDENTICAL", "ONE-COLUMN")) else (0 if verdict in ("REFUSED", "NOT-RUN") else None)
        if require_columns and n_real is not None and n_real < require_columns:
            short.append((a, p, verdict, n_real))
        head = verdict.split(" ")[0]
        counts[head] = counts.get(head, 0) + 1
        per_arm.setdefault(a, {})
        per_arm[a][head] = per_arm[a].get(head, 0) + 1
        print(f"| {a + '/' + p:<20} | {verdict:<13} | " + " | ".join(f"{s:<18}" for s in shown) + " |")
    print()
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for a in sorted(per_arm):
        print(f"summary ({a}): " + ", ".join(f"{k}={v}" for k, v in sorted(per_arm[a].items())))
    if require_columns:
        for a, p, verdict, n in short:
            print(f"REQUIRE FAIL {a}/{p}: {verdict} rests on {n} real hash(es), --require-columns "
                  f"{require_columns} demands that many; a column that should cover this cell is refusing, "
                  "which is not a pass")
        if require_columns > len(cols):
            bad += 1
        bad += len(short)
        print(f"require-columns {require_columns}: {'OK' if not short and require_columns <= len(cols) else str(len(short)) + ' short'}")
    return 1 if bad else 0


def _box_name(j):
    b = j.get("box") or {}
    return b.get("gpu") or b.get("cpu_model") or b.get("hostname") or "unnamed box"


def _ratios(ours_arm, theirs_arm):
    """Per prompt: (prefill ratio, decode ratio) over prompts both hashed
    STABLE with a positive time; the rest are named as excluded."""
    rows, excluded = [], []
    for pid, oc in sorted((ours_arm.get("prompts") or {}).items()):
        tc = (theirs_arm.get("prompts") or {}).get(pid)
        if tc is None or oc.get("verdict") != "STABLE" or tc.get("verdict") != "STABLE":
            excluded.append(pid)
            continue
        op, od = oc.get("prefill_ms_per_token"), oc.get("decode_ms_per_token")
        tp, td = tc.get("prefill_ms_per_token"), tc.get("decode_ms_per_token")
        if not all(isinstance(x, (int, float)) and x > 0 for x in (op, od, tp, td)):
            excluded.append(pid)
            continue
        rows.append((pid, op / tp, od / td, op, tp, od, td))
    return rows, excluded


def _line(subject, arm_name, rows, phase, box, idx):
    vals = [r[idx] for r in rows]
    if not vals:
        return f"{subject} takes an unknown multiple of the incumbent's time ({arm_name}) for {phase} on {box}: no prompt both columns hashed STABLE"
    med = statistics.median(vals)
    return (f"{subject} takes {med:.2f} times the incumbent's time ({arm_name}) for {phase} on {box}: "
            f"median over {len(vals)} prompts, range {min(vals):.2f} to {max(vals):.2f}")


def ratio(ours_path, torch_path, arms=None):
    (on, ours), (tn, theirs) = load([ours_path, torch_path])
    if ours.get("kind") != "ours" or theirs.get("kind") != "torch":
        raise SystemExit("--ratio takes ours.json then torch.json (kind ours, then kind torch)")
    box = _box_name(ours)
    ob, tb = ours.get("box") or {}, theirs.get("box") or {}
    if ob.get("hostname") != tb.get("hostname") or ob.get("gpu") != tb.get("gpu"):
        print(f"NOTE: the two records name different boxes (ours {ob.get('hostname')} {ob.get('gpu')}; "
              f"incumbent {tb.get('hostname')} {tb.get('gpu')}); a ratio across boxes is not a result")
    if (ours.get("model") or {}).get("config_sha256") != (theirs.get("model") or {}).get("config_sha256"):
        raise SystemExit("REFUSING: the two records ran different models")
    if (ours.get("protocol") or {}).get("prompts_sha256") != (theirs.get("protocol") or {}).get("prompts_sha256"):
        raise SystemExit("REFUSING: the two records were not handed the same prompt file")
    lib = theirs.get("library") or {}
    print(f"incumbent: torch {lib.get('torch')} transformers {lib.get('transformers')} device {lib.get('device')} "
          f"{lib.get('device_name') or ''} dtype {lib.get('dtype')}; ours: numeric_mode "
          f"{(ours.get('library') or {}).get('numeric_mode')} vendor {(ours.get('library') or {}).get('vendor')} "
          f"commit {ours.get('commit')}")
    our_arms = [a for a in (ours.get("arms") or {}) if not arms or a in set(arms)]
    their_arms = list(theirs.get("arms") or {})
    fast_arms = [a for a in their_arms if a != "torch-deterministic"]
    for fa in fast_arms:
        for oa in our_arms:
            rows, excluded = _ratios(ours["arms"][oa], theirs["arms"][fa])
            print()
            print(_line(f"identical mode ({oa})", fa, rows, "prefill", box, 1))
            print(_line(f"identical mode ({oa})", fa, rows, "decode", box, 2))
            if excluded:
                print(f"  excluded prompts (not STABLE on both columns, or no time): {', '.join(excluded)}")
            print(f"  | {'prompt':<6} | {'prefill x':>9} | {'decode x':>9} | {'ours pre ms/tok':>15} | {'theirs pre':>10} | {'ours dec ms/tok':>15} | {'theirs dec':>10} |")
            for pid, rp, rd, op, tp, od, td in rows:
                print(f"  | {pid:<6} | {rp:>9.2f} | {rd:>9.2f} | {op:>15.3f} | {tp:>10.3f} | {od:>15.3f} | {td:>10.3f} |")
    if "torch-deterministic" in their_arms:
        for fa in fast_arms:
            rows, excluded = _ratios(theirs["arms"]["torch-deterministic"], theirs["arms"][fa])
            print()
            print(_line("the incumbent's own determinism setting (torch-deterministic)", fa, rows, "prefill", box, 1)
                  .replace("the incumbent's time", "its fast default's time"))
            print(_line("the incumbent's own determinism setting (torch-deterministic)", fa, rows, "decode", box, 2)
                  .replace("the incumbent's time", "its fast default's time"))
            if excluded:
                print(f"  excluded prompts: {', '.join(excluded)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--diff", nargs="+", metavar="RECORD.json")
    g.add_argument("--ratio", nargs=2, metavar=("OURS.json", "TORCH.json"))
    ap.add_argument("--require-columns", type=int, default=0)
    ap.add_argument("--arms", default=None, help="comma separated formats to restrict to")
    args = ap.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
    if args.diff:
        return diff(args.diff, args.require_columns, arms)
    return ratio(args.ratio[0], args.ratio[1], arms)


if __name__ == "__main__":
    sys.exit(main())
