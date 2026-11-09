#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/measure_owed.py -- the measurements tonight's lanes left owed (bench/results/paper_compare/owed-pre0867.json),
planned through the board and reported beside the paper compare (lane measure-prep, 2026-11-09).

  plan   --owed JSON --paper-recipes JSON --planner tools/bench_board.py --vendor V --out-dir DIR [--b-arms a,b]
        ON THE BOX: resolves every board / ab row to the race ids the current board plans (bench_board.plan_races),
        per ARM (candidate, prior = the published 0.8.66 ours-only venv, and each B wheel set named in --b-arms) and
        per ROWS group ('full', or a neighbouring row count: its own bench_board --rows process). Writes into DIR:
          board-args.txt        the union of the paper's and the owed rows' families, lanes and datasets (one arg
                                per line), so ONE board process can time the paper races and the owed races together
          units.tsv             arm <TAB> rows <TAB> ids-file, one line per board process the job runs, in order
          ids-<arm>-<rows>.txt  the race ids of that process
          smoke-ids.txt         the smoke ids (rows=<smoke rows>, shape=small) of the owed full-row races the paper
                                smoke does not already cover; the job smokes them with the paper's
          drivers.json          the driver rows (tools/measure_owed_driver.py), with the arms present on this box
          not-planned.json      {row id or race: why}: a lane the board no longer has, a B set not staged, ...
          rowmap.json           {row id: [race ids]} for the report
        A race id the paper recipes hold is not planned again for the candidate or the prior arm: the paper board
        times it for the candidate, and its 0.8.66 number is the paper compare's middle column.
  report --owed JSON --plan-dir DIR --results RES [--prior-compare JSON] --vendor V [--out-md --out-json --summary]
        per owed row and race cell: candidate ms, prior ms (the prior board, or the paper compare's 0.8.66 cell),
        each B arm's ms, the ratios, and the row's flags: REGRESSION / FASTER / SAME vs the prior, B-FASTER / B-SLOWER
        / B-SAME for an A/B, NOT-RUN with its reason, and the driver rows from drivers.jsonl. Prints one screen.

Our arms only, GPU only, one warm-up and one timed run per arm (the board's --rounds 1). Tests:
tools/test_measure_owed.py.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_compare as PC  # noqa: E402

SCHEMA = "mojolearn-measure-owed/1"
REPORT_SCHEMA = "mojolearn-measure-owed-report/1"
ARMS_FIXED = ("candidate", "prior")


def load_planner(path):
    spec = importlib.util.spec_from_file_location("measure_owed_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def rows_tag(rows):
    return "full" if rows in (None, "full") else str(int(rows))


def family_of(planner):
    out = {}
    for f in planner.FAMILIES:
        for lane in planner.family_lanes(f):
            out.setdefault(lane, f)
    return out


def row_arms(row, b_present):
    """The board arms of an owed row: board rows race the candidate and the prior; ab rows the candidate and every
    B set the row names that is staged on this box (the missing ones are returned apart)."""
    if row["kind"] == "board":
        return ["candidate", "prior"], []
    if row["kind"] == "ab":
        want = as_list(row.get("b_arms") or [])
        return ["candidate"] + [b for b in want if b in b_present], [b for b in want if b not in b_present]
    return [], []


def plan(doc, paper, planner, vendor, b_present):
    """(units {(arm, rows): [race ids]}, smoke ids, drivers, not_planned, rowmap, board args)."""
    if doc.get("schema") != SCHEMA:
        raise SystemExit("measure_owed: not a %s document" % SCHEMA)
    fam = family_of(planner)
    modes = planner.modes_for(vendor, "identical")
    paper_ids = set(paper.get("race_ids") or [])
    pb = paper.get("bench_board_args") or {}
    families, lanes, datasets = list(pb.get("families") or []), list(pb.get("lanes") or []), list(pb.get("datasets") or [])
    units, smoke, drivers, missing, rowmap = {}, [], [], {}, {}
    cache = {}
    smoke_rows = int(getattr(planner, "SMOKE_ROWS", 2000))

    def planned(lane_list, rows, shape="full"):
        fams = tuple(sorted({fam[l] for l in lane_list}))
        key = (fams, tuple(lane_list), rows, shape)
        if key not in cache:
            races = planner.plan_races(vendor, modes, fams, tuple(lane_list), tuple(planner.DATASETS), rows, shape)
            cache[key] = planner.gpu_opponents_only(races, vendor)
        return cache[key]

    for row in doc["rows"]:
        rid = row["id"]
        if row["kind"] == "driver":
            drivers.append(row)
            continue
        if row["kind"] not in ("board", "ab"):
            missing[rid] = "unknown kind %r" % row["kind"]
            continue
        want_lanes = as_list(row["lane"])
        gone = [l for l in want_lanes if l not in fam]
        for l in gone:
            missing["%s %s" % (rid, l)] = "lane %s is no longer a board lane" % l
        lane_list = [l for l in want_lanes if l in fam]
        if not lane_list:
            continue
        rows = None if row.get("rows") in (None, "full") else int(row["rows"])
        ds = row.get("dataset") or "*"
        races = [r for r in planned(lane_list, rows) if ds == "*" or r["dataset"] == ds]
        if not races:
            missing[rid] = "the board plans no race for %s on %s (dataset %s, rows %s)" % (
                ",".join(lane_list), vendor, ds, rows_tag(rows))
            continue
        ids = sorted({r["id"] for r in races})
        rowmap[rid] = ids
        arms, absent = row_arms(row, b_present)
        for b in absent:
            missing["%s %s" % (rid, b)] = "B wheel set %s is not staged on this box (PC_B_ARMS)" % b
        for arm in arms:
            todo = [i for i in ids if not (arm in ARMS_FIXED and rows is None and i in paper_ids)]
            if todo:
                units.setdefault((arm, rows_tag(rows)), set()).update(todo)
        for l in lane_list:
            f = fam[l]
            if f not in families:
                families.append(f)
            if l not in lanes:
                lanes.append(l)
        for r in races:
            if r["family"] != "neural" and r["dataset"] not in datasets and r["dataset"] in planner.DATASETS:
                datasets.append(r["dataset"])
        if rows is None:
            sm = {r["id"]: r for r in planned(lane_list, smoke_rows, PC.SMOKE_SHAPE)}
            for i in ids:
                if i in paper_ids:
                    continue
                sid = PC.smoke_id(i, smoke_rows)
                if sid in sm:
                    smoke.append(sid)
                else:
                    missing["%s %s" % (rid, i)] = "the board plans no smoke race %s" % sid
    bargs = ["--vendor", vendor, "--families", ",".join(families), "--lanes", ",".join(lanes),
             "--modes", "identical", "--gpu-opponents-only"]
    if datasets:
        bargs += ["--datasets", ",".join(datasets)]
    return ({k: sorted(v) for k, v in units.items()}, sorted(set(smoke)), drivers, missing, rowmap, bargs)


def unit_order(key):
    arm, rows = key
    a = 0 if arm == "candidate" else 1 if arm == "prior" else 2
    return (a, arm, rows != "full", rows)


def cmd_plan(a):
    doc = PC.load_json(a.owed)
    paper = PC.load_json(a.paper_recipes)
    planner = load_planner(a.planner)
    b_present = [b for b in (a.b_arms or "").split(",") if b]
    unknown = [b for b in b_present if b not in (doc.get("b_arms") or {})]
    if unknown:
        raise SystemExit("measure_owed: B set(s) %s are not in %s" % (",".join(unknown), a.owed))
    units, smoke, drivers, missing, rowmap, bargs = plan(doc, paper, planner, a.vendor, b_present)
    os.makedirs(a.out_dir, exist_ok=True)
    lines = []
    for key in sorted(units, key=unit_order):
        arm, rows = key
        f = os.path.join(a.out_dir, "ids-%s-%s.txt" % (arm, rows))
        with open(f, "w") as fh:
            fh.write("".join(i + "\n" for i in units[key]))
        lines.append("%s\t%s\t%s\n" % (arm, rows, f))
    with open(os.path.join(a.out_dir, "units.tsv"), "w") as fh:
        fh.write("".join(lines))
    with open(os.path.join(a.out_dir, "smoke-ids.txt"), "w") as fh:
        fh.write("".join(i + "\n" for i in smoke))
    with open(os.path.join(a.out_dir, "board-args.txt"), "w") as fh:
        fh.write("".join(x + "\n" for x in bargs))
    for name, obj in (("drivers.json", drivers), ("not-planned.json", missing), ("rowmap.json", rowmap)):
        with open(os.path.join(a.out_dir, name), "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
    print("OWED-PLAN vendor=%s rows=%d units=%d races=%d smoke=%d drivers=%d not_planned=%d b_arms=%s"
          % (a.vendor, len(doc["rows"]), len(units), sum(len(v) for v in units.values()), len(smoke),
             len(drivers), len(missing), ",".join(b_present) or "-"))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------

def board_dir(results, arm, rows):
    """Where the job writes the board of one unit (tools/ops/paper_compare_job.sh uses the same names)."""
    if arm == "candidate" and rows == "full":
        return os.path.join(results, "board")
    return os.path.join(results, "board-%s-%s" % (arm, rows))


def our_cells(race):
    """{(phase, arm, batch): cell} of our arms in one board race."""
    out = {}
    for phase, cells in (("fit", race.get("cells") or []), ("infer", race.get("infer_cells") or [])):
        for c in cells:
            if str(c.get("arm", "")).startswith("ours") or c.get("library") == "mojolearn":
                out[(phase, c.get("arm"), c.get("batch"))] = c
    return out


def cell_ms(cell):
    if not cell:
        return None
    return cell.get("median_ms") if PC.timed(cell.get("status"), cell.get("median_ms")) else None


def load_board(path):
    try:
        return (PC.load_json(path).get("races") or {})
    except (OSError, ValueError):
        return None


def verdict(ratio, low_flag, high_flag, same_flag):
    if ratio is None:
        return None
    return high_flag if ratio > PC.SLOWER else low_flag if ratio < PC.FASTER else same_flag


def report(doc, plan_dir, results, prior_rows, vendor):
    rowmap = PC.load_json(os.path.join(plan_dir, "rowmap.json"))
    missing = PC.load_json(os.path.join(plan_dir, "not-planned.json"))
    boards = {}

    def board(arm, rows):
        k = (arm, rows)
        if k not in boards:
            boards[k] = load_board(os.path.join(board_dir(results, arm, rows), "board.json")) or {}
        return boards[k]

    out = []
    for row in doc["rows"]:
        rid = row["id"]
        if row["kind"] == "driver":
            continue
        rows = rows_tag(row.get("rows"))
        why = [v for k, v in missing.items() if k == rid or k.startswith(rid + " ")]
        b_arms = as_list(row.get("b_arms") or []) if row["kind"] == "ab" else []
        for race_id in rowmap.get(rid, []):
            cand_race = board("candidate", rows).get(race_id) or {}
            prior_race = board("prior", rows).get(race_id) or {}
            cells = our_cells(cand_race)
            if not cells:
                out.append({"row": rid, "race_id": race_id, "key": race_id, "flags": ["NOT-RUN"],
                            "why": "; ".join(why) or "no candidate cell (smoke failed or not reached)"})
                continue
            for (phase, arm, batch), cell in sorted(cells.items(), key=lambda kv: str(kv[0])):
                key = PC.recipe_key(race_id, phase, arm, batch)
                rec = {"row": rid, "race_id": race_id, "key": key, "phase": phase, "arm": arm, "batch": batch,
                       "candidate_ms": cell_ms(cell), "candidate_status": PC.short_status(cell.get("status")),
                       "flags": [], "why": "; ".join(why) or None}
                if row["kind"] == "board":
                    pc = our_cells(prior_race).get((phase, arm, batch))
                    pms = cell_ms(pc)
                    if pms is None and prior_rows and key in prior_rows:
                        pms = prior_rows[key].get("new_ms") if PC.number(prior_rows[key].get("new_ms")) else None
                        rec["prior_from"] = "paper compare"
                    rec["prior_ms"] = pms
                    r = rec["candidate_ms"] / pms if PC.number(rec["candidate_ms"]) and pms else None
                    rec["ratio_vs_prior"] = r
                    f = verdict(r, "FASTER", "REGRESSION", "SAME")
                    rec["flags"].append(f or "NOT-COMPARED")
                for b in b_arms:
                    bc = our_cells(board(b, rows).get(race_id) or {}).get((phase, arm, batch))
                    bms = cell_ms(bc)
                    rec.setdefault("b", {})[b] = bms
                    r = bms / rec["candidate_ms"] if PC.number(bms) and rec["candidate_ms"] else None
                    rec.setdefault("b_ratio", {})[b] = r
                    rec["flags"].append("%s:%s" % (b, verdict(r, "B-FASTER", "B-SLOWER", "B-SAME") or "B-NOT-RUN"))
                out.append(rec)
    drivers = []
    path = os.path.join(results, "drivers", "drivers.jsonl")
    if os.path.isfile(path):
        with open(path) as fh:
            drivers = [json.loads(x) for x in fh if x.strip()]
    counts = {}
    for r in out:
        for f in r["flags"]:
            f = f.split(":")[-1]
            counts[f] = counts.get(f, 0) + 1
    return {"schema": REPORT_SCHEMA, "vendor": vendor, "rows": out, "drivers": drivers, "flag_counts": counts,
            "not_planned": missing}


def render_summary(rep, top=10):
    out = ["OWED-REPORT vendor=%s cells=%d drivers=%d" % (rep["vendor"], len(rep["rows"]), len(rep["drivers"])),
           "flags: " + " ".join("%s=%d" % kv for kv in sorted(rep["flag_counts"].items()))]
    worse = sorted([r for r in rep["rows"] if "REGRESSION" in r["flags"]], key=lambda r: -r["ratio_vs_prior"])
    if worse:
        out.append("slower than the prior:")
        out += ["  %.2fx %s [%s] (%s -> %s ms)" % (r["ratio_vs_prior"], r["key"], r["row"], PC.fmt_ms(r["prior_ms"]),
                                                   PC.fmt_ms(r["candidate_ms"])) for r in worse[:top]]
    ab = [r for r in rep["rows"] if any(str(f).endswith("B-FASTER") for f in r["flags"])]
    if ab:
        out.append("B arm faster than the candidate (%d): %s" % (len(ab), ", ".join(
            "%s %s" % (r["key"], [f for f in r["flags"] if f.endswith("B-FASTER")]) for r in ab[:6])))
    for d in rep["drivers"]:
        out.append("  driver %s %s %s ms=%s status=%s" % (d.get("row"), d.get("arm"), d.get("case"),
                                                         PC.fmt_ms(d.get("ms")), d.get("status")))
    nr = [r["key"] for r in rep["rows"] if "NOT-RUN" in r["flags"]]
    if nr:
        out.append("NOT-RUN (%d): %s%s" % (len(nr), ", ".join(nr[:6]), " ..." if len(nr) > 6 else ""))
    if rep["not_planned"]:
        out.append("not planned (%d): %s" % (len(rep["not_planned"]), ", ".join(sorted(rep["not_planned"])[:6])))
    return "\n".join(out)


def render_md(rep):
    lines = ["# Owed measurements on %s" % rep["vendor"], "",
             "Candidate against the published prior (board rows) and each B wheel set (A/B rows); ratio = "
             "candidate / prior, B ratio = B / candidate. REGRESSION above %s, FASTER below %s."
             % (PC.SLOWER, PC.FASTER), "",
             "| row | cell | candidate ms | prior ms | ratio | B arms (ms, B/A) | flags | note |",
             "|---|---|---:|---:|---:|---|---|---|"]
    for r in rep["rows"]:
        b = ", ".join("%s %s (%s)" % (k, PC.fmt_ms(v), PC.fmt_ratio((r.get("b_ratio") or {}).get(k)))
                      for k, v in sorted((r.get("b") or {}).items())) or "-"
        note = (r.get("why") or "") + (" prior from the paper compare" if r.get("prior_from") else "")
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["row"], r["key"], PC.fmt_ms(r.get("candidate_ms")), PC.fmt_ms(r.get("prior_ms")),
            PC.fmt_ratio(r.get("ratio_vs_prior")), b, " ".join(r["flags"]), note.replace("|", "/")[:200]))
    if rep["drivers"]:
        lines += ["", "| driver row | arm | case | ms | status |", "|---|---|---|---:|---|"]
        for d in rep["drivers"]:
            lines.append("| %s | %s | %s | %s | %s |" % (d.get("row"), d.get("arm"), d.get("case"),
                                                         PC.fmt_ms(d.get("ms")), str(d.get("status"))[:120]))
    lines.append("")
    return "\n".join(lines)


def cmd_report(a):
    doc = PC.load_json(a.owed)
    prior_rows = PC.load_prior(a.prior_compare)[0] if a.prior_compare else None
    rep = report(doc, a.plan_dir, a.results, prior_rows, a.vendor)
    md = a.out_md or os.path.join(a.results, "OWED-%s.md" % a.vendor)
    js = a.out_json or os.path.join(a.results, "owed-%s.json" % a.vendor)
    sm = a.summary or os.path.join(a.results, "OWED-SUMMARY-%s.txt" % a.vendor)
    text = render_summary(rep)
    for path, body in ((md, render_md(rep)), (js, json.dumps(rep, indent=1, sort_keys=True) + "\n"), (sm, text + "\n")):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(body)
    print(text)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="measure_owed", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="(box) the owed rows' race ids per arm and rows group")
    pl.add_argument("--owed", required=True)
    pl.add_argument("--paper-recipes", required=True)
    pl.add_argument("--planner", default=os.path.join(HERE, "bench_board.py"))
    pl.add_argument("--vendor", required=True, choices=PC.VENDORS)
    pl.add_argument("--b-arms", default="", help="comma list of the B wheel sets staged on this box")
    pl.add_argument("--out-dir", required=True)
    r = sub.add_parser("report", help="the owed rows' numbers from the job's boards and drivers")
    r.add_argument("--owed", required=True)
    r.add_argument("--plan-dir", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--prior-compare", default=None, help="the 0.8.66 compare JSON (prior numbers of paper races)")
    r.add_argument("--vendor", required=True, choices=PC.VENDORS)
    r.add_argument("--out-md", default=None)
    r.add_argument("--out-json", default=None)
    r.add_argument("--summary", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"plan": cmd_plan, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
