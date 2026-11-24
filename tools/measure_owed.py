#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/measure_owed.py -- the measurements tonight's lanes left owed (bench/results/paper_compare/owed-pre0867.json),
planned through the board and reported beside the paper compare (lane measure-prep, 2026-11-09).

  plan   --owed JSON --paper-recipes JSON --planner tools/bench_board.py --vendor V --out-dir DIR [--b-arms a,b]
         [--prior]
        ON THE BOX: resolves every board / ab row to the race ids the current board plans (bench_board.plan_races),
        per ARM (candidate; prior = a last-release ours-only venv, only with --prior; each B wheel set named in
        --b-arms) and per ROWS group ('full', or a neighbouring row count: its own bench_board --rows process).
        ONLY RACES A REPORT READS ARE PLANNED (lane paper-plan-trim, 2026-11-16: the L40S job of d95f805cce timed 197
        candidate races for a compare that read 61, and planned 0.8.55 on 159 races it lacks 98 of): a board row
        (candidate against the last release) is planned only with --prior, i.e. when the job times a last release
        beside the paper baseline (PC_PRIOR_VERSION other than 0.8.55; Andrew, 2026-11-16: a measurement job times
        ONE prior, 0.8.55, so by default none is); an ab row only when at least one of its B sets is staged (the
        candidate alone is read by nobody); a driver row only when at least two of its arms run here. Every row left
        out is named in not-planned.json. Writes into DIR:
          board-args.txt        the union of the paper's and the owed rows' families, lanes and datasets (one arg
                                per line), so ONE board process can time the paper races and the owed races together
          units.tsv             arm <TAB> rows <TAB> ids-file, one line per board process the job runs, in order
          ids-<arm>-<rows>.txt  the race ids of that process
          smoke-ids.txt         the smoke ids (rows=<smoke rows>, shape=small) of the owed full-row races the paper
                                smoke does not already cover; the job smokes them with the paper's
          smoke-ids-prior.txt   the same for the prior arm's owed full rows (empty without --prior): the prior's
                                smoke covers only the races it is compared on
          drivers.json          the driver rows (tools/measure_owed_driver.py), with the arms present on this box
          not-planned.json      {row id or race: why}: a lane the board no longer has, a B set not staged, ...
          rowmap.json           {row id: [race ids]} for the report
        A race id the paper recipes hold is not planned again for the candidate or the prior arm: the paper board
        times it for the candidate, and the prior release's on-box board (board-prior-<version>/) for the prior.
        The job times the prior's full-row units in that same board (no second timing of the prior, 2026-11-14).
  plan-check --plan-dir DIR --b-arms a,b --vendor V [--prior]
        whether DIR's plan was made for these B sets, this vendor and this prior choice (plan.json, written by plan): prints
        OWED-PLAN-CURRENT (rc 0) or OWED-PLAN-STALE <why> (rc 1). The job re-makes a stale plan before its owed
        phase: job 0003 of ba08dac006 (L40S, 2026-11-13) ran without the plan part, reused the plan job 0001 made
        without PC_B_ARMS (b_arms=-), and PC_OWED_ARMS=candidate,<b names> ran no B arm though all five were installed.
  data-keys --owed JSON --paper-recipes JSON --planner tools/bench_board.py --vendor V --data-root DIR
            [--b-arms a,b] [--prior]
        the R2 dataset keys (bench/results/dataset_store/manifest.tsv) the paper and the PLANNED owed races read, one
        `<key> TAB <path>` line each, from the board's own data_needs (data_keys): what the stage step stages
        (tools/ops/measure_candidate.py) and what the job checks after its plan.
  report --owed JSON --plan-dir DIR --results RES --prior-version V --vendor V [--out-md --out-json --summary]
        per owed row and race cell: candidate ms, prior ms (the prior release's on-box board board-prior-<version>/
        for full rows, board-prior-<rows>/ for a neighbouring shape: the same job on the same box; a board of another
        host refuses, one without a host probe is NOT-COMPARED),
        each B arm's ms, the ratios, and the row's flags: REGRESSION / FASTER / SAME vs the prior, B-FASTER / B-SLOWER
        / B-SAME for an A/B, NOT-RUN with its reason, and the driver rows from drivers.jsonl. Prints one screen.

IN-JOB BASE CAP (lane fx-same-pod-prior, 2026-11-19; it replaces the STORED BASE of 2026-11-17, which capped a B arm
at a base time stored from another host: on host bd9c355c19c0 the DBSCAN brute base itself takes 1,170 s against the
stored 994 s, so a B arm faster than base there could hit the cap and read "not faster"; fx-board-dbscan-RESULT.md).
An ab row with "cap_base": true has its base (the candidate) and its B arms timed back to back in the same job, like
every ab row; each B race's rounds are then capped at the time the base took for that race IN THAT JOB (bench_board
--race-round-seconds from `base-caps` over the base board just written): a run that reaches the cap stops and is
reported NOT-FASTER-THAN-BASE. plan writes, beside ids-<arm>-<rows>.txt, ids-<arm>-<rows>.cap.txt (the races of the
unit whose B rounds are capped). A neighbouring shape is the row's own "rows" (a row count): it needs no other option.
  base-caps --board BASE_BOARD.json --ids FILE --cap-ids FILE --out JSON
        ON THE BOX: {"round_seconds": {race: s}} for the races of --ids listed in --cap-ids whose base cell in
        BASE_BOARD (this job's board of the candidate) is timed: the slowest of its timed cells, whole seconds up.

SHARED BASE (lane ab-harness-speed, 2026-11-24). nvc-ab-20261124T070727Z-69dac2 and ...T095727Z-92fd9f timed the
candidate base once PER ARM: 165 base races for 122 distinct ones, 852 s of repeated base wall and a base board process
per unit. plan now orders the B units so arms on the same races run next to each other (overlap_order), and before
each base chunk the job runs
  base-share --target BASE_DIR --ids CHUNK --source DIR... [--max-age-s S] --out TODO
        ON THE BOX: every race of CHUNK some base board of this job (another unit's base of the same rows; for full
        rows also the listed rows' candidate board) holds as done, finished at most S seconds ago (BASE_MAX_AGE_S,
        PC_AB_BASE_MAX_AGE_S), is copied into BASE_DIR's board with a "shared_base" record (board, finished, age); the
        rest go to TODO and are timed now, so a base older than the bound is timed again (the drift guard). The A/B
        summary (tools/ops/targeted_rows.py) prints for every pair the base it used and the gap from that base's end
        to the arm's start.

Our arms only, GPU only, one warm-up and one timed run per arm (the board's --rounds 1). Tests:
tools/test_measure_owed.py.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

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


def row_arms(row, b_present, prior_present=False):
    """(the board arms of an owed row, the arms it needs that this job does not run). A board row races the candidate
    and the last-release prior: none without that prior (PRIOR_PRESENT), since its report compares the two. An ab row
    races the candidate and every B set it names that is staged on this box: none when no B set is (the report reads
    the candidate cell only for a B ratio). Remove what the change made unnecessary (2026-11-16): the candidate-only
    units of rows nobody compares are no longer timed."""
    if row["kind"] == "board":
        return (["candidate", "prior"], []) if prior_present else ([], ["prior"])
    if row["kind"] == "ab":
        want = as_list(row.get("b_arms") or [])
        have = [b for b in want if b in b_present]
        return (["candidate"] + have if have else []), [b for b in want if b not in b_present]
    return [], []


def driver_arms(row, b_present, prior_present=False):
    """The arms of a driver row that run on this box (candidate[:variant], prior with PRIOR_PRESENT, staged B sets)."""
    present = {"candidate"} | ({"prior"} if prior_present else set()) | set(b_present)
    return [a for a in as_list(row.get("arms") or []) if str(a).split(":")[0] in present]


def race_planner(planner, vendor):
    """(lane -> family, planned(lane_list, rows, shape)): the board's races for some lanes, cached."""
    fam = family_of(planner)
    modes = planner.modes_for(vendor, "identical")
    cache = {}

    def planned(lane_list, rows, shape="full"):
        fams = tuple(sorted({fam[l] for l in lane_list}))
        key = (fams, tuple(lane_list), rows, shape)
        if key not in cache:
            races = planner.plan_races(vendor, modes, fams, tuple(lane_list), tuple(planner.DATASETS), rows, shape)
            cache[key] = planner.gpu_opponents_only(races, vendor)
        return cache[key]
    return fam, planned


def row_races(row, fam, planned):
    """(lanes the board still has, lanes it lost, rows (None = full), the races) of one board / ab row."""
    want_lanes = as_list(row["lane"])
    gone = [l for l in want_lanes if l not in fam]
    lane_list = [l for l in want_lanes if l in fam]
    rows = None if row.get("rows") in (None, "full") else int(row["rows"])
    ds = row.get("dataset") or "*"
    races = [r for r in planned(lane_list, rows) if ds == "*" or r["dataset"] == ds] if lane_list else []
    return lane_list, gone, rows, races


def data_keys(doc, paper, planner, vendor, data_root, b_present=(), prior_present=False):
    """[(R2 key, path the board reads it from)] for every input the job's races read: the paper recipes and the owed
    board / ab rows the plan times (row_arms: B_PRESENT, PRIOR_PRESENT, the same rule as plan), through the board's
    own data_needs (bench_board.py) and the algos corpus keys. The stage step
    (tools/ops/measure_candidate.py) stages these before a submit and the job checks them after its plan (2026-11-12:
    a hand-written list left out higgs and half the smoke shards refused)."""
    fam, planned = race_planner(planner, vendor)
    races = [r for r in (paper.get("recipes") or []) if r.get("family") and r.get("lane") and r.get("dataset")]
    for row in doc.get("rows") or []:
        if row.get("kind") in ("board", "ab") and row_arms(row, b_present, prior_present)[0]:
            races += row_races(row, fam, planned)[3]
    needed, _ = planner.data_needs(races)
    out = [(planner.R2_KEYS[n], os.path.join(data_root, planner.DATA_FILES[n])) for n in needed if n in planner.R2_KEYS]
    corpus = sorted({k for r in races if r["family"] == "algos"
                     for k in planner.ALGOS.r2_keys(r["lane"], r["dataset"]) if k.startswith("corpus/")})
    return out + [(k, planner.ALGOS.corpus_path(k)) for k in corpus]


def plan(doc, paper, planner, vendor, b_present, prior_present=False):
    """(units {(arm, rows): [race ids]}, smoke ids {"candidate": [...], "prior": [...]}, drivers, not_planned, rowmap,
    board args). Only the rows whose compared arms run in this job are planned (row_arms, driver_arms)."""
    if doc.get("schema") != SCHEMA:
        raise SystemExit("measure_owed: not a %s document" % SCHEMA)
    fam, planned = race_planner(planner, vendor)
    paper_ids = set(paper.get("race_ids") or [])
    pb = paper.get("bench_board_args") or {}
    families, lanes, datasets = list(pb.get("families") or []), list(pb.get("lanes") or []), list(pb.get("datasets") or [])
    units, smoke, drivers, missing, rowmap = {}, {"candidate": set(), "prior": set()}, [], {}, {}
    smoke_rows = int(getattr(planner, "SMOKE_ROWS", 2000))

    for row in doc["rows"]:
        rid = row["id"]
        if row["kind"] == "driver":
            have = driver_arms(row, b_present, prior_present)
            if len(have) >= 2:
                drivers.append(row)
            else:
                missing[rid] = "driver row: needs two of its arms %s; this job runs only %s" % (
                    ",".join(as_list(row.get("arms") or [])), ",".join(have) or "none")
            continue
        if row["kind"] not in ("board", "ab"):
            missing[rid] = "unknown kind %r" % row["kind"]
            continue
        arms, absent = row_arms(row, b_present, prior_present)
        for b in absent:
            missing["%s %s" % (rid, b)] = (
                "B wheel set %s is not staged on this box (PC_B_ARMS)" % b if b != "prior" else
                "the board row compares the candidate with the last release; this job times no last release "
                "(PC_PRIOR_VERSION is the paper baseline)")
        if not arms:
            continue
        lane_list, gone, rows, races = row_races(row, fam, planned)
        for l in gone:
            missing["%s %s" % (rid, l)] = "lane %s is no longer a board lane" % l
        if not lane_list:
            continue
        ds = row.get("dataset") or "*"
        if not races:
            missing[rid] = "the board plans no race for %s on %s (dataset %s, rows %s)" % (
                ",".join(lane_list), vendor, ds, rows_tag(rows))
            continue
        ids = sorted({r["id"] for r in races})
        rowmap[rid] = ids
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
                    smoke["candidate"].add(sid)
                    if "prior" in arms:
                        smoke["prior"].add(sid)
                else:
                    missing["%s %s" % (rid, i)] = "the board plans no smoke race %s" % sid
    bargs = ["--vendor", vendor, "--families", ",".join(families), "--lanes", ",".join(lanes),
             "--modes", "identical", "--gpu-opponents-only"]
    if datasets:
        bargs += ["--datasets", ",".join(datasets)]
    return ({k: sorted(v) for k, v in units.items()}, {k: sorted(v) for k, v in smoke.items()}, drivers, missing,
            rowmap, bargs)


def unit_caps(doc, rowmap, units):
    """{(B arm, rows tag): [race id]} of the planned B units whose rounds are capped at this job's base time (an ab row
    with "cap_base": true; IN-JOB BASE CAP above)."""
    out = {}
    for row in doc.get("rows") or []:
        if not row.get("cap_base") or row.get("kind") != "ab" or row["id"] not in rowmap:
            continue
        rt = rows_tag(row.get("rows"))
        for b in as_list(row.get("b_arms") or []):
            if (b, rt) in units:
                out.setdefault((b, rt), set()).update(i for i in rowmap[row["id"]] if i in units[(b, rt)])
    return {k: sorted(v) for k, v in out.items() if v}


def base_caps(board, ids, cap_ids):
    """{race: s}: the per-round cap of each race of IDS in CAP_IDS whose base cell in BOARD (this job's candidate
    board) is timed: the slowest of its timed cells, whole seconds up. A race whose base has no time gets no cap
    (the family's own ceiling applies)."""
    races = (board or {}).get("races") or {}
    want = set(cap_ids)
    out = {}
    for rid in ids:
        if rid not in want:
            continue
        ms = [m for m in (cell_ms(c) for c in our_cells(races.get(rid) or {}).values()) if PC.number(m)]
        if ms:
            out[rid] = int(-(-max(ms) // 1000))
    return out


#: SHARED BASE (lane ab-harness-speed, 2026-11-24): the default drift bound, seconds from the shared base race's end to
#: the start of the chunk that takes it. Today's back-to-back unit puts a base race and its arm race up to two chunks
#: apart (the base chunk's later races, then the arm chunk's earlier ones: PC_TG_CHUNK_S 600 s of base time each), so
#: 1200 s keeps every arm no further from its base than a back-to-back unit already allowed. paper_compare_job.sh
#: passes PC_AB_BASE_MAX_AGE_S.
BASE_MAX_AGE_S = 1200


def _iso_s(text):
    """Epoch seconds of a board's ISO UTC stamp ('2026-11-24T11:27:02Z'), or None."""
    import datetime
    try:
        return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def share_base(target, ids, sources, now_s, max_age_s=BASE_MAX_AGE_S):
    """(board, shared {race: provenance}, todo [race]) for one base chunk of an owed A/B unit.

    TARGET is the unit's base board (a document or None), SOURCES [(board dir, board document)] the base boards this
    job already wrote on this box (the other units' bases of the same rows, and the listed rows' candidate board for
    full rows: the same candidate venv and board arguments). A race of IDS the target already holds as done is left
    alone; a race some source holds as done, finished at most MAX_AGE_S before NOW_S, is copied into the target with a
    "shared_base" record (its board, finish time and age), the freshest when several hold it; every other race is in
    TODO, to be timed now. The target board is created from the freshest source's document when it does not exist
    (same box, same candidate bytes: bench_board resumes it)."""
    races = dict(((target or {}).get("races") or {}))
    shared, todo = {}, []
    for rid in ids:
        if (races.get(rid) or {}).get("status") == "done":
            continue
        best = None
        for bdir, doc in sources:
            rec = ((doc or {}).get("races") or {}).get(rid)
            if not rec or rec.get("status") != "done" or rec.get("shared_base"):
                continue      # a shared copy is never shared again: the timed original is the source
            fin = _iso_s(rec.get("finished"))
            if fin is None or now_s - fin > max_age_s or now_s < fin:
                continue
            if best is None or fin > best[0]:
                best = (fin, bdir, doc, rec)
        if best is None:
            todo.append(rid)
            continue
        fin, bdir, doc, rec = best
        prov = {"board": bdir, "finished": rec.get("finished"), "age_s": int(now_s - fin), "max_age_s": max_age_s}
        races[rid] = dict(json.loads(json.dumps(rec)), shared_base=prov)
        shared[rid] = prov
    if not shared:
        return target, shared, todo
    if target is None:
        # the board document (box, config, boot ids) of the source of the freshest shared race; its own run lists
        # (plan, preflight, selections) stay with it
        fresh = max(shared.values(), key=lambda p: _iso_s(p["finished"]) or 0)["board"]
        src = next(doc for d, doc in sources if d == fresh)
        target = {k: v for k, v in src.items() if k not in ("races", "plan", "preflight", "superseded",
                                                             "measurement_selection", "skipped_recipes",
                                                             "failed_preflight_recipes")}
        target["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_s))
    target = dict(target, races=races)
    return target, shared, todo


def cmd_base_share(a):
    read = lambda p: [x.strip() for x in open(p) if x.strip()] if p and os.path.isfile(p) else []  # noqa: E731
    ids = read(a.ids)
    tpath = os.path.join(a.target, "board.json")
    target = load_board(tpath)
    if target is not None and "placeholder" in target:
        target = None
    sources = []
    for d in a.source:
        d = d.rstrip("/")
        if os.path.abspath(d) == os.path.abspath(a.target):
            continue
        doc = load_board(os.path.join(d, "board.json"))
        if doc is not None and "placeholder" not in doc:
            sources.append((d, doc))
    now_s = time.time() if a.now is None else a.now
    board, shared, todo = share_base(target, ids, sources, now_s, a.max_age_s)
    if shared:
        os.makedirs(a.target, exist_ok=True)
        PC.write_atomic(tpath, PC.json_text(board))
    PC.write_atomic(a.out, "".join(i + "\n" for i in todo))
    srcs = sorted({os.path.basename(p["board"]) for p in shared.values()})
    print("OWED-BASE-SHARE races=%d shared=%d to_time=%d max_age_s=%d%s" % (
        len(ids), len(shared), len(todo), a.max_age_s, (" from " + ",".join(srcs)) if srcs else ""))
    return 0


def overlap_order(units):
    """The B units of UNITS ({(arm, rows): [ids]}) in run order: candidate and prior units first (unit_order), then
    each next B unit the one sharing the most races with the unit just placed (same rows), ties by unit_order, so
    arms timed on the same races run next to each other and share one base (SHARED BASE)."""
    keys = sorted(units, key=unit_order)
    head = [k for k in keys if k[0] in ARMS_FIXED]
    rest = [k for k in keys if k[0] not in ARMS_FIXED]
    out = list(head)
    while rest:
        nxt = rest[0]
        if out and out[-1][0] not in ARMS_FIXED:
            last = out[-1]
            prev = set(units[last])
            best = 0
            for k in rest:      # the first unit (unit_order) with the largest overlap; none: the next in unit_order
                n = len(prev & set(units[k])) if k[1] == last[1] else 0
                if n > best:
                    best, nxt = n, k
        out.append(nxt)
        rest.remove(nxt)
    return out


def cmd_base_caps(a):
    read = lambda p: [x.strip() for x in open(p) if x.strip()] if p and os.path.isfile(p) else []  # noqa: E731
    caps = base_caps(load_board(a.board), read(a.ids), read(a.cap_ids))
    PC.write_atomic(a.out, PC.json_text({"round_seconds": caps, "base_board": a.board}))
    print("OWED-BASE-CAPS races=%d from %s" % (len(caps), a.board))
    return 0


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
    units, smoke, drivers, missing, rowmap, bargs = plan(doc, paper, planner, a.vendor, b_present, a.prior)
    os.makedirs(a.out_dir, exist_ok=True)
    lines = []
    for key in overlap_order(units):     # arms on the same races next to each other: they share one base
        arm, rows = key
        f = os.path.join(a.out_dir, "ids-%s-%s.txt" % (arm, rows))
        PC.write_atomic(f, "".join(i + "\n" for i in units[key]))
        lines.append("%s\t%s\t%s\n" % (arm, rows, f))
    for f in os.listdir(a.out_dir):     # a re-made plan keeps no stale cap list (or pre-2026-11-19 stored base)
        if f.startswith("ids-") and f.endswith((".cap.txt", ".base.json", ".stored.txt")):
            os.remove(os.path.join(a.out_dir, f))
    for (arm, rows), ids in unit_caps(doc, rowmap, units).items():
        PC.write_atomic(os.path.join(a.out_dir, "ids-%s-%s.cap.txt" % (arm, rows)), "".join(i + "\n" for i in ids))
    PC.write_atomic(os.path.join(a.out_dir, "units.tsv"), "".join(lines))
    PC.write_atomic(os.path.join(a.out_dir, "smoke-ids.txt"), "".join(i + "\n" for i in smoke["candidate"]))
    PC.write_atomic(os.path.join(a.out_dir, "smoke-ids-prior.txt"), "".join(i + "\n" for i in smoke["prior"]))
    PC.write_atomic(os.path.join(a.out_dir, "board-args.txt"), "".join(x + "\n" for x in bargs))
    for name, obj in (("drivers.json", drivers), ("not-planned.json", missing), ("rowmap.json", rowmap)):
        PC.write_atomic(os.path.join(a.out_dir, name), PC.json_text(obj))
    # written last: plan-check reads it as "this plan is complete for these B sets"
    PC.write_atomic(os.path.join(a.out_dir, PLAN_META),
                    PC.json_text({"vendor": a.vendor, "b_arms": b_present, "prior": bool(a.prior)}))
    print("OWED-PLAN vendor=%s rows=%d units=%d races=%d smoke=%d drivers=%d not_planned=%d b_arms=%s prior=%s"
          % (a.vendor, len(doc["rows"]), len(units), sum(len(v) for v in units.values()), len(smoke["candidate"]),
             len(drivers), len(missing), ",".join(b_present) or "-", "yes" if a.prior else "no"))
    return 0


#: what a plan was made for (cmd_plan writes it; plan_stale reads it)
PLAN_META = "plan.json"


def plan_stale(plan_dir, b_arms, vendor, prior=False):
    """None when PLAN_DIR's plan was made for these B sets (a comma list, order free), VENDOR and PRIOR (a last-release
    prior arm); else why not. A plan made before 2026-11-16 records no "prior" and planned the prior arm."""
    want = sorted(b for b in (b_arms or "").split(",") if b)
    if not os.path.isfile(os.path.join(plan_dir, "units.tsv")):
        return "no plan (units.tsv missing)"
    try:
        with open(os.path.join(plan_dir, PLAN_META)) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        if not want and prior:
            return None     # a plan from before plan.json was made without B sets and with the prior arm
        if not want:
            return "the plan has no %s (made before 2026-11-13, with the prior arm); this job has prior=no" % PLAN_META
        return "the plan records no B sets (no %s: made before 2026-11-13); this job has b_arms=%s" % (
            PLAN_META, ",".join(want))
    have = sorted(meta.get("b_arms") or [])
    if meta.get("vendor") != vendor:
        return "the plan is for vendor %s, this job is %s" % (meta.get("vendor"), vendor)
    if have != want:
        return "the plan has b_arms=%s, this job has b_arms=%s" % (",".join(have) or "-", ",".join(want) or "-")
    if bool(meta.get("prior", True)) != bool(prior):
        return "the plan has prior=%s, this job has prior=%s" % (
            "yes" if meta.get("prior", True) else "no", "yes" if prior else "no")
    return None


def cmd_plan_check(a):
    why = plan_stale(a.plan_dir, a.b_arms, a.vendor, a.prior)
    print("OWED-PLAN-STALE " + why if why else "OWED-PLAN-CURRENT b_arms=%s" % (a.b_arms or "-"))
    return 1 if why else 0


def cmd_data_keys(a):
    """One `<key> TAB <path>` line per input (tools/ops/paper_compare_job.sh checks each path after its plan)."""
    doc = PC.load_json(a.owed) if a.owed else {"rows": []}
    keys = data_keys(doc, PC.load_json(a.paper_recipes), load_planner(a.planner), a.vendor, a.data_root,
                     [b for b in (a.b_arms or "").split(",") if b], a.prior)
    sys.stdout.write("".join("%s\t%s\n" % kp for kp in keys))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------

def board_dir(results, arm, rows, prior_version=None):
    """Where the job writes the board of one unit (tools/ops/paper_compare_job.sh uses the same names). The prior
    arm's full rows are timed in the prior release's own on-box board (board-prior-<version>/, the paper compare's
    prior column): no second timing of the prior (2026-11-14)."""
    if arm == "candidate" and rows == "full":
        return os.path.join(results, "board")
    if arm == "prior" and rows == "full" and prior_version:
        return os.path.join(results, "board-prior-%s" % prior_version)
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
    """The board document, or None (not run)."""
    try:
        return PC.load_json(path)
    except (OSError, ValueError):
        return None


def verdict(ratio, low_flag, high_flag, same_flag):
    if ratio is None:
        return None
    return high_flag if ratio > PC.SLOWER else low_flag if ratio < PC.FASTER else same_flag


def report(doc, plan_dir, results, vendor, prior_version=None):
    """The owed rows' cells. Every ratio compares two boards of this job that carry the same host id (the box's host
    probe, AGENTS.md "Times are compared only on the same box"): a board of another host refuses (SystemExit), and a
    board without a probe gives NOT-COMPARED, never a flag. The prior's numbers come only from its on-box board."""
    rowmap = PC.load_json(os.path.join(plan_dir, "rowmap.json"))
    missing = PC.load_json(os.path.join(plan_dir, "not-planned.json"))
    docs, hosts = {}, {}

    def board(arm, rows):
        k = (arm, rows)
        if k not in docs:
            docs[k] = load_board(os.path.join(board_dir(results, arm, rows, prior_version), "board.json")) or {}
        return docs[k].get("races") or {}

    def verified(arm, rows):
        """Whether ARM's board and the candidate's (same rows) are one box; refuses two hosts."""
        k = (arm, rows)
        if k not in hosts:
            board("candidate", rows), board(arm, rows)
            if not docs[k].get("races"):
                hosts[k] = False      # not run: no cell, NOT-RUN on that side
            else:
                h, _ = PC.same_box([docs[("candidate", rows)]], [docs[k]], "the %s %s board" % (arm, rows))
                hosts[k] = h is not None
        return hosts[k]

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
                    pms = cell_ms(our_cells(prior_race).get((phase, arm, batch)))
                    rec["prior_ms"] = pms
                    r = rec["candidate_ms"] / pms if PC.number(rec["candidate_ms"]) and pms else None
                    rec["ratio_vs_prior"] = r
                    if r is not None and not verified("prior", rows):
                        rec["flags"].append("NOT-COMPARED")     # no host probe on one side: shown, no verdict
                    else:
                        rec["flags"].append(verdict(r, "FASTER", "REGRESSION", "SAME") or "NOT-COMPARED")
                for b in b_arms:
                    bc = our_cells(board(b, rows).get(race_id) or {}).get((phase, arm, batch))
                    bms = cell_ms(bc)
                    rec.setdefault("b", {})[b] = bms
                    r = bms / rec["candidate_ms"] if PC.number(bms) and rec["candidate_ms"] else None
                    rec.setdefault("b_ratio", {})[b] = r
                    v = verdict(r, "B-FASTER", "B-SLOWER", "B-SAME") if r is None or verified(b, rows) else \
                        "B-NOT-COMPARED"
                    if r is None and row.get("cap_base") and capped(bc) and PC.number(rec["candidate_ms"]):
                        v = "B-NOT-FASTER-THAN-BASE"     # stopped at this job's base time (IN-JOB BASE CAP)
                    rec["flags"].append("%s:%s" % (b, v or "B-NOT-RUN"))
                out.append(rec)
    drivers = []
    path = os.path.join(results, "drivers", "drivers.jsonl")
    if os.path.isfile(path):
        # a line cut short by a full disk is reported as UNREADABLE, never a crash of the whole report (2026-11-14:
        # the owed report died on `JSONDecodeError: line 1 column 87` after the L40S pod filled)
        with open(path, errors="replace") as fh:
            for n, x in enumerate(fh, 1):
                if not x.strip():
                    continue
                try:
                    drivers.append(json.loads(x))
                except ValueError as exc:
                    drivers.append({"status": "UNREADABLE", "line": n, "error": str(exc)[:200],
                                    "file": os.path.relpath(path, results)})
    counts = {}
    for r in out:
        for f in r["flags"]:
            f = f.split(":")[-1]
            counts[f] = counts.get(f, 0) + 1
    return {"schema": REPORT_SCHEMA, "vendor": vendor, "rows": out, "drivers": drivers, "flag_counts": counts,
            "not_planned": missing, "prior_version": prior_version,
            "host_id": PC.candidate_host([d for (a, _), d in docs.items() if a == "candidate" and d])[0]}


def capped(cell):
    """Whether a B cell stopped at its round cap (the classical racer's per-round timeout)."""
    return bool(cell) and "timeout" in str(cell.get("status") or "")


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
    nf = [r["key"] for r in rep["rows"] if any(str(f).endswith("B-NOT-FASTER-THAN-BASE") for f in r["flags"])]
    if nf:
        out.append("B arm stopped at this job's base time, not faster than base (%d): %s" % (len(nf), ", ".join(nf[:6])))
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
        note = r.get("why") or ""
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
    rep = report(doc, a.plan_dir, a.results, a.vendor, a.prior_version)
    md = a.out_md or os.path.join(a.results, "OWED-%s.md" % a.vendor)
    js = a.out_json or os.path.join(a.results, "owed-%s.json" % a.vendor)
    sm = a.summary or os.path.join(a.results, "OWED-SUMMARY-%s.txt" % a.vendor)
    text = render_summary(rep)
    for path, body in ((md, render_md(rep)), (js, json.dumps(rep, indent=1, sort_keys=True) + "\n"), (sm, text + "\n")):
        PC.write_atomic(path, body)
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
    pl.add_argument("--prior", action="store_true",
                    help="the job times a last release beside the paper baseline (PC_PRIOR_VERSION is not 0.8.55): "
                         "the owed board rows and the prior arm are planned; without it they are not")
    pl.add_argument("--out-dir", required=True)
    c = sub.add_parser("plan-check", help="whether a plan dir was made for these B sets and vendor")
    c.add_argument("--plan-dir", required=True)
    c.add_argument("--b-arms", default="")
    c.add_argument("--vendor", required=True, choices=PC.VENDORS)
    c.add_argument("--prior", action="store_true", help="the job times a last-release prior arm (see plan --prior)")
    d = sub.add_parser("data-keys", help="the R2 dataset keys the paper and owed races read, with their paths")
    d.add_argument("--owed", default=None, help="the owed file (omitted: the paper races only)")
    d.add_argument("--paper-recipes", required=True)
    d.add_argument("--planner", default=os.path.join(HERE, "bench_board.py"))
    d.add_argument("--vendor", required=True, choices=PC.VENDORS)
    d.add_argument("--data-root", required=True, help="the board's --data-root (e.g. /root/datasets/gbm-bench)")
    d.add_argument("--b-arms", default="", help="comma list of the B wheel sets the job stages (plan --b-arms)")
    d.add_argument("--prior", action="store_true", help="the job times a last-release prior arm (plan --prior)")
    bc = sub.add_parser("base-caps", help="(box) the B rounds' caps from this job's base board (IN-JOB BASE CAP)")
    bc.add_argument("--board", required=True, help="this job's board.json of the base (the candidate)")
    bc.add_argument("--ids", required=True, help="the races of the chunk about to run")
    bc.add_argument("--cap-ids", required=True, help="the unit's ids-<arm>-<rows>.cap.txt")
    bc.add_argument("--out", required=True)
    bs = sub.add_parser("base-share", help="(box) copy this job's fresh base races into a unit's base board "
                        "(SHARED BASE); write the races still to time")
    bs.add_argument("--target", required=True, help="the unit's base board dir (board-tg-ab-<arm>-base-<rows>)")
    bs.add_argument("--ids", required=True, help="the base chunk about to run")
    bs.add_argument("--source", action="append", default=[], help="a base board dir of this job (repeat it)")
    bs.add_argument("--max-age-s", type=int, default=BASE_MAX_AGE_S,
                    help="share a base race finished at most this many seconds ago (default %(default)s)")
    bs.add_argument("--now", type=float, default=None, help=argparse.SUPPRESS)
    bs.add_argument("--out", required=True, help="the races still to time (the chunk's .base file)")
    r = sub.add_parser("report", help="the owed rows' numbers from the job's boards and drivers")
    r.add_argument("--owed", required=True)
    r.add_argument("--plan-dir", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--prior-version", default=None,
                   help="the prior release (PC_PRIOR_VERSION): its full rows are read from board-prior-<version>/, "
                        "the on-box prior board the paper compare also reads")
    r.add_argument("--vendor", required=True, choices=PC.VENDORS)
    r.add_argument("--out-md", default=None)
    r.add_argument("--out-json", default=None)
    r.add_argument("--summary", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"plan": cmd_plan, "plan-check": cmd_plan_check, "data-keys": cmd_data_keys,
            "base-caps": cmd_base_caps, "base-share": cmd_base_share, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
