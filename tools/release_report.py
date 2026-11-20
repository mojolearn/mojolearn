#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE RELEASE STATUS REPORT: one authoritative row per release requirement and release step.

    pixi run release 0.8.66 --report [--report-json FILE] [--owner-waiver W.json]   (tools/release.py)
    python3 tools/release_report.py --version V --commit C --rows rows.json [--state state.json]
                                    [--waiver W.json ...] [--plan plan.json] [--json out] [--markdown out]
    python3 tools/release_report.py --version V --commit C --blocked-tsv blocked-rows.tsv --waiver W.json

Every row has exactly one STATE (docs/release/STATUS_REPORT.md):

  passed              executed at this freeze with these inputs, evidence verified
  reused              passed on older evidence the ledger proved still applies (same inputs, or the
                      admitted reference with an unchanged closure, or a retained receipt)
  failed              an execution failed (or contradicts the admitted reference): a bug to fix
  missing-execution   no evidence for these inputs: something must run (named by NEXT ACTION)
  blocked-provenance  evidence exists but cannot be linked to these inputs (provenance incomplete,
                      evidence to restore, the reference analysis unavailable)
  not-applicable      not required for publication (the lane's applicability excludes the column, a
                      non-gating campaign row, a step this release skips)
  waived              an owner waiver names a row that is not passed or reused; the row stays
                      `waived` with the waiver's reason and approver and its underlying state, and is
                      never counted as passed

and every row that is not passed or reused has a NEXT ACTION naming the single command to run.
The totals line: `REPORT <v> passed=.. reused=.. failed=.. missing=.. blocked=.. n/a=.. waived=..  ready=YES|NO`
(ready = zero failed, missing and blocked rows; waived rows are listed, never hidden).

Pure: the report reads evaluated ledger rows (tools/release_requirements.py evaluate, the same rows
the gate and --status read), the release state's step records, the waivers and the saved plan
(tools/release_plan.py). It runs, rents and publishes nothing.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "mojolearn.release-report.v1"
STATES = ("passed", "reused", "failed", "missing-execution", "blocked-provenance", "not-applicable", "waived")
#: totals-line key per state (the line is grep-stable: `REPORT <v> passed=..`)
TOTAL_KEYS = (("passed", "passed"), ("reused", "reused"), ("failed", "failed"), ("missing", "missing-execution"),
              ("blocked", "blocked-provenance"), ("n/a", "not-applicable"), ("waived", "waived"))
UNRESOLVED = ("failed", "missing-execution", "blocked-provenance")
ID_RE = re.compile(r"identity-(apple|nvidia|amd|cpu)/(.+)")
#: "closure changed since <c> (<class> <backend>)": the class a reference verdict judged (PTX lives in identity-nvidia)
CLASS_IN_REASON = re.compile(r"\((apple|nvidia|ptx|amd|cpu) [a-z ]+\)")
#: dependency decisions that mean "passed on older evidence"
REUSE_DECISIONS = ("retained", "reference-closure-unchanged")
#: ledger actions that mean the evidence cannot be linked to the inputs
PROVENANCE_ACTIONS = ("complete-provenance", "restore-evidence")
STEP_PASSED_REUSE = re.compile(r"\b(REUSE|reused|adopted|taken from|retained verdict)\b")
STEP_SKIPPED = re.compile(r"^\s*SKIPPED\b")
POST_STEPS = re.compile(r"^(published-|publish-|finish-line$|record$|boxes-down$)")


# ---------------------------------------------------------------- waivers
def _sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def load_waiver(path):
    """An owner waiver file ({prefixes, reason, approved_by}, optionally `waived` names)."""
    doc = json.loads(Path(path).read_text())
    return dict(file=str(path), sha256=_sha(path), prefixes=sorted(x for x in doc.get("prefixes") or () if x),
                names=sorted(x for x in doc.get("waived") or () if x), approved_by=doc.get("approved_by") or "unknown",
                reason=doc.get("reason") or "")


def state_waivers(state):
    """The waivers a release state logged (Release.waived, Release.gate_waiver): what was waived stays visible."""
    out = []
    for entry in (state or {}).get("owner_waivers") or ():
        w = dict(file=entry.get("file"), sha256=entry.get("sha256"), prefixes=[], names=sorted(entry.get("waived") or ()),
                 approved_by=entry.get("approved_by") or "unknown", reason=entry.get("reason") or "", at=entry.get("at"))
        if entry.get("file") and Path(entry["file"]).is_file():
            try:
                w["prefixes"] = load_waiver(entry["file"])["prefixes"]
            except (OSError, ValueError):
                pass
        out.append(w)
    return out


def waivers_from(paths=(), state=None):
    """Every waiver that applies: the files named (--owner-waiver, committed docs/release/<v>-*waiver*.json)
    and the ones the release state logged. Duplicates (same file and names) collapse."""
    out, seen = [], set()
    for w in [load_waiver(p) for p in paths if p] + state_waivers(state):
        key = (w.get("sha256") or w.get("file"), tuple(w["prefixes"]), tuple(w["names"]))
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def waiver_for(waivers, name):
    """(waiver, matching rule) naming NAME, else (None, None)."""
    for w in waivers or ():
        if name in w["names"]:
            return w, name
        for p in w["prefixes"]:
            if name.startswith(p):
                return w, p
    return None, None


# ---------------------------------------------------------------- rows
def ledger_state(name, row, commit=None):
    """(state, why) of one evaluated ledger row, before waivers."""
    status, action = row.get("status"), row.get("action")
    decision = row.get("dependency_decision")
    reasons = list(row.get("reasons") or ())
    why = "; ".join(reasons)
    if status == "passed":
        tested = row.get("tested_revision")
        older = bool(tested and (commit or row.get("source_revision")) and tested != (commit or row.get("source_revision")))
        if decision in REUSE_DECISIONS or older:
            return "reused", why or "passed on retained evidence"
        return "passed", why or "passed at this freeze"
    if status == "failed" or action == "repair":
        return "failed", why or "the recorded attempt failed"
    if row.get("required") is False:
        return "not-applicable", why or "not required for publication"
    if action in PROVENANCE_ACTIONS or decision == "blocked":
        return "blocked-provenance", why or "provenance incomplete"
    return "missing-execution", why or "no qualification evidence recorded for this requirement"


def step_state(step, record):
    """(state, why) of one release step from tools/release.py step_state: done / failed / held / owed."""
    st, detail = record
    if st == "done":
        if STEP_SKIPPED.search(detail.split(": ", 1)[-1] if detail else ""):
            return "not-applicable", detail
        if STEP_PASSED_REUSE.search(detail or ""):
            return "reused", detail
        return "passed", detail or "recorded for this commit"
    if st == "failed":
        return "failed", detail
    if st == "held":
        return "missing-execution", detail or "held for --publish"
    return "missing-execution", detail or "not recorded for this commit"


def producing_step(name, section="ledger"):
    """The release step (or tool) whose execution produces this row's evidence."""
    if section == "step":
        return name
    m = ID_RE.fullmatch(name)
    if m:
        return "reference-refresh (tools/reference_refresh.sh %s column; admitted at the freeze)" % m.group(1)
    kind, _, vendor = name.partition("-")
    vendor = vendor.split("/")[0]
    if kind == "compile":
        return "macos-build" if vendor == "apple" else "linux-builds"
    if kind in ("runtime", "adapters", "candidate"):
        if vendor in ("apple", "nvidia", "amd"):
            return "gate-" + vendor
        return "none (no producer in the release step graph)"
    if kind == "published":
        return "published-" + vendor
    if name == "coverage-source":
        return "tools/verify_coverage_audit.py (admitted by tools/release_ledger_admission.py coverage-source)"
    if kind == "performance":
        return "performance campaign (outside the release)"
    return "none (no producer in the release step graph)"


def identity_column(name, row):
    """The identity column a reference refresh records for this row (ptx when the reason names the PTX set)."""
    vendor = ID_RE.fullmatch(name).group(1)
    for reason in row.get("reasons") or ():
        m = CLASS_IN_REASON.search(reason)
        if m:
            return m.group(1)
    return vendor


def evidence_of(row):
    """Evidence paths (with the revision a path is read at) of a ledger row; R2 tags when only retained."""
    out = []
    for item in row.get("evidence") or ():
        if not isinstance(item, dict):
            continue
        if item.get("tag"):
            out.append("r2:" + str(item["tag"]))
        elif item.get("path"):
            out.append(str(item["path"]) + ("@" + str(item["revision"])[:12] if item.get("revision") else ""))
    return out


def missing_evidence(row):
    """Paths the ledger said are unavailable or differ (restore-evidence)."""
    out = []
    for reason in row.get("reasons") or ():
        for prefix in ("evidence unavailable: ", "evidence digest differs: "):
            if reason.startswith(prefix):
                out.append(reason[len(prefix):])
    return out


def restore_action(name, row, plan_restore):
    """The single restore command for a row whose evidence is retained but not here."""
    tags = sorted({str(i["tag"]) for i in row.get("evidence") or () if isinstance(i, dict) and i.get("tag")})
    if tags:
        return "restore retained evidence: tools/box_persist.py fetch --tag %s" % tags[0]
    source = (plan_restore or {}).get(name)
    if source:
        m = re.search(r"--tag (\S+)", source)
        if m:
            return "restore retained evidence: tools/box_persist.py fetch --tag %s" % m.group(1)
        return "restore retained evidence from %s" % source
    paths = missing_evidence(row)
    return "restore retained evidence: %s (no retained copy named: rerun its producer, %s)" % (
        ", ".join(paths[:3]) or "the cited evidence", producing_step(name))


def next_action(version, commit, name, state, row, section="ledger", plan_restore=None, ledger_path=None):
    """THE NEXT-ACTION CATALOGUE (docs/release/STATUS_REPORT.md): one command per unresolved row."""
    if state in ("passed", "reused", "not-applicable"):
        return None
    c12 = (commit or "<commit>")[:12]
    rel = "tools/release.py %s" % version
    if section == "step":
        if state == "failed":
            return "%s --only %s (after fixing the failure it recorded)" % (rel, name)
        if (row or {}).get("held"):
            return "%s --only %s --publish pypi" % (rel, name)
        return "%s --only %s" % (rel, name)
    if row.get("action") == "restore-evidence":
        return restore_action(name, row, plan_restore)
    if row.get("action") == "resume" and row.get("active_run"):
        return "wait for or resume the unfinished run %s: %s --status" % (row["active_run"], rel)
    m = ID_RE.fullmatch(name)
    if m:
        col = identity_column(name, row)
        refresh = "tools/reference_refresh.sh %s %s" % (c12, col)
        if state == "failed":
            return "fix the code (a digest mismatch is a bug, never a rerun), then %s" % refresh
        if state == "blocked-provenance":
            why = " ".join(row.get("reasons") or ())
            if "selector provenance" in why or "dependency equivalence" in why:
                # the pre-2026-11-04 ledger path: identity rows are judged by the admitted reference now
                return "re-judge with the reference admission (tooling with release_identity_reference.py): %s --report" % rel
            if "reference admission unavailable" in why:
                return "repair the frozen reference analysis: python3 tools/release_requirements.py plan --ledger %s " \
                       "--explain --json" % (ledger_path or "<release>/requirements.json")
            return refresh
        return refresh
    kind, _, vendor = name.partition("-")
    vendor = vendor.split("/")[0]
    if state == "blocked-provenance" and row.get("action") == "complete-provenance" and kind not in ("identity",):
        step = producing_step(name)
        if step.startswith("none"):
            return "add a producer for %s (tools/release_ledger_admission.py ADAPTERS), then %s --report" % (name, rel)
        if step.startswith("gate-") or step in ("macos-build", "linux-builds") or step.startswith("published-"):
            return "%s --only %s (its producer report reconciles the provenance)" % (rel, step)
    step = producing_step(name)
    if state == "failed":
        prefix = "fix the failure it recorded, then "
    else:
        prefix = ""
    if step.startswith("gate-") or step in ("macos-build", "linux-builds") or step.startswith("published-"):
        return prefix + "%s --only %s" % (rel, step)
    if name == "coverage-source":
        return prefix + ("python3 tools/release_ledger_admission.py admit --ledger %s --root . --adapter coverage-source "
                         "--producer audit=<tools/verify_coverage_audit.py --json output>"
                         % (ledger_path or "<release>/requirements.json"))
    return prefix + "add a producer for %s (none in the release step graph), then %s --report" % (name, rel)


def build(version, commit, rows=None, steps=None, waivers=(), plan=None, stage=None, ledger_path=None):
    """The report from evaluated ledger ROWS ({name: row}), release STEPS ({step: (state, detail[, held])}),
    WAIVERS (waivers_from) and the saved PLAN (release_plan.decide output; its restore sources)."""
    plan_restore = {r["id"]: r.get("source") for r in (plan or {}).get("restore") or ()}
    out = []
    for name, row in sorted((rows or {}).items()):
        if stage and row.get("stage", "pre") != stage:
            continue
        state, why = ledger_state(name, row, commit)
        out.append(dict(id=name, section="ledger", stage=row.get("stage", "pre"), required=row.get("required", True),
                        state=state, reason=why, evidence=evidence_of(row), producing_step=producing_step(name),
                        tested_revision=row.get("tested_revision"), action=row.get("action"),
                        next_action=next_action(version, commit, name, state, row, plan_restore=plan_restore,
                                                ledger_path=ledger_path)))
    for name, rec in sorted((steps or {}).items()):
        st_stage = "post" if POST_STEPS.match(name) else "pre"
        if stage and st_stage != stage:
            continue
        held = len(rec) > 2 and rec[2]
        state, why = step_state(name, rec[:2])
        out.append(dict(id=name, section="step", stage=st_stage, required=True, state=state, reason=why,
                        evidence=[], producing_step=name, action=None,
                        next_action=next_action(version, commit, name, state, dict(held=held), section="step")))
    for r in out:
        if r["state"] in ("passed", "reused", "not-applicable"):
            w, rule = waiver_for(waivers, r["id"])
            if w is not None:
                r["waiver_unused"] = dict(rule=rule, file=w.get("file"))   # evidence passed it; the waiver did not
            continue
        w, rule = waiver_for(waivers, r["id"])
        if w is not None:
            r.update(underlying_state=r["state"], state="waived",
                     waiver=dict(rule=rule, file=w.get("file"), sha256=w.get("sha256"), approved_by=w["approved_by"],
                                 reason=w["reason"]))
    totals = {s: sum(r["state"] == s for r in out) for s in STATES}
    underlying = {}
    for r in out:
        if r["state"] == "waived":
            underlying[r["underlying_state"]] = underlying.get(r["underlying_state"], 0) + 1
    ready = not any(totals[s] for s in UNRESOLVED)
    report = dict(schema=SCHEMA, version=version, commit=commit, stage=stage or "all", rows=out, totals=totals,
                  waived_underlying=dict(sorted(underlying.items())), ready=ready,
                  waivers=[dict(file=w.get("file"), sha256=w.get("sha256"), approved_by=w["approved_by"],
                                reason=w["reason"], prefixes=len(w["prefixes"]), names=len(w["names"]))
                           for w in waivers or ()],
                  runs_nothing=True)
    report["totals_line"] = totals_line(report)
    return report


def totals_line(report):
    t = report["totals"]
    return "REPORT %s %s  ready=%s" % (report.get("version"), " ".join("%s=%d" % (k, t[s]) for k, s in TOTAL_KEYS),
                                      "YES" if report["ready"] else "NO")


# ---------------------------------------------------------------- output
def _cell(text, width=240):
    text = str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= width else text[:width - 3] + "..."


def markdown(report):
    """The Markdown table: unresolved rows first, then waived, then the rest (one row each)."""
    order = {s: i for i, s in enumerate(("failed", "blocked-provenance", "missing-execution", "waived",
                                         "not-applicable", "reused", "passed"))}
    rows = sorted(report["rows"], key=lambda r: (order[r["state"]], r["section"], r["id"]))
    out = ["# Release status: %s at %s" % (report.get("version"), (report.get("commit") or "")[:12]), "",
           "`%s`" % report["totals_line"], ""]
    if report["waived_underlying"]:
        out += ["Waived rows by underlying state: " + ", ".join("%s %d" % kv for kv in report["waived_underlying"].items()),
                ""]
    for w in report["waivers"]:
        out.append("- waiver `%s` (approved by %s): %s" % (w.get("file"), w["approved_by"], _cell(w["reason"], 400)))
    if report["waivers"]:
        out.append("")
    out += ["| row | state | stage | producing step | evidence | next action |", "|---|---|---|---|---|---|"]
    for r in rows:
        state = r["state"] + (" (%s)" % r["underlying_state"] if r["state"] == "waived" else "")
        if r["state"] == "waived":
            action = "WAIVED by %s (%s): %s" % (r["waiver"]["approved_by"], r["waiver"]["rule"], r["waiver"]["reason"])
        else:
            action = r.get("next_action") or ""
        ev = ", ".join(r["evidence"][:3]) + (" (+%d)" % (len(r["evidence"]) - 3) if len(r["evidence"]) > 3 else "")
        out.append("| %s | %s | %s | %s | %s | %s |" % (_cell(r["id"]), state, r["stage"], _cell(r["producing_step"], 80),
                                                     _cell(ev or ("-" if r["section"] == "ledger" else r["reason"]), 160),
                                                     _cell(action or r["reason"])))
    out += ["", "The report runs, rents and publishes nothing."]
    return "\n".join(out) + "\n"


def lines(report, limit=40):
    """Terminal lines: the totals line, then the unresolved rows (bounded), then the waived count."""
    out = [report["totals_line"]]
    shown = 0
    for r in report["rows"]:
        if r["state"] in UNRESOLVED and shown < limit:
            out.append("  %-18s %s: %s -> %s" % (r["state"], r["id"], _cell(r["reason"], 120), r.get("next_action")))
            shown += 1
    rest = sum(r["state"] in UNRESOLVED for r in report["rows"]) - shown
    if rest > 0:
        out.append("  ... %d more unresolved row(s) in report.json" % rest)
    if report["totals"]["waived"]:
        out.append("  waived %d row(s) (%s), listed in report.md" % (
            report["totals"]["waived"], ", ".join("%s %d" % kv for kv in report["waived_underlying"].items())))
    return out


def write(report, json_path, md_path=None):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    tmp.replace(json_path)
    md = Path(md_path) if md_path else json_path.with_suffix(".md")
    md.write_text(markdown(report))
    return json_path, md


# ---------------------------------------------------------------- retained blocked-row reports
def rows_from_blocked_tsv(path):
    """Ledger-like rows from a retained blocked-row report (`stage<TAB>row<TAB>tested_revision<TAB>reason`,
    as the 0.8.65 publish wrote it): the rows the gate refused, in the ledger's own vocabulary."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        stage, name, tested, reason = (line.split("\t") + ["", "", "", ""])[:4]
        row = dict(status="untested", stage=stage or "pre", required=True, reasons=[reason] if reason else [],
                   action="run", evidence=[], tested_revision=tested or None)
        if "blocked" in reason:
            row.update(action="complete-provenance", dependency_decision="blocked")
        elif "affected" in reason:
            row["dependency_decision"] = "affected"
        rows[name] = row
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--commit", default="")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rows", help="evaluated ledger rows JSON ({name: row})")
    src.add_argument("--blocked-tsv", help="a retained blocked-row report (stage, row, tested_revision, reason)")
    ap.add_argument("--state", help="the release state.json (step records, logged owner waivers)")
    ap.add_argument("--waiver", action="append", default=[], help="an owner waiver JSON (repeatable)")
    ap.add_argument("--plan", help="the saved plan.json (restore sources)")
    ap.add_argument("--stage", choices=("pre", "post"))
    ap.add_argument("--json", help="write report.json here (and report.md beside it)")
    ap.add_argument("--markdown", help="write the Markdown table here")
    a = ap.parse_args(argv)
    rows = json.loads(Path(a.rows).read_text()) if a.rows else rows_from_blocked_tsv(a.blocked_tsv)
    state = json.loads(Path(a.state).read_text()) if a.state else None
    plan = json.loads(Path(a.plan).read_text()) if a.plan else None
    report = build(a.version, a.commit or (state or {}).get("commit"), rows, None, waivers_from(a.waiver, state),
                   plan, stage=a.stage)
    if a.json:
        write(report, a.json, a.markdown)
    elif a.markdown:
        Path(a.markdown).write_text(markdown(report))
    print("\n".join(lines(report)))
    return 0 if report["ready"] else 3


if __name__ == "__main__":
    sys.exit(main())
