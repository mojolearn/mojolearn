#!/usr/bin/env python3
"""Compare saved GPU columns across vendors; never imports native bindings.

Each directory contains <lane>.gpu.json files. Repeat --column with the same
backend to overlay a later directory explicitly; its whole lane record replaces
the earlier record. Source commits must match across vendors for each lane,
not across unrelated lanes. Missing, invalid or empty coverage cannot pass.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("consolidated_reference", ROOT / "python/mojolearn/_verify_reference.py")
vref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vref)
PARTS = ("train", "infer", "model", "reload", "batch", "batchgrad", "batchscale", "ragged", "stepfull", "rlpair")
HEX = re.compile(r"[0-9a-f]{16}\Z")


def _value(cell, part, repeats):
    """(state,value,reason); keep numeric disagreement distinct from refusal."""
    values = cell.get("hashes" if part == "train" else part)
    verdict = cell.get("verdict" if part == "train" else part + "_verdict")
    if not isinstance(values, list) or not values:
        return "INCOMPLETE", None, "missing values"
    if len(values) != repeats:
        return "INCOMPLETE", None, "missing or extra repeats"
    if all(isinstance(v, str) and HEX.fullmatch(v) for v in values) and len(set(values)) > 1:
        return "DIFFERENT", None, "numeric value moved between repeats"
    if isinstance(verdict, str) and ("MOVED" in verdict or verdict in ("DIVERGENT", "DIFFERENT")):
        return "DIFFERENT", None, "record reports " + verdict
    if not all(isinstance(v, str) for v in values) or len(set(values)) != 1:
        return "INCOMPLETE", None, "invalid or inconsistent values"
    value = values[0]
    if value.startswith("n/a:"):
        if value.startswith(vref._SKIPPED_NA) or len(value) == 4:
            return "INCOMPLETE", value, "skipped or undeclared property"
        if verdict == "N/A" or (part == "reload" and verdict is None):
            return "NA", value, None
    # reload is recorded without a separate verdict; its parent model and
    # inference verdicts are independently required and compared below.
    if part == "reload" and verdict is None and HEX.fullmatch(value):
        return "NUMERIC", value, None
    valid = vref._part_value(cell, part)
    if valid is not None and HEX.fullmatch(valid):
        return "NUMERIC", valid, None
    return "INCOMPLETE", value, "part is not stable numerical evidence or explicit N/A"


def _input_witness(record, fixture):
    x = record.get("fixtures", {}).get(fixture)
    held = record.get("heldout", {}).get(fixture)
    if (not isinstance(x, dict) or set(x) != {"X", "y_clf", "y_reg"}
            or not isinstance(held, dict) or set(held) != {"X"}
            or not all(isinstance(v, str) and HEX.fullmatch(v) for v in (*x.values(), *held.values()))):
        return None
    return x, held


def _read_columns(lanes, columns):
    selected, ignored = {}, 0
    for backend, directory in columns:
        if backend not in ("metal", "hip", "cuda"):
            raise ValueError("columns must name metal, hip or cuda")
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"column directory does not exist: {directory}")
        layer = {}
        for path in sorted(directory.rglob("*.gpu.json")):
            lane = path.name[:-len(".gpu.json")]
            if lane not in lanes:
                ignored += 1
                continue
            if lane in layer:
                raise ValueError(f"ambiguous duplicate {backend}/{lane} in {directory}; use separate explicit overlays")
            metadata = dict(path=str(path.resolve()))
            try:
                raw = path.read_bytes()
                metadata.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
                record = json.loads(raw)
                metadata["source_commit"] = record.get("commit")
                metadata["batch_revision"] = (record.get("batch_revisions") or {}).get(lane)
                why = vref.admit(record, str(path), known_lanes=lanes)
                cls, backend_error = vref.record_device_class(record, str(path))
                if why or backend_error:
                    raise ValueError(why or backend_error)
                if cls != vref.VENDOR_CLASS[backend]:
                    raise ValueError(f"expected {backend}, record class is {cls}")
                if record.get("complete") is not True:
                    raise ValueError("record is not explicitly complete")
                if type(record.get("repeats")) is not int or record["repeats"] < 1:
                    raise ValueError("invalid repetition count")
                scope = {key.split("/", 1)[0] for key in record["cells"]}
                if scope != {lane}:
                    raise ValueError("filename and record lane scope disagree")
                layer[lane] = (record, metadata, None)
            except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
                # An invalid explicit overlay must shadow the older result,
                # not quietly fall back to a stale passing record.
                layer[lane] = (None, metadata, str(exc))
        selected.setdefault(backend, {}).update(layer)
    if len(selected) < 2:
        raise ValueError("at least two distinct GPU backend columns are required")
    return selected, ignored


def compare(plan, columns, fixtures=("base",), progress=False):
    lanes = plan.get("lanes")
    if (not isinstance(lanes, list) or not lanes
            or not all(isinstance(l, str) and l for l in lanes)
            or len(set(lanes)) != len(lanes)):
        raise ValueError("plan must contain a nonempty unique lane list")
    if not fixtures or len(set(fixtures)) != len(fixtures) or any(not f for f in fixtures):
        raise ValueError("fixtures must be a nonempty unique list")
    expected_batch = plan.get("batch_revisions", {})
    if not isinstance(expected_batch, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or not v
            for k, v in expected_batch.items()):
        raise ValueError("plan batch_revisions must map lane names to nonempty revisions")
    selected, ignored = _read_columns(set(lanes), columns)
    backends = sorted(selected)
    rows, issues, artifacts, lane_results = [], [], [], {}
    counts = dict(numeric_matches=0, na_parts=0, unrecorded_parts=0, different_parts=0,
                  incomplete_parts=0, expected_cells=len(lanes)*len(fixtures), compared_cells=0)
    for lane in lanes:
        records, lane_issues = {}, []
        for backend in backends:
            entry = selected[backend].get(lane)
            if entry is None:
                lane_issues.append(dict(lane=lane, backend=backend, reason="missing record"))
                continue
            record, metadata, error = entry
            artifacts.append(dict(lane=lane, backend=backend, **metadata))
            if error:
                lane_issues.append(dict(lane=lane, backend=backend, reason=error))
            else:
                records[backend] = record
        if len(records) != len(backends):
            issues.extend(lane_issues)
            lane_results[lane] = "INCOMPLETE"
            continue
        commits = {j["commit"] for j in records.values()}
        revisions = {j.get("lane_revisions", {}).get(lane) for j in records.values()}
        if len(commits) != 1 or len(revisions) != 1:
            issues.append(dict(lane=lane, reason="source commits or lane revisions differ across columns"))
            lane_results[lane] = "INCOMPLETE"
            continue
        batch_revisions = {b: j.get("batch_revisions", {}).get(lane)
                           if isinstance(j.get("batch_revisions", {}), dict) else "INVALID"
                           for b, j in records.items()}
        batch_revision_bad = (any(v is not None and (not isinstance(v, str) or not v or v == "INVALID")
                                  for v in batch_revisions.values())
                              or any(v != next(iter(batch_revisions.values())) for v in batch_revisions.values())
                              or (lane in expected_batch and any(v != expected_batch[lane]
                                                                 for v in batch_revisions.values())))
        lane_states = []
        for fixture in fixtures:
            key = lane + "/" + fixture
            cells = {b:j["cells"].get(key) for b,j in records.items()}
            witness = [_input_witness(j, fixture) for j in records.values()]
            if any(not isinstance(c, dict) for c in cells.values()) or any(w is None for w in witness):
                issues.append(dict(lane=lane, fixture=fixture, reason="missing fixture cell or input witness"))
                lane_states.append("INCOMPLETE")
                continue
            if any(w != witness[0] for w in witness[1:]):
                issues.append(dict(lane=lane, fixture=fixture, reason="fixture or heldout input hashes differ"))
                lane_states.append("INCOMPLETE")
                continue
            numeric = 0
            for part in PARTS:
                present = [part in c for c in cells.values()]
                if part == "rlpair" and not any(present):
                    # The harness emits rlpair only for declared sampler pairs;
                    # vref's declared-part contract deliberately permits this.
                    counts["unrecorded_parts"] += 1
                    rows.append(dict(lane=lane, fixture=fixture, part=part, state="UNRECORDED",
                                     reason="no sampler/trainer part recorded in any column"))
                    continue
                if part == "reload" and not any(present):
                    states = [_value(c, "model", records[b]["repeats"])[0] for b,c in cells.items()]
                    if all(s == "NA" for s in states):
                        counts["na_parts"] += 1
                        rows.append(dict(lane=lane, fixture=fixture, part=part, state="NA",
                                         reason="all columns declare no saved-model output"))
                        continue
                values = {b:_value(c, part, records[b]["repeats"]) for b,c in cells.items()}
                states = {s for s, _, _ in values.values()}
                reason = None
                protocol_key = part + "_protocol"
                if "DIFFERENT" in states:
                    state, reason = "DIFFERENT", "within-column numerical instability"
                elif part == "batch" and batch_revision_bad:
                    state, reason = "INCOMPLETE", "batch revisions differ, are missing, or do not match plan"
                elif "INCOMPLETE" in states:
                    state, reason = "INCOMPLETE", "; ".join(f"{b}: {why}" for b,(s,_,why) in values.items() if why)
                elif part in ("batch", "batchgrad", "batchscale", "ragged", "stepfull", "rlpair") and (
                        any(not isinstance(j.get(protocol_key), dict) for j in records.values()) or
                        len({json.dumps(j.get(protocol_key), sort_keys=True) for j in records.values()}) != 1):
                    state, reason = "INCOMPLETE", "property protocols missing or different"
                elif states == {"NA"}:
                    state = "NA" if len({v for _,v,_ in values.values()}) == 1 else "INCOMPLETE"
                    if state == "INCOMPLETE":
                        reason = "N/A declarations differ across columns"
                elif states != {"NUMERIC"}:
                    state, reason = "INCOMPLETE", "numerical versus N/A scope differs"
                elif part == "reload" and any(
                        _value(cells[b], "infer", records[b]["repeats"])[0] == "NUMERIC"
                        and value != _value(cells[b], "infer", records[b]["repeats"])[1]
                        for b,(_,value,_) in values.items()):
                    state, reason = "DIFFERENT", "reloaded model differs from its original inference"
                elif len({v for _,v,_ in values.values()}) != 1:
                    state, reason = "DIFFERENT", "numerical hashes differ across vendors"
                else:
                    state = "MATCH"
                    numeric += 1
                bucket = {"MATCH":"numeric_matches", "NA":"na_parts", "INCOMPLETE":"incomplete_parts", "DIFFERENT":"different_parts"}[state]
                counts[bucket] += 1
                rows.append(dict(lane=lane, fixture=fixture, part=part, state=state,
                                 values={b:v for b,(_,v,_) in values.items()}, reason=reason))
                lane_states.append(state)
            if not numeric:
                issues.append(dict(lane=lane, fixture=fixture, reason="no stable numerical part compared"))
                lane_states.append("INCOMPLETE")
            else:
                counts["compared_cells"] += 1
        lane_results[lane] = ("DIFFERENT" if "DIFFERENT" in lane_states else
                                "INCOMPLETE" if "INCOMPLETE" in lane_states else "AGREE")
    counts.update(missing_records=sum(i["reason"] == "missing record" for i in issues),
                  invalid_records=sum("backend" in i and i["reason"] != "missing record" for i in issues),
                  uncompleted_cells=counts["expected_cells"] - counts["compared_cells"],
                  agree_lanes=sum(s == "AGREE" for s in lane_results.values()),
                  incomplete_lanes=sum(s == "INCOMPLETE" for s in lane_results.values()),
                  different_lanes=sum(s == "DIFFERENT" for s in lane_results.values()))
    verdict = ("DIFFERENT" if counts["different_parts"] else
               "INCOMPLETE" if issues or counts["incomplete_parts"] or counts["compared_cells"] != counts["expected_cells"] else "AGREE")
    return dict(format="mojolearn.consolidated-crossvendor.v1", verdict=verdict,
                exit={"AGREE":0,"DIFFERENT":1,"INCOMPLETE":2}[verdict], progress=progress,
                lanes=lanes, fixtures=list(fixtures), columns=backends, counts=counts,
                column_directories=[dict(backend=b, path=str(Path(d).resolve())) for b,d in columns],
                lane_results=lane_results, issues=issues, parts=rows, records=artifacts,
                expected_batch_revisions=expected_batch,
                ignored_records=ignored, source_commits=sorted({a["source_commit"] for a in artifacts if a.get("source_commit")}),
                scope="saved GPU columns only; no native execution; numeric hashes exclude timings")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--column", action="append", required=True, metavar="BACKEND=DIR")
    parser.add_argument("--fixtures", default="base", help="comma-separated expected fixtures")
    parser.add_argument("--progress", action="store_true", help="label a partial snapshot; missing coverage still exits 2")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)
    try:
        columns = [x.split("=", 1) for x in args.column]
        if any(len(x) != 2 or not x[1] for x in columns):
            raise ValueError("--column needs BACKEND=DIR")
        raw = Path(args.plan).read_bytes()
        report = compare(json.loads(raw), columns, args.fixtures.split(","), args.progress)
        report["plan"] = dict(path=str(Path(args.plan).resolve()), sha256=hashlib.sha256(raw).hexdigest())
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"CROSSVENDOR {report['verdict']}: {report['counts']}; evidence {args.json_out}")
    return report["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
