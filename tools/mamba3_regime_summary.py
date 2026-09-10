#!/usr/bin/env python3
"""Validate and summarize same-binary regime logs; diagnostic, never a price gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(path):
    raw = Path(path).read_bytes()
    environment = None
    fixtures = {}
    calls = []
    active = None
    passed = None
    for number, line in enumerate(raw.decode().splitlines(), 1):
        if line.startswith("M3_PHASE "):
            if active is None:
                raise ValueError(f"phase outside a call at line {number}")
            _, label, value = line.split()
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid phase duration")
            if label in active["phases_ms"]:
                raise ValueError(f"duplicate phase {label}; cannot aggregate overlapping scopes")
            active["phases_ms"][label] = value
            continue
        if not line.startswith("{"):
            continue  # Fixture witness lines are retained in the hashed original log.
        event = json.loads(line)
        kind = event["kind"]
        if passed is not None and kind != "telemetry_exit":
            raise ValueError("event after final pass")
        if kind == "environment":
            if environment is not None or calls or fixtures:
                raise ValueError("multiple/out-of-order environment records")
            environment = event
        elif kind == "fixture":
            if environment is None or active is not None or event["shape"] in fixtures:
                raise ValueError("duplicate/out-of-order fixture")
            fixtures[event["shape"]] = event
        elif kind == "call_begin":
            if environment is None or active is not None or event["shape"] not in fixtures:
                raise ValueError("missing fixture or overlapping calls")
            active = dict(event, phases_ms={})
        elif kind == "call_end":
            if active is None or any(event[k] != active[k] for k in ("trial", "position", "shape", "call")):
                raise ValueError("call begin/end mismatch")
            if event["wall_ns"] < active["wall_ns"] or not math.isfinite(event["elapsed_ms"]) or event["elapsed_ms"] <= 0:
                raise ValueError("invalid call duration")
            calls.append(dict(event, phases_ms=active["phases_ms"], begin_wall_ns=active["wall_ns"]))
            active = None
        elif kind == "pass":
            if active is not None:
                raise ValueError("pass inside unfinished call")
            passed = event
    if environment is None or passed is None or active is not None:
        raise ValueError("incomplete capture")
    order = environment["order"]
    expected = [(trial, position, shape, call)
                for trial in range(environment["passes"])
                for position, shape in enumerate(order)
                for call in range(environment["rounds"])]
    actual = [(c["trial"], c["position"], c["shape"], c["call"]) for c in calls]
    if not expected or actual != expected or set(fixtures) != set(order):
        raise ValueError("missing, duplicated, or reordered scheduled calls")
    if any(f["binary_sha256"] != passed["binary_sha256"] for f in fixtures.values()):
        raise ValueError("binary changed across fixtures")
    if set(passed["output_hashes"]) != set(fixtures) or any(
            c["output_sha256"] != passed["output_hashes"][c["shape"]] for c in calls):
        raise ValueError("output hash changed")
    phase_sets = {tuple(sorted(c["phases_ms"])) for c in calls}
    if len(phase_sets) != 1:
        raise ValueError("inconsistent phase inventory; capture may be buffered or incomplete")
    visits = []
    previous = None
    seen = set()
    for trial in range(environment["passes"]):
        for position, shape in enumerate(order):
            rows = [c for c in calls if c["trial"] == trial and c["position"] == position]
            first = rows[0]
            visits.append(dict(trial=trial, position=position, shape=shape,
                predecessor=previous, first_shape_visit=shape not in seen,
                elapsed_ms=[r["elapsed_ms"] for r in rows],
                first_call_ms=first["elapsed_ms"],
                later_calls_median_ms=statistics.median([r["elapsed_ms"] for r in rows[1:]]) if len(rows) > 1 else None,
                cpu_ms=[1000 * (r["resource_delta"]["user_s"] + r["resource_delta"]["system_s"]) for r in rows],
                resource_deltas=[r["resource_delta"] for r in rows],
                phases_ms={label: [r["phases_ms"][label] for r in rows] for label in first["phases_ms"]}))
            seen.add(shape)
            previous = shape
    return dict(schema="mojolearn.mamba3.regime-summary.v1",
        scope="Diagnostic only; cold calls retained, no warmup assumption, opponent ratio, or promotion decision. Nested phase durations must not be summed.",
        log=str(Path(path).resolve()), log_sha256=hashlib.sha256(raw).hexdigest(),
        environment=environment, fixtures=fixtures, output_hashes=passed["output_hashes"],
        instrumented=bool(calls[0]["phases_ms"]), visits=visits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
