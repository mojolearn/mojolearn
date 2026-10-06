#!/usr/bin/env python3
"""Publish captured opponent-only boards without changing their results or ratios.

The config has sources [{name, vendor, board_dir, harness_source_sha}], where name
is one of the three owned overnight-20261006 hardware directories. Runtime config
and local process state stay outside the repository. A source JSON/Markdown pair
must agree on coverage before replacing the previously published pair.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

NAMES = {"nvidia-specific", "nvidia-default", "amd"}
COVERAGE = re.compile(r"Races: (\d+) planned, (\d+) done, (\d+) failed, (\d+) unsupported, (\d+) pending\. Cells: (\d+) \(([^\n]*)\)\.")


def atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".incoming")
    tmp.write_bytes(raw)
    tmp.replace(path)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def inspect_pair(raw: bytes, markdown: bytes, vendor: str) -> tuple[dict, dict]:
    board = json.loads(raw)
    if board.get("schema") != "mojolearn-bench-board/1":
        raise ValueError("unsupported board schema")
    config = board.get("config", {})
    if config.get("vendor") != vendor or config.get("opponents_only") is not True:
        raise ValueError("source must be the requested vendor's opponent-only board")
    races = board.get("races", {})
    if not isinstance(races, dict):
        raise ValueError("races must be keyed by race id")
    cells = [cell for race in races.values() for cell in race.get("cells", [])]
    for cell in cells:
        arm = cell.get("arm", "")
        if arm.startswith("ours") or re.search(r"-cpu(?:-|$)", arm):
            raise ValueError("source contains an own-algorithm or CPU cell: " + arm)
    states = Counter(race.get("status", "unknown") for race in races.values())
    cell_states = Counter(str(cell.get("status", "unknown")).split("(")[0] for cell in cells)
    planned = len(board.get("plan", []))
    done, failed, unsupported = (states[x] for x in ("done", "failed", "unsupported"))
    pending = max(0, planned - done - failed - unsupported)
    coverage = dict(planned=planned, done=done, failed=failed, unsupported=unsupported,
                    pending=pending, cells=len(cells), cell_statuses=dict(cell_states),
                    race_statuses=dict(states))
    match = COVERAGE.search(markdown.decode())
    if not match or tuple(map(int, match.groups()[:6])) != (planned, done, failed, unsupported, pending, len(cells)):
        raise ValueError("captured JSON and Markdown coverage differ; await next capture")
    expected = ", ".join(f"{key} {value}" for key, value in sorted(cell_states.items())) or "none"
    if match.group(7) != expected:
        raise ValueError("captured JSON and Markdown cell statuses differ")
    return board, coverage


def publish(source: dict, destination: Path) -> dict:
    name = source["name"]
    if name not in NAMES:
        raise ValueError("destination is outside this campaign's owned directories")
    src = Path(source["board_dir"])
    out = destination / "bench/results/bench_board" / ("overnight-20261006-" + name)
    paths = [src / "board.json", src / "BOARD.md"]
    if not all(path.exists() for path in paths):
        if not (out / "BOARD.md").exists():
            atomic(out / "BOARD.md", ("# Overnight GPU opponent measurements\n\n"
                   + name + ": pending the applicable candidate queue and first captured opponent board.\n"
                   + "No opponent timings or comparisons have been published for this machine yet.\n").encode())
        return dict(name=name, status="WAITING_FOR_CAPTURE", changed=False)
    raw, markdown = (path.read_bytes() for path in paths)
    if paths[0].read_bytes() != raw:
        raise ValueError("source board changed while reading; await stable capture")
    board, coverage = inspect_pair(raw, markdown, source["vendor"])
    hashes = dict(board_json=hashlib.sha256(raw).hexdigest(), board_markdown=hashlib.sha256(markdown).hexdigest())
    prior_path = out / "publication.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    changed = prior.get("source_sha256") != hashes
    if changed:
        receipt = dict(schema="mojolearn-opponent-publication/1", source_sha256=hashes,
                       source_directory=str(src), harness_source_sha=source["harness_source_sha"],
                       coverage=coverage, source_updated=board.get("updated"), published_at=time.time(),
                       policy="Exact opponent-only source files, hardware and failures retained. No A/B component ratios are derived.")
        # The receipt is written last, so readers can verify both files by hash.
        atomic(out / "board.json", raw)
        atomic(out / "BOARD.md", markdown)
        atomic(prior_path, json_bytes(receipt))
    return dict(name=name, status="PUBLISHED", changed=changed, coverage=coverage, source_sha256=hashes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--thread")
    parser.add_argument("--codex", default="/opt/homebrew/bin/codex")
    args = parser.parse_args()
    pending_notifications = {}
    while True:
        rows = []
        for source in json.loads(args.config.read_text())["sources"]:
            try:
                rows.append(publish(source, args.destination_root))
            except (ValueError, OSError, KeyError, TypeError) as error:
                rows.append(dict(name=source.get("name"), status="CAPTURE_NOT_PUBLISHED", error=str(error), changed=False))
        changed = [row for row in rows if row.get("changed")]
        state = dict(pid=os.getpid(), updated=time.time(), sources=rows)
        pending_notifications.update({row["name"]: row for row in changed})
        if pending_notifications and args.thread:
            message = "Captured GPU opponent boards updated through performance_opponents_publish.py: " + "; ".join(
                row["name"] + " " + json.dumps(row["coverage"], sort_keys=True) for row in pending_notifications.values())
            message += ". Review and incrementally commit/push owned overnight-20261006 hardware directories. Files preserve source hardware/failures; no component A/B opponent ratios."
            try:
                result = subprocess.run([args.codex, "queue", "--thread", args.thread, "--message", message], capture_output=True, timeout=30)
                state["notification_returncode"] = result.returncode
                if result.returncode == 0:
                    pending_notifications.clear()
            except (OSError, subprocess.TimeoutExpired) as error:
                state["notification_error"] = str(error)
        atomic(args.state, json_bytes(state))
        if not args.watch:
            print(json.dumps(state))
            return int(any(row["status"] == "CAPTURE_NOT_PUBLISHED" for row in rows))
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
