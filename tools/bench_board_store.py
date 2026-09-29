# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE OPPONENT STORE: an opponent is measured once per KEY and reused.

Andrew (2026-09-29): "we don't always run the opponent, only when we need a
new datapoint; we store the opponent with the device it was on and the
time".

Every finished opponent cell of a board run is appended to one JSONL file
(tools/bench_board.py --opponent-store, default <out>/../opponent-store.jsonl)
with its KEY and its provenance. Before a race runs, each opponent arm is
looked up by KEY; a hit is not run again, and its stored cell goes on the
board marked `source: stored (measured <UTC> on <box>, <device>)`.

THE KEY (every field must be known, or the cell is neither reused nor
stored):

    box, machine          hostname and machine model (GPU name, else CPU model)
    vendor, device, os    the board's vendor, the arm's device (and its name), the OS
    library, library_version   the version the arm's own worker imported
    family, lane, dataset, rows, neural_shape, arm
    params_sha256         sha256 of the arm's parameters READ BACK from the
                          constructed object (the BOARD-PARAMS check): before a
                          race the board constructs each candidate opponent
                          (the drivers' --params-only), so a stored cell is
                          reused only when the read-back is the same
    settings_sha256       sha256 of the race's settings (lane config, seed,
                          harness source), kept as an extra key field
    data_sha256           sha256 of the data file(s) the race reads
    rounds

Provenance kept beside it: measured_at, commit, the arm's canonical
parameters from the BOARD-PARAMS check, the full cell. A capped, refused or
failed opponent is not stored; only a successful measurement (status ok with
a time) is stored and reused, so every failure is attempted again. Standard library only.
"""
import hashlib
import json
import os

KEY_FIELDS = ("box", "machine", "vendor", "device", "os", "library", "library_version",
              "family", "lane", "dataset", "rows", "neural_shape", "arm",
              "params_sha256", "settings_sha256", "data_sha256", "rounds")
#: the fields known before any arm is constructed (the pre-filter: is there
#: anything stored that could match, so constructing the arm is worth it?)
PRE_FIELDS = ("box", "machine", "vendor", "os", "library", "family", "lane", "dataset", "rows",
              "neural_shape", "arm", "settings_sha256", "data_sha256", "rounds")
#: key fields that may legitimately be None (not every race has them)
OPTIONAL = ("rows", "neural_shape")


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def missing(key):
    """Key fields with no value (the cell can be neither reused nor stored)."""
    return [f for f in KEY_FIELDS if f not in OPTIONAL and key.get(f) in (None, "")]


def key_id(key):
    return sha256_json({f: key.get(f) for f in KEY_FIELDS})


def load(path):
    """{key_id: the latest record} of the store (no file: empty)."""
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("key"), dict):
                    out[key_id(rec["key"])] = rec
    except OSError:
        pass
    return out


def append(path, record):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def reusable(record):
    """ONLY a successful measurement is reused: status ok with a time
    (orchestrator, 2026-09-29). Every error, timeout, crash and refusal
    (race-level or arm-level, library errors included) is attempted again
    on the next run: a failure is often the driver's (the L40S board's cuML
    ETS host copy, fixed in the driver, would otherwise have been served back
    as an error), and a race-level parameter refusal says nothing about the
    opponent at all. Planned refusals come from the plan, not the store."""
    cell = record.get("cell") or {}
    return str(cell.get("status") or "") == "ok" and cell.get("median_ms") is not None


def lookup(store, key):
    """The stored record for `key`, or None (a key with a missing field never hits)."""
    if missing(key):
        return None
    rec = store.get(key_id(key))
    return rec if rec is not None and reusable(rec) else None


def candidates(store, partial):
    """The reusable stored records whose PRE_FIELDS equal `partial`'s."""
    return [r for r in store.values() if reusable(r)
            and all(r["key"].get(f) == partial.get(f) for f in PRE_FIELDS)]


def source_text(record):
    k = record["key"]
    return "stored (measured %s on %s, %s)" % (record.get("measured_at"), k.get("box"), k.get("device"))


def stored_cell(record):
    """The stored cell as it goes on a board: `source` says it was not run."""
    cell = json.loads(json.dumps(record["cell"]))
    cell["source"] = source_text(record)
    cell["stored"] = {"measured_at": record.get("measured_at"), "commit": record.get("commit"),
                      "box": record["key"].get("box"), "device": record["key"].get("device"),
                      "key_id": key_id(record["key"])}
    return cell


def record(key, cell, *, measured_at, commit, params=None, infer_cells=None):
    return {"key": {f: key.get(f) for f in KEY_FIELDS}, "measured_at": measured_at,
            "commit": commit, "params": params, "cell": cell,
            "infer_cells": infer_cells or []}
