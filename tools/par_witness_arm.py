#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The TWO-DEVICE arm of a `par-*` lane check, run under the device witness
(lane/par-harness, 2026-09-28).

    python tools/par_witness_arm.py --witness-json W.json -- <identity_break.py arguments>

MOJOLEARN_PAR_DEVICES must name at least two distinct devices. This runs
tools/identity_break.py IN THIS PROCESS, exactly as `algos_lane_check.run_arm`
would run it, and opens one `_verify_par.PoolWitness` PER CELL (lane/fixture):
every `DevicePool` started and every native byte-LM session opened while that
cell runs is inventoried, and the cell's witness refusal is the sentence
`PoolWitness.refusal()` gives (no pool at all, a pool on other devices, two
workers on one physical GPU, a stale ONE_DEVICE_BY_DESIGN declaration). The
per-cell record goes to --witness-json; the column JSON is the harness's own.

WHY IN PROCESS. The witness wraps `DevicePool._start` and the byte-LM trainers'
`_open` in the process that fits. A child that ran the harness would start its
pools where no witness could see them. This is the same witness `verify --par`
uses (`_verify_par.par_check`), lifted onto the lane check's column so that
the lane gates and the consolidation check read a two-device column only when
it was shown to run on two devices.

A CELL IS DELIMITED BY `identity_break.CellTimer`, which the harness creates
once per computed cell just before its first fit. The witness of one cell is
closed when the next cell's timer is created and at exit, so placement from
another lane or fixture never certifies this one. A cell REUSED from a resumed
JSON creates no timer and gets no witness; `algos_lane_check.compare` then
finds no record for it and refuses the column.

Exit status: the harness's own. A witness that could not be written is exit 3.
"""
import argparse
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools" / "identity_break.py"


def parse_devices(raw):
    try:
        devs = tuple(int(x) for x in (raw or "").split(",") if x.strip() != "")
    except ValueError:
        devs = ()
    if len(devs) < 2 or len(set(devs)) != len(devs) or any(d < 0 for d in devs):
        raise SystemExit(f"REFUSING: MOJOLEARN_PAR_DEVICES={raw!r} must name at least two distinct "
                         "nonnegative device indices for the two-device arm")
    return devs


class CellWitnesses:
    """One `PoolWitness` per cell, opened and closed around the harness's own
    cell boundaries."""

    def __init__(self, devices):
        self.vendor = None
        self.devices = devices
        self.cells = {}
        self._stack = None
        self._label = None
        self._witness = None

    def open(self, label):
        self.close()
        from mojolearn._verify_par import ONE_DEVICE_BY_DESIGN, PoolWitness
        if self.vendor is None:
            # Asked at the first cell, after the harness imported the package
            # in its own order, never before it.
            from mojolearn import _backend
            self.vendor = _backend.vendor()
        lane = label.split("/", 1)[0]
        declared = ONE_DEVICE_BY_DESIGN.get(lane)
        self._witness = PoolWitness(self.vendor, self.devices,
                                    one_device_driver=declared[0] if declared else None)
        self._label = label
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(self._witness.watching())

    def close(self):
        if self._stack is None:
            return
        try:
            self._stack.close()
        finally:
            self.cells[self._label] = dict(witness=self._witness.summary(),
                                           refusal=self._witness.refusal(),
                                           one_device_by_design=bool(self._witness.one_device_driver))
            self._stack = self._witness = self._label = None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: par_witness_arm.py --witness-json W.json -- <identity_break.py arguments>", file=sys.stderr)
        return 2
    cut = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--witness-json", required=True)
    a = ap.parse_args(argv[:cut])
    devices = parse_devices(os.environ.get("MOJOLEARN_PAR_DEVICES", ""))
    out = Path(a.witness_json)
    if out.exists():
        out.unlink()

    spec = importlib.util.spec_from_file_location("identity_break_par_witness", HARNESS)
    ib = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ib
    spec.loader.exec_module(ib)
    cells = CellWitnesses(devices)
    base = ib.CellTimer

    class WitnessedCellTimer(base):
        def __init__(self, label, *args, **kwargs):
            cells.open(label)
            super().__init__(label, *args, **kwargs)

    ib.CellTimer = WitnessedCellTimer
    sys.argv = [str(HARNESS)] + argv[cut + 1:]
    rc = 1
    try:
        rc = ib.main()
    finally:
        cells.close()
        try:
            out.write_text(json.dumps(dict(format="mojolearn.par-witness-arm.v1", vendor=cells.vendor,
                                           devices=list(devices), cells=cells.cells),
                                      indent=1, sort_keys=True))
        except OSError as exc:
            print(f"par_witness_arm: could not write {out}: {exc}", file=sys.stderr, flush=True)
            rc = 3
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
