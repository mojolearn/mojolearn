#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The gate boxes' rent check and the idle-box watchdog (tools/release.py).

0.8.66 run h (2026-11-07): gate-nvidia and gate-amd spent 55+ minutes on the Mac with both rented boxes idle,
running the requirements ledger's reuse analysis before any job was submitted. The ledger is now off by default
(Release.ledger_on), the box steps only submit, and:

* ready_to_rent lists every input the box steps need before gate-boxes rents, so a missing one stops the
  release before the bill;
* IdleBoxWatch reports and totals the minutes a held box runs no box job.
"""
import os
import re
import threading
from pathlib import Path


# ------------------------------------------------------------ before the bill
def ready_to_rent(needs, *, wheels, payload, plan):
    """The inputs the box steps of NEEDS ({vendor: [why]}) lack, as one line each; [] when renting is safe.

    WHEELS(vendor), PAYLOAD(vendor) and PLAN() each return None when their input is there, or a
    line naming what is missing; an exception counts as missing with its message."""
    missing = []

    def check(label, probe):
        try:
            problem = probe()
        except Exception as error:  # noqa: BLE001 - whatever stops the input stops the rental
            problem = f"{type(error).__name__}: {error}"
        if problem:
            missing.append(f"{label}: {problem}")

    check("plan", plan)
    for vendor in sorted(needs):
        check(f"{vendor} wheels", lambda: wheels(vendor))
        check(f"{vendor} payload", lambda: payload(vendor))
    return missing


# ------------------------------------------------------------ the idle-box watchdog
class IdleBoxWatch:
    """While a box is held and runs no box job, print `IDLE BOX <name> <minutes> min (~$<cost>)` every
    REPORT_EVERY_S and total the idle minutes and their cost into the release state (idle_box_minutes,
    idle_box_cost, idle_boxes per vendor). CLOCK is the release's clock (seconds); RATE(vendor) its hourly
    price or None when unknown."""

    REPORT_EVERY_S = 120

    def __init__(self, *, say, clock, rate, state, save):
        self.say, self.clock, self.rate, self.state, self.save = say, clock, rate, state, save
        self._streak = {}     # vendor -> start of the current idle streak
        self._last = {}       # vendor -> previous tick
        self._reported = {}   # vendor -> last IDLE BOX line
        self._lock = threading.Lock()

    def tick(self, held, busy):
        """HELD: {vendor: box record}; BUSY: the vendors with a box job running now. Returns the lines said."""
        now = float(self.clock())
        lines = []
        with self._lock:
            totals = self.state.setdefault("idle_boxes", {})
            for vendor in list(self._last):
                if vendor not in held:
                    for table in (self._streak, self._last, self._reported):
                        table.pop(vendor, None)
            for vendor, rec in sorted(held.items()):
                last = self._last.get(vendor)
                self._last[vendor] = now
                if vendor in busy:
                    self._streak.pop(vendor, None)
                    self._reported.pop(vendor, None)
                    continue
                start = self._streak.setdefault(vendor, now if last is None else last)
                if last is not None:
                    seconds = max(0.0, now - max(last, start))
                    row = totals.setdefault(vendor, dict(minutes=0.0, cost=0.0))
                    row["minutes"] = round(row["minutes"] + seconds / 60.0, 3)
                    rate = self.rate(vendor)
                    if rate is not None:
                        row["cost"] = round(row["cost"] + rate * seconds / 3600.0, 4)
                    else:
                        row["rate_unknown"] = True
                if now - self._reported.get(vendor, start) >= self.REPORT_EVERY_S:
                    self._reported[vendor] = now
                    minutes = (now - start) / 60.0
                    rate = self.rate(vendor)
                    cost = f"{rate * minutes / 60.0:.2f}" if rate is not None else "?"
                    name = rec.get("pod") or rec.get("droplet") or vendor
                    lines.append(f"IDLE BOX {vendor} {name} {minutes:.0f} min (~${cost})")
            self.state["idle_box_minutes"] = round(sum(r["minutes"] for r in totals.values()), 1)
            self.state["idle_box_cost"] = round(sum(r["cost"] for r in totals.values()), 2)
        for line in lines:
            self.say("  " + line)
        self.save()
        return lines


def summary_line(state):
    """The SUMMARY's idle-box line, or None when no box sat idle."""
    minutes = state.get("idle_box_minutes") or 0
    if not minutes:
        return None
    parts = [f"{v} {r['minutes']:.0f} min" + (f" ~${r['cost']:.2f}" if not r.get("rate_unknown") else " (rate unknown)")
             for v, r in sorted((state.get("idle_boxes") or {}).items())]
    return f"idle boxes: {minutes:.0f} min, ~${state.get('idle_box_cost') or 0:.2f} ({'; '.join(parts)})"


def hourly_rate(vendor, state_env, environ=None):
    """The box's hourly price: MOJOLEARN_BOX_HOURLY_<VENDOR>, else the provider state file's COST_HR (RunPod,
    tools/dev_pod.sh) or PRICE (DigitalOcean, tools/do_amd_steward.sh); None when neither is known."""
    environ = os.environ if environ is None else environ
    value = environ.get("MOJOLEARN_BOX_HOURLY_" + vendor.upper())
    if value is None and state_env is not None:
        try:
            text = Path(state_env).read_text()
        except OSError:
            text = ""
        m = re.search(r"^(?:COST_HR|PRICE)=['\"]?([0-9.]+)", text, re.M)
        value = m.group(1) if m else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
