#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The requirements ledger evaluated once per freeze and inputs, the gate boxes' rent check and the idle-box
watchdog (tools/release.py).

0.8.66 run h (2026-11-07): gate-nvidia and gate-amd spent 55+ minutes on the Mac with both rented boxes idle,
running the ledger's dependency rounds, the retained-producer reconstructions and a serial evaluation that
linux-builds had already run and the publish preflight and the record would run again. Now:

* EvaluationStore keeps one evaluation document per key under <release dir>/requirements-eval/<key>.json.
  The key (evaluation_key) is the source commit, the tooling commit, the requirements catalog, the sha256 of
  every candidate wheel known, the identity admission commit and the rest of the qualification configuration.
  A caller whose key matches, with an unchanged ledger history and unchanged retained producers, reads the
  document and computes nothing. A new key (a wheel after linux-pack or macos-build, the admission commit
  after identity-admit) is computed by the step that produced it, and only for the rows whose inputs changed:
  the document's rows seed release_requirements.EvaluationMemo, whose rows are keyed by their own digest.
* ready_to_rent lists every input the box steps need before gate-boxes rents.
* IdleBoxWatch reports and totals the minutes a held box runs no box job.
"""
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "mojolearn.release-requirements-eval.v1"
#: where a release keeps its evaluation documents, under its release dir
DIRNAME = "requirements-eval"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def admission_commit(commit, environ=None):
    """The identity admission commit MOJOLEARN_IDENTITY_REFERENCE_AT names for the freeze COMMIT, or None."""
    environ = os.environ if environ is None else environ
    for pair in (environ.get("MOJOLEARN_IDENTITY_REFERENCE_AT") or "").split(","):
        freeze, _, admitted = pair.partition(":")
        if freeze == commit and re.fullmatch(r"[0-9a-f]{40}", admitted or ""):
            return admitted
    return None


def wheel_hashes(configuration):
    """The sha256 of every candidate wheel the configuration knows (its artifacts:<platform> entries)."""
    return sorted({sha for key, wheels in configuration.items() if key.startswith("artifacts:")
                   and isinstance(wheels, dict) for sha in wheels.values() if isinstance(sha, str)})


def evaluation_key(commit, tooling_commit, catalog_sha256, configuration, admission=None, version=None):
    """(key, inputs): the key of one evaluation of the ledger. INPUTS are recorded in the document."""
    inputs = dict(source_commit=commit, tooling_commit=tooling_commit, catalog_sha256=catalog_sha256,
                  wheels=wheel_hashes(configuration), admission_commit=admission, version=version,
                  configuration_digest=_digest(configuration))
    return _digest(inputs)[:24], inputs


def history_digest(ledger_doc):
    """What a stored evaluation was computed against besides its key: the declared requirements' fingerprints
    and the ids of every attempt, run and interruption (the history is append-only)."""
    return _digest(dict(version=ledger_doc.get("version"),
                        requirements={k: (v or {}).get("fingerprint")
                                      for k, v in (ledger_doc.get("requirements") or {}).items()},
                        attempts=[a.get("id") for a in ledger_doc.get("attempts", [])],
                        runs=[r.get("id") for r in ledger_doc.get("runs", [])],
                        interruptions=[i.get("id") for i in ledger_doc.get("interruptions", [])]))


def producer_signature(producers):
    """The retained producers (release.retained_producers) by path, size and mtime: a changed or new
    producer artifact admits its evidence again; an unchanged one is not re-read."""
    out = []
    for name, paths in producers:
        row = {}
        for key, value in sorted(paths.items()):
            try:
                st = Path(value).stat()
                row[key] = [str(value), st.st_size, st.st_mtime_ns]
            except OSError:
                row[key] = [str(value), None, None]
        out.append([name, row])
    return out


class EvaluationStore:
    """<release dir>/requirements-eval/<key>.json, written atomically; the newest one seeds the memo."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def path(self, key):
        return self.directory / (key + ".json")

    def load(self, key):
        try:
            doc = json.loads(self.path(key).read_text())
        except (OSError, ValueError):
            return None
        return doc if doc.get("schema") == SCHEMA and doc.get("key") == key else None

    def latest(self):
        found = []
        for path in self.directory.glob("*.json"):
            try:
                found.append((path.stat().st_mtime_ns, path))
            except OSError:
                continue
        for _, path in sorted(found, reverse=True):
            doc = self.load(path.stem)
            if doc is not None:
                return doc
        return None

    def save(self, doc):
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.path(doc["key"])
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            tmp.replace(path)
        return path


def memo_rows(doc):
    """The EvaluationMemo rows of a stored document ({name: {digest, row}})."""
    if not doc:
        return {}
    status, digests = doc.get("status") or {}, doc.get("digests") or {}
    return {name: dict(digest=digests[name], row=row) for name, row in status.items() if name in digests}


def document(key, inputs, manifest, rows, digests, *, ledger_history, producers, stats, computed_by, notes=()):
    return dict(schema=SCHEMA, key=key, inputs=inputs, manifest=manifest, status=rows, digests=digests,
                ledger_history=ledger_history, producers=producers, stats=stats, computed_by=computed_by,
                notes=list(notes), updated_at=stamp())


# ------------------------------------------------------------ before the bill
def ready_to_rent(needs, *, wheels, payload, plan, ledger):
    """The inputs the box steps of NEEDS ({vendor: [why]}) lack, as one line each; [] when renting is safe.

    WHEELS(vendor), PAYLOAD(vendor), PLAN() and LEDGER() each return None when their input is there, or a
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
    check("ledger document", ledger)
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
