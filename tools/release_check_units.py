#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DEVICE CHECKS AS LANE-LIKE UNITS (Andrew 2026-11-05: "include autograd entries, framework adapters and the
install/load check IN the verification process as additional things; then if verify works we release").

tools/release.py --identity-on-wheels only. The 24 autograd entries (tools/autograd_smoke.py ENTRIES), the framework
adapter smoke (its three phases torch-compile, torch-training, max-training run as one process, so one unit) and the
install/load check (plugin and kernel-set load, PTX selection, the GPU language-model check, API coverage:
tools/qualify_verifier_wheel.py --no-identity-lanes) are UNITS with what an identity lane has:

  a record per device class   <store>/<device class>/<unit>/<key12>.json: status, the unit's key, and a digest of
                              its output where it prints numbers (autograd DIGEST / PASS lines, the adapter phase
                              results); a pass verdict where it only loads;
  a key                       what the unit exercises, by the existing closure tooling:
                                autograd compiled checks  the Mojo import closure of their source
                                                          (tools/compile_closures.py closure_of) + pixi.lock + the
                                                          builder and runner tools; they are built on the box from
                                                          source, so no wheel member is in the key;
                                autograd Python entries   the test file / probe runner + the package's Python import
                                                          closure (runtime_qualifier_units.product_closure, the walk
                                                          the runtime units and the identity lanes use) + the bytes
                                                          of every native member of the vendor's wheels;
                                adapters                  the adapter runner tools + the package's Python closure of
                                                          `mojolearn` + every native member's bytes;
                                install/load              no closure can be proved (it exercises the whole wheel): the
                                                          digest of every wheel member EXCEPT the reference table,
                                                          identity_columns/COMMIT, *.dist-info/* and _version.py, plus
                                                          the qualifier's own bytes, so a table- or metadata-only
                                                          change never forces it;
  the identity rule           the newest PASS record of the same device class and unit with the same key is
                              REUSED and the unit does not run; anything else runs, once.

    python3 tools/release_check_units.py plan --vendor nvidia --device-class nvidia-sm_89 --source-root SRC \\
        --wheel W ... --store DIR --json owed.json          (on the Mac, before any rental; runs nothing)
    python3 tools/release_check_units.py run --vendor nvidia --plan owed.json --python VENV_PY --wheel W ... \\
        --bin-dir BIN --out DIR                             (on the box, in the one box job)

Tests: tools/test_release_check_units.py (no device: the runners are faked).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

SCHEMA = "mojolearn.release-check-unit.v1"
PLAN_SCHEMA = "mojolearn.release-check-plan.v1"
SUMMARY = "check-units.json"
INSTALL_LOAD = "install-load"
ADAPTERS = "adapters"
NATIVE = (".so", ".dylib", ".metallib", ".ptx", ".hsaco", ".cubin", ".bin")
#: wheel members a table- or metadata-only change touches, never part of the install/load key
NOT_LOADED = ("mojolearn/verify_reference/table.json", "mojolearn/identity_columns/COMMIT", "mojolearn/_version.py")
ADAPTER_TOOLS = ("tools/adapter_gate.py", "tools/adapter_wheel_smoke.py", "tools/check_native_adapters.py")
AUTOGRAD_TOOLS = ("tools/autograd_smoke.py", "tools/gate_autograd_artifacts.py")
LOAD_TOOLS = ("tools/qualify_verifier_wheel.py",)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def safe(unit):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", unit)


def units():
    """Every unit name: install-load, adapters, autograd/<entry> (the registry's 24, in order)."""
    import autograd_smoke as AS
    return [INSTALL_LOAD, ADAPTERS] + ["autograd/" + e["name"] for e in AS.ENTRIES]


def entry(unit):
    import autograd_smoke as AS
    name = unit.split("/", 1)[1]
    return next(e for e in AS.ENTRIES if e["name"] == name)


def wheel_members(wheels):
    """{member: sha256} over WHEELS (members of several wheels never collide: core and plugins are disjoint)."""
    out = {}
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            for info in z.infolist():
                if not info.filename.endswith("/"):
                    out[info.filename] = sha256_bytes(z.read(info.filename))
    return out


def natives(members):
    return {m: d for m, d in members.items() if m.endswith(NATIVE)}


def files_of(root, rels):
    out = {}
    for rel in sorted(set(rels)):
        p = Path(root) / rel
        out[rel] = sha256_file(p) if p.is_file() else None
    return out


def python_closure(root, seeds_from):
    """The package's Python import closure of the mojolearn modules SEEDS_FROM (files) import, by the runtime
    units' walk over F's lane_select; None when it cannot be computed (then the unit simply runs)."""
    import runtime_qualifier_units as RQU
    try:
        import lane_select
        if Path(lane_select.ROOT).resolve() != Path(root).resolve():
            return None
        names = set()
        for f in seeds_from:
            found = RQU._package_imports(Path(root) / f)
            if found is None:
                return None
            names.update(found or ["mojolearn"])
        seeds = RQU.package_seeds(sorted(names), lane_select._python_files())
        return RQU.product_closure(seeds, lane_select)
    except Exception:   # an unknown closure is never a reason to reuse
        return None


def unit_key(unit, root, members):
    """{files, natives, members, closure_available}: what UNIT exercises (module docstring), and its digest."""
    key = dict(unit=unit, files={}, natives={}, members={}, closure_available=True)
    if unit == INSTALL_LOAD:
        key["files"] = files_of(root, LOAD_TOOLS)
        key["members"] = {m: d for m, d in members.items() if m not in NOT_LOADED and ".dist-info/" not in m}
    elif unit == ADAPTERS:
        closure = python_closure(root, ["tools/adapter_wheel_smoke.py"])
        key["closure_available"] = closure is not None
        key["files"] = files_of(root, list(ADAPTER_TOOLS) + list(closure or []))
        key["natives"] = natives(members)
    else:
        e = entry(unit)
        if "source" in e:
            import compile_closures
            try:
                mojo = compile_closures.closure_of(Path(root), [Path(root) / e["source"]])
                key["files"] = dict(sorted(mojo.items()))
            except Exception:
                key["closure_available"] = False
            key["files"].update(files_of(root, list(AUTOGRAD_TOOLS) + ["pixi.lock"]))
        else:
            test = e.get("pytest", "").split("::")[0]
            seeds = [test] if test else ["tools/autograd_smoke.py"]
            closure = python_closure(root, seeds)
            key["closure_available"] = closure is not None
            key["files"] = files_of(root, list(AUTOGRAD_TOOLS) + seeds + list(closure or []))
            key["natives"] = natives(members)
    key["digest"] = digest({k: key[k] for k in ("unit", "files", "natives", "members")})
    return key


def records(store, device, unit):
    d = Path(store) / device / safe(unit)
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return sorted(out, key=lambda r: r.get("at", ""), reverse=True)


def decide(key, history, device):
    """("reuse", record) or ("run", reason): the identity lanes' rule over the unit's own key."""
    if not key.get("closure_available"):
        return "run", "the unit's closure cannot be computed"
    why = []
    for rec in history:
        if rec.get("schema") != SCHEMA or rec.get("device_class") != device:
            continue
        if rec.get("status") != "passed":
            why.append("%s: %s" % (rec.get("at"), rec.get("status")))
            continue
        if (rec.get("key") or {}).get("digest") == key["digest"]:
            return "reuse", rec
        old = rec.get("key") or {}
        changed = sorted(k for part in ("files", "natives", "members")
                         for k in set(old.get(part) or {}) | set(key.get(part) or {})
                         if (old.get(part) or {}).get(k) != (key.get(part) or {}).get(k))
        why.append("changed since %s: %s" % (rec.get("at"), ", ".join(changed[:5])))
        break   # the newest passing record decides, as for a lane
    return "run", "; ".join(why) or "no record for this device class"


def plan(vendor, device, root, wheels, store):
    members = wheel_members(wheels)
    rows = []
    for unit in units():
        if vendor == "cpu" and unit.startswith("autograd/") and "source" in entry(unit):
            continue   # compiled device-check executables: GPU only
        key = unit_key(unit, root, members)
        decision, detail = decide(key, records(store, device, unit), device)
        rows.append(dict(unit=unit, decision=decision, key=key,
                         **({"reused": detail.get("at"), "record_digest": detail.get("digest")} if decision == "reuse"
                            else {"reason": detail})))
    owed = [r["unit"] for r in rows if r["decision"] == "run"]
    return dict(schema=PLAN_SCHEMA, vendor=vendor, device_class=device, units=rows, owed=owed,
                reused=len(rows) - len(owed))


# ---------------------------------------------------------------- running the owed units (on the box)
def output_digest(log):
    """A digest of the deterministic numeric lines a check printed (DIGEST / PASS lines), or None."""
    try:
        lines = [l.strip() for l in Path(log).read_text(errors="replace").splitlines()
                 if re.search(r"DIGEST|^PASS\b", l.strip())]
    except OSError:
        return None
    return sha256_bytes("\n".join(sorted(lines)).encode()) if lines else None


class Runner:
    def __init__(self, a, run=subprocess.run):
        self.a, self.run = a, run
        self.out = Path(a.out)

    def record(self, row, status, digest_=None, detail=""):
        rec = dict(schema=SCHEMA, unit=row["unit"], device_class=self.plan["device_class"], vendor=self.plan["vendor"],
                   status=status, digest=digest_, detail=detail[:400], key=row["key"],
                   source_commit=self.a.source_commit, at=self.a.at)
        d = self.out / "records" / self.plan["device_class"] / safe(row["unit"])
        d.mkdir(parents=True, exist_ok=True)
        (d / (row["key"]["digest"][:12] + ".json")).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        return rec

    def install_load(self, row):
        out = self.out / "install-load"
        cmd = [sys.executable, str(TOOLS / "qualify_verifier_wheel.py"), "--installed-python", self.a.python,
               "--expected-version", self.a.version, "--expected-source-commit", self.a.source_commit,
               "--scope", "cpu-only" if self.plan["vendor"] == "cpu" else "expanded", "--no-identity-lanes",
               "--output", str(out)]
        p = self.run(cmd, capture_output=True, text=True)
        doc = {}
        try:
            doc = json.loads((out / "results.json").read_text())
        except (OSError, ValueError):
            pass
        ok = p.returncode == 0 and doc.get("status") in ("PASSED", "PASSED_CPU_ONLY")
        return self.record(row, "passed" if ok else "failed", None, doc.get("reason", "") if not ok else "")

    def autograd(self, rows):
        if not rows:
            return []
        out = self.out / "autograd" / "autograd-smoke.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        names = [r["unit"].split("/", 1)[1] for r in rows]
        self.run([sys.executable, str(TOOLS / "autograd_smoke.py"), "run", "--vendor", self.plan["vendor"],
                  "--bin-dir", self.a.bin_dir, "--python", self.a.python, "--out", str(out),
                  "--entries", ",".join(names)], capture_output=True, text=True)
        try:
            got = {r["name"]: r for r in json.loads(out.read_text()).get("rows", [])}
        except (OSError, ValueError, AttributeError):
            got = {}
        recs = []
        for row in rows:
            r = got.get(row["unit"].split("/", 1)[1]) or {}
            ok = r.get("status") == "ok" and r.get("rc") == 0
            recs.append(self.record(row, "passed" if ok else "failed", output_digest(r.get("log", "")) if ok else None,
                                    "" if ok else "status %s rc %s" % (r.get("status"), r.get("rc"))))
        return recs

    def adapters(self, row):
        import adapter_gate as AG
        out = self.out / "adapters"
        rec = AG.run(self.plan["vendor"], sys.executable, out, self.a.wheel)
        ok = bool(rec.get("ok"))
        phases = {}
        for name in AG.PHASES:
            for p in sorted(out.glob(name + "*/*.json")) + sorted(out.glob(name + ".json")):
                try:
                    phases[p.name] = json.loads(p.read_text())
                except ValueError:
                    pass
        return self.record(row, "passed" if ok else "failed", digest(phases) if ok and phases else None,
                           "" if ok else str(rec.get("why", "")))

    def main(self):
        self.plan = json.loads(Path(self.a.plan).read_text())
        if self.plan.get("schema") != PLAN_SCHEMA:
            raise SystemExit("release_check_units: not a check plan")
        self.out.mkdir(parents=True, exist_ok=True)
        owed = [r for r in self.plan["units"] if r["decision"] == "run"]
        print("check-units: %s: REUSED %d / RUN %d" % (self.plan["device_class"], self.plan["reused"], len(owed)),
              flush=True)
        results = {}
        for row in owed:
            if row["unit"] == INSTALL_LOAD:
                results[row["unit"]] = self.install_load(row)
        for rec in self.autograd([r for r in owed if r["unit"].startswith("autograd/")]):
            results[rec["unit"]] = rec
        for row in owed:
            if row["unit"] == ADAPTERS:
                results[row["unit"]] = self.adapters(row)
        summary = dict(schema=SCHEMA + "+summary", device_class=self.plan["device_class"], vendor=self.plan["vendor"],
                       source_commit=self.a.source_commit, plan=self.plan,
                       units={r["unit"]: dict(decision=r["decision"], status=(results.get(r["unit"]) or {}).get(
                           "status", "reused" if r["decision"] == "reuse" else "not run")) for r in self.plan["units"]})
        bad = sorted(u for u, s in summary["units"].items() if s["status"] not in ("passed", "reused"))
        summary["ok"] = not bad
        (self.out / SUMMARY).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        print("check-units: %s%s" % ("PASSED" if not bad else "FAILED ", ", ".join(bad[:8])), flush=True)
        return 0 if not bad else 1


def summary_errors(doc, plan_doc=None):
    """Why a check-units.json does not prove every unit passed or was reused, or []."""
    if not doc or doc.get("schema") != SCHEMA + "+summary":
        return ["no check-units summary"]
    bad = sorted(u for u, s in (doc.get("units") or {}).items() if s.get("status") not in ("passed", "reused"))
    if plan_doc is not None and set((doc.get("units") or {})) != {r["unit"] for r in plan_doc.get("units", [])}:
        bad.append("the summary covers other units than the plan")
    return ["unit %s did not pass" % u if not u.startswith("the ") else u for u in bad]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--vendor", required=True, choices=("nvidia", "amd", "apple", "cpu"))
    p.add_argument("--device-class", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--wheel", action="append", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--json", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--python", required=True, help="the venv interpreter the wheels are installed in")
    r.add_argument("--wheel", action="append", required=True)
    r.add_argument("--bin-dir", default="")
    r.add_argument("--version", required=True)
    r.add_argument("--source-commit", required=True)
    r.add_argument("--at", default="")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        doc = plan(a.vendor, a.device_class, a.source_root, a.wheel, a.store)
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print("check-units plan %s: REUSED %d / RUN %d" % (a.device_class, doc["reused"], len(doc["owed"])))
        return 0
    if not a.at:
        import datetime as dt
        a.at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return Runner(a).main()


if __name__ == "__main__":
    sys.exit(main())
