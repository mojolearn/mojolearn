"""Installed-wheel framework adapter smoke as ledger evidence for adapters-<vendor>.

Used by tools/release_ledger_admission.py; never installs or runs anything. The
actual producer is tools/adapter_wheel_smoke.py, launched by the release gate
(tools/gate_smoke_box.py on NVIDIA/AMD, release.py gate-apple on the Mac). Each
declared coverage ID needs its own complete checker record:

  torch-inference    torch-compile phase:  torch-compile/adapter-results.json
  torch-training     torch-training phase: torch-training/training-results.json
  max-train          max-training phase:   max-training/max-training-results.json
  max-train-resume   the same MAX record's ``<case>: resume exact`` checks

The gate's own adapter judgment, every phase exit code, framework state, the
installed wheel hashes and the checker records must all agree. The counts are
the checker's actual check records, not a fixed phase count.
"""
from pathlib import Path
import re
import zipfile

from release_ledger_admission import require, load, reference, checked

TOOL = "tools/release_adapter_evidence.py"
COVERAGE = ("torch-inference", "torch-training", "max-train", "max-train-resume")
RESULTS = {"torch-compile": "torch-compile/adapter-results.json",
           "torch-training": "torch-training/training-results.json",
           "max-training": "max-training/max-training-results.json"}
RESUME = ": resume exact"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def supports_requirement(wanted):
    vendor = str(wanted.get("inputs", {}).get("contract", {}).get("id", "")).rpartition("-")[2]
    return (wanted.get("kind") == "adapters" and vendor in ("nvidia", "amd", "apple")
            and wanted.get("inputs", {}).get("contract", {}).get("id") == "adapters-" + vendor
            and wanted.get("coverage") == list(COVERAGE) and not wanted.get("missing_inputs"))


def _smoke(directory, vendor, version):
    """Re-judge one adapter smoke directory from its own records."""
    import adapter_gate
    directory = Path(directory).resolve()
    smoke_ref = reference(directory / adapter_gate.RECEIPT)
    smoke = checked(smoke_ref)
    status, ok, why = adapter_gate.judge(smoke)
    require(ok and status == "PASSED", "adapter smoke did not pass: " + why)
    require(smoke.get("device") == adapter_gate.DEVICE[vendor] and smoke.get("problems") == []
            and smoke.get("frameworks") == {"torch": "ran", "max": "ran"},
            "adapter smoke did not run both frameworks on the vendor device")
    installed = smoke.get("installed_after_install") or smoke.get("installed") or {}
    versions = {installed.get(name) for name in ("mojolearn", "mojolearn-core") if installed.get(name)}
    require(versions == {version}, "adapter smoke installed another mojolearn version")
    refs, records, phases = [smoke_ref], {}, {}
    for phase, relative in RESULTS.items():
        record = smoke["phases"].get(phase) or {}
        require(type(record.get("exit")) is int and record["exit"] == 0 and record.get("timed_out") is not True,
                "adapter phase did not complete: " + phase)
        log = reference(directory / (phase + ".log"))
        result_ref = reference(directory / relative)
        result = checked(result_ref)
        checks = result.get("checks")
        require(result.get("complete") is True and result.get("failures") == []
                and isinstance(checks, list) and checks and all(isinstance(c, str) and c for c in checks)
                and len(set(checks)) == len(checks), "adapter checker record is incomplete: " + phase)
        refs += [log, result_ref]
        records[phase] = checks
        phases[phase] = record
    resume = [c for c in records["max-training"] if c.endswith(RESUME)]
    counts = {"torch-inference": len(records["torch-compile"]), "torch-training": len(records["torch-training"]),
              "max-train": len(records["max-training"]) - len(resume), "max-train-resume": len(resume)}
    require(all(count > 0 for count in counts.values()), "an adapter coverage ID has no actual check record")
    return dict(smoke=smoke_ref, phases=phases, counts=counts, evidence=refs)


# ---------------------------------------------------------------- NVIDIA / AMD release gate
def _gate_facts(root, producer):
    import adapter_gate
    import release_reuse
    directory = Path(producer["gate"]).resolve()
    gate_ref = reference(directory / "gate-smoke.json")
    gate = checked(gate_ref)
    identity = gate.get("candidate") or {}
    vendor, version = gate.get("vendor"), gate.get("version")
    revision = identity.get("source_commit")
    require(vendor in ("nvidia", "amd") and gate.get("input_mode") == "candidate"
            and identity.get("vendor") == vendor and identity.get("version") == version
            and isinstance(identity.get("wheels"), dict) and identity["wheels"]
            and all(HEX64.fullmatch(str(v)) for v in identity["wheels"].values())
            and isinstance(revision, str) and HEX40.fullmatch(revision),
            "gate report does not identify exact candidate wheels and source")
    record = gate.get("adapters") or {}
    require(gate.get("adapters_enabled") is True and record.get("ok") is True and record.get("status") == "PASSED"
            and type(record.get("rc")) is int and record["rc"] == 0
            and record.get("phases") == {phase: 0 for phase in adapter_gate.PHASES}
            and type(gate.get("adapters_finished")) in (int, float) and gate["adapters_finished"] > 0,
            "gate adapter stage did not pass every installed-wheel phase")
    require(Path(str(record.get("report", ""))).parts[-2:] == ("adapter-smoke", adapter_gate.RECEIPT),
            "gate adapter record names another receipt")
    smoke = _smoke(directory / "adapter-smoke", vendor, version)
    require(record.get("phase_records", {}) and all(record["phase_records"].get(p) == smoke["phases"][p]
                                                    for p in adapter_gate.PHASES),
            "gate adapter record differs from the retained smoke receipt")
    return dict(producer="gate-adapters", source_root=str(Path(root).resolve()), source_revision=revision,
                vendor=vendor, version=version, gate=gate_ref, wheels=dict(identity["wheels"]),
                release_modes=list(release_reuse.record_modes(identity)), counts=smoke["counts"],
                evidence=[gate_ref] + smoke["evidence"])


# ---------------------------------------------------------------- Apple gate-apple
def _apple_facts(root, producer):
    directory = Path(producer["adapters"]).resolve()
    wheel = Path(producer["wheel"]).resolve()
    stamp_ref = reference(directory / "wheel.json")
    stamp = checked(stamp_ref)
    wheel_ref = reference(wheel)
    require(stamp.get("wheel") == wheel.name and stamp.get("wheel_sha256") == wheel_ref["sha256"]
            and "macosx" in wheel.name, "adapter smoke receipt belongs to another macOS wheel")
    with zipfile.ZipFile(wheel) as archive:
        revision = archive.read("mojolearn/identity_columns/COMMIT").decode().strip()
        version = next((n.split("-")[1] for n in archive.namelist()
                        if n.endswith(".dist-info/METADATA") and n.startswith("mojolearn-")), None)
    require(HEX40.fullmatch(revision) and version and wheel.name.split("-")[1] == version,
            "macOS wheel lacks its packaged source commit/version")
    smoke = _smoke(directory, "apple", version)
    return dict(producer="apple-gate-adapters", source_root=str(Path(root).resolve()), source_revision=revision,
                vendor="apple", version=version, wheels={wheel.name: wheel_ref["sha256"]},
                counts=smoke["counts"], evidence=[stamp_ref, wheel_ref] + smoke["evidence"])


def _collect(facts_of):
    def collect(root, document, producer):
        facts = facts_of(root, producer)
        vendor = facts["vendor"]
        declared = ["build_flags"]
        configuration = dict(document.get("configuration", {}))
        require("build_flags" in configuration, "ledger configuration lacks build_flags")
        modes = facts.get("release_modes")
        if modes is None:
            # The macOS wheel records no mode list; the ledger's declared modes apply.
            modes = configuration.get("release_modes")
            declared.append("release_modes")
        require(modes == ["identical"], "only the identical release is admitted")
        return dict(source_revision=facts["source_revision"], facts=facts,
                    units={"adapters-" + vendor: dict(vendor=vendor)}, spec_ids=["adapters-" + vendor],
                    declared=declared,
                    configuration={"release_modes": modes, "build_flags": configuration["build_flags"],
                                   "artifacts:" + vendor: facts["wheels"]})
    return collect


def _check(requirement_id, wanted, facts, unit):
    vendor = facts["vendor"]
    require(supports_requirement(wanted) and requirement_id == "adapters-" + vendor == "adapters-" + unit["vendor"],
            "adapter evidence belongs to another vendor")
    configuration = wanted["inputs"].get("configuration", {})
    require(configuration.get("artifacts:" + vendor) == facts["wheels"],
            "installed adapter wheels differ from the requirement artifacts")
    if "release_modes" in facts:
        require(configuration.get("release_modes") == facts["release_modes"], "release modes differ")
    rows = [dict(id=name, status="passed", exit_code=0, expected=facts["counts"][name],
                 completed=facts["counts"][name], failed=0, skipped=0) for name in wanted["coverage"]]
    return rows, list(facts["evidence"])


class _Adapter:
    def __init__(self, facts_of):
        self._facts, self.collect, self.check_unit = facts_of, _collect(facts_of), _check

    def verify(self, report):
        return self._facts(report["facts"]["source_root"], report["producer"])


ADAPTERS = {"gate-adapters": _Adapter(_gate_facts), "apple-gate-adapters": _Adapter(_apple_facts)}
