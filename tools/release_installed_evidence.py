"""Installed-wheel runtime, multi-GPU and published-install reports as ledger evidence.

Used by tools/release_ledger_admission.py; never installs or runs anything. Each
adapter re-derives its producer's report from the retained raw files (junit,
verifier captures, driver samples, manifests, checker records) and admits only a
complete pass whose wheels are the ledger's qualified candidate bytes:

  installed-runtime  runtime-<vendor>   tools/release_runtime_installed.py   --producer report=DIR/runtime-report.json
  multi-gpu          multi-gpu-<vendor> tools/release_multi_gpu_parallel.py  --producer report=DIR/multi-gpu-report.json
  published-install  published-<vendor> tools/release_published_install.py  --producer report=DIR/published-report.json
"""
from pathlib import Path

from release_ledger_admission import require, load, reference, checked

import release_multi_gpu_parallel as multi
import release_published_install as published
import release_runtime_installed as runtime

ROW_KEYS = ("id", "status", "exit_code", "expected", "completed", "failed", "skipped")
FAMILIES = {
    # kind: (requirement prefix, vendors, coverage IDs, producer module, adapter name)
    "runtime": ("runtime-", ("nvidia", "amd", "apple", "cpu"), runtime.COVERAGE_IDS, runtime, "installed-runtime"),
    "multi": ("multi-gpu-", ("nvidia", "amd"), multi.COVERAGE_IDS, multi, "multi-gpu"),
    "published": ("published-", ("nvidia", "amd", "apple"), published.COVERAGE_IDS, published, "published-install"),
}


def supports_requirement(wanted):
    family = FAMILIES.get(wanted.get("kind"))
    if family is None:
        return False
    prefix, vendors, coverage, _, _ = family
    name = str(wanted.get("inputs", {}).get("contract", {}).get("id", ""))
    return (name.startswith(prefix) and name[len(prefix):] in vendors and wanted.get("required", True) is not False
            and wanted.get("coverage") == list(coverage))


def _rederive(kind, out):
    """The producer's own report function over the retained directory."""
    if kind == "runtime":
        return runtime.summarize(out)
    if kind == "multi":
        return multi.judge(out)
    return published.judge(out)


def _facts(kind):
    prefix, vendors, coverage, module, name = FAMILIES[kind]

    def facts_of(root, producer):
        report_ref = reference(producer["report"])
        report = checked(report_ref)
        out = Path(producer["report"]).resolve().parent
        require(report.get("schema") == module.REPORT_SCHEMA and report.get("tool") == "tools/" + module.__name__ + ".py",
                "not a " + name + " producer report")
        fresh = _rederive(kind, out)
        stored = {k: v for k, v in report.items() if k not in ("qualifier_exit",)}
        require(fresh == stored, "retained " + name + " report differs from its raw producer files")
        require(report.get("status") == "passed" and report.get("exit_code") == 0
                and (kind != "runtime" or report.get("qualifier_exit") == 0),
                name + " producer did not pass")
        vendor = report.get("vendor")
        require(vendor in vendors, name + " report names another vendor")
        rows = [{k: row[k] for k in ROW_KEYS} for row in report["coverage"]]
        require([row["id"] for row in rows] == list(coverage), name + " coverage IDs differ from the declaration")
        require(all(row["status"] == "passed" and row["expected"] > 0 and row["completed"] == row["expected"]
                    and not row["failed"] and not row["skipped"] for row in rows),
                name + " coverage is incomplete")
        revision = report.get("source_revision")
        require(isinstance(revision, str) and len(revision) == 40, name + " report has no full source commit")
        wheels = report.get("published") if kind == "published" else report.get("wheels")
        require(isinstance(wheels, dict) and wheels, name + " report names no wheels")
        evidence = [report_ref]
        for ref in report["evidence"]:
            relative = Path(ref["path"])
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "producer evidence must lie inside its retained directory")
            local = reference(out / relative)
            require(local["sha256"] == ref["sha256"], "retained producer evidence digest differs: " + ref["path"])
            if local not in evidence:
                evidence.append(local)
        return dict(producer=name, source_root=str(Path(root).resolve()), source_revision=revision,
                    **({"test_revision": report.get("test_revision", revision)} if kind == "runtime" else {}),
                    vendor=vendor, wheels=dict(wheels), version=report.get("version"),
                    coverage=rows, evidence=evidence)
    return facts_of


def _qualified(configuration, wheels):
    """Every installed wheel is a qualified candidate artifact of this release."""
    mappings = [mapping for key, mapping in configuration.items()
                if key.startswith("artifacts:") and isinstance(mapping, dict)]
    return all(any(mapping.get(name) == digest for mapping in mappings) for name, digest in wheels.items())


def _collect(kind):
    prefix, _, _, _, name = FAMILIES[kind]
    facts_of = _facts(kind)

    def collect(root, document, producer):
        facts = facts_of(root, producer)
        vendor = facts["vendor"]
        configuration = dict(document.get("configuration", {}))
        require("build_flags" in configuration, "ledger configuration lacks build_flags")
        require(configuration.get("release_modes") == ["identical"], "only the identical release is admitted")
        reconstruction = {"release_modes": ["identical"], "build_flags": configuration["build_flags"]}
        declared = ["release_modes", "build_flags"]
        if kind == "runtime":
            test_revision = facts["test_revision"]
            require(isinstance(test_revision, str) and len(test_revision) == 40,
                    "runtime test revision must be a full committed revision")
            configured_tests = configuration.get("runtime_test_source_revision")
            require((configured_tests is None and test_revision == facts["source_revision"])
                    or configured_tests == test_revision,
                    "runtime test source override needs the exact ledger configuration")
            if configured_tests is not None:
                reconstruction["runtime_test_source_revision"] = configured_tests
                declared.append("runtime_test_source_revision")
        if kind == "published":
            artifacts = configuration.get("artifacts:" + vendor)
            require(isinstance(artifacts, dict) and artifacts, "ledger configuration lacks artifacts:" + vendor)
            require(facts["version"] == document.get("version"), "published report is another release version")
            reconstruction["artifacts:" + vendor] = artifacts
            declared.append("artifacts:" + vendor)
        else:
            require(_qualified(configuration, facts["wheels"]),
                    name + " ran wheels that are not the ledger's qualified candidate artifacts")
        unit = prefix + vendor
        return dict(source_revision=facts["source_revision"], facts=facts, units={unit: dict(vendor=vendor)},
                    spec_ids=[unit], declared=declared, configuration=reconstruction)
    return collect


def _check(kind):
    prefix = FAMILIES[kind][0]

    def check_unit(requirement_id, wanted, facts, unit):
        vendor = facts["vendor"]
        require(supports_requirement(wanted) and requirement_id == prefix + vendor == prefix + unit["vendor"],
                "installed evidence belongs to another requirement or vendor")
        configuration = wanted["inputs"].get("configuration", {})
        require(configuration.get("release_modes") == ["identical"], "release modes differ")
        if kind == "runtime":
            expected_tests = configuration.get("runtime_test_source_revision") or facts["source_revision"]
            require(facts["test_revision"] == expected_tests, "runtime test revision differs from the requirement")
        if kind == "published":
            artifacts = configuration.get("artifacts:" + vendor) or {}
            require(all(artifacts.get(file) == digest for file, digest in facts["wheels"].items()),
                    "published bytes differ from the requirement's qualified artifacts")
        return [dict(row) for row in facts["coverage"]], list(facts["evidence"])
    return check_unit


class _Adapter:
    def __init__(self, kind):
        self._facts, self.collect, self.check_unit = _facts(kind), _collect(kind), _check(kind)

    def verify(self, report):
        return self._facts(report["facts"]["source_root"], report["producer"])


ADAPTERS = {family[4]: _Adapter(kind) for kind, family in FAMILIES.items()}
