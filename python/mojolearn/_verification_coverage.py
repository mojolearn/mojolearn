"""Installed-package verification scope; inspection never executes a lane."""
import hashlib
from pathlib import Path

from . import host_surface
from . import _verify_reference as vref
from ._verification_catalog import ENTRIES, PROVENANCE
from ._verification_ctr_models import MODEL_SHA256 as CTR_MODELS
from ._verification_evidence_data import DATA as HISTORICAL_EVIDENCE


def declaration(spec):
    if callable(spec):
        return dict(kind="check", reason=None)
    if isinstance(spec, str) and spec.startswith("n/a:"):
        return dict(kind="not_applicable", reason=spec)
    return dict(kind="undeclared", reason="no explicit verification contract")


def inventory(harness, table, vendor_class):
    public = set(host_surface.public_reference_lanes())
    covered = set(host_surface.covered_lanes()) | set(host_surface.PUBLIC_HOST_ONLY_LANES)
    pending = host_surface.PUBLIC_PENDING_LANES
    candidates = set(host_surface.PUBLIC_REFERENCE_CANDIDATES)
    stale = set(vref.stale_reference_lanes(table, harness))
    lanes = {}
    for name in harness.LANES:
        status, reason = "available", None
        if vendor_class == "cpu" and name not in public:
            if name.startswith(host_surface.PUBLIC_EXCLUDED_PREFIXES):
                status, reason = "excluded", "parallel driver excluded from the public CPU verifier"
            elif name in covered:
                status = "withheld"
                reason = pending.get(name, "reference qualification pending" if name in candidates
                                     else "CPU route not admitted to the public verifier")
            else:
                status, reason = "unavailable", "no declared public CPU verification route"
        if status == "available" and name in stale:
            status, reason = "withheld", "stale reference"
        properties = {"batch": declaration(getattr(harness, "BATCH", {}).get(name))}
        for part, (specs, default, *_rest) in getattr(harness, "EXTRA_PARTS", {}).items():
            properties[part] = declaration(specs.get(name, default))
            properties[part]["run_by_default"] = part in vref.PARTS
            properties[part]["command"] = "verify --all" if part in vref.PARTS else "verify --batch-checks"
        properties["batch"]["run_by_default"] = True
        properties["rlpair"] = declaration(getattr(harness, "RLPAIR", {}).get(name, "n/a:no-sampler-trainer-pair"))
        properties["rlpair"].update(run_by_default=False, command="verify --batch-checks")
        refs = {}
        for part in dict.fromkeys((*vref.PARTS, *properties)):
            refs[part] = sum(bool((entry := vref.entry(table, name, fixture, part))
                                 and entry.get("ref") is not None and not entry.get("conflict"))
                             for fixture in harness.FIXTURES)
        lanes[name] = dict(status=status, reason=reason, properties=properties,
                           reference_fixtures=refs, fixtures=len(harness.FIXTURES))
    for name, lane in lanes.items():
        evidence = HISTORICAL_EVIDENCE['lanes'].get(name, {})
        lane['historical_evidence'] = evidence
        lane['multi_gpu_scope'] = ('parallel_lane' if name.startswith('par-')
                                   else 'no_parallel_lane_declared_here')
        lane['release_qualified'] = False
    harness_path = getattr(harness, '__file__', None)
    snapshot_matches = bool(harness_path and hashlib.sha256(Path(harness_path).read_bytes()).hexdigest()
                            == HISTORICAL_EVIDENCE['harness_sha256'])
    entries = []
    mapped = set()
    for entry in ENTRIES:
        row = dict(entry)
        row["lane_status"] = {name: lanes.get(name, dict(status="missing_lane", reason="not registered"))
                              for name in entry["lanes"]}
        row["verification"] = "mapped" if entry["lanes"] else "portable_models_and_dedicated_gate"
        mapped.update(entry["lanes"])
        relevant = [lanes[n]['historical_evidence'] for n in entry['lanes'] if n in lanes]
        row['evidence_summary'] = dict(
            release_qualified=False,
            build_negative_control_for_each_mapped_lane=bool(relevant) and len(relevant) == len(entry['lanes']) and all(
                all(any(c['kind'] == 'build' and c['part'] == part
                        for c in e.get('negative_controls', [])) for part in entry['parts'])
                for e in relevant),
            backend_records_for_each_mapped_lane=[vendor for vendor in ('cpu', 'apple', 'nvidia', 'amd')
                if relevant and all(e.get('backend_records', {}).get(vendor) for e in relevant)],
            multi_gpu_lanes=[n for n in entry['lanes'] if n.startswith('par-')])
        entries.append(row)
    return dict(format="mojolearn.verification-coverage.v1", execution="not run",
                scope="registered harness and the 246-entry MLSys appendix; mapping is not certification",
                vendor_class=vendor_class, provenance=PROVENANCE, entries=entries, lanes=lanes,
                evidence_provenance={k: v for k, v in HISTORICAL_EVIDENCE.items() if k != 'lanes'},
                evidence_snapshot_matches_harness=snapshot_matches,
                declared_ctr_model_sha256=CTR_MODELS,
                additional_lanes=sorted(set(lanes) - mapped),
                counts=dict(appendix_entries=len(entries), registered_lanes=len(lanes),
                            **{status: sum(r["status"] == status for r in lanes.values())
                               for status in ("available", "withheld", "excluded", "unavailable")}),
                limitations=["References describe recorded fixtures, not all possible inputs or hardware.",
                             "A single-device parallel driver run does not certify multiple GPUs.",
                             "Gradient, batch-size, ragged and sampler/replay checks require --batch-checks; they are not implicit in --all.",
                             "The appendix's seasonal-difference selection label means select_d with caller-supplied D."])


def format_human(report):
    c = report["counts"]
    lines = ["Verification coverage inventory — no algorithms executed",
             f"{c['appendix_entries']} appendix entries; {c['registered_lanes']} registered lanes; "
             f"backend {report['vendor_class']}",
             f"{c['available']} available, {c['withheld']} withheld, {c['excluded']} excluded, "
             f"{c['unavailable']} unavailable", "",
             "Lane | status | batch check | reason", "--- | --- | --- | ---"]
    for name, lane in report["lanes"].items():
        batch = lane["properties"]["batch"]
        lines.append(f"{name} | {lane['status']} | {batch['kind']} | "
                     f"{lane['reason'] or batch['reason'] or ''}")
    strong = sum(e['evidence_summary']['build_negative_control_for_each_mapped_lane'] for e in report['entries'])
    lines += ["", f"Historical build negative controls cover every mapped lane/part for {strong} of {c['appendix_entries']} appendix entries.",
              "This is historical evidence on listed fixtures, not qualification of this wheel.",
              "Use --json for sabotage pairs, backend records and one-versus-multiple GPU comparisons."]
    if not report['evidence_snapshot_matches_harness']:
        lines.append("Historical snapshot was generated from a different harness; newer lanes may lack evidence entries.")
    lines += ["", "Saved-model entries (portable CPU probes plus dedicated gates):"]
    lines += [f"- {e['title']}: {e.get('alternative_gate', 'unmapped')}"
              for e in report["entries"] if not e["lanes"]]
    lines += ["", *report["limitations"], "Use --json for all 246 entry mappings and per-property reference counts."]
    return "\n".join(lines)
