"""Installed-package verification scope; inspection never executes a lane."""
import hashlib
import json
import re
from pathlib import Path

from . import host_surface
from . import _verify_reference as vref
from . import _verify_small as small_profile
from ._verification_catalog import ENTRIES, PROVENANCE
from ._verification_ctr_models import MODEL_SHA256 as CTR_MODELS
from ._verification_evidence_data import DATA as HISTORICAL_EVIDENCE


def declaration(spec):
    if callable(spec):
        return dict(kind="check", reason=None)
    if isinstance(spec, str) and spec.startswith("n/a:"):
        return dict(kind="not_applicable", reason=spec)
    return dict(kind="undeclared", reason="no explicit verification contract")


def reference_support(table, lane, fixtures, parts, stale=False, vendor_class=None):
    """Expose which current numerical answers each device class supports.

    An old, conflicting or N/A entry is not a numerical witness. These are
    reference-table counts, not executions of the installed wheel.
    """
    result = {}
    for part in parts:
        row = dict(numerical_fixtures=0, not_applicable_fixtures=0,
                   missing_or_conflicted_fixtures=0, stale_fixtures=0,
                   agreeing_device_classes={c: 0 for c in ('cpu', 'apple', 'nvidia', 'amd')})
        for fixture in fixtures:
            ent = vref.entry(table, lane, fixture, part, device_class=vendor_class)
            if stale:
                row['stale_fixtures'] += 1
                continue
            ref = ent.get('ref') if ent else None
            if not ent or ent.get('conflict') or not isinstance(ref, str):
                row['missing_or_conflicted_fixtures'] += 1
            elif ref.startswith('n/a:'):
                row['not_applicable_fixtures'] += 1
            elif re.fullmatch('[0-9a-f]{16}', ref):
                row['numerical_fixtures'] += 1
                for cls, column in vref.columns_of(table, ent).items():
                    if cls in row['agreeing_device_classes'] and column['agrees']:
                        row['agreeing_device_classes'][cls] += 1
            else:
                row['missing_or_conflicted_fixtures'] += 1
        result[part] = row
    return result


def inventory(harness, table, vendor_class):
    public = set(host_surface.public_reference_lanes())
    covered = set(host_surface.covered_lanes()) | set(host_surface.PUBLIC_HOST_ONLY_LANES)
    pending = host_surface.PUBLIC_PENDING_LANES
    candidates = set(host_surface.PUBLIC_REFERENCE_CANDIDATES)
    stale = set(vref.stale_reference_lanes(table, harness))
    parallel_cpu = {name for name in covered if name.startswith("par-")}
    lanes = {}
    # NOTHING IS HIDDEN, SO NOTHING IS `excluded` (Andrew, 2026-09-20). Every
    # lane is in the public surface; the inventory now says what the verifier
    # DOES with each one. `not_applicable` replaced `excluded` because the
    # older word described a decision we no longer make: these lanes were
    # never absent for want of merit, they are claims a one-device run cannot
    # state. `withheld` survives only for a lane whose comparison is blocked,
    # and it carries the reason verbatim.
    exposure = host_surface.lane_exposure(list(harness.LANES), vendor_class or "cpu")
    for name in harness.LANES:
        status, reason = "available", None
        row = exposure[name]
        if row["status"] == host_surface.LANE_NOT_APPLICABLE:
            status, reason = "not_applicable", row["reason"]
        elif not row["comparable"]:
            status = "withheld"
            reason = row["reason"] or pending.get(
                name, "reference qualification pending" if name in candidates
                else "CPU route not admitted to the public verifier")
        if status == "available" and name in stale:
            status, reason = "withheld", "stale reference"
        elif reason == "stale reference" and name not in stale:
            # A fresh CPU recording can repair the revision before independent
            # GPU qualification closes the explicit hold in host_surface.
            reason = "reference qualification pending"
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
                           execution=dict(
                               cpu_route_declared=name in covered,
                               cpu_logical_shards=name in parallel_cpu,
                               command=(f"verify --include-pending --lanes {name}" if name in covered and
                                        (name not in public or name in stale) else f"verify --lanes {name}"),
                               requires_gpu_for_execution=name not in covered,
                               physical_multi_gpu_measured_by_cpu=False),
                           reference_fixtures=refs, fixtures=len(harness.FIXTURES),
                           reference_support=reference_support(table, name, harness.FIXTURES,
                                                               refs, stale=name in stale, vendor_class=vendor_class),
                           reference_admission=table.get('lane_admission', {}).get(name,
                               dict(policy=table.get('admission_policy', dict(status='legacy')))))
    for name, lane in lanes.items():
        evidence = HISTORICAL_EVIDENCE['lanes'].get(name, {})
        lane['historical_evidence'] = evidence
        lane['multi_gpu_scope'] = ('parallel_lane' if name.startswith('par-')
                                   else 'no_parallel_lane_declared_here')
        lane['release_qualified'] = False
    harness_path = getattr(harness, '__file__', None)
    snapshot_matches = bool(harness_path and hashlib.sha256(Path(harness_path).read_bytes()).hexdigest()
                            == HISTORICAL_EVIDENCE['harness_sha256'])
    manifest_path = Path(__file__).parent / vref.TABLE_DIR / "models" / "models.json"
    model_keys = set()
    try:
        manifest = json.loads(manifest_path.read_text())
        model_keys = {m["lane"] + "/" + m["fixture"] for m in manifest["models"]
                      if (manifest_path.parent / m["file"]).is_file()}
        model_error = None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        model_error = f"portable model assets unavailable: {exc}"
    entries = []
    mapped = set()
    for entry in ENTRIES:
        row = dict(entry)
        row["lane_status"] = {name: lanes.get(name, dict(status="missing_lane", reason="not registered"))
                              for name in entry["lanes"]}
        row["verification"] = "mapped" if entry["lanes"] else "portable_models_and_dedicated_gate"
        if entry.get("portable_models"):
            missing = sorted(set(entry["portable_models"]) - model_keys)
            row["installed_check"] = dict(command="verify --models-only", run_by_default=True,
                status="available" if not missing and not model_error else "unavailable",
                models=entry["portable_models"], missing_models=missing, error=model_error,
                scope="representative bundled GPU-trained models through the CPU saved-model loader; model bytes and batch invariance")
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
                reference_admission_policy=table.get('admission_policy',
                    dict(status='legacy', reason='predates repeated-value and protocol admission; regenerate before release')),
                declared_ctr_model_sha256=CTR_MODELS,
                supplemental_checks=[dict(command='verify-causal-lm',
                    scope='tiny whole loaded-model inference composition',
                    reference_admitted=False, release_qualified=False),
                    dict(command='verify-distributed',
                    scope='two-GPU forecasting, GPC and disjoint IVF numerical checks with transport controls',
                    reference_admitted=False, release_qualified=False),
                    dict(command='verify-cross-validation',
                    scope='GPU fold scheduling, model, prediction and score checks',
                    reference_admitted=False, release_qualified=False)],
                experimental_profiles=[dict(profile=small_profile.PROFILE,
                    lanes=list(small_profile.LANES), cases=len(small_profile.CASES),
                    max_rows=small_profile.MAX_ROWS,
                    default_reference_compatible=False, reference_admitted=False)],
                additional_lanes=sorted(set(lanes) - mapped),
                counts=dict(appendix_entries=len(entries), registered_lanes=len(lanes),
                            **{status: sum(r["status"] == status for r in lanes.values())
                               for status in ("available", "withheld", "not_applicable", "excluded", "unavailable")}),
                cpu_execution_counts=dict(declared=len(set(lanes) & covered),
                    logical_shard_drivers=len(set(lanes) & parallel_cpu),
                    parallel_drivers_requiring_gpu=sum(name.startswith("par-") and name not in covered for name in lanes)),
                limitations=["References describe recorded fixtures, not all possible inputs or hardware.",
                             "A single-device parallel driver run does not certify multiple GPUs.",
                             "Gradient, batch-size, ragged and sampler/replay checks require --batch-checks; they are not implicit in --all.",
                             "The appendix's seasonal-difference selection label means select_d with caller-supplied D."])


def format_human(report):
    c = report["counts"]
    lines = ["Verification coverage inventory — no algorithms executed",
             f"{c['appendix_entries']} appendix entries; {c['registered_lanes']} registered lanes; "
             f"backend {report['vendor_class']}",
             f"{c['available']} available, {c['not_applicable']} not applicable here, "
             f"{c['withheld']} withheld (comparison blocked); every lane is public", "",
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
    policy = report['reference_admission_policy']
    if policy.get('status') == 'legacy':
        lines.append("Bundled reference admission: legacy; regeneration under the stricter policy is still owed.")
    lines += ["", "Saved-model entries (portable CPU probes plus dedicated gates):"]
    lines += [f"- {e['title']}: {e.get('installed_check', {}).get('command', 'unmapped')} "
              f"({e.get('installed_check', {}).get('status', 'unavailable')}); additional gate: {e.get('alternative_gate', 'unmapped')}"
              for e in report["entries"] if not e["lanes"]]
    execution = report['cpu_execution_counts']
    lines += ["", f"CPU execution routes: {execution['declared']}, including "
              f"{execution['logical_shard_drivers']} logical-shard drivers; "
              f"{execution['parallel_drivers_requiring_gpu']} parallel drivers still require GPUs.",
              "Use verify --include-pending --batch-checks to exercise pending CPU routes and extended properties; unqualified results remain incomplete."]
    lines += ["", *report["limitations"], "Use --json for all 246 entry mappings and per-property reference counts."]
    return "\n".join(lines)
