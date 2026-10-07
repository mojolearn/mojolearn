#!/usr/bin/env python3
"""Glue for tools/six_lane_grid_run.sh: run the IDENTICAL switch grid on one Linux GPU box.

Metadata and process orchestration only. It never imports an estimator, compiles, or
measures by itself: compiles are `six_lane_ab.py compile`, admission is
`six_lane_materialize.materialize` + `six_lane_ab.queue`, execution is
`performance_full_ab_queue.py`, timing/identity/decisions are the existing tools.

Subcommands (see tools/six_lane_grid_run.md for the end-to-end runbook):
  kit           (laptop) gather the retained Oct 6 saved full-workload facts, the per-workload loaded
                bindings and every small sha-bound evidence file the registered input variants
                reference, into one directory the orchestrator copies to each box
  install-kit   (box) put each kit evidence file at its original absolute path (sha-checked);
                list which full-input data files are present
  stage         (box) one A/B queue per grid cell (facts + isolated packages + materialize + queue),
                ordered factorial regime first, then pairwise; idempotent per configuration
  run           (box) run the staged cells one pair at a time with performance_full_ab_queue.py;
                skips measured cells, keeps going past a failed cell, resumes an interrupted one
  quality       receipts -> candidate (A) vs incumbent (B) task-quality rows (tools/af_quality.py)
  manifest      NVIDIA + AMD receipts -> six_lane_compare_results.py --manifest input
  status        one-line summary of a run directory
"""
# cpu-route: offline orchestration of retained evidence files; no product runtime.
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

IDENTICAL_DEFINE = 'MOJOLEARN_NUMERIC_IDENTICAL=1'
TRACK = {'nvidia': 'nvidia-native', 'amd': 'amd-gfx942'}
COLUMN = {'nvidia': 'nvidia-native', 'amd': 'amd'}
REGIME_ORDER = {'factorial': 0, 'pairwise': 1}
OCT6 = 'six-lane-full-ab-20261006'
CANON = (('classical/', 'classical:'), ('classical2/', 'more:'), ('algos/', 'expanded:'))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def canon(wid):
    for old, new in CANON:
        wid = wid.replace(old, new)
    return wid


def stem(binding):
    return Path(binding).stem


def safe(name):
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name)[:120] + '-' + sha_value(name)[:8]


# ------------------------------------------------------------------ kit (laptop)

def sha_refs(value, out):
    """Every {path, sha256} reference inside a JSON value."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            out.append((value['path'], value['sha256']))
        for child in value.values():
            sha_refs(child, out)
    elif isinstance(value, list):
        for child in value:
            sha_refs(child, out)
    return out


def kit(args):
    evidence = Path(args.evidence).expanduser()
    out = Path(args.out).expanduser()
    roots = {'nvidia': evidence / OCT6 / 'nvidia-native', 'amd': evidence / OCT6 / 'amd'}
    search = [Path(p).expanduser() for p in (args.search or [])] + [evidence / OCT6]
    facts, bindings, sources = {}, {}, {}
    for vendor, base in roots.items():
        facts[vendor], bindings[vendor] = {}, {}
        for path in sorted(base.rglob('*facts.json')):
            if 'artifacts' not in path.parts:
                continue
            try:
                doc = read(path)
            except (OSError, ValueError):
                continue
            for wid, fact in (doc.items() if isinstance(doc, dict) else []):
                if isinstance(fact, dict) and fact.get('full_dataset_coverage') is True and isinstance(fact.get('workload'), dict):
                    facts[vendor][canon(wid)] = fact
                    sources.setdefault(vendor, {})[canon(wid)] = dict(path=str(path), sha256=sha_file(path))
        # Loaded bindings per workload: the retained deployments (what each scored worker attested).
        for path in sorted(base.rglob('deployments.json')):
            try:
                doc = read(path)
            except (OSError, ValueError):
                continue
            for dep in doc if isinstance(doc, list) else []:
                wid = dep.get('workload_id')
                if wid and isinstance(dep.get('artifacts'), dict):
                    names = sorted({stem(a['path']) for a in dep['artifacts'].get('A', [])})
                    if names:
                        bindings[vendor].setdefault(canon(wid), names)
    # Committed Oct 6 receipts carry the same attestation for workloads whose deployment file is gone.
    for receipt in sorted((ROOT / 'experiments/six_lane_integration/measurements/20261006/receipts').glob('*/*/*/attempt-*/receipt.json')):
        vendor = receipt.parts[-5]
        if vendor not in bindings:
            continue
        job = read(receipt).get('workload', {})
        names = sorted({stem(a['path']) for a in job.get('artifact_provenance', {}).get('A', [])})
        if job.get('workload_id') and names:
            bindings[vendor].setdefault(canon(job['workload_id']), names)
    # Small sha-bound evidence the registered input variants read (proposals, receipts, plans, contracts),
    # followed one level into retained JSON documents. Large data inputs are listed, never copied.
    by_name = {}
    for root in search:
        for path in root.rglob('*'):
            if path.is_file() and not path.is_symlink():
                by_name.setdefault(path.name, []).append(path)
    files, data, missing = {}, {}, []

    def find(original, sha):
        candidates = ([Path(original)] if Path(original).is_file() else []) + by_name.get(Path(original).name, [])
        for cand in candidates:
            if cand.stat().st_size < 64 * 1024 * 1024 and sha_file(cand) == sha:
                return cand
        return None

    pending = []
    for vendor in facts:
        for wid, fact in facts[vendor].items():
            for item in fact['workload']['input_files']:
                data.setdefault((vendor, item['sha256']), dict(vendor=vendor, name=Path(item['path']).name,
                                                               sha256=item['sha256'], original_path=item['path']))
            if fact.get('registered_input_variant'):
                pending += sha_refs(fact['registered_input_variant'], [])
    seen = set()
    while pending:
        original, sha = pending.pop()
        if sha in seen:
            continue
        seen.add(sha)
        if any(d['sha256'] == sha for d in data.values()) and original.endswith('.npz'):
            continue
        local = find(original, sha)
        if local is None:
            missing.append(dict(original_path=original, sha256=sha))
            continue
        dest = out / 'files' / (sha[:16] + '-' + Path(original).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(local, dest)
        files[sha] = dict(sha256=sha, original_path=original, kit_path=str(dest.relative_to(out)))
        if local.suffix == '.json':
            try:
                pending += sha_refs(read(local), [])
            except ValueError:
                pass
    for entry in data.values():
        entry['laptop_candidates'] = [str(p) for p in by_name.get(entry['name'], [])]
    write(out / 'vendor-workload-facts.json', facts)
    write(out / 'facts-sources.json', sources)
    write(out / 'required-bindings.json', bindings)
    write(out / 'files.json', dict(files=sorted(files.values(), key=lambda f: f['original_path']), missing=missing))
    write(out / 'data-manifest.json', sorted(data.values(), key=lambda d: (d['vendor'], d['name'], d['sha256'])))
    summary = dict(kit=str(out), facts={v: len(f) for v, f in facts.items()},
                   bindings={v: len(b) for v, b in bindings.items()}, evidence_files=len(files),
                   evidence_missing=len(missing), data_files=len(data))
    write(out / 'kit.json', dict(summary, schema='mojolearn.grid-run-kit/1', created=time.time()))
    print(json.dumps(summary))
    return 0


def install_kit(args):
    kit_dir = Path(args.kit)
    manifest = read(kit_dir / 'files.json')
    placed = present = conflicts = 0
    for item in manifest['files']:
        if not any(item['original_path'].startswith(p) for p in args.prefix):
            continue  # laptop-only original location: the stager relocates this reference to the kit copy
        target = Path(item['original_path'])
        source = kit_dir / item['kit_path']
        if target.is_file():
            if sha_file(target) != item['sha256']:
                conflicts += 1
                print('CONFLICT ' + str(target), file=sys.stderr)
            else:
                present += 1
            continue
        if args.dry_run:
            placed += 1  # would place
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        placed += 1
    data = [d for d in read(kit_dir / 'data-manifest.json') if d['vendor'] == args.vendor]
    data_dirs = [Path(p) for p in (args.data_dir or [])]
    absent = []
    for d in data:
        if not Path(d['original_path']).is_file() and not any((p / d['name']).is_file() for p in data_dirs):
            absent.append(d['original_path'])
    print(json.dumps(dict(evidence_placed=placed, evidence_present=present, evidence_conflicts=conflicts,
                          data_files=len(data), data_absent=len(absent), first_absent=absent[:3],
                          dry_run=bool(args.dry_run))))
    return 1 if conflicts or (absent and not args.dry_run) else 0


# ------------------------------------------------------------------ grid helpers

def load_grid(grid):
    from six_lane_matrix_io import read_matrix
    grid = Path(grid)
    matrix = read_matrix(grid / 'grid-matrix.json.gz')
    plan = read(grid / 'grid-plan.json')
    build_plan = read(grid / 'grid-build-plan.json')
    return matrix, plan, build_plan


def regime_of(plan, algorithm):
    return (plan.get('algorithms', {}).get(algorithm) or {}).get('regime', 'pairwise')


def ordered_cells(matrix, plan, vendor, phase='all'):
    """Factorial-regime algorithms first, then pairwise; inside: algorithm, priority, configuration, workload."""
    configs = {c['id']: c for c in matrix['configurations']}
    rows = []
    for cell in matrix['cells']:
        if cell['vendor'] != vendor:
            continue
        cfg = configs[cell['configuration']]
        algorithm = cfg.get('grid', {}).get('algorithm') or cell['workload'].get('grid_algorithm')
        regime = regime_of(plan, algorithm)
        if phase != 'all' and regime != phase:
            continue
        rows.append((REGIME_ORDER.get(regime, 9), algorithm, cfg.get('priority', 0), cfg['id'], cell['workload_id'],
                     dict(cell=cell, regime=regime, algorithm=algorithm)))
    rows.sort(key=lambda r: r[:5])
    return [r[-1] for r in rows]


def build_index(builds_root):
    """Compile receipts under the build root(s): key -> {path, receipt, status, binding, defines}."""
    index = {}
    for root in builds_root:
        for path in sorted(Path(root).glob('**/receipt.json')):
            try:
                receipt = read(path)
            except (OSError, ValueError):
                continue
            key = receipt.get('key')
            if not key:
                continue
            entry = dict(receipt=str(path), status=receipt.get('status'), path=receipt.get('artifact'),
                         binding=receipt.get('binding'), defines=receipt.get('defines'),
                         artifact_sha256=receipt.get('artifact_sha256'))
            if index.get(key, {}).get('status') != 'COMPILED':
                index[key] = entry
    return index


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def package_template(repo, out, math_lib=None):
    """The Python package shell of the frozen checkout: mojolearn/ without binding .so files, no bytecode,
    plus the portable math library (built outside the checkout, which must stay clean). Built once per run."""
    dest = Path(out) / 'template'
    if (dest / '.complete').exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    src = Path(repo) / 'python' / 'mojolearn'

    def ignore(directory, names):
        skip = {n for n in names if n == '__pycache__' or n.endswith('.pyc')}
        if Path(directory).name != '.libs':
            skip |= {n for n in names if n.endswith(('.so', '.dylib'))}
        return skip
    shutil.copytree(src, dest / 'mojolearn', ignore=ignore)
    if math_lib:
        (dest / 'mojolearn' / '.libs').mkdir(exist_ok=True)
        shutil.copy2(math_lib, dest / 'mojolearn' / '.libs' / 'libMojolearnMath.so')
    (dest / '.complete').write_text('ok\n')
    return dest


def package_for(out, template, vendor, artifacts):
    """Content-addressed isolated package: every incumbent binding plus this arm's overrides."""
    name = sha_value(sorted((s, a['artifact_sha256']) for s, a in artifacts.items()))[:20]
    dest = Path(out) / 'packages' / vendor / name
    if (dest / '.complete').exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(template, dest, copy_function=link_or_copy, ignore=shutil.ignore_patterns('.complete'))
    target = dest / 'mojolearn' / 'identical'
    target.mkdir(parents=True, exist_ok=True)
    for s, a in sorted(artifacts.items()):
        link_or_copy(a['path'], target / (s + '.so'))
    (dest / '.complete').write_text('ok\n')
    return dest


# ------------------------------------------------------------------ stage (box)

def cell_facts(original, cell, source, kit_files, data_dirs, audit_dir):
    """Saved full-workload facts admitted for one grid cell (the retained recipe, unchanged numerics)."""
    from six_lane_targeted_variants import SCHEMA
    fact = copy.deepcopy(original)
    fact['source_sha'] = source
    work = fact['workload']
    variant = fact.get('registered_input_variant', {}).get('variant')
    for item in work['input_files']:
        if Path(item['path']).is_file():
            continue
        if variant == 'tsvd-full-v1':
            raise ValueError('tsvd-full-v1 inputs cannot be relocated; place them at ' + item['path'])
        for directory in data_dirs:
            cand = Path(directory) / Path(item['path']).name
            if cand.is_file():
                item['path'] = str(cand)
                break
        else:
            raise ValueError('Full input missing: ' + item['path'])
    parents = {str(Path(x['path']).parent) for x in work['input_files']}
    if len(parents) == 1:
        work['data_directory'] = parents.pop()
    work.pop('model_state_paths', None)
    audit_dir.mkdir(parents=True, exist_ok=True)
    old_path = audit_dir / ('original-facts-' + safe(cell['workload_id']) + '.json')
    if not old_path.exists():
        write(old_path, original)
    audit = audit_dir / ('full-recipe-audit-' + safe(cell['workload_id']) + '.json')
    if not audit.exists():
        write(audit, dict(source_sha=source, original_facts=dict(path=str(old_path), sha256=sha_file(old_path)),
                          input_files=work['input_files'], unchanged_shapes=fact['dimensions'],
                          unchanged_settings=fact['estimator_settings'], unchanged_boundary=fact['timed_boundary'],
                          prior_cap_audit=original['workload'].get('intrinsic_cap_audit'),
                          scope='Retained Oct 6 full recipe reused unchanged for a grid configuration; '
                                'new source and control-specific artifacts admitted independently by the materializer.'))
    work['intrinsic_cap_audit'] = dict(work['intrinsic_cap_audit'], evidence=str(audit))
    if fact.get('registered_input_variant'):
        reg = relocate(fact['registered_input_variant'], kit_files)
        fact['registered_input_variant'] = reg
        transfer = dict(schema=SCHEMA, measurement_source_sha=source,
                        target={k: cell[k] for k in ('configuration', 'vendor', 'mode', 'workload_id', 'key')},
                        original_facts=dict(path=str(old_path), sha256=sha_file(old_path)))
        if reg.get('contract_sha256'):
            located = locate(reg['contract_sha256'], None, kit_files)
            if located:
                transfer['original_contract'] = dict(path=located, sha256=reg['contract_sha256'])
        fact['input_variant_control_transfer'] = transfer
    fact['coverage_resolutions'] = {
        reason: dict(path=str(audit), sha256=sha_file(audit),
                     conclusion='Exact retained full input identities, unchanged numerical recipe/settings/boundary; '
                                'grid A/B artifacts admitted independently. Broader caller gaps remain explicit.')
        for reason in cell['blockers']}
    fact['resource_policy'] = dict(policy='Full actual allocation, serial arms under one GPU; worker captures effective '
                                          'thread/resource metadata; no laptop diagnostic caps')
    return fact


def locate(sha, original, kit_files):
    if original and Path(original).is_file():
        return original
    item = kit_files.get(sha)
    if item and Path(item['local']).is_file():
        return item['local']
    return original


def relocate(value, kit_files):
    if isinstance(value, dict):
        out = {k: relocate(v, kit_files) for k, v in value.items()}
        if isinstance(out.get('path'), str) and isinstance(out.get('sha256'), str):
            out['path'] = locate(out['sha256'], out['path'], kit_files)
        return out
    if isinstance(value, list):
        return [relocate(v, kit_files) for v in value]
    return value


def cell_artifacts(cfg, vendor, wid, algorithm_plan, required, index, b_keys):
    """A/B deployed artifact keys for one cell; raises with the blocking reason."""
    builds = cfg['grid']['builds'][vendor]
    changed_stems = [stem(b) for b in cfg['grid']['bindings']]
    a_keys = dict(zip(changed_stems, builds['A']))
    loaded = list(required.get(wid) or algorithm_plan.get('workload_bindings') or [])
    reach = set(loaded) | set(algorithm_plan.get('workload_bindings') or [])
    changed = [s for s in changed_stems if s in reach]
    if not changed:
        raise ValueError('No changed binding is loaded by this workload (grid bindings %s, loaded %s)'
                         % (changed_stems, sorted(reach)))
    listed = sorted(set(loaded) | set(changed))
    arms = {}
    for arm in ('A', 'B'):
        arms[arm] = []
        for s in listed:
            key = a_keys[s] if (arm == 'A' and s in changed) else b_keys.get(s)
            entry = index.get(key) if key else None
            if not entry or entry['status'] != 'COMPILED':
                raise ValueError('Missing compiled %s artifact for %s (key %s, status %s)'
                                 % (arm, s, key, entry and entry['status']))
            arms[arm].append(dict(stem=s, key=key, prerequisite=s not in changed, **entry))
    return arms


def stage(args):
    from six_lane_ab import queue as write_queue
    from six_lane_materialize import materialize
    vendor = args.vendor
    run = Path(args.run).resolve()
    if run.is_relative_to(ROOT):
        raise SystemExit('The run directory must be outside the frozen checkout')
    matrix, plan, build_plan = load_grid(args.grid)
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    kit_dir = Path(args.kit)
    facts = read(kit_dir / 'vendor-workload-facts.json')[vendor]
    required = read(kit_dir / 'required-bindings.json').get(vendor, {})
    kit_files = {f['sha256']: dict(f, local=str(kit_dir / f['kit_path'])) for f in read(kit_dir / 'files.json')['files']}
    index = build_index(args.builds)
    if args.dry_run:
        # Laptop / pre-compile check: a planned, not yet compiled job counts as present.
        for job in build_plan['jobs']:
            if job['vendor'] == vendor and job['key'] not in index:
                index[job['key']] = dict(status='COMPILED', planned_only=True, receipt=None, path=None,
                                         binding=job['binding'], defines=job['defines'], artifact_sha256=None)
    b_keys = {stem(j['binding']): j['key'] for j in build_plan['jobs'] if j['vendor'] == vendor and j.get('pack') == 'B'}
    configs = {c['id']: c for c in matrix['configurations']}
    cells = ordered_cells(matrix, plan, vendor, args.phase)
    out = run / 'stage'
    by_config = {}
    for row in cells:
        by_config.setdefault(row['cell']['configuration'], []).append(row)
    template = None
    status = []
    for n, (cid, rows) in enumerate(by_config.items()):
        if args.limit and n >= args.limit:
            break
        cdir = out / 'configs' / safe(cid)
        record = cdir / 'staged.json'
        if record.exists():
            prior = read(record)
            if prior.get('source_sha') == source:
                status += prior['cells']
                continue
            raise SystemExit('Staged under another source freeze; use a new run directory: ' + str(cdir))
        cfg = configs[cid]
        algorithm_plan = plan['algorithms'].get(rows[0]['algorithm'], {})
        results, cell_facts_doc, deployments, admitted = [], {}, [], []
        for row in rows:
            cell = row['cell']
            entry = dict(key=cell['key'], configuration=cid, workload_id=cell['workload_id'], regime=row['regime'],
                         algorithm=row['algorithm'], state='BLOCKED')
            try:
                wid = cell['workload_id']
                if wid not in facts:
                    raise ValueError('No retained saved full-workload facts for this workload on ' + vendor)
                arms = cell_artifacts(cfg, vendor, wid, algorithm_plan, required, index, b_keys)
                if args.dry_run:
                    entry['state'] = 'DRY_RUN_ADMISSIBLE'
                    entry['inputs_present_here'] = all(
                        Path(i['path']).is_file() or any((Path(d) / Path(i['path']).name).is_file() for d in args.data_dir or [])
                        for i in facts[wid]['workload']['input_files'])
                    results.append(entry)
                    continue
                fact = cell_facts(facts[wid], cell, source, kit_files, args.data_dir or [], cdir / 'audits')
                template = template or package_template(ROOT, run, args.math_lib)
                dep = dict(configuration=cid, vendor=vendor, target_track=TRACK[vendor], workload_id=wid,
                           packages={}, artifacts={})
                every_b = {s: index[k] for s, k in b_keys.items() if index.get(k, {}).get('status') == 'COMPILED'}
                for arm in ('A', 'B'):
                    overrides = dict(every_b)
                    overrides.update({a['stem']: a for a in arms[arm]})
                    package = package_for(run, template, vendor, overrides)
                    dep['packages'][arm] = str(package)
                    dep['artifacts'][arm] = [dict(path=str(package / 'mojolearn' / 'identical' / (a['stem'] + '.so')),
                                                  receipt=a['receipt'], prerequisite=a['prerequisite'])
                                             for a in arms[arm]]
                cell_facts_doc[wid] = fact
                deployments.append(dep)
                admitted.append(cell)
                entry['state'] = 'FACTS_READY'
            except (ValueError, KeyError, OSError) as exc:
                entry['reason'] = str(exc)
            results.append(entry)
        if admitted:
            slim = dict(matrix, configurations=[cfg], cells=admitted)
            write(cdir / 'matrix.json', slim)
            write(cdir / 'facts.json', cell_facts_doc)
            write(cdir / 'deployments.json', deployments)
            with (cdir / 'materialize.log').open('w') as log:
                stdout, sys.stdout = sys.stdout, log
                try:
                    materialize(argparse.Namespace(matrix=cdir / 'matrix.json', workloads=cdir / 'facts.json',
                                                   deployments=cdir / 'deployments.json', vendor=vendor,
                                                   target_track=TRACK[vendor], select=[cid], output=cdir / 'materialized'))
                finally:
                    sys.stdout = stdout
            coverage = {c['key']: c for c in read(cdir / 'materialized' / 'coverage.json')['cells']}
            ok = [c for c in admitted if coverage.get(c['key'], {}).get('status') == 'MATERIALIZED_NOT_EXECUTED']
            for entry in results:
                if entry['state'] == 'FACTS_READY' and entry['key'] not in {c['key'] for c in ok}:
                    entry.update(state='BLOCKED', reason='materialize: ' + coverage.get(entry['key'], {}).get('reason', 'not materialized'))
            if ok:
                write(cdir / 'queue-matrix.json', dict(slim, cells=ok))
                with (cdir / 'queue.log').open('w') as log:
                    stdout, sys.stdout = sys.stdout, log
                    try:
                        write_queue(argparse.Namespace(matrix=cdir / 'queue-matrix.json', recipes=cdir / 'materialized' / 'recipes.json',
                                                       vendor=vendor, select=[cid], output=cdir / 'queue.json'))
                    finally:
                        sys.stdout = stdout
                q = read(cdir / 'queue.json')
                for job in q['jobs']:
                    single = authorize(dict(q, jobs=[job]), args.authorize, args.cell_timeout)
                    path = cdir / 'cells' / (job['key'] + '.json')
                    write(path, single)
                    for entry in results:
                        if entry['key'] == job['key']:
                            entry.update(state='READY', queue=str(path))
        if not args.dry_run:
            write(record, dict(source_sha=source, vendor=vendor, configuration=cid, cells=results, staged=time.time()))
        status += results
    ready = sum(entry['state'] in ('READY', 'DRY_RUN_ADMISSIBLE') for entry in status)
    blocked = sum(entry['state'] == 'BLOCKED' for entry in status)
    reasons = {}
    for entry in status:
        if entry['state'] == 'BLOCKED':
            head = entry.get('reason', '?').split(' (')[0][:90]
            reasons[head] = reasons.get(head, 0) + 1
    summary = dict(vendor=vendor, phase=args.phase, source_sha=source, cells=len(status), ready=ready, blocked=blocked,
                   inputs_absent_here=sum(1 for e in status if e.get('inputs_present_here') is False),
                   dry_run=bool(args.dry_run), blocked_reasons=dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:8]))
    if not args.dry_run:
        write(out / ('cells-' + args.phase + '.json'), dict(summary, cells=status))
    print(json.dumps(summary))
    return 0


def authorize(queue_doc, text, timeout):
    queue_doc = copy.deepcopy(queue_doc)
    bench = ROOT / '.pixi/envs/bench/lib'
    default = ROOT / '.pixi/envs/default/lib'
    queue_doc.update(execution_authorized=True, authorization=text,
                     environment=dict(queue_doc.get('environment') or {}, PYTHONDONTWRITEBYTECODE='1',
                                      LD_LIBRARY_PATH=str(bench) + ':' + str(default)))
    for job in queue_doc['jobs']:
        job['timeout_seconds'] = timeout
        for arm in ('A', 'B'):
            argv = job['arms'][arm]['argv']
            worker_path = Path(argv[argv.index('--recipe') + 1])
            worker = read(worker_path)
            worker.update(execution_authorized=True, authorization=text)
            write(worker_path, worker)
    return queue_doc


# ------------------------------------------------------------------ run (box)

def cell_result(results_dir, key):
    path = Path(results_dir) / key / 'results.json'
    if not path.exists():
        return None
    try:
        return read(path).get(key)
    except (OSError, ValueError):
        return None


def run_cells(args):
    run = Path(args.run).resolve()
    staged = read(run / 'stage' / ('cells-' + args.phase + '.json'))
    cells = [c for c in staged['cells'] if c['state'] == 'READY']
    results = run / 'results'
    logs = run / 'logs' / 'cells'
    logs.mkdir(parents=True, exist_ok=True)
    runner = args.runner or str(ROOT / 'tools/performance_full_ab_queue.py')
    python = args.python or sys.executable
    counts = dict(measured=0, failed=0, skipped_failed=0, ran=0)
    status_line = run / 'status.txt'

    def note(phase, current=''):
        line = ('grid-run vendor=%s phase=%s step=run state=%s cells=%d measured=%d failed=%d current=%s updated=%s'
                % (staged['vendor'], args.phase, phase, len(cells), counts['measured'], counts['failed'], current,
                   time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        status_line.write_text(line + '\n')
    for n, cell in enumerate(cells):
        key = cell['key']
        prior = cell_result(results, key)
        if prior and prior.get('status') == 'MEASURED_FULL':
            counts['measured'] += 1
            continue
        retry = False
        if prior and not args.retry_failed:
            counts['failed'] += 1
            counts['skipped_failed'] += 1
            continue
        if prior or (results / key / key / 'attempts').exists():
            retry = True  # failed (with --retry-failed) or interrupted: a fresh attempt, prior evidence kept
        if args.limit and counts['ran'] >= args.limit:
            break
        note('MEASURING', key)
        argv = [python, runner, '--config', cell['queue'], '--output', str(results / key)] + (['--retry-failed'] if retry else [])
        with (logs / (key + '.log')).open('a') as log:
            rc = subprocess.run(argv, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT).returncode
        counts['ran'] += 1
        result = cell_result(results, key)
        if rc == 0 and result and result.get('status') == 'MEASURED_FULL':
            counts['measured'] += 1
        else:
            counts['failed'] += 1
    note('DONE')
    print(json.dumps(dict(vendor=staged['vendor'], phase=args.phase, cells=len(cells), **counts)))
    return 0


# ------------------------------------------------------------------ evidence (quality, manifest, status)

def latest_receipts(results_dirs):
    """(configuration, workload_id) -> (path, receipt) of the newest MEASURED_FULL attempt."""
    out = {}
    for root in results_dirs:
        for path in sorted(Path(root).glob('*/*/attempts/attempt-*/receipt.json')):
            try:
                receipt = read(path)
            except (OSError, ValueError):
                continue
            if receipt.get('status') != 'MEASURED_FULL':
                continue
            job = receipt.get('workload', {})
            key = (job.get('master_selection', {}).get('id'), job.get('workload_id'))
            out[key] = (path, receipt)
    return out


def scored_results(receipt):
    return {r['arm']: r['result'] for r in receipt.get('runs', [])
            if r.get('phase') == 'scored' and isinstance(r.get('result'), dict)}


def quality(args):
    from af_quality import compare
    rows = []
    for (cid, wid), (path, receipt) in sorted(latest_receipts(args.results).items()):
        res = scored_results(receipt)
        metrics = {arm: ((res.get(arm, {}).get('task_quality') or {}).get('metrics') or {}).get('ours') or {} for arm in 'AB'}
        verdict = compare(metrics['A'], metrics['B']) if metrics['A'] and metrics['B'] else dict(verdict='PENDING', metrics={}, unknown=[], worst=0.0)
        rows.append(dict(vendor=res.get('A', {}).get('vendor'), configuration=cid, workload_id=wid, receipt=str(path),
                         receipt_sha256=sha_file(path), source_sha=receipt.get('source_sha'), metrics=metrics,
                         candidate_vs_baseline=verdict))
    write(args.out, dict(schema='mojolearn.grid-quality-rows/1', rule='tools/af_quality.py compare: candidate A vs incumbent B, '
                         'rel 1e-3 / abs 1e-6 on the saved task metrics; WORSE holds the arm', rows=rows))
    counts = {}
    for r in rows:
        counts[r['candidate_vs_baseline']['verdict']] = counts.get(r['candidate_vs_baseline']['verdict'], 0) + 1
    print(json.dumps(dict(rows=len(rows), verdicts=counts, out=str(args.out))))
    return 0


def expected_of(receipt):
    res = scored_results(receipt)
    a = res['A']
    capture = dict(outputs=[m['path'] for m in a['outputs']['manifest']])
    states = [r.get('model_state') or {} for r in res.values()]
    if all(s.get('status') == 'CAPTURED' and s.get('manifest') for s in states):
        capture['model_state'] = [m['path'] for m in a['model_state']['manifest']]
    expected = {k: a[k] for k in ('dataset_sha256', 'dataset_version', 'dataset_split', 'seed', 'dimensions',
                                  'estimator_settings', 'harness_sha256', 'timed_boundary')}
    expected.update(configurations={arm: res[arm]['configuration'] for arm in ('A', 'B')}, capture_paths=capture,
                    repeated_operations=len(a.get('repeated_use') or []))
    return expected


def manifest(args):
    columns = {'nvidia-native': latest_receipts(args.nvidia), 'amd': latest_receipts(args.amd)}
    cases, unpaired = [], []
    for key in sorted(set(columns['nvidia-native']) | set(columns['amd'])):
        cid, wid = key
        have = {col: columns[col][key] for col in columns if key in columns[col]}
        if len(have) < 2:
            unpaired.append(dict(configuration=cid, workload_id=wid, columns=sorted(have)))
        anchor_path, anchor = have.get('nvidia-native') or have['amd']
        job = anchor['workload']
        try:
            expected = expected_of(anchor)
        except (KeyError, TypeError):
            unpaired.append(dict(configuration=cid, workload_id=wid, reason='anchor receipt lacks scored capture'))
            continue
        cases.append(dict(id=sha_value([cid, wid])[:20], configuration_id=cid, implementation_ids=job.get('implementation_ids', []),
                          source_sha=anchor['source_sha'], workload_id=wid, mode=job.get('mode', 'identical'),
                          expected=expected,
                          columns={col: dict(receipt=str(Path(p).resolve()), transport_environment={'A': {}, 'B': {}}, history=[])
                                   for col, (p, _r) in have.items()}))
    write(args.out, dict(schema='mojolearn.six-lane-comparison-input/1', cases=cases))
    write(Path(args.out).with_name(Path(args.out).stem + '-unpaired.json'), unpaired)
    print(json.dumps(dict(cases=len(cases), paired=len(cases) - sum(1 for u in unpaired if 'columns' in u), unpaired=len(unpaired), out=str(args.out))))
    return 0 if cases else 1


def build_keys(args):
    """Compile shards: one line per binding, `<binding> <key,key,...>`, covering the A builds of the phase's
    configurations and every incumbent (B) build of the vendor (each isolated package carries all of them)."""
    matrix, plan, build_plan = load_grid(args.grid)
    wanted = {cell['cell']['configuration'] for cell in ordered_cells(matrix, plan, args.vendor, args.phase)}
    keys = {j['key'] for j in build_plan['jobs'] if j['vendor'] == args.vendor and j.get('pack') == 'B'}
    for cfg in matrix['configurations']:
        if cfg['id'] in wanted:
            keys.update(cfg['grid']['builds'][args.vendor]['A'])
    shards = {}
    for job in build_plan['jobs']:
        if job['vendor'] == args.vendor and job['key'] in keys:
            shards.setdefault(job['binding'], []).append(job['key'])
    for binding in sorted(shards):
        print(binding + ' ' + ','.join(sorted(shards[binding])))
    return 0


def status(args):
    path = Path(args.run) / 'status.txt'
    print(path.read_text().strip() if path.exists() else 'grid-run state=NOT_STARTED run=' + str(args.run))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='command', required=True)
    k = s.add_parser('kit', help='(laptop) gather retained facts, loaded bindings and variant evidence files')
    k.add_argument('--evidence', default='~/mojolearn-evidence')
    k.add_argument('--search', action='append', help='extra directories searched for referenced evidence files')
    k.add_argument('--out', required=True)
    i = s.add_parser('install-kit', help='(box) place kit evidence files at their original absolute paths')
    i.add_argument('--kit', required=True)
    i.add_argument('--vendor', choices=('nvidia', 'amd'), required=True)
    i.add_argument('--data-dir', action='append')
    i.add_argument('--prefix', action='append', default=None, help='original-path prefixes to place (default /root/)')
    i.add_argument('--dry-run', action='store_true')
    st = s.add_parser('stage', help='(box) admit grid cells into one authorized A/B queue per cell')
    for sub in (st,):  # noqa: B007
        sub.add_argument('--vendor', choices=('nvidia', 'amd'), required=True)
        sub.add_argument('--grid', required=True)
        sub.add_argument('--kit', required=True)
        sub.add_argument('--builds', action='append', required=True, help='compile output root(s) (repeatable)')
        sub.add_argument('--run', required=True)
        sub.add_argument('--phase', choices=('factorial', 'pairwise', 'all'), default='all')
        sub.add_argument('--data-dir', action='append', help='where full inputs live when not at their original paths')
        sub.add_argument('--authorize', default='', help='authorization text recorded in every queue and worker')
        sub.add_argument('--cell-timeout', type=int, default=3600)
        sub.add_argument('--math-lib', help='libMojolearnMath.so built outside the checkout (packaging/portable_math)')
        sub.add_argument('--limit', type=int, help='stage only the first N configurations')
        sub.add_argument('--dry-run', action='store_true', help='check admissibility only; write nothing')
    r = s.add_parser('run', help='(box) run staged cells one pair at a time')
    r.add_argument('--run', required=True)
    r.add_argument('--phase', choices=('factorial', 'pairwise', 'all'), default='all')
    r.add_argument('--retry-failed', action='store_true', help='retry failed cells in fresh attempts (prior evidence kept)')
    r.add_argument('--limit', type=int, help='run at most N cells this invocation')
    r.add_argument('--python', help='interpreter for the controller (default: this one)')
    r.add_argument('--runner', help=argparse.SUPPRESS)
    q = s.add_parser('quality', help='receipts -> candidate vs incumbent quality rows')
    q.add_argument('results', nargs='+')
    q.add_argument('--out', required=True)
    m = s.add_parser('manifest', help='NVIDIA + AMD receipts -> comparison manifest')
    m.add_argument('--nvidia', action='append', required=True)
    m.add_argument('--amd', action='append', required=True)
    m.add_argument('--out', required=True)
    b = s.add_parser('build-keys', help='compile shards for a phase: `<binding> <key,...>` per line')
    b.add_argument('--grid', required=True)
    b.add_argument('--vendor', choices=('nvidia', 'amd'), required=True)
    b.add_argument('--phase', choices=('factorial', 'pairwise', 'all'), default='all')
    t = s.add_parser('status')
    t.add_argument('--run', required=True)
    args = p.parse_args(argv)
    if args.command == 'install-kit' and not args.prefix:
        args.prefix = ['/root/']
    if args.command == 'stage' and not args.dry_run and not args.authorize:
        p.error('stage needs --authorize "<text>" (recorded in every queue and worker) unless --dry-run')
    return dict(kit=kit, stage=stage, run=run_cells, quality=quality, manifest=manifest, status=status,
                **{'install-kit': install_kit, 'build-keys': build_keys})[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
