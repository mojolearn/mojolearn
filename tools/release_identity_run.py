#!/usr/bin/env python3
"""Plan or run only identity work the release ledger actually needs.

This frontdoor records a before-execution launch; cross-vendor admission remains
release_identity_evidence's job. An exited producer is never a release PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import release_identity_evidence as identity
from release_requirements import RequirementsLedger

SCHEMA = 'mojolearn.release-identity-execution.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def vendor_for(label):
    require(bool(re.fullmatch(r'[a-z0-9][a-z0-9_.-]*', label)), 'invalid producer label')
    if label == 'cpu':
        return 'cpu'
    for vendor in ('apple', 'nvidia', 'amd'):
        if label.startswith(vendor + '-'):
            return vendor
    raise ValueError('expected apple-*, nvidia-*, amd-* or cpu label')


def policy_environment(environment):
    """The release producer always collects the ordinary full fixture/part set."""
    for key in ('FIXTURES', 'PARTIAL_COLUMN', 'SKIP', 'ONLY_CHANGED', 'SHARD',
                'MOJOLEARN_IDENTITY_N', 'MOJOLEARN_IDENTITY_WIDE', 'MOJOLEARN_PAR_DEVICES'):
        require(not environment.get(key), key + ' narrows or overrides the release column; remove it')
    require(environment.get('REPEATS', '1') == '1', 'release identity needs exactly one repeat')
    require(environment.get('ALLOW_DIRTY', '0') != '1', 'release identity needs committed source')
    for key, value in environment.items():
        if key.startswith('MOJOLEARN_') and 'SABOTAGE' in key:
            require(value in ('', '0'), 'sabotage is not a release identity column: ' + key)


def source_preflight(root, expected):
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == expected, f'execution checkout {head} differs from ledger source {expected}')
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                    '--untracked-files=no'], text=True).strip()
    require(not dirty, 'execution checkout has tracked changes; commit or restore them before identity')
    return head


def applicability(root, vendor, source):
    """Use the frozen release selector's existing device/host applicability rule."""
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == source, 'plan checkout must name the ledger source so applicability is not guessed')
    backend = dict(apple='metal', nvidia='cuda', amd='hip', cpu='cpu')[vendor]
    code = ("import json,sys; sys.path.insert(0,sys.argv[1]); "
            "import lane_applicability,verify_lanes; "
            "print(json.dumps(lane_applicability.degenerate(verify_lanes.PASS_COLUMN[sys.argv[2]])))")
    return json.loads(subprocess.check_output([sys.executable, '-c', code, str(Path(root) / 'tools'), backend],
                                             cwd=root, text=True))


def plan(ledger, vendor, lanes=None):
    doc, rows = ledger.read(), ledger.plan()
    units = {name: wanted for name, wanted in doc['requirements'].items()
             if wanted.get('kind') == 'identity' and wanted.get('inputs', {}).get('unit', {}).get('vendor') == vendor}
    dropped = applicability(ledger.source_root or identity.ROOT, vendor, doc['source_revision'])
    asked = {f'identity-{vendor}/{lane}' for lane in lanes} if lanes is not None else set(units)
    require(asked and not asked - set(units), 'empty or unknown identity lane selection')
    excluded = {name: dropped[identity.unit_lane(units[name])] for name in sorted(asked)
                if identity.unit_lane(units[name]) in dropped}
    names = sorted(asked - set(excluded))
    require(names, 'no applicable identity lanes in this selection')
    selected = {name: dict(action=rows[name]['action'], reasons=rows[name].get('reasons', []),
                           fingerprint=units[name]['fingerprint']) for name in names}
    for name in selected:
        if not identity.supports_requirement(units[name]):
            selected[name] = dict(selected[name], action='complete-provenance',
                                  reasons=['identity requirement lacks complete source/coverage provenance'])
    blocked = [name for name, row in selected.items() if row['action'] not in ('run', 'repair', 'reuse')]
    return dict(schema=SCHEMA, source_revision=doc['source_revision'], vendor=vendor,
                selected=selected, counts=dict(Counter(row['action'] for row in selected.values())),
                excluded=excluded, blocked=blocked, executable=not blocked)


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('another identity producer owns ' + str(path)) from error
        try:
            yield lock
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def labels_for(label, wanted):
    require(not label.startswith('nvidia-ptx-'), 'start the NVIDIA column with its native label; PTX is included automatically')
    labels = [label]
    if vendor_for(label) == 'nvidia' and any(identity.PTX_CLASS in identity.required_classes(w) for w in wanted):
        stem = label[len('nvidia-'):].rsplit('-', 1)[0]
        labels.append('nvidia-ptx-' + stem + '-sm80')
    return labels


def launch(ledger, root, label, out, lanes=None, *, environment=None, runner=subprocess.run):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(not out.is_relative_to(root), 'retain identity output outside the execution checkout')
    env = dict(os.environ if environment is None else environment)
    policy_environment(env)
    vendor = vendor_for(label)
    require(not label.startswith('nvidia-ptx-'), 'start the NVIDIA column with its native label; PTX is included automatically')
    planned = plan(ledger, vendor, lanes)
    require(planned['executable'], 'selected identity units are active or need restored provenance: ' + ', '.join(planned['blocked']))
    if planned['counts'].get('run', 0) + planned['counts'].get('repair', 0):
        source_preflight(root, planned['source_revision'])
    out.mkdir(parents=True, exist_ok=True)
    with exclusive(out / ('identity-producer-' + vendor + '.lock')):
        # Reevaluate all selected units atomically. A concurrent active unit
        # refuses the whole batch; unchanged PASS units return None.
        ids = ledger.begin_many(planned['selected'])
        doc = ledger.read()
        runs = {r['requirement']: r for r in doc['runs'] if r['id'] in ids.values()}
        if not runs:
            return dict(planned, status='reused', launched=0)
        stamp = identity.stamp().replace(':', '')
        path = out / f'identity-launch-{vendor}-{stamp}.json'
        selected_lanes = sorted(identity.unit_lane(doc['requirements'][name]) for name in runs)
        labels = labels_for(label, [doc['requirements'][name] for name in runs])
        execution = dict(schema=SCHEMA, root=str(root), out=str(out), labels=labels,
                         owner_pid=os.getpid(), lock=str(out / ('identity-producer-' + vendor + '.lock')),
                         all_parts=list(identity.ALL_PARTS), fixtures='all', repeats=1,
                         reasons={name: r['reasons'] for name, r in runs.items()})
        record = dict(schema=identity.LAUNCH_SCHEMA, phase='before-execution', vendor=vendor,
                      ledger=str(ledger.path.resolve()), source_revision=planned['source_revision'],
                      launched_at=identity.stamp(), lanes=selected_lanes, execution=execution,
                      runs={name: dict(run_id=r['id'], fingerprint=r['fingerprint'], source_revision=r['source_revision'])
                            for name, r in runs.items()})
        identity.write_json(path, record)
        outcome = dict(schema=SCHEMA, launch=identity.reference(path), status='unstarted', columns=[])
        outcome_path = out / f'identity-execution-{vendor}-{stamp}.json'
        try:
            require(all(r['source_revision'] == planned['source_revision'] for r in runs.values()),
                    'ledger source changed during admission; no producer was started')
            source_preflight(root, planned['source_revision'])
            env.update(LANES=','.join(selected_lanes), REPEATS='1', SCOPE='all',
                       MOJOLEARN_IDENTITY_LAUNCH=str(path),
                       MOJOLEARN_IDENTITY_GUARD=str(Path(__file__).resolve()))
            for column_label in labels:
                log = out / f'identity-controller-{column_label}-{stamp}.log'
                outcome['status'] = 'running'
                identity.write_json(outcome_path, outcome)
                with log.open('w') as stream:
                    completed = runner(['bash', str(root / identity.TOOL), column_label, str(out)],
                                       cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
                outcome['columns'].append(dict(label=column_label, exit_code=completed.returncode, log=str(log)))
                if completed.returncode:
                    outcome['status'] = 'producer-failed-awaiting-admission'
                    break
            else:
                outcome['status'] = 'columns-recorded-awaiting-cross-vendor-admission'
        except BaseException as error:
            outcome['error'] = str(error)
            outcome['status'] = ('unstarted' if isinstance(error, OSError) and not outcome['columns']
                                 else 'interrupted' if outcome['status'] == 'running' else 'unstarted')
            raise
        finally:
            identity.write_json(outcome_path, outcome)
            if outcome['status'] == 'unstarted':
                # No product execution occurred. Close only this failed launch,
                # with zero runtime coverage, so the next normal invocation can
                # repair it without manual ledger editing.
                ledger.record_many([dict(run_id=r['id'], status='failed',
                    paths=[str(path), str(outcome_path)],
                    summary='Identity producer was not started: ' + outcome.get('error', 'controller refusal')
                            + '; zero identity slots executed',
                    exit_code=2, expected=0, completed=0, failed=0, skipped=0)
                    for r in runs.values()])
        return dict(planned, launched=len(runs), launch=str(path), outcome=str(outcome_path), status=outcome['status'])


def validate_launch(path, root, label, out, *, environment=None, consume=False):
    """Producer preflight: bind the running controller, active units and full coverage."""
    env = dict(os.environ if environment is None else environment)
    policy_environment(env)
    record = identity.load(path)
    execution = record.get('execution', {})
    require(record.get('schema') == identity.LAUNCH_SCHEMA and record.get('phase') == 'before-execution'
            and execution.get('schema') == SCHEMA, 'producer needs a guarded before-execution launch')
    require(execution.get('root') == str(Path(root).resolve()) and execution.get('out') == str(Path(out).resolve())
            and label in execution.get('labels', []) and record.get('vendor') == vendor_for(label),
            'launch belongs to another checkout, output or column')
    require(execution.get('all_parts') == list(identity.ALL_PARTS) and execution.get('fixtures') == 'all'
            and execution.get('repeats') == 1, 'launch must collect every part and fixture once')
    require(env.get('LANES', '').split(',') == record.get('lanes'), 'producer lanes differ from guarded launch')
    source_preflight(root, record['source_revision'])
    try:
        os.kill(execution['owner_pid'], 0)
        with exclusive(Path(execution['lock'])):
            raise ValueError('identity controller lock is not held')
    except ValueError as error:
        require(str(error).startswith('another identity producer owns '), str(error))
    ledger = RequirementsLedger(record['ledger'])
    doc = ledger.read()
    require(doc.get('source_revision') == record['source_revision'], 'ledger source changed after launch')
    active = {r['id']: r for r in doc['runs'] if r['id'] not in {a.get('run_id') for a in doc.get('attempts', [])}}
    require(record.get('runs'), 'launch contains no active identity units')
    require(sorted(identity.unit_lane(doc['requirements'][n]) for n in record['runs']) == record.get('lanes'),
            'launch lane list differs from its active units')
    for name, claimed in record['runs'].items():
        r = active.get(claimed.get('run_id'), {})
        require(r.get('requirement') == name and r.get('fingerprint') == claimed.get('fingerprint')
                and r.get('source_revision') == record['source_revision']
                and doc['requirements'][name]['fingerprint'] == r.get('fingerprint'),
                'launch unit is no longer active for these inputs: ' + name)
    if consume:
        claim = Path(str(path) + '.' + label + '.claimed')
        try:
            with claim.open('x') as stream:
                json.dump(dict(launch=identity.reference(path), label=label, controller=execution['owner_pid']), stream)
        except FileExistsError as error:
            raise ValueError('this launch already started column ' + label) from error
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan', 'run', 'validate'))
    p.add_argument('--root', type=Path, default=identity.ROOT)
    p.add_argument('--ledger', type=Path)
    p.add_argument('--label', required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--lanes', help='optional explicit subset; default derives from release requirements')
    p.add_argument('--launch', type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == 'validate':
            require(args.launch and args.out, 'validate needs --launch and --out')
            validate_launch(args.launch, args.root, args.label, args.out, consume=True)
            print('identity launch validated: all parts, all fixtures, active units')
            return 0
        require(args.ledger is not None, '--ledger is required')
        ledger = RequirementsLedger(args.ledger, source_root=args.root)
        lanes = args.lanes.split(',') if args.lanes else None
        if args.command == 'plan':
            result = plan(ledger, vendor_for(args.label), lanes)
        else:
            require(args.out is not None, 'run needs --out')
            result = launch(ledger, args.root, args.label, args.out, lanes)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get('executable') and result.get('status') != 'producer-failed-awaiting-admission' else 1
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('identity launch refused: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
