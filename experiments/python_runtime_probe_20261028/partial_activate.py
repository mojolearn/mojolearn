#!/usr/bin/env python3
"""Activate the host groups that DID build when tools/development_build.py
refused to activate an incomplete set. Probe-only: the index it writes lists
only built groups, so missing families load as the package's no-binding
stubs and their endpoints are reported as refused/errored by the probe.
Run with the same environment as the build so the plan keys match.
PROBE_PLANS (JSON list of {"cache": dir, "flags": MOJOLEARN_MOJO_BUILD_FLAGS})
names several caches built with different flags; the first one holding a
group's artifact serves it. Writes partial_activation.json to argv[1]."""
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools'))
import development_build as db  # noqa: E402
import merged_bindings as merged  # noqa: E402

out = Path(sys.argv[1])
specs = json.loads(os.environ.get('PROBE_PLANS') or json.dumps(
    [dict(cache=str(Path.home() / '.cache/mojolearn-development'),
          flags=os.environ.get('MOJOLEARN_MOJO_BUILD_FLAGS', ''))]))
plans = []
for spec in specs:  # driver-loop: over build caches
    env = dict(os.environ, MOJOLEARN_MOJO_BUILD_FLAGS=spec['flags'])
    if not spec['flags']:
        env.pop('MOJOLEARN_MOJO_BUILD_FLAGS')
    if spec.get('jobs'):
        env['MOJOLEARN_COMPILE_JOBS'] = str(spec['jobs'])
    plans.append((Path(spec['cache']), spec['flags'], db.make_plan(REPO, Path(spec['cache']), ['host'], env)))
plan = plans[0][2]
built, missing, files, flags_used = [], [], {}, {}
host = REPO / 'python/mojolearn/host'
for row0 in plan['groups']:  # driver-loop: over build groups
    source = None
    for cache, flags, other in plans:  # driver-loop: over build caches
        row = next(r for r in other['groups'] if r['stem'] == row0['stem'])
        source = db.cached_artifact(cache, row)
        if source is not None:
            flags_used[row['stem']] = flags
            break
    if source is None:
        missing.append(row0['stem'])
        continue
    relative = '.development/' + row['key'] + '/' + source.name
    destination = host / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    files[relative] = row['members']
    built.append(row['stem'])
support = {}
for row0 in plan['support']:  # driver-loop: over support artifacts
    support[row0['name']] = None
    for cache, flags, other in plans:  # driver-loop: over build caches
        row = next(r for r in other['support'] if r['name'] == row0['name'])
        source = db.cached_artifact(cache, row)
        if source is not None:
            support[row0['name']] = db.install_support(row, source, REPO)
            break
db.atomic_json(host / '_mojolearn_merged.json', {'format': merged.INDEX_FORMAT, 'layout': 'development',
               'files': files, 'provenance': {'partial_probe_activation': True}})
out.write_text(json.dumps(dict(built=built, missing=missing, support=support, flags=flags_used), indent=1) + '\n')
print('partial activation: built=%d missing=%d' % (len(built), len(missing)))
