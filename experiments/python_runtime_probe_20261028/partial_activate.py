#!/usr/bin/env python3
"""Activate the host groups that DID build when tools/development_build.py
refused to activate an incomplete set. Probe-only: the index it writes lists
only built groups, so missing families load as the package's no-binding
stubs and their endpoints are reported as refused/errored by the probe.
Run with the same environment as the build so the plan keys match.
Writes partial_activation.json (built / missing groups) to argv[1]."""
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
cache = Path(os.environ.get('PROBE_DEV_CACHE', str(Path.home() / '.cache/mojolearn-development')))
plan = db.make_plan(REPO, cache, ['host'], dict(os.environ))
built, missing, files = [], [], {}
host = REPO / 'python/mojolearn/host'
for row in plan['groups']:  # driver-loop: over build groups
    source = db.cached_artifact(cache, row)
    if source is None:
        missing.append(row['stem'])
        continue
    relative = '.development/' + row['key'] + '/' + source.name
    destination = host / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    files[relative] = row['members']
    built.append(row['stem'])
support = {}
for row in plan['support']:  # driver-loop: over support artifacts
    source = db.cached_artifact(cache, row)
    support[row['name']] = db.install_support(row, source, REPO) if source is not None else None
db.atomic_json(host / '_mojolearn_merged.json', {'format': merged.INDEX_FORMAT, 'layout': 'development',
               'files': files, 'provenance': {'partial_probe_activation': True}})
out.write_text(json.dumps(dict(built=built, missing=missing, support=support), indent=1) + '\n')
print('partial activation: built=%d missing=%d' % (len(built), len(missing)))
