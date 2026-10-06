#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit source-only Apple FAST A/B configurations. Never build, check or run them.

This is offline experiment glue, not estimator runtime code. The output is a
configuration description; it conveys no compilation, reach, quality or speed
result. It deliberately has no subprocess, network, build or execution stage.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# G02's required MMA rollback prevents G01's MMA geometry from being reached.
# These are source-known route constraints, not measured interaction results.
EXCLUSIVE_CARDS = (frozenset(('AFCL-G01', 'AFCL-G02')),)


def catalog() -> dict[str, dict]:
    entries = {}
    for path in sorted((HERE / 'lanes').glob('*.json')):
        for entry in json.loads(path.read_text())['entries']:
            entries[entry['id']] = entry
    return entries


def configuration(entries: list[dict], choices: tuple[str, ...]) -> dict:
    defines = set()
    excluded = {"MOJOLEARN_NUMERIC_IDENTICAL", "MOJOLEARN_NUMERIC_DETERMINISTIC"}
    env = {}
    for entry, arm in zip(entries, choices):
        defines.update(entry[f'{arm}_defines'])
        excluded.update(entry.get('defines_absent_in_both_arms', []))
        for key, value in entry.get(f'{arm}_env', {}).items():
            if key in env and env[key] != value:
                raise ValueError(f'Conflicting {key} requirements; emit these cards separately')
            env[key] = value
    conflicts = {define.split('=', 1)[0] for define in defines} & excluded
    if conflicts:
        raise ValueError('Selected caller routes require these defines absent: ' + ', '.join(sorted(conflicts)))
    return {
        'cards': {entry['id']: arm for entry, arm in zip(entries, choices)},
        'defines': sorted(defines),
        'environment': env,
        'status': 'source_configuration_only_not_executed',
        'required_numeric_mode': 'fast',
        'required_target': 'apple',
        'exclude_defines': sorted(excluded),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ids', nargs='*', help='Card IDs, e.g. AFCL-L01 AFCL-L02')
    parser.add_argument('--list', action='store_true', help='Print source catalog descriptions only')
    parser.add_argument('--factorial', action='store_true', help='Emit all A/B combinations for up to six interacting cards')
    parser.add_argument('--output', type=Path, help='Write JSON configuration; default stdout')
    args = parser.parse_args()
    entries = catalog()
    if args.list:
        result = [{'id': e['id'], 'title': e.get('title', ''), 'status': e['status']} for e in entries.values()]
    else:
        if not args.ids:
            parser.error('Select one or more card IDs, or use --list')
        if len(set(args.ids)) != len(args.ids):
            parser.error('Specify each card once')
        unknown = sorted(set(args.ids) - entries.keys())
        if unknown:
            parser.error('Unknown card IDs: ' + ', '.join(unknown))
        chosen = [entries[id] for id in args.ids]
        for group in EXCLUSIVE_CARDS:
            if group <= set(args.ids):
                parser.error('Mutually exclusive caller routes: ' + ', '.join(sorted(group)))
        if args.factorial and len(chosen) > 6:
            parser.error('Factorial descriptions are limited to six cards (64 configurations)')
        arms = itertools.product(('baseline', 'candidate'), repeat=len(chosen)) if args.factorial else (
            ('baseline',) * len(chosen), ('candidate',) * len(chosen)
        )
        try:
            configs = [configuration(chosen, arm) for arm in arms]
        except ValueError as exc:
            parser.error(str(exc))
        result = {
            'schema': 1,
            'scope': 'Apple FAST classical ML',
            'status': 'uncompiled_unverified_unmeasured',
            'execution_supported': False,
            'configurations': configs,
            'card_details': chosen,
            'requirements': [
                'Use the intended FAST Apple binding; emitted defines are additions to its existing recipe.',
                'Start from a clean per-arm configuration; do not inherit unrelated AFCL flags or OFF overrides.',
                'Resolve and record every affected full-workload recipe and all intrinsic caps before future execution.',
                'Matching prerequisite flags enable an existing path in both arms; they are not candidate wins.',
                'All quality, actual route reach, compilation and performance evidence remains pending.',
            ],
        }
    output = json.dumps(result, indent=2) + '\n'
    if args.output is None:
        print(output, end='')
    else:
        args.output.write_text(output)


if __name__ == '__main__':
    main()
