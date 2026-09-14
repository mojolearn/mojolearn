#!/usr/bin/env python3
"""Generate citation version/date from the package version and dated changelog.

No network or wall clock: rebuilding a release must preserve its citation.
An unreleased heading removes date-released until publication is recorded.
"""
import argparse
from datetime import date
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def expected(root):
    text = (root / 'CITATION.cff').read_text(encoding='utf-8')
    versions = re.findall(r'^__version__ = "([^"]+)"$',
                          (root / 'python/mojolearn/_version.py').read_text(), re.M)
    if len(versions) != 1:
        raise ValueError('Expected one package __version__')
    version = versions[0]
    entries = re.findall(r'^## ' + re.escape(version)
                         + r' \((published|unreleased) (\d{4}-\d{2}-\d{2})\)\s*$',
                         (root / 'CHANGELOG.md').read_text(), re.M)
    if len(entries) != 1:
        raise ValueError(f'Expected one dated CHANGELOG heading for {version}')
    status, release_date = entries[0]
    date.fromisoformat(release_date)
    if len(re.findall(r'^version:.*$', text, re.M)) != 1:
        raise ValueError('Expected one top-level CITATION.cff version')
    if len(re.findall(r'^date-released:.*$', text, re.M)) > 1:
        raise ValueError('Duplicate top-level CITATION.cff date-released')
    text = re.sub(r'^date-released:.*\n?', '', text, flags=re.M)
    replacement = f'version: "{version}"'
    if status == 'published':
        replacement += f'\ndate-released: "{release_date}"'
    return re.sub(r'^version:.*$', lambda _: replacement, text, flags=re.M)


def sync(root=ROOT, write=False):
    root = Path(root)
    path = root / 'CITATION.cff'
    try:
        new = expected(root)
    except (ValueError, OSError) as exc:
        print(f'citation_metadata: {exc}')
        return 1
    if path.read_text(encoding='utf-8') != new:
        if not write:
            print('citation_metadata: stale citation version/date; run '
                  '`python3 tools/docs_facts.py --write`')
            return 1
        path.write_text(new, encoding='utf-8')
        print('citation_metadata: updated CITATION.cff from package version and CHANGELOG')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    raise SystemExit(sync(args.root, args.write))
