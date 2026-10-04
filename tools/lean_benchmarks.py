#!/usr/bin/env python3
"""Sparse worktrees retaining recent results; never overwrite benchmark evidence."""
import argparse
import collections
import datetime as dt
import json
import pathlib
import re
import subprocess

DATE = re.compile(r'(?<!\d)(20\d{2})-?(\d{2})-?(\d{2})(?!\d)')
DEFAULT_REPO = str(pathlib.Path(__file__).resolve().parents[1])


def git(repo, *args, input=None):
    return subprocess.check_output(['git', '-C', str(repo), *args], input=input, text=True)


def select(entries, today, days, count, pins):
    runs = {}
    for path, size in entries:
        parts = pathlib.PurePosixPath(path).parts
        # Date-bearing directories identify whole runs; leave undated data and
        # loose files included rather than guessing whether they are obsolete.
        for i, part in enumerate(parts[2:-1], 2):
            match = DATE.search(part)
            if not match:
                continue
            try:
                date = dt.date(*map(int, match.groups()))
            except ValueError:
                continue
            run = '/'.join(parts[:i+1])
            family = '/'.join(parts[:i]) if i > 2 else DATE.sub('{date}', part)
            item = runs.setdefault(run, {'path': run, 'family': family,
                                         'date': date.isoformat(), 'bytes': 0})
            item['bytes'] += size
            break
    groups = collections.defaultdict(list)
    for item in runs.values():
        groups[item['family']].append(item)
    recent = set()
    for group in groups.values():
        group.sort(key=lambda r: (r['date'], r['path']), reverse=True)
        recent.update(r['path'] for r in group[:count])
    cutoff = (today - dt.timedelta(days=days-1)).isoformat()
    for item in runs.values():
        path = item['path']
        reasons = []
        if path in recent:
            reasons.append('latest-per-family')
        if item['date'] >= cutoff:
            reasons.append('recent-date')
        if any(path == p or path.startswith(p+'/') or p.startswith(path+'/') for p in pins):
            reasons.append('pinned')
        item['included'] = bool(reasons)
        item['reasons'] = reasons
    return sorted(runs.values(), key=lambda r: (r['family'], r['date'], r['path']))


def literal(path):
    return ''.join('\\'+c if c in '\\*?[]!# ' else c for c in path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='new worktree path, or existing helper-managed worktree with --refresh')
    parser.add_argument('branch', nargs='?')
    parser.add_argument('start', nargs='?', default='HEAD')
    parser.add_argument('--repo', default=DEFAULT_REPO)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--days', type=int)
    parser.add_argument('--count', type=int)
    parser.add_argument('--keep', action='append', default=[], help='pin a tracked bench/results path; repeatable and retained on refresh')
    args = parser.parse_args()
    target = pathlib.Path(args.path).expanduser().absolute()
    old = {}
    if args.refresh:
        if args.branch:
            parser.error('--refresh does not take a branch')
        meta = pathlib.Path(git(target, 'rev-parse', '--absolute-git-dir').strip())/'lean-benchmarks.json'
        if not meta.exists():
            parser.error('refresh requires a worktree created by this helper')
        old = json.loads(meta.read_text())
        if git(target, 'status', '--porcelain', '--untracked-files=all').strip():
            parser.error('commit or preserve working changes before refreshing')
        repo, ref = target, 'HEAD'
    else:
        if not args.branch:
            parser.error('a new branch is required')
        if target.exists():
            parser.error('new worktree path must not exist')
        repo, ref = args.repo, args.start
    days = args.days if args.days is not None else old.get('days', 14)
    count = args.count if args.count is not None else old.get('count', 3)
    if days < 1 or count < 1:
        parser.error('--days and --count must be positive')
    # Resolve once so a concurrently moving branch cannot change our selection.
    head = git(repo, 'rev-parse', '--verify', ref+'^{commit}').strip()
    entries = []
    for record in git(repo, 'ls-tree', '-r', '-l', '-z', head, 'bench/results').split('\0'):
        if record:
            fields, path = record.split('\t', 1)
            size = fields.split()[-1]
            if size.isdigit():
                entries.append((path, int(size)))
    pins = sorted(set(old.get('pins', []) + [p.rstrip('/') for p in args.keep]))
    for pin in pins:
        if not pin.startswith('bench/results/') or '..' in pathlib.PurePosixPath(pin).parts:
            parser.error('pins must be repo-relative paths under bench/results/')
        if not any(p == pin or p.startswith(pin+'/') for p, _ in entries):
            parser.error('pin not found in selected commit: '+pin)
    runs = select(entries, dt.date.today(), days, count, pins)
    excluded = [r for r in runs if not r['included']]
    # Include by default, excluding specific old runs. Future runs stay visible
    # on checkout/pull without needing to regenerate an allowlist.
    patterns = '/*\n' + ''.join('!/'+literal(r['path'])+'/\n' for r in excluded)
    if not args.refresh:
        subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--no-checkout', '-b', args.branch, str(target), head], check=True)
        meta = pathlib.Path(git(target, 'rev-parse', '--absolute-git-dir').strip())/'lean-benchmarks.json'
    if args.refresh:
        ignored = git(target, 'ls-files', '--others', '--ignored', '--exclude-standard', '-z').split('\0')
        prefixes = tuple(r['path']+'/' for r in excluded)
        if prefixes and any(p.startswith(prefixes) for p in ignored if p):
            parser.error('ignored files exist in history being excluded; preserve them before refreshing')
    git(target, 'sparse-checkout', 'set', '--no-cone', '--stdin', input=patterns)
    if not args.refresh:
        git(target, 'checkout', args.branch)
    report = {'head': head, 'as_of': dt.date.today().isoformat(), 'days': days,
              'count': count, 'pins': pins, 'runs': runs,
              'excluded_bytes': sum(r['bytes'] for r in excluded)}
    meta.write_text(json.dumps(report, indent=2)+'\n')
    latest = collections.defaultdict(list)
    for r in sorted(runs, key=lambda r: (r['date'], r['path']), reverse=True):
        if r['included']:
            latest[r['family']].append(r['path'])
    (meta.parent/'latest-benchmarks.json').write_text(json.dumps(dict(latest), indent=2)+'\n')
    print(f'Included {len(runs)-len(excluded)} dated runs; excluded {len(excluded)} older runs ({report["excluded_bytes"]/2**20:.1f} MiB). Undated results remain included.')
    print(f'Latest included runs: {meta.parent / "latest-benchmarks.json"}')
    print('Selection uses dates in paths, not benchmark success or comparability. Pin comparison baselines with --keep.')


if __name__ == '__main__':
    main()
