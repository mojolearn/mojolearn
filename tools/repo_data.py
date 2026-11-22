#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Bulk data that tools read, kept OUT of the repository and resolved by its repo path.

    python3 tools/repo_data.py path   <repo-relative path>   # print the local path (fetches when absent)
    python3 tools/repo_data.py fetch  <bucket>               # R2 -> evidence root, every file verified
    python3 tools/repo_data.py verify <bucket>               # local copy against the tracked manifest
    python3 tools/repo_data.py pack   <bucket> [--from DIR]  # copy into the evidence root, write the
                                                             # tracked manifest, build + pin the tar
    python3 tools/repo_data.py push   <bucket>               # upload the pinned tar, read it back

WHY (Andrew, 2026-11-25): measurements are kept, they are costly, but their bulk does not belong in
the checkout. A bucket is a repo directory whose files a tool still reads. Its bytes live in the
evidence root (`$MOJOLEARN_EVIDENCE_ROOT/repo-data/<same repo path>`, default
`~/mojolearn-evidence/repo-data/`) and in R2; the repository keeps only the per-file manifest
`bench/results/dataset_store/repo-data/<bucket with / as @>.tsv` (path, bytes, sha256), so the
claim stays reviewable in a diff while the bytes leave.

ONE STORE. R2 transport and pins are `tools/dataset_store.sh`, the store the boxes already use for
datasets: each bucket travels as ONE deterministic tar under key `repo-data/<bucket>.tar`, pinned
in `bench/results/dataset_store/manifest.tsv`. A rented box gets it the way it gets a dataset:
`sh tools/dataset_store.sh stage "<ssh target>" repo-data/<bucket>.tar` (a presigned URL, no
credential leaves this machine), then `python3 tools/repo_data.py install <bucket> <tar>` there.

RESOLVED ONCE PER RUN. `resolve()` checks a bucket once per process (cached): every file's size on
each run, every sha256 on first use and after a fetch (a stamp file records the manifest digest it
verified). A listed file that is missing or changed is refused, never re-fetched over the evidence
copy: the evidence root is never overwritten or deleted by this tool.
"""
import argparse
import concurrent.futures
import functools
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / 'bench/results/dataset_store/repo-data'
STORE_MANIFEST = ROOT / 'bench/results/dataset_store/manifest.tsv'
STORE_TOOL = ROOT / 'tools/dataset_store.sh'
#: Transport staging for tars (download and pack): the cache directory `tools/dataset_store.sh`'s catalog maps
#: each `repo-data/*.tar` key to (HOME/.cache/mojolearn/repo-data/, /root/.cache/... on a box), so a transient
#: tar never lands in (and never has to be deleted from) the evidence root.
STAGING = Path(os.path.expanduser('~/.cache/mojolearn/repo-data'))
STAMP = '.repo-data-verified'

#: The buckets that left the tree: repo-relative directory -> R2 key of its tar.
BUCKETS = {
    'experiments/six_lane_integration/measurements':
        'repo-data/experiments/six_lane_integration/measurements.tar',
}


class RepoDataError(SystemExit):
    """A refusal in words: what is missing and the exact command that fetches it."""


def evidence_root():
    return Path(os.environ.get('MOJOLEARN_EVIDENCE_ROOT') or os.path.expanduser('~/mojolearn-evidence')) / 'repo-data'


def workers():
    # Hashing is independent per file and hashlib releases the GIL, so the files are split over the cores.
    return max(1, int(os.environ.get('MOJOLEARN_REPO_DATA_WORKERS') or os.cpu_count() or 1))


def manifest_path(bucket):
    return MANIFEST_DIR / (bucket.replace('/', '@') + '.tsv')


def bucket_of(relpath):
    rel = Path(relpath).as_posix().strip('/')
    for bucket in BUCKETS:
        if rel == bucket or rel.startswith(bucket + '/'):
            return bucket, rel[len(bucket):].lstrip('/')
    raise RepoDataError(f'repo_data: {relpath} is not under a moved bucket ({", ".join(sorted(BUCKETS))})')


def fetch_command(bucket):
    return f'python3 tools/repo_data.py fetch {bucket}'


def read_manifest(bucket):
    path = manifest_path(bucket)
    if not path.is_file():
        raise RepoDataError(f'repo_data: no tracked manifest {path.relative_to(ROOT)} for bucket {bucket}')
    raw = path.read_bytes()
    rows = []
    for line in raw.decode('utf-8').splitlines():
        if not line or line.startswith('#') or line == 'path\tbytes\tsha256':
            continue
        rel, size, sha = line.split('\t')
        rows.append((rel, int(size), sha))
    return rows, hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_tree(base, rows, full):
    """Problems of the tree at `base` against the manifest rows; sha256 only when `full`."""
    problems = []
    for rel, size, _ in rows:
        p = base / rel
        if not p.is_file():
            problems.append(f'{rel} is missing')
        elif p.stat().st_size != size:
            problems.append(f'{rel} has {p.stat().st_size} bytes, manifest {size}')
    if full and not problems:
        with concurrent.futures.ThreadPoolExecutor(workers()) as pool:
            for (rel, _, sha), got in zip(rows, pool.map(sha_file, [base / r for r, _, _ in rows])):
                if got != sha:
                    problems.append(f'{rel} sha256 {got[:16]}, manifest {sha[:16]}')
    return problems


@functools.cache
def bucket_root(bucket):
    """The verified local copy of `bucket`, fetched from R2 when absent. Once per process."""
    rows, digest = read_manifest(bucket)
    base = evidence_root() / bucket
    if not base.exists():
        fetch(bucket)
    stamp = base / STAMP
    full = not stamp.is_file() or stamp.read_text().strip() != digest
    problems = check_tree(base, rows, full)
    if problems:
        raise RepoDataError(
            f'repo_data: REFUSED: the local copy of {bucket} at {base} does not match '
            f'{manifest_path(bucket).relative_to(ROOT)}: ' + '; '.join(problems[:8])
            + f'{" ..." if len(problems) > 8 else ""}. Nothing was changed. Move that copy aside yourself and run '
            f'`{fetch_command(bucket)}`.')
    if full:
        stamp.write_text(digest + '\n')
    return base


def resolve(relpath):
    """Repo-relative path of moved data -> its local path under the evidence root (verified once per run)."""
    bucket, inner = bucket_of(relpath)
    base = bucket_root(bucket)
    return base / inner if inner else base


def pinned(key):
    for line in STORE_MANIFEST.read_text().splitlines():
        fields = line.split('\t')
        if fields and fields[0] == key:
            return int(fields[1]), fields[2]
    return None


def store(*args, check=True):
    return subprocess.run(['sh', str(STORE_TOOL), *args], check=check)


def install(bucket, tar_path):
    """Extract a verified bucket tar into the evidence root; every member checked first."""
    rows, digest = read_manifest(bucket)
    want = {rel: (size, sha) for rel, size, sha in rows}
    base = evidence_root() / bucket
    if base.exists():
        raise RepoDataError(f'repo_data: {base} already exists; `python3 tools/repo_data.py verify {bucket}` checks it')
    partial = base.with_name(base.name + '.partial')
    if partial.exists():
        raise RepoDataError(f'repo_data: {partial} is left from an interrupted install; move it aside and retry')
    seen = set()
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if not m.isfile() or m.name not in want or m.name in seen:
                raise RepoDataError(f'repo_data: {tar_path}: unexpected member {m.name!r}')
            body = tar.extractfile(m).read()
            size, sha = want[m.name]
            if len(body) != size or hashlib.sha256(body).hexdigest() != sha:
                raise RepoDataError(f'repo_data: {tar_path}: {m.name} does not match the manifest')
            target = partial / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            seen.add(m.name)
    if seen != set(want):
        raise RepoDataError(f'repo_data: {tar_path} lacks {len(set(want) - seen)} listed file(s)')
    (partial / STAMP).write_text(digest + '\n')
    partial.rename(base)
    return base


def tar_name(bucket):
    return bucket.replace('/', '@') + '.tar'


def fetch(bucket):
    key = BUCKETS[bucket]
    if pinned(key) is None:
        raise RepoDataError(f'repo_data: {key} has no pin in {STORE_MANIFEST.relative_to(ROOT)}')
    STAGING.mkdir(parents=True, exist_ok=True)
    tar_path = STAGING / tar_name(bucket)
    # dataset_store.sh pull presigns with ~/.mojolearn_r2, downloads resumably and verifies size + sha256.
    if store('pull', key, str(tar_path), check=False).returncode != 0:
        raise RepoDataError(
            f'repo_data: REFUSED: {bucket} is not in the checkout and could not be fetched from R2 key {key}. '
            f'It needs the aws CLI and ~/.mojolearn_r2 on this machine: `{fetch_command(bucket)}`. On a rented '
            f'box: from the laptop `sh tools/dataset_store.sh stage "<ssh target>" {key}`, then on the box '
            f'`python3 tools/repo_data.py install {bucket} /root/.cache/mojolearn/repo-data/{tar_name(bucket)}`.')
    base = install(bucket, tar_path)
    tar_path.unlink()
    return base


def write_tar(base, rows, tar_path):
    """A deterministic tar of the manifest's files (sorted, mtime 0, owner 0, mode 0644), streamed to disk."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for rel, size, _ in sorted(rows):
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode = size, 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with open(base / rel, 'rb') as f:
                tar.addfile(info, f)


def write_pin(key, size, sha):
    lines = [line for line in STORE_MANIFEST.read_text().splitlines() if line and line.split('\t')[0] != key]
    lines.append(f'{key}\t{size}\t{sha}')
    STORE_MANIFEST.write_text('\n'.join(sorted(lines)) + '\n')


def pack(bucket, source=None):
    """Copy the bucket into the evidence root (verified), write its tracked manifest, build and pin the tar."""
    src = Path(source) if source else ROOT / bucket
    base = evidence_root() / bucket
    files = sorted(p for p in src.rglob('*') if p.is_file() and p.name != STAMP)
    with concurrent.futures.ThreadPoolExecutor(workers()) as pool:
        shas = list(pool.map(sha_file, files))
    rows = [(p.relative_to(src).as_posix(), p.stat().st_size, s) for p, s in zip(files, shas)]
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path(bucket).write_text('path\tbytes\tsha256\n' + ''.join(f'{r}\t{n}\t{s}\n' for r, n, s in rows))
    if src.resolve() != base.resolve():
        if base.exists():
            problems = check_tree(base, rows, True)
            if problems:
                raise RepoDataError(f'repo_data: {base} exists and differs from {src}: ' + '; '.join(problems[:8]))
        else:
            for p in files:
                target = base / p.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(p.read_bytes())
    problems = check_tree(base, rows, True)
    if problems:
        raise RepoDataError(f'repo_data: copy of {bucket} did not verify: ' + '; '.join(problems[:8]))
    (base / STAMP).write_text(read_manifest(bucket)[1] + '\n')
    tar_path = STAGING / tar_name(bucket)
    write_tar(base, rows, tar_path)
    size, sha = tar_path.stat().st_size, sha_file(tar_path)
    write_pin(BUCKETS[bucket], size, sha)
    print(f'packed {bucket}: {len(rows)} files, {sum(n for _, n, _ in rows)} bytes -> {base}; '
          f'tar {size} bytes sha256 {sha} at {tar_path}')
    return tar_path


def push(bucket):
    """Upload the pinned tar with dataset_store.sh, then pull it back and verify it against the pin."""
    key = BUCKETS[bucket]
    tar_path = STAGING / tar_name(bucket)
    pin = pinned(key)
    if pin is None or not tar_path.is_file() or tar_path.stat().st_size != pin[0] or sha_file(tar_path) != pin[1]:
        raise RepoDataError(f'repo_data: {tar_path} is absent or not the pinned tar; run `python3 tools/repo_data.py pack {bucket}`')
    readback = STAGING / (tar_path.name + '.readback')
    # The catalog maps the key to the staged tar (HOME/.cache/mojolearn/repo-data/...), so `push <key>` uploads it;
    # the pull re-downloads it and dataset_store.sh checks size + sha256 against the pin before anything is removed.
    store('push', key)
    store('pull', key, str(readback))
    readback.unlink()
    tar_path.unlink()
    print(f'pushed and read back {key}: {pin[0]} bytes sha256 {pin[1]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('path').add_argument('relpath')
    for name in ('fetch', 'verify', 'push'):
        sub.add_parser(name).add_argument('bucket', choices=sorted(BUCKETS))
    p = sub.add_parser('pack'); p.add_argument('bucket', choices=sorted(BUCKETS)); p.add_argument('--from', dest='source')
    i = sub.add_parser('install'); i.add_argument('bucket', choices=sorted(BUCKETS)); i.add_argument('tar')
    args = ap.parse_args(argv)
    if args.cmd == 'path':
        print(resolve(args.relpath))
    elif args.cmd == 'fetch':
        print(fetch(args.bucket))
    elif args.cmd == 'verify':
        rows, _ = read_manifest(args.bucket)
        problems = check_tree(evidence_root() / args.bucket, rows, True)
        print('\n'.join(problems[:20]) if problems else f'ok: {args.bucket} {len(rows)} files')
        return 1 if problems else 0
    elif args.cmd == 'pack':
        pack(args.bucket, args.source)
    elif args.cmd == 'push':
        push(args.bucket)
    elif args.cmd == 'install':
        print(install(args.bucket, args.tar))
    return 0


if __name__ == '__main__':
    sys.exit(main())
