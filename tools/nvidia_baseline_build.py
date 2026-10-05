#!/usr/bin/env python3
"""Plan, or explicitly rent, one CPU-only experimental sm_80 build.

The frozen commit must be the tip of an advertised origin ref before renting.
The existing RunPod CPU runner owns staging, watchdogs, fetching and teardown.
The remote build restores a real sparse Git checkout so its manifest can bind
the original commit. No native release route or admission rule is relaxed.

  python tools/nvidia_baseline_build.py FULL_SHA --out /absolute/new-directory
  python tools/nvidia_baseline_build.py FULL_SHA --out /absolute/new-directory --rent
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'rocm/dev-ubuntu-22.04@sha256:a3850e6638c6c390436ef1aacd72fd1359af36083ac823d5136818206998c484'
BODY = 'tools/nvidia_baseline_cpu_box.sh'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, timeout=60).strip()


def transport_origin(raw):
    """A credential-free HTTPS URL; never copy a local/credentialed origin."""
    match = re.fullmatch(r'git@([A-Za-z0-9.-]+):([A-Za-z0-9_./-]+)', raw)
    if match:
        raw = 'https://' + match[1] + '/' + match[2]
    parsed = urlsplit(raw)
    require(parsed.scheme == 'https' and parsed.hostname and parsed.username is None
            and parsed.password is None and not parsed.query and not parsed.fragment
            and parsed.port is None and re.fullmatch(r'/[A-Za-z0-9_./-]+', parsed.path)
            and '..' not in parsed.path.split('/'),
            'Origin must be credential-free HTTPS (or git@host:path); credentials are never transported')
    return raw


def plan(commit, out, *, seconds=6000, lease=120, vcpu=32, jobs='auto'):
    require(re.fullmatch('[0-9a-f]{40}', commit), 'A full frozen commit SHA is required')
    require(30 <= lease <= 180 and 120 <= seconds <= 6000 and seconds <= lease * 60 - 900,
            'Lease must leave at least 15 minutes outside the 120..6000 second build bound')
    require(4 <= vcpu <= 128, 'vCPU count must be 4..128')
    require(jobs == 'auto' or (str(jobs).isdigit() and 1 <= int(jobs) <= 16), 'Jobs must be auto or 1..16')
    out = Path(out).expanduser().resolve()
    require(not out.exists(), 'Output directory already exists')
    require(git('rev-parse', commit + '^{commit}') == commit, 'Frozen commit is unavailable locally')
    git('cat-file', '-e', commit + ':' + BODY)
    origin = transport_origin(git('remote', 'get-url', 'origin'))
    refs = git('ls-remote', '--refs', origin)
    advertised = [row.split('\t', 1)[1] for row in refs.splitlines()
                  if '\t' in row and row.split('\t', 1)[0] == commit]
    require(advertised, 'Frozen commit is not an advertised origin ref tip; push the finished branch before renting')
    return dict(schema='mojolearn.ptx-baseline-cpu-plan.v1', source_commit=commit,
                origin=origin, advertised_refs=advertised, output=str(out), image=IMAGE,
                lease_minutes=lease, build_seconds=seconds, vcpu=vcpu, jobs=str(jobs),
                code_format='ptx-baseline', architecture='sm_80',
                experimental=True, identical_qualified=False, release_qualified=False)


def body_command(spec):
    # The pinned ROCm development image lacks Git. Bootstrap transport tools
    # outside the frozen checkout: source_commit and its compiler checks remain
    # untouched, including when building an older advertised source revision.
    preflight = """set -euo pipefail
: "${LEG_OUT:?existing leased runner output directory required}"
if ! command -v git >/dev/null 2>&1; then
  command -v apt-get >/dev/null 2>&1 || { echo 'Missing git and apt-get' >&2; exit 2; }
  command -v timeout >/dev/null 2>&1 || { echo 'Missing timeout for bounded Git bootstrap' >&2; exit 2; }
  {
    timeout -k 10 180 apt-get -o Acquire::Retries=2 update
    DEBIAN_FRONTEND=noninteractive timeout -k 10 180 apt-get -o Acquire::Retries=2 install -y --no-install-recommends --no-upgrade git ca-certificates
  } > "$LEG_OUT/ptx-prerequisites.log" 2>&1
fi
git --version > "$LEG_OUT/ptx-prerequisites-readback.txt"
"""
    return preflight + shlex.join(['bash', BODY, spec['source_commit'], spec['origin'],
                       str(spec['build_seconds']), spec['jobs']]) + '\n'


def run(spec):
    # Detached worktree pins exactly what the existing runner archives.
    with tempfile.TemporaryDirectory(prefix='mojolearn-ptx-build-') as tmp:
        tree = Path(tmp) / 'source'
        command = Path(tmp) / 'command.sh'
        command.write_text(body_command(spec))
        git('worktree', 'add', '--detach', str(tree), spec['source_commit'])
        try:
            args = ['bash', str(ROOT / 'tools/runpod_cpu_leg.sh'), '--lane', 'nvidia-ptx80',
                    '--worktree', str(tree), '--cmd-file', str(command), '--envs', 'default,pkg',
                    '--image', spec['image'], '--vcpu', str(spec['vcpu']), '--flavors', 'cpu5g,cpu3g',
                    '--lease', str(spec['lease_minutes']), '--disk', '80', '--out', spec['output'],
                    '--no-bincache', '--rent']
            # Cold initial experiment avoids promoting unqualified compiler
            # artifacts. Native release/cache behavior remains unchanged.
            return subprocess.run(args, check=False).returncode
        finally:
            git('worktree', 'remove', '--force', str(tree))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seconds', type=int, default=6000)
    parser.add_argument('--lease', type=int, default=120)
    parser.add_argument('--vcpu', type=int, default=32)
    parser.add_argument('--jobs', default='auto')
    parser.add_argument('--rent', action='store_true')
    args = parser.parse_args()
    spec = plan(args.commit, args.out, seconds=args.seconds, lease=args.lease, vcpu=args.vcpu, jobs=args.jobs)
    print(json.dumps(spec, indent=2, sort_keys=True), flush=True)
    if args.rent:
        return run(spec)
    print('DRY RUN: no pod created; --rent is required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
