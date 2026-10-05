#!/usr/bin/env python3
"""Dispatch the registered Linux workflow's isolated experimental PTX job.

The workflow ref selects tooling; the full source SHA selects the unchanged
build source. Dry-run by default. No GPU rental or package publication.
"""
import argparse
import json
import re
import shlex
import subprocess

REPO = 'mojolearn/mojolearn'
WORKFLOW = 'release-linux-build.yml'


def command(source, tooling_ref, jobs, tag):
    if not re.fullmatch('[0-9a-f]{40}', source):
        raise ValueError('Source must be a full 40-hex commit SHA')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_./-]*', tooling_ref) or '..' in tooling_ref:
        raise ValueError('Invalid tooling ref')
    if jobs not in (1, 2):
        raise ValueError('Experimental jobs must be 1 or 2')
    if not re.fullmatch('[A-Za-z0-9_.-]{1,64}', tag):
        raise ValueError('Tag must be 1..64 safe identifier characters')
    return ['gh', 'workflow', 'run', WORKFLOW, '--repo', REPO, '--ref', tooling_ref,
            '-f', 'code_format=ptx-baseline', '-f', 'commit=' + source,
            '-f', 'jobs=' + str(jobs), '-f', 'tag=' + tag]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--tooling-ref', required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--dispatch', action='store_true')
    args = parser.parse_args()
    argv = command(args.source, args.tooling_ref, args.jobs, args.tag)
    print(json.dumps(dict(source_commit=args.source, tooling_ref=args.tooling_ref,
                         jobs=args.jobs, experimental=True, identical_qualified=False,
                         command=shlex.join(argv)), indent=2), flush=True)
    if not args.dispatch:
        return 0
    # The tooling ref must be pushed, and must include this input/job. GitHub
    # resolves both refs independently; neither is substituted for the other.
    return subprocess.run(argv, check=False, timeout=60).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit('REFUSED: ' + str(exc))
