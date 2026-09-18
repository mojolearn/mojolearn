#!/usr/bin/env python3
"""Rebuild main's source in place on a box with no git and maybe no patch(1).

    python3 tools/lm_attention_replay_vendors_before.py make <base> <out.json>   (host, needs git)
    python3 tools/lm_attention_replay_vendors_before.py apply <in.json>          (box)
    python3 tools/lm_attention_replay_vendors_before.py restore <in.json>        (box)

`make` records, per source file this lane changed, each diff hunk as the
exact (after, before) text blocks, context included. `apply` replaces each
after-block with its before-block and REFUSES unless every block occurs
exactly once, so a drifted tree cannot be half reverted. `restore` is the
inverse. Only numerical source is reverted; tools and docs are not.
"""
import json
import subprocess
import sys
from pathlib import Path

FILES = ('checks/kernel_matrix.mojo', 'transformer/impl/llama/fused_attention.mojo')


def hunks(diff):
    blocks, old, new, inside = [], [], [], False
    for line in diff.splitlines(keepends=True):
        if line.startswith('@@'):
            if inside:
                blocks.append((''.join(new), ''.join(old)))
            old, new, inside = [], [], True
        elif not inside or line.startswith('\\'):
            continue
        elif line.startswith('-'):
            old.append(line[1:])
        elif line.startswith('+'):
            new.append(line[1:])
        else:
            old.append(line[1:])
            new.append(line[1:])
    if inside:
        blocks.append((''.join(new), ''.join(old)))
    return blocks


def make(base, out):
    data = {}
    for f in FILES:
        diff = subprocess.run(['git', 'diff', '-U3', base, 'HEAD', '--', f],
                              check=True, capture_output=True, text=True).stdout
        data[f] = hunks(diff)
    data['_base'] = subprocess.run(['git', 'rev-parse', base], check=True,
                                   capture_output=True, text=True).stdout.strip()
    Path(out).write_text(json.dumps(data, indent=1))
    print('recorded', {f: len(data[f]) for f in FILES}, 'hunks against', data['_base'])


def swap(path, reverse):
    data = json.loads(Path(path).read_text())
    for f in FILES:
        text = Path(f).read_text()
        for after, before in data[f]:
            src, dst = (after, before) if not reverse else (before, after)
            if text.count(src) != 1:
                sys.exit(f'REFUSING: a block of {f} occurs {text.count(src)} times')
            text = text.replace(src, dst)
        Path(f).write_text(text)
        print(('restored ' if reverse else 'reverted ') + f, len(data[f]), 'hunks')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'make':
        make(sys.argv[2], sys.argv[3])
    elif mode == 'apply':
        swap(sys.argv[2], False)
    elif mode == 'restore':
        swap(sys.argv[2], True)
    else:
        sys.exit('mode: make, apply or restore')
