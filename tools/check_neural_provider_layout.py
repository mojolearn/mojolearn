#!/usr/bin/env python3
"""Reject Python/provider dependencies from the six neural native source trees.

Source architecture guard only: it does not establish runtime callback freedom
for Python-facing orchestration that remains in bindings/providers.
"""
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
NATIVE_TREES = ('autograd', 'sequence', 'training', 'transformer', 'mamba', 'core')
TEST_FIXTURES = frozenset({
    'training/checks/cross_vendor_resume.mojo',
    'training/checks/peer_copy_check.mojo',
    'training/checks/train_gradient_capture.mojo',
})


def native_roots(root=ROOT):
    return tuple(path.relative_to(root).with_suffix('').as_posix().replace('/', '.')
                 for tree in NATIVE_TREES for path in sorted((root / tree).rglob('*.mojo'))
                 if path.relative_to(root).as_posix() not in TEST_FIXTURES)


def check(root=ROOT):
    roots = native_roots(root)
    seen, problems = audit(root, roots)
    return roots, seen, problems


def main():
    roots, seen, problems = check()
    for chain, reason in problems:
        print(f'FAIL {reason}: {" -> ".join(chain)}')
    print(f'neural native layout: {len(roots)} roots, {len(seen)} closure modules, '
          f'{len(problems)} violations; {len(TEST_FIXTURES)} explicit testing exclusions')
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
