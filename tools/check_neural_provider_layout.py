#!/usr/bin/env python3
"""Reject Python/provider dependencies from the six neural native source trees.

Source architecture guard only: it does not establish runtime callback freedom
for Python-facing orchestration that remains in bindings/providers.
"""
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
NATIVE_TREES = ('autograd', 'sequence', 'training', 'transformer', 'mamba', 'core')
# The three training checks that once imported Python (cross_vendor_resume,
# peer_copy_check, train_gradient_capture) are native now, so the testing
# exclusion list they needed is gone.


def native_roots(root=ROOT):
    return tuple(path.relative_to(root).with_suffix('').as_posix().replace('/', '.')
                 for tree in NATIVE_TREES for path in sorted((root / tree).rglob('*.mojo')))


def check(root=ROOT, *, require_native_only=False):
    # The default reports missing local source dependencies;
    # require_native_only adds the existing opt-in
    # import-purity assertion of check_generation_native_boundary.audit.
    roots = native_roots(root)
    seen, problems = audit(root, roots, require_native_only=require_native_only)
    return roots, seen, problems


def main():
    roots, seen, problems = check()
    for chain, reason in problems:
        print(f'FAIL {reason}: {" -> ".join(chain)}')
    print(f'neural native layout: {len(roots)} roots, {len(seen)} closure modules, '
          f'{len(problems)} violations')
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
