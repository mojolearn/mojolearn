#!/usr/bin/env python3
"""Local product-family source dependency guard; no runtime-free claim implied."""
from pathlib import Path
from check_generation_native_boundary import audit
from check_neural_provider_layout import NATIVE_TREES, TEST_FIXTURES

ROOT = Path(__file__).resolve().parents[1]
# Explicit numerical ownership trees. bindings/providers and framework adapters
# retain their separate admission contracts; testing exclusions are inherited.
FAMILY_TREES = ('arima', 'cholesky', 'cluster', 'dbscan', 'decomposition', 'embedding', 'ensemble', 'extratrees', 'gaussian_process', 'gbdt', 'gemm', 'glm', 'hdbscan', 'hierarchy', 'holtwinters', 'isolation_forest', 'ivf', 'kde', 'kernel_methods', 'metrics', 'mixture', 'naive_bayes', 'neighbors', 'preprocessing', 'resample', 'solver', 'spectral', 'svm', 'tokenizer', 'tsa', 'umap', 'x_ann', 'x_cluster', 'x_cnn', 'x_decomp', 'x_linear', 'x_metrics', 'x_neighbors', 'x_prep', 'xtrees')


def native_roots(root=ROOT):
    return tuple(path.relative_to(root).with_suffix('').as_posix().replace('/', '.')
                 for tree in NATIVE_TREES + FAMILY_TREES
                 for path in sorted((root / tree).rglob('*.mojo'))
                 if path.relative_to(root).as_posix() not in TEST_FIXTURES)


def check(root=ROOT):
    roots = native_roots(root)
    seen, problems = audit(root, roots)
    return roots, seen, problems


def main():
    roots, seen, problems = check()
    for chain, reason in problems:
        print(f'FAIL {reason}: {" -> ".join(chain)}')
    print(f'library native layout: {len(NATIVE_TREES) + len(FAMILY_TREES)} trees, '
          f'{len(roots)} roots, {len(seen)} closure modules, {len(problems)} violations; '
          f'{len(TEST_FIXTURES)} explicit testing exclusions')
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
