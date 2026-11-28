#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The AnnData front-end's smoke against an INSTALLED mojolearn wheel. A test controller, never product code.

    VENV/bin/python tools/check_anndata_adapter.py --out DIR

Run with the interpreter of the venv that holds the installed wheel and the `mojolearn[anndata]` extra, from outside
any checkout (tools/adapter_wheel_smoke.py phase `anndata` does both). It imports `mojolearn.adapters.anndata` from
the installed package, runs every live row of its table (`FUNCTIONS` rows whose lane is "main") on a small AnnData
with a dense X, a CSR X and a categorical obs column, and checks for each:

  - the slots are the ones Scanpy writes (obsm['X_pca'], varm['PCs'], uns['pca'], obsm['X_umap'], obs['louvain'], ...),
    with the shapes and dtypes Mojolearn returns (float32 embeddings, a categorical of cluster labels);
  - the values are the bytes of the same estimator called directly on the same buffer (the front-end computes
    nothing: anything else is a front-end bug);
  - copy=True leaves the object untouched and returns the modified copy; a view is refused in place and accepted
    with copy=True.

The adapter has no identity lanes of its own (AGENTS.md "Framework adapters are the same API in another format"):
the estimators' lanes are the identity evidence. Writes DIR/anndata-smoke.json {status PASSED|FAILED, checks,
problems}; exit 0 for PASSED, 1 for FAILED.
"""
import argparse
import json
import pathlib
import sys
import traceback

SCHEMA = "mojolearn.anndata-smoke.v1"


def _data():
    import anndata
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    n, g = 240, 30
    centers = rng.normal(0, 4, (3, g)).astype(np.float32)
    group = np.repeat(np.arange(3), n // 3)
    x = (centers[group] + rng.normal(0, 1, (n, g))).astype(np.float32)
    obs = pd.DataFrame({"group": pd.Categorical(group.astype(str))}, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(g)])
    dense = anndata.AnnData(X=x, obs=obs, var=var)
    sparse = anndata.AnnData(X=sp.csr_matrix(np.where(x > 0, x, 0).astype(np.float32)), obs=obs.copy(), var=var.copy())
    # A symmetric graph of three blocks, to stand for obsp['connectivities'].
    a = (group[:, None] == group[None, :]) & (rng.random((n, n)) < 0.2)
    a = np.triu(a, 1)
    graph = sp.csr_matrix((a | a.T).astype(np.float32))
    graph.sort_indices()
    dense.obsp["connectivities"] = graph
    return dense, sparse


def _same(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def run_checks():
    """{name: 'ok' | 'FAILED: why'} for every check."""
    import numpy as np
    import mojolearn
    from mojolearn.adapters import anndata as mla
    checks = {}

    def check(name, fn):
        try:
            fn()
            checks[name] = "ok"
        except Exception as exc:  # recorded, never hidden
            checks[name] = "FAILED: %s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()

    live = sorted(row.name for row in mla.FUNCTIONS if row.lane == "main")

    def table():
        exposed = sorted([f"pp.{n}" for n in vars(mla.pp)] + [f"tl.{n}" for n in vars(mla.tl)])
        assert exposed == live, (exposed, live)

    def pca():
        dense, _ = _data()
        mla.pp.pca(dense, n_comps=5)
        direct = mojolearn.PCA(n_components=5, random_state=0)
        z = direct.fit_transform(dense.X)
        assert _same(dense.obsm["X_pca"], z)
        assert dense.obsm["X_pca"].dtype == np.float32 and dense.varm["PCs"].shape == (dense.n_vars, 5)
        assert _same(np.ascontiguousarray(dense.varm["PCs"].T), direct.components_)
        assert set(dense.uns["pca"]) == {"variance", "variance_ratio", "params"}

    def pca_sparse():
        _, sparse = _data()
        mla.pp.pca(sparse, n_comps=4)
        assert _same(sparse.obsm["X_pca"], mojolearn.PCA(n_components=4, random_state=0).fit_transform(sparse.X))

    def umap():
        dense, _ = _data()
        mla.pp.pca(dense, n_comps=8)
        mla.tl.umap(dense, n_neighbors=10, maxiter=50)
        z = mojolearn.UMAP(n_neighbors=10, n_epochs=50, min_dist=0.5).fit_transform(dense.obsm["X_pca"])
        assert _same(dense.obsm["X_umap"], z) and dense.obsm["X_umap"].shape == (dense.n_obs, 2)

    def tsne():
        dense, _ = _data()
        mla.pp.pca(dense, n_comps=8)
        mla.tl.tsne(dense, perplexity=20.0)
        z = mojolearn.TSNE(perplexity=20.0, learning_rate=1000.0).fit_transform(dense.obsm["X_pca"])
        assert _same(dense.obsm["X_tsne"], z)

    def louvain():
        dense, _ = _data()
        mla.tl.louvain(dense)
        graph = dense.obsp["connectivities"]
        labels = mojolearn.Louvain().fit((graph.indptr, graph.indices, graph.shape[0])).labels_
        column = dense.obs["louvain"]
        assert str(column.dtype) == "category"
        assert _same(column.cat.codes.to_numpy().astype(np.int32), labels)
        assert dense.uns["louvain"]["params"] == {"resolution": 1.0, "random_state": 0}

    def scale():
        dense, _ = _data()
        x = dense.X.copy()
        mla.pp.scale(dense)
        assert _same(dense.X, mojolearn.StandardScaler().fit_transform(x))
        assert "mean" in dense.var and "std" in dense.var

    def copy_and_views():
        dense, _ = _data()
        out = mla.pp.pca(dense, n_comps=3, copy=True)
        assert "X_pca" not in dense.obsm and "X_pca" in out.obsm
        view = dense[:120]
        try:
            mla.pp.pca(view, n_comps=3)
        except ValueError:
            pass
        else:
            raise AssertionError("a view was modified in place")
        assert "X_pca" in mla.pp.pca(view, n_comps=3, copy=True).obsm

    for name, fn in (("table", table), ("pp.pca", pca), ("pp.pca-csr", pca_sparse), ("tl.umap", umap),
                     ("tl.tsne", tsne), ("tl.louvain", louvain), ("pp.scale", scale),
                     ("copy-and-views", copy_and_views)):
        check(name, fn)
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import mojolearn
    here = pathlib.Path(__file__).resolve().parents[1]
    problems = []
    if pathlib.Path(mojolearn.__file__).resolve().is_relative_to(here):
        problems.append("mojolearn imported from this checkout, not from the installed wheel: " + mojolearn.__file__)
    checks = run_checks()
    problems += [f"{name}: {state}" for name, state in checks.items() if state != "ok"]
    report = dict(schema=SCHEMA, mojolearn=getattr(mojolearn, "__version__", None), module=mojolearn.__file__,
                  checks=checks, problems=problems, status="FAILED" if problems else "PASSED")
    (out / "anndata-smoke.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"anndata smoke: {report['status']} " + " ".join(f"{k}={v.split(':')[0]}" for k, v in checks.items()))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
