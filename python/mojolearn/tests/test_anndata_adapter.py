# SPDX-License-Identifier: Apache-2.0
"""mojolearn.adapters.anndata at the Python boundary (lane sc-anndata-1, 2026-11-28): the AnnData front-end reads
its input from the slots Scanpy reads, makes one call of the public estimator, and writes the slots Scanpy writes.

The front-end computes nothing, so every value check here compares its output with the bytes of the same estimator
called directly on the same buffer; the estimators' own numerical and identity checks live with their lanes. Needs
the `mojolearn[anndata]` extra (anndata, which brings NumPy, pandas, SciPy and h5py); skipped without it."""
import inspect
import os
import string
import subprocess
import sys
from pathlib import Path

import pytest

anndata = pytest.importorskip("anndata")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")

import mojolearn  # noqa: E402
from mojolearn.adapters import anndata as mla  # noqa: E402

PACKAGE_PARENT = str(Path(mojolearn.__file__).resolve().parents[1])
REPO = Path(__file__).resolve().parents[3]


def _adata(n=180, g=24, seed=0, dtype=np.float32, sparse=False):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(3), n // 3)
    centers = rng.normal(0, 4, (3, g))
    x = (centers[group] + rng.normal(0, 1, (n, g))).astype(dtype)
    if sparse:
        x = sp.csr_matrix(np.where(x > 0, x, 0).astype(dtype))
    obs = pd.DataFrame({"group": pd.Categorical(group.astype(str)), "depth": rng.random(n)},
                       index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(g)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    a = np.triu((group[:, None] == group[None, :]) & (rng.random((n, n)) < 0.2), 1)
    graph = sp.csr_matrix(((a | a.T) * (1 + rng.integers(0, 3, (n, n)))).astype(np.float32))
    graph = sp.csr_matrix(np.maximum(graph.toarray(), graph.toarray().T))
    graph.sort_indices()
    adata.obsp["connectivities"] = graph
    return adata


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _child(code):
    env = dict(os.environ, PYTHONPATH=PACKAGE_PARENT)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)


# ------------------------------------------------------------------------------------------------- the module

def test_import_mojolearn_never_imports_anndata():
    r = _child("import sys, mojolearn; print('anndata' in sys.modules)")
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "False"


def test_a_missing_anndata_is_refused_naming_the_extra():
    r = _child("import sys; sys.modules['anndata'] = None\nimport mojolearn.adapters.anndata")
    assert r.returncode != 0
    assert "mojolearn.adapters.anndata needs anndata." in r.stderr
    assert 'pip install "mojolearn[anndata]"' in r.stderr


def test_the_table_is_well_formed():
    names = [row.name for row in mla.FUNCTIONS]
    assert len(names) == len(set(names))
    packers = {"result", "item", "attr", "attr?", "attr_T", "labels", "args", "args1", "const", "key", "rank_genes"}
    for row in mla.FUNCTIONS:
        assert row.name.split(".")[0] in ("pp", "tl"), row.name
        assert row.source in mla._SOURCES, row.name
        assert row.lane and row.doc.strip(), row.name
        params = [p.name for p in row.params]
        assert len(params) == len(set(params)) and "copy" in params, row.name
        for output in row.outputs:
            assert output.slot in ("obsm", "varm", "obsp", "obs", "var", "uns", "X"), row.name
            assert output.value[0] in packers, (row.name, output.value)
            for template in (output.key, output.added or ""):
                # every field a key names is a parameter of the row
                fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
                assert fields <= set(params), (row.name, template)


def test_only_live_rows_are_exposed():
    live = sorted(row.name for row in mla.FUNCTIONS if row.lane == "main")
    exposed = sorted([f"pp.{n}" for n in vars(mla.pp)] + [f"tl.{n}" for n in vars(mla.tl)])
    assert exposed == live
    assert {"pp.pca", "pp.scale", "tl.umap", "tl.tsne", "tl.louvain"} <= set(live)


@pytest.mark.parametrize("row", [r for r in mla.FUNCTIONS if r.lane == "main"], ids=lambda r: r.name)
def test_live_rows_resolve(row):
    # A row goes live (lane "main") only when every keyword it passes is a parameter of the public API it calls.
    api = getattr(mojolearn, row.api)
    accepted = inspect.signature(api).parameters
    for p in row.params:
        if p.target is not None:
            assert p.target in accepted, (row.name, p.target)
    if row.method is not None:
        assert callable(getattr(api, row.method)), row.name
    function = getattr(getattr(mla, row.name.split(".")[0]), row.name.split(".")[1])
    assert list(inspect.signature(function).parameters) == ["adata"] + [p.name for p in row.params]
    assert function.__doc__ and "Parameters" in function.__doc__


def test_an_unknown_keyword_is_a_type_error():
    with pytest.raises(TypeError):
        mla.pp.pca(_adata(), zero_center=False)


def test_not_an_anndata_is_refused():
    with pytest.raises(TypeError, match="anndata.AnnData"):
        mla.pp.pca(np.zeros((10, 4), np.float32))


# ------------------------------------------------------------------------------------------------- pp.pca

def test_pca_writes_scanpys_slots_with_the_estimators_bytes():
    adata = _adata()
    assert mla.pp.pca(adata, n_comps=5) is None
    direct = mojolearn.PCA(n_components=5, random_state=0)
    z = direct.fit_transform(adata.X)
    assert _same(adata.obsm["X_pca"], z) and adata.obsm["X_pca"].dtype == np.float32
    assert adata.varm["PCs"].shape == (adata.n_vars, 5)
    assert _same(np.ascontiguousarray(adata.varm["PCs"].T), direct.components_)
    assert _same(adata.uns["pca"]["variance"], direct.explained_variance_)
    assert _same(adata.uns["pca"]["variance_ratio"], direct.explained_variance_ratio_)
    assert adata.uns["pca"]["params"] == {"zero_center": True, "use_highly_variable": False, "mask_var": None}


def test_pca_sparse_x():
    adata = _adata(sparse=True)
    mla.pp.pca(adata, n_comps=4)
    assert _same(adata.obsm["X_pca"], mojolearn.PCA(n_components=4, random_state=0).fit_transform(adata.X))


def test_pca_float64_x_gives_float32():
    adata = _adata(dtype=np.float64)
    mla.pp.pca(adata, n_comps=3)
    assert adata.obsm["X_pca"].dtype == np.float32
    assert _same(adata.obsm["X_pca"], mojolearn.PCA(n_components=3, random_state=0).fit_transform(adata.X))


def test_pca_layer_and_key_added():
    adata = _adata()
    adata.layers["raw"] = adata.X * 2
    mla.pp.pca(adata, n_comps=3, layer="raw", key_added="raw_pca")
    assert "X_pca" not in adata.obsm and "pca" not in adata.uns
    assert _same(adata.obsm["raw_pca"], mojolearn.PCA(n_components=3, random_state=0).fit_transform(adata.layers["raw"]))
    assert adata.varm["raw_pca"].shape == (adata.n_vars, 3) and "variance" in adata.uns["raw_pca"]


def test_pca_results_survive_an_h5ad_round_trip(tmp_path):
    adata = _adata()
    mla.pp.pca(adata, n_comps=4)
    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    assert _same(back.obsm["X_pca"], adata.obsm["X_pca"])
    assert np.array_equal(back.varm["PCs"], adata.varm["PCs"])
    assert _same(back.uns["pca"]["variance"], adata.uns["pca"]["variance"])


# ------------------------------------------------------------------------------------------------- pp.scale

def test_scale_replaces_x_and_writes_mean_and_std():
    adata = _adata()
    x = adata.X.copy()
    mla.pp.scale(adata)
    direct = mojolearn.StandardScaler()
    assert _same(adata.X, direct.fit_transform(x))
    assert _same(adata.var["mean"].to_numpy(), direct.mean_) and _same(adata.var["std"].to_numpy(), direct.scale_)


def test_scale_without_centering_on_a_layer():
    adata = _adata()
    adata.layers["counts"] = adata.X.copy()
    x = adata.X.copy()
    mla.pp.scale(adata, zero_center=False, layer="counts")
    assert _same(adata.X, x)
    assert _same(adata.layers["counts"], mojolearn.StandardScaler(with_mean=False).fit_transform(x))


# ------------------------------------------------------------------------------------------------- tl.umap / tl.tsne

def test_umap_takes_the_neighbor_settings_pp_neighbors_recorded():
    adata = _adata()
    mla.pp.pca(adata, n_comps=6)
    adata.uns["neighbors"] = {"params": {"n_neighbors": 8, "metric": "cosine", "use_rep": "X_pca"}}
    mla.tl.umap(adata, maxiter=30)
    z = mojolearn.UMAP(n_neighbors=8, metric="cosine", n_epochs=30, min_dist=0.5).fit_transform(adata.obsm["X_pca"])
    assert _same(adata.obsm["X_umap"], z) and adata.obsm["X_umap"].shape == (adata.n_obs, 2)
    assert set(adata.uns["umap"]["params"]) == {"a", "b"}


def test_umap_explicit_arguments_win_and_n_pcs_slices():
    adata = _adata()
    mla.pp.pca(adata, n_comps=6)
    adata.uns["neighbors"] = {"params": {"n_neighbors": 8}}
    mla.tl.umap(adata, n_neighbors=12, n_pcs=4, maxiter=30, min_dist=0.3, key_added="umap4")
    z = mojolearn.UMAP(n_neighbors=12, n_epochs=30, min_dist=0.3).fit_transform(
        np.ascontiguousarray(adata.obsm["X_pca"][:, :4]))
    assert _same(adata.obsm["umap4"], z) and "X_umap" not in adata.obsm


def test_umap_without_pca_uses_x():
    adata = _adata()
    mla.tl.umap(adata, n_neighbors=10, maxiter=20)
    assert _same(adata.obsm["X_umap"], mojolearn.UMAP(n_neighbors=10, n_epochs=20, min_dist=0.5).fit_transform(adata.X))


def test_tsne():
    adata = _adata()
    mla.pp.pca(adata, n_comps=6)
    mla.tl.tsne(adata, perplexity=15.0)
    z = mojolearn.TSNE(perplexity=15.0, learning_rate=1000.0).fit_transform(adata.obsm["X_pca"])
    assert _same(adata.obsm["X_tsne"], z)
    assert adata.uns["tsne"]["params"] == {"perplexity": 15.0, "early_exaggeration": 12.0, "learning_rate": 1000.0}


# ------------------------------------------------------------------------------------------------- tl.louvain

def test_louvain_unweighted_by_default_as_scanpy():
    adata = _adata()
    mla.tl.louvain(adata)
    g = adata.obsp["connectivities"]
    labels = mojolearn.Louvain().fit((g.indptr, g.indices, g.shape[0])).labels_
    column = adata.obs["louvain"]
    assert isinstance(column.dtype, pd.CategoricalDtype)
    assert _same(column.cat.codes.to_numpy().astype(np.int32), labels)
    assert list(column.cat.categories) == [str(i) for i in range(len(column.cat.categories))]
    assert adata.uns["louvain"]["params"] == {"resolution": 1.0, "random_state": 0}


def test_louvain_weighted_adjacency_and_key_added():
    adata = _adata()
    g = adata.obsp["connectivities"]
    mla.tl.louvain(adata, use_weights=True, resolution=0.7, adjacency=g, key_added="clusters")
    labels = mojolearn.Louvain(resolution=0.7).fit(g).labels_
    assert _same(adata.obs["clusters"].cat.codes.to_numpy().astype(np.int32), labels)
    assert "louvain" not in adata.obs and adata.uns["clusters"]["params"]["resolution"] == 0.7


def test_louvain_reads_the_connectivities_key_pp_neighbors_recorded():
    adata = _adata()
    adata.obsp["knn_connectivities"] = adata.obsp.pop("connectivities")
    adata.uns["knn"] = {"connectivities_key": "knn_connectivities"}
    mla.tl.louvain(adata, neighbors_key="knn")
    assert "louvain" in adata.obs


def test_louvain_without_a_graph_says_what_to_run():
    adata = _adata()
    del adata.obsp["connectivities"]
    with pytest.raises(ValueError, match="pp.neighbors"):
        mla.tl.louvain(adata)


# ------------------------------------------------------------------------------------------------- copy, views, backed

def test_copy_returns_a_modified_copy_and_leaves_the_object():
    adata = _adata()
    out = mla.pp.pca(adata, n_comps=3, copy=True)
    assert isinstance(out, anndata.AnnData) and out is not adata
    assert "X_pca" in out.obsm and "X_pca" not in adata.obsm


def test_a_view_is_refused_in_place_and_accepted_with_copy():
    adata = _adata()
    view = adata[:90]
    with pytest.raises(ValueError, match="view"):
        mla.pp.pca(view, n_comps=3)
    out = mla.pp.pca(view, n_comps=3, copy=True)
    assert out.n_obs == 90 and not out.is_view and "X_pca" in out.obsm
    assert _same(out.obsm["X_pca"], mojolearn.PCA(n_components=3, random_state=0).fit_transform(adata.X[:90]))


def test_backed_mode_is_refused(tmp_path):
    path = tmp_path / "b.h5ad"
    _adata().write_h5ad(path)
    backed = anndata.read_h5ad(path, backed="r")
    try:
        with pytest.raises(ValueError, match="to_memory"):
            mla.pp.pca(backed, n_comps=3)
        with pytest.raises(ValueError, match="to_memory"):
            mla.pp.pca(backed, n_comps=3, copy=True)
    finally:
        backed.file.close()


# ------------------------------------------------------------------------------------------------- docs

def test_the_user_guide_names_live_functions_only():
    guide = REPO / "docs-site/user_guide/anndata.md"
    if not guide.exists():
        pytest.skip("source tree only: docs-site is not installed with the wheel")
    text = guide.read_text()
    for row in mla.FUNCTIONS:
        named = f"{row.name}(" in text
        assert named == (row.lane == "main"), row.name
    assert "In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs." in text
