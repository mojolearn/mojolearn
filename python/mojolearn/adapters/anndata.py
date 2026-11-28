# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# Engineering notes (lane sc-anndata-1, 2026-11-28; Andrew: "a Scanpy user should be able to pass their AnnData
# and get results where Scanpy puts them").
#
# WHAT THIS IS. A front-end in the sense of AGENTS.md "Framework adapters are the same API in another format": it
# computes nothing. Each function UNPACKS buffers from the AnnData object (X or a layer, an obsm array, an obsp
# graph, obs category codes), makes ONE call of a public Mojolearn estimator or function, and PACKS the returned
# arrays into the slots Scanpy writes. No arithmetic on data, no loop or comprehension over values, no NumPy math:
# NumPy is used only for zero-copy views (`numpy.asarray` of a returned `Array`, a transpose view, a column slice of
# an obsm array) and pandas only to wrap returned codes as a categorical. A value a kernel does not return in the
# form Scanpy stores is a gap for the kernel's lane (FUNCTIONS rows with `lane` other than "main"), never computed
# here. The loops below run over the table's rows, parameters and output slots (model structure), never over data.
#
# ONE TABLE. `FUNCTIONS` maps each Scanpy-named function to its estimator, its input slots, its parameters (Scanpy
# names, mapped onto the estimator's keywords) and its output slots; `pp` and `tl` are generated from it. A new
# estimator is one row. A row whose `lane` is not "main" names the lane that owns its API: the row is written to that
# lane's brief and is NOT exposed until the integrator sets its lane to "main" when the API merges (the boundary
# test `test_anndata_adapter.py::test_live_rows_resolve` then checks every keyword against the estimator's signature).
#
# QUALIFICATION. Like the Torch and MAX adapters: no identity lanes, verifier rows or sabotage arms of its own (the
# estimators' lanes are the identity evidence). Its tests are Python boundary tests
# (python/mojolearn/tests/test_anndata_adapter.py) and the installed-wheel smoke phase `anndata` of
# tools/adapter_wheel_smoke.py (tools/check_anndata_adapter.py).
"""Run Mojolearn on AnnData objects, with Scanpy's function and slot names.

``mojolearn.adapters.anndata`` gives an :class:`anndata.AnnData` the same
Mojolearn estimators the rest of the library offers, called the way Scanpy
calls its own functions: ``pp.pca(adata)``, ``tl.umap(adata)``,
``tl.louvain(adata)``. Each function reads its input from the AnnData object
(``X``, a ``layer``, an ``obsm`` representation or an ``obsp`` graph), runs one
Mojolearn estimator, and writes the results where Scanpy writes them
(``obsm['X_pca']``, ``varm['PCs']``, ``uns['pca']``, ``obsm['X_umap']``,
``obs['louvain']`` and so on).

Reading and writing ``.h5ad`` files is anndata's own job::

    import anndata
    from mojolearn.adapters.anndata import pp, tl

    adata = anndata.read_h5ad("pbmc3k.h5ad")
    pp.pca(adata, n_comps=50)
    tl.umap(adata, n_neighbors=15)
    adata.write_h5ad("pbmc3k_mojolearn.h5ad")

The functions modify ``adata`` in place and return None; with ``copy=True``
they work on a copy and return it. A view (``adata[mask]``) is accepted with
``copy=True``; to modify it in place, make it an object of its own first
(``adata = adata[mask].copy()``). An object opened in backed mode is refused:
load it with ``adata.to_memory()``.

Computation is float32, as everywhere in Mojolearn: a float64 matrix is
converted once by the estimator, and results are float32.

Importing ``mojolearn`` never imports anndata. This module needs it, and
refuses with an install hint when it is missing
(``pip install "mojolearn[anndata]"``). Scanpy is not needed.

In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
"""
import inspect as _inspect
from collections import namedtuple as _namedtuple
from types import SimpleNamespace as _Namespace

from .._optional_numpy import require_numpy as _require_numpy

__all__ = ["pp", "tl", "FUNCTIONS"]


def _import_anndata():
    try:
        import anndata
    except ImportError as exc:
        raise ImportError(
            "mojolearn.adapters.anndata needs anndata. Install it with: pip install \"mojolearn[anndata]\"") from exc
    import pandas  # an anndata dependency: present whenever anndata imports
    return anndata, pandas


_anndata, _pandas = _import_anndata()
_np = _require_numpy("mojolearn.adapters.anndata")

import mojolearn as _ml  # noqa: E402  (after the refusal above, so a missing anndata is named first)

#: One parameter of a function: its Scanpy name and default, the keyword of the estimator (or function) it is
#: passed to (None: the front-end itself reads it, such as `layer`, `use_rep`, `key_added`, `copy`), and
#: `inherit`: True for a parameter that, when None, takes the value `pp.neighbors` recorded in
#: `uns[neighbors_key]['params']` (as Scanpy's `tl.umap` reads its neighbor settings), and is left out of the
#: estimator call when neither gives one (the estimator's own default applies).
P = _namedtuple("P", "name default target inherit", defaults=(None, False))

#: One output: the AnnData slot ('obsm', 'varm', 'obsp', 'obs', 'var', 'uns', or 'X' for the matrix the input was
#: read from: X or `layer`), its key (a format string over the call's arguments; 'uns' keys are paths 'a/b'), the
#: key used when `key_added` is given (None: the same key), and the value: a pair (packer, argument) from _PACK.
O = _namedtuple("O", "slot key added value")

#: One function: `name` 'pp.<f>' or 'tl.<f>'; `api` the public Mojolearn name; `method` the estimator method to
#: call (None: `api` is a function); `source` the input form (_SOURCES); `params`, `outputs`; `lane` "main" when
#: the API is on main (the row is exposed), otherwise the lane that owns it (pending, not exposed); `doc` the
#: function's docstring.
Row = _namedtuple("Row", "name api method source params outputs lane doc")

_COPY = P("copy", False)
_LAYER = P("layer", None)
_KEY_ADDED = P("key_added", None)
_NEIGHBORS_KEY = P("neighbors_key", None)
_ADJACENCY = P("adjacency", None)


def _uns_params(*names):
    return ("args", names)


FUNCTIONS = (
    # ------------------------------------------------------------------ on main
    Row("pp.pca", "PCA", "fit_transform", "X",
        (P("n_comps", 50, "n_components"), P("svd_solver", "auto", "svd_solver"),
         P("random_state", 0, "random_state"), _LAYER, _KEY_ADDED, _COPY),
        (O("obsm", "X_pca", "{key_added}", ("result", None)),
         O("varm", "PCs", "{key_added}", ("attr_T", "components_")),
         O("uns", "pca/variance", "{key_added}/variance", ("attr", "explained_variance_")),
         O("uns", "pca/variance_ratio", "{key_added}/variance_ratio", ("attr", "explained_variance_ratio_")),
         O("uns", "pca/params", "{key_added}/params",
           ("const", {"zero_center": True, "use_highly_variable": False, "mask_var": None}))),
        "main",
        """Principal component analysis of ``adata.X`` (or ``layer``), as Scanpy's ``pp.pca``.

        Runs :class:`mojolearn.PCA` and writes ``obsm['X_pca']`` (cells x ``n_comps``), ``varm['PCs']``
        (genes x ``n_comps``) and ``uns['pca']`` (``variance``, ``variance_ratio``, ``params``). With
        ``key_added`` the three are written under that key instead. The data are always centered; to use
        only the highly variable genes, pass ``adata[:, adata.var['highly_variable']].copy()``. A
        matrix with ``n_comps`` or fewer genes or cells needs a smaller ``n_comps``.

        Parameters
        ----------
        adata : AnnData
        n_comps : int, default 50
            Number of principal components.
        svd_solver : str, default "auto"
            The :class:`mojolearn.PCA` solver.
        random_state : int, default 0
            Seed of the ``"randomized"`` solver.
        layer : str or None, default None
            Read ``adata.layers[layer]`` instead of ``adata.X``.
        key_added : str or None, default None
            Write ``obsm[key_added]``, ``varm[key_added]`` and ``uns[key_added]``.
        copy : bool, default False
            Return a modified copy instead of modifying ``adata``.
        """),
    Row("pp.scale", "StandardScaler", "fit_transform", "X",
        (P("zero_center", True, "with_mean"), _LAYER, _COPY),
        (O("X", "", None, ("result", None)),
         O("var", "mean", None, ("attr?", "mean_")),
         O("var", "std", None, ("attr?", "scale_"))),
        "main",
        """Scale each gene to unit variance, as Scanpy's ``pp.scale``.

        Runs :class:`mojolearn.StandardScaler` on a dense ``adata.X`` (or ``layer``), replaces it with the
        scaled matrix and writes ``var['mean']`` and ``var['std']``. The standard deviation is the
        population one (divided by the number of cells), as in scikit-learn's ``StandardScaler``; Scanpy
        divides by the number of cells minus one.

        Parameters
        ----------
        adata : AnnData
        zero_center : bool, default True
            Subtract each gene's mean before scaling.
        layer : str or None, default None
            Scale ``adata.layers[layer]`` instead of ``adata.X``.
        copy : bool, default False
            Return a modified copy instead of modifying ``adata``.
        """),
    Row("tl.umap", "UMAP", "fit_transform", "rep",
        (P("n_neighbors", None, "n_neighbors", True), P("metric", None, "metric", True),
         P("use_rep", None, None, True), P("n_pcs", None, None, True),
         P("min_dist", 0.5, "min_dist"), P("spread", 1.0, "spread"), P("n_components", 2, "n_components"),
         P("maxiter", None, "n_epochs"), P("alpha", 1.0, "learning_rate"), P("gamma", 1.0, "repulsion_strength"),
         P("negative_sample_rate", 5, "negative_sample_rate"), P("init_pos", "spectral", "init"),
         P("random_state", 0, "random_state"), P("a", None, "a"), P("b", None, "b"),
         _NEIGHBORS_KEY, _KEY_ADDED, _COPY),
        (O("obsm", "X_umap", "{key_added}", ("result", None)),
         O("uns", "umap/params", "{key_added}/params", ("args", ("a", "b")))),
        "main",
        """UMAP embedding of the cells, as Scanpy's ``tl.umap``.

        Runs :class:`mojolearn.UMAP` on the representation ``use_rep`` (``obsm['X_pca']`` when present,
        else ``X``) and writes ``obsm['X_umap']`` and ``uns['umap']``. ``n_neighbors``, ``metric``,
        ``use_rep`` and ``n_pcs`` left as None take the values ``pp.neighbors`` recorded in
        ``uns['neighbors']['params']``; Mojolearn's UMAP builds its own exact neighbor graph from that
        representation. ``min_dist`` defaults to 0.5, Scanpy's choice (``mojolearn.UMAP`` itself defaults
        to umap-learn's 0.1).

        Parameters
        ----------
        adata : AnnData
        n_neighbors, metric : optional
            Neighborhood size and distance (see :class:`mojolearn.UMAP`).
        use_rep : str or None, default None
            ``'X'`` or an ``obsm`` key.
        n_pcs : int or None, default None
            Use only the first ``n_pcs`` columns of the representation.
        min_dist, spread, n_components, negative_sample_rate, random_state, a, b : optional
            As in :class:`mojolearn.UMAP`.
        maxiter : int or None, default None
            Number of epochs (``n_epochs``).
        alpha : float, default 1.0
            Initial learning rate (``learning_rate``).
        gamma : float, default 1.0
            Weight of the negative samples (``repulsion_strength``).
        init_pos : {"spectral", "random", "pca"} or array, default "spectral"
            Initial layout (``init``).
        neighbors_key : str or None, default None
            The ``uns`` key ``pp.neighbors`` wrote (``'neighbors'``).
        key_added : str or None, default None
            Write ``obsm[key_added]`` and ``uns[key_added]``.
        copy : bool, default False
            Return a modified copy instead of modifying ``adata``.
        """),
    Row("tl.tsne", "TSNE", "fit_transform", "rep",
        (P("use_rep", None), P("n_pcs", None), P("perplexity", 30.0, "perplexity"),
         P("early_exaggeration", 12.0, "early_exaggeration"), P("learning_rate", 1000.0, "learning_rate"),
         P("random_state", 0, "random_state"), _KEY_ADDED, _COPY),
        (O("obsm", "X_tsne", "{key_added}", ("result", None)),
         O("uns", "tsne/params", "{key_added}/params",
           ("args", ("perplexity", "early_exaggeration", "learning_rate")))),
        "main",
        """t-SNE embedding of the cells, as Scanpy's ``tl.tsne``.

        Runs :class:`mojolearn.TSNE` on ``use_rep`` (``obsm['X_pca']`` when present, else ``X``) and writes
        ``obsm['X_tsne']`` and ``uns['tsne']``.

        Parameters
        ----------
        adata : AnnData
        use_rep : str or None, default None
            ``'X'`` or an ``obsm`` key.
        n_pcs : int or None, default None
            Use only the first ``n_pcs`` columns of the representation.
        perplexity, early_exaggeration, learning_rate, random_state : optional
            As in :class:`mojolearn.TSNE`; ``learning_rate`` defaults to Scanpy's 1000.
        key_added : str or None, default None
            Write ``obsm[key_added]`` and ``uns[key_added]``.
        copy : bool, default False
            Return a modified copy instead of modifying ``adata``.
        """),
    Row("tl.louvain", "Louvain", "fit", "graph",
        (P("resolution", 1.0, "resolution"), P("random_state", 0, "seed"), P("use_weights", False),
         _ADJACENCY, _NEIGHBORS_KEY, P("key_added", "louvain"), _COPY),
        (O("obs", "{key_added}", None, ("labels", None)),
         O("uns", "{key_added}/params", None, ("args", ("resolution", "random_state")))),
        "main",
        """Louvain clustering of the neighbor graph, as Scanpy's ``tl.louvain``.

        Runs :class:`mojolearn.Louvain` on ``obsp['connectivities']`` (the graph ``pp.neighbors`` wrote, or
        ``adjacency``) and writes the clusters to ``obs['louvain']`` as a categorical with categories
        ``'0'``, ``'1'``, ... numbered in order of each cluster's first cell, and ``uns['louvain']``.

        Parameters
        ----------
        adata : AnnData
        resolution : float, default 1.0
            Larger values give more, smaller clusters.
        random_state : int, default 0
            Accepted for Scanpy compatibility; the result does not depend on it.
        use_weights : bool, default False
            Use the edge weights of the graph; False treats every edge as weight 1, as Scanpy does.
        adjacency : sparse matrix or None, default None
            A symmetric graph to cluster instead of ``obsp['connectivities']``.
        neighbors_key : str or None, default None
            The ``uns`` key ``pp.neighbors`` wrote (``'neighbors'``).
        key_added : str, default "louvain"
            The ``obs`` column and ``uns`` key written.
        copy : bool, default False
            Return a modified copy instead of modifying ``adata``.
        """),
    # ------------------------------------------------------------------ pending: written to the owning lane's brief
    Row("tl.leiden", "Leiden", "fit", "graph",
        (P("resolution", 1.0, "resolution"), P("random_state", 0, "random_state"),
         P("n_iterations", -1, "n_iterations"), P("objective_function", "modularity", "objective_function"),
         P("use_weights", True), _ADJACENCY, _NEIGHBORS_KEY, P("key_added", "leiden"), _COPY),
        (O("obs", "{key_added}", None, ("labels", None)),
         O("uns", "{key_added}/params", None, ("args", ("resolution", "random_state", "n_iterations")))),
        "leiden-umap-1",
        """Leiden clustering of the neighbor graph, as Scanpy's ``tl.leiden``.

        Runs :class:`mojolearn.Leiden` on ``obsp['connectivities']`` (or ``adjacency``) and writes the
        clusters to ``obs['leiden']`` as a categorical, and ``uns['leiden']``.
        """),
    Row("pp.neighbors", "neighbors_graph", None, "rep",
        (P("n_neighbors", 15, "n_neighbors"), P("n_pcs", None), P("use_rep", None),
         P("method", "exact", "method"), P("metric", "euclidean", "metric"), P("random_state", 0, "random_state"),
         _KEY_ADDED, _COPY),
        (O("obsp", "distances", "{key_added}_distances", ("item", "distances")),
         O("obsp", "connectivities", "{key_added}_connectivities", ("item", "connectivities")),
         O("uns", "neighbors/distances_key", "{key_added}/distances_key",
           ("key", ("distances", "{key_added}_distances"))),
         O("uns", "neighbors/connectivities_key", "{key_added}/connectivities_key",
           ("key", ("connectivities", "{key_added}_connectivities"))),
         O("uns", "neighbors/params", "{key_added}/params",
           ("args", ("n_neighbors", "method", "metric", "random_state", "use_rep", "n_pcs")))),
        "sc-ann-1",
        """The cells' neighbor graph, as Scanpy's ``pp.neighbors``: ``obsp['distances']`` (the kNN distances)
        and ``obsp['connectivities']`` (UMAP's fuzzy graph), and ``uns['neighbors']``.
        """),
    Row("pp.normalize_total", "NormalizeTotal", "fit_transform", "X",
        (P("target_sum", None, "target_sum"), P("exclude_highly_expressed", False, "exclude_highly_expressed"),
         P("max_fraction", 0.05, "max_fraction"), _LAYER, _COPY),
        (O("X", "", None, ("result", None)),),
        "sc-prep-1",
        """Scale each cell to the same total count, as Scanpy's ``pp.normalize_total``."""),
    Row("pp.log1p", "Log1pTransformer", "fit_transform", "X",
        (P("base", None, "base"), _LAYER, _COPY),
        (O("X", "", None, ("result", None)),
         O("uns", "log1p/base", None, ("args1", "base"))),
        "sc-prep-1",
        """``log(1 + x)`` of every entry, keeping sparsity, as Scanpy's ``pp.log1p``."""),
    Row("pp.highly_variable_genes", "HighlyVariableGenes", "fit", "X+batch",
        (P("flavor", "seurat", "flavor"), P("n_top_genes", None, "n_top_genes"), P("min_mean", 0.0125, "min_mean"),
         P("max_mean", 3.0, "max_mean"), P("min_disp", 0.5, "min_disp"), P("max_disp", None, "max_disp"),
         P("span", 0.3, "span"), P("n_bins", 20, "n_bins"), P("batch_key", None), _LAYER, _COPY),
        (O("var", "highly_variable", None, ("attr", "highly_variable_")),
         O("var", "means", None, ("attr?", "means_")),
         O("var", "dispersions", None, ("attr?", "dispersions_")),
         O("var", "dispersions_norm", None, ("attr?", "dispersions_norm_")),
         O("var", "variances", None, ("attr?", "variances_")),
         O("var", "variances_norm", None, ("attr?", "variances_norm_")),
         O("var", "highly_variable_rank", None, ("attr?", "highly_variable_rank_")),
         O("uns", "hvg/flavor", None, ("args1", "flavor"))),
        "sc-prep-1",
        """Mark the highly variable genes in ``var['highly_variable']``, as Scanpy's ``pp.highly_variable_genes``."""),
    Row("pp.regress_out", "RegressOut", "fit_transform", "X+covariates",
        (P("keys", None), _LAYER, _COPY),
        (O("X", "", None, ("result", None)),),
        "sc-prep-1",
        """Replace each gene by its residuals on the ``obs`` columns ``keys``, as Scanpy's ``pp.regress_out``."""),
    Row("tl.rank_genes_groups", "rank_genes_groups", None, "X+groups",
        (P("groupby", None), P("method", "wilcoxon", "method"), P("reference", "rest", "reference"),
         P("tie_correct", False, "tie_correct"), P("corr_method", "benjamini-hochberg", "corr_method"),
         _LAYER, P("key_added", "rank_genes_groups"), _COPY),
        (O("uns", "{key_added}", None, ("rank_genes", ("groupby", "reference", "method", "corr_method", "layer"))),),
        "sc-stats-1",
        """Marker genes of each group of ``obs[groupby]``, as Scanpy's ``tl.rank_genes_groups``."""),
    Row("pp.harmony_integrate", "Harmony", "fit_transform", "rep+batch",
        (P("key", None), P("basis", "X_pca"), P("theta", 2.0, "theta"), P("sigma", 0.1, "sigma"),
         P("nclust", None, "nclust"), P("max_iter_harmony", 10, "max_iter_harmony"), P("lamb", 1.0, "lamb"),
         P("random_state", 0, "random_state"), P("adjusted_basis", "X_pca_harmony"), _COPY),
        (O("obsm", "{adjusted_basis}", None, ("result", None)),),
        "sc-stats-1",
        """Harmony batch correction of ``obsm[basis]``, as Scanpy's ``external.pp.harmony_integrate``."""),
    Row("tl.diffmap", "DiffusionMap", "fit", "graph",
        (P("n_comps", 15, "n_components"), P("use_weights", True), _ADJACENCY, _NEIGHBORS_KEY, _COPY),
        (O("obsm", "X_diffmap", None, ("attr", "embedding_")),
         O("uns", "diffmap_evals", None, ("attr", "eigenvalues_"))),
        "sc-graph-1",
        """Diffusion map of the neighbor graph, as Scanpy's ``tl.diffmap``."""),
    Row("tl.paga", "PAGA", "fit", "graph+groups",
        (P("groups", None), P("model", "v1.2", "model"), P("use_weights", True), _ADJACENCY, _NEIGHBORS_KEY, _COPY),
        (O("uns", "paga/connectivities", None, ("attr", "connectivities_")),
         O("uns", "paga/connectivities_tree", None, ("attr", "connectivities_tree_")),
         O("uns", "paga/groups", None, ("args1", "groups")),
         O("uns", "{groups}_sizes", None, ("attr", "group_sizes_"))),
        "sc-graph-1",
        """Cluster-level connectivity of ``obs[groups]``, as Scanpy's ``tl.paga``."""),
    Row("pp.shared_neighbors", "shared_neighbors_graph", None, "knn",
        (P("prune_snn", 1.0 / 15.0, "prune_snn"), _NEIGHBORS_KEY, P("key_added", "snn"), _COPY),
        (O("obsp", "{key_added}", None, ("result", None)),),
        "sc-graph-1",
        """Shared-nearest-neighbor graph of the kNN graph (Seurat's ``FindNeighbors``), written to ``obsp['snn']``."""),
)


# ---------------------------------------------------------------------------------------------------- the driver

def _admit(adata, copy):
    """The AnnData object the call works on (UNPACK: type, backed mode, views)."""
    if not isinstance(adata, _anndata.AnnData):
        raise TypeError(f"expected an anndata.AnnData, got {type(adata).__name__}")
    if adata.isbacked:
        raise ValueError("adata is opened in backed mode; load it into memory first with adata.to_memory()")
    if copy:
        return adata.copy()
    if adata.is_view:
        raise ValueError("adata is a view of another AnnData object; pass copy=True, or make it an object of its "
                         "own first: adata = adata[...].copy()")
    return adata


def _matrix(adata, args):
    layer = args.get("layer")
    return adata.X if layer is None else adata.layers[layer]


def _neighbors_meta(adata, args):
    key = args.get("neighbors_key") or "neighbors"
    return adata.uns[key] if key in adata.uns else {}


def _representation(adata, args):
    use_rep, n_pcs = args.get("use_rep"), args.get("n_pcs")
    if use_rep is None:
        use_rep = "X_pca" if "X_pca" in adata.obsm else "X"
    rep = adata.X if use_rep == "X" else adata.obsm[use_rep]
    # A column slice is a view of the same buffer; the estimator admits its strides natively.
    return rep if n_pcs is None else rep[:, :n_pcs]


def _graph(adata, args):
    graph = args.get("adjacency")
    if graph is None:
        key = _neighbors_meta(adata, args).get("connectivities_key", "connectivities")
        if key not in adata.obsp:
            raise ValueError(f"adata.obsp has no {key!r} graph; build it with pp.neighbors or pass adjacency=")
        graph = adata.obsp[key]
    if args.get("use_weights", True):
        return graph
    if not hasattr(graph, "indptr"):
        raise ValueError("use_weights=False needs a sparse graph")
    # Every edge weighs 1: the estimators' (indptr, indices, n) form. For the symmetric graphs these
    # estimators take, CSR and CSC store the same index arrays.
    return (graph.indptr, graph.indices, graph.shape[0])


def _codes(adata, key):
    column = adata.obs[key]
    if not isinstance(column.dtype, _pandas.CategoricalDtype):
        raise ValueError(f"adata.obs[{key!r}] must be categorical (adata.obs[{key!r}] = "
                         f"adata.obs[{key!r}].astype('category'))")
    return column.cat.codes.to_numpy()


def _knn(adata, args):
    key = _neighbors_meta(adata, args).get("distances_key", "distances")
    if key not in adata.obsp:
        raise ValueError(f"adata.obsp has no {key!r} graph; build it with pp.neighbors")
    return adata.obsp[key]


#: Input form -> the positional arguments of the one call.
_SOURCES = {
    "X": lambda adata, args: (_matrix(adata, args),),
    "rep": lambda adata, args: (_representation(adata, args),),
    "graph": lambda adata, args: (_graph(adata, args),),
    "graph+groups": lambda adata, args: (_graph(adata, args), _codes(adata, args["groups"])),
    "X+groups": lambda adata, args: (_matrix(adata, args), _codes(adata, args["groupby"])),
    "X+batch": lambda adata, args: ((_matrix(adata, args),) if args.get("batch_key") is None
                                    else (_matrix(adata, args), _codes(adata, args["batch_key"]))),
    "X+covariates": lambda adata, args: (_matrix(adata, args),
                                         tuple(adata.obs[k].to_numpy() for k in args["keys"])),  # over obs keys
    "rep+batch": lambda adata, args: (adata.obsm[args["basis"]], _codes(adata, args["key"])),
    "knn": lambda adata, args: (_knn(adata, args),),
}


def _key(output, args):
    template = output.key if output.added is None or args.get("key_added") is None else output.added
    return template.format(**args)


def _rank_genes(adata, args, result, names):
    # PACK of rank_genes_groups into Scanpy's layout: one record array per statistic, one field per group, each
    # field in that group's order of decreasing score. The kernel returns the statistics already in that order
    # and the gene order itself (`order`); this only names the fields and the genes (gathers of labels).
    categories = adata.obs[args["groupby"]].cat.categories
    groups = [str(g) for g in categories.take(_np.asarray(result["groups"]))]  # over groups
    value = {"params": {n: args.get(n) for n in names}}  # over parameter names
    order = _np.asarray(result["order"])
    value["names"] = _np.rec.fromarrays([adata.var_names.take(row) for row in order], names=groups)  # over groups
    for stat in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        value[stat] = _np.rec.fromarrays(list(_np.asarray(result[stat])), names=groups)  # over groups
    return value


def _pack(adata, args, model, result, value):
    kind, what = value
    if kind == "result":
        return _np.asarray(result)
    if kind == "item":
        item = result[what]
        return item if hasattr(item, "indptr") else _np.asarray(item)
    if kind in ("attr", "attr?"):
        item = getattr(model, what, None) if kind == "attr?" else getattr(model, what)
        return None if item is None else item if hasattr(item, "indptr") else _np.asarray(item)
    if kind == "attr_T":
        return _np.asarray(getattr(model, what)).T  # a transposed view, no copy
    if kind == "labels":
        return _pandas.Categorical.from_codes(_np.asarray(model.labels_),
                                              categories=_pandas.RangeIndex(model.n_communities_).astype(str))
    if kind == "args":
        return {n: args.get(n) for n in what}  # over parameter names
    if kind == "args1":
        return args.get(what)
    if kind == "const":
        return dict(what)
    if kind == "key":
        return _key(O(None, what[0], what[1], None), args)
    if kind == "rank_genes":
        return _rank_genes(adata, args, result, what)
    raise AssertionError(kind)


def _write(adata, args, output, item, fresh):
    if item is None:
        return
    slot = output.slot
    if slot == "X":
        if args.get("layer") is None:
            adata.X = item
        else:
            adata.layers[args["layer"]] = item
        return
    key = _key(output, args)
    if slot != "uns":
        getattr(adata, slot)[key] = item
        return
    path = key.split("/")
    if path[0] not in fresh:  # each call replaces the uns entries it writes, as Scanpy does
        if len(path) > 1:
            adata.uns[path[0]] = {}
        fresh.add(path[0])
    node = adata.uns
    for part in path[:-1]:  # over the path's parts
        node = node[part]
    node[path[-1]] = item


#: Public Mojolearn name -> the class or function, resolved on the first call of a row and kept.
_RESOLVED = {}


def _api(name):
    found = _RESOLVED.get(name)
    if found is None:
        found = _RESOLVED[name] = getattr(_ml, name)
    return found


def _run(row, args):
    adata = _admit(args["adata"], args.get("copy", False))
    args["adata"] = adata
    meta = _neighbors_meta(adata, args).get("params", {}) if any(p.inherit for p in row.params) else {}
    keywords = {}
    for p in row.params:  # over the row's parameters
        value = args[p.name]
        if value is None and p.inherit:
            value = args[p.name] = meta.get(p.name)
        if p.target is not None and not (value is None and p.inherit):
            keywords[p.target] = value
    inputs = _SOURCES[row.source](adata, args)
    api = _api(row.api)
    if row.method is None:
        model, result = None, api(*inputs, **keywords)
    else:
        model = api(**keywords)
        result = getattr(model, row.method)(*inputs)
    fresh = set()
    for output in row.outputs:  # over the row's output slots
        _write(adata, args, output, _pack(adata, args, model, result, output.value), fresh)
    return adata if args.get("copy", False) else None


def _function(row):
    parameters = [_inspect.Parameter("adata", _inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    parameters += [_inspect.Parameter(p.name, _inspect.Parameter.KEYWORD_ONLY, default=p.default)
                   for p in row.params]  # over the row's parameters
    signature = _inspect.Signature(parameters)

    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return _run(row, dict(bound.arguments))

    call.__name__ = row.name.split(".", 1)[1]
    call.__qualname__ = row.name
    call.__module__ = __name__
    call.__signature__ = signature
    call.__doc__ = _inspect.cleandoc(row.doc)
    return call


def _namespace(prefix):
    return _Namespace(**{row.name.split(".", 1)[1]: _function(row) for row in FUNCTIONS  # over the table's rows
                         if row.lane == "main" and row.name.startswith(prefix + ".")})


#: Preprocessing, as Scanpy's ``sc.pp``: ``pp.pca``, ``pp.scale``.
pp = _namespace("pp")
#: Tools, as Scanpy's ``sc.tl``: ``tl.umap``, ``tl.tsne``, ``tl.louvain``.
tl = _namespace("tl")
