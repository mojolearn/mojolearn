"""THE DECOMP LANE'S HOST SURFACE FRAGMENT.

Owned by the `decomp` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_decomp",) once bindings/build_x_decomp.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_decomp", binding="_mojolearn_x_decomp_host",
                        routes="_mojolearn_x_decomp", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_decomp",)
FAMILIES = (
    dict(
        family="x_decomp",
        binding="_mojolearn_x_decomp_host",
        routes="_mojolearn_x_decomp",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("x-decomp-ipca", "x-decomp-grp", "x-decomp-srp", "x-decomp-nmf", "x-decomp-fastica", "x-decomp-factor-analysis", "x-decomp-spectral-rbf", "x-decomp-lu", "x-decomp-lstsq-rsvd", "x-decomp-pls", "x-decomp-dict-learning", "x-decomp-sparse-pca", "x-decomp-lda", "x-decomp-manifold", "x-decomp-robust-cov", "x-decomp-als", "x-decomp-pca-randomized", "x-decomp-umap-options"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("IncrementalPCA", "GaussianRandomProjection", "SparseRandomProjection", "NMF", "FastICA", "FactorAnalysis", "SpectralEmbedding", "lu_factor", "lstsq", "PLSRegression", "PLSCanonical", "CCA", "DictionaryLearning", "MiniBatchDictionaryLearning", "SparsePCA", "MiniBatchSparsePCA", "LatentDirichletAllocation", "Isomap", "MDS", "ClassicalMDS", "LocallyLinearEmbedding", "MinCovDet", "EllipticEnvelope", "AlternatingLeastSquares", "SparseCoder"),
        display="the decomposition and linear algebra expansion",
        host_modules=("x_decomp/host.mojo", "x_decomp/host_simd.mojo", "x_decomp/host_graph.mojo", "x_decomp/host_qr.mojo", "x_decomp/host_jacobi.mojo", "x_decomp/host_lda.mojo", "x_decomp/host_ew.mojo", "x_decomp/cells.mojo", "x_decomp/api.mojo", "x_decomp/exec_trait.mojo", "x_decomp/kit.mojo", "x_decomp/mcd.mojo", "x_decomp/lda_online.mojo", "x_decomp/moves.mojo", "x_decomp/graph_cells.mojo", "x_decomp/graph_host.mojo"),
        exports=(
            "x_decomp_host_numeric_mode", "x_decomp_host_vendor", "x_decomp_host_column", "x_decomp_host_sabotage",
            "x_decomp_gemm", "x_decomp_ew", "x_decomp_colsum", "x_decomp_rowsum", "x_decomp_sqdist",
            "x_decomp_rand", "x_decomp_lu", "x_decomp_lu_solve", "x_decomp_chol", "x_decomp_eigh", "x_decomp_cd_rows", "x_decomp_orth", "x_decomp_orth_diag", "x_decomp_svd", "x_decomp_lasso_rows", "x_decomp_omp_rows", "x_decomp_rand_gamma", "x_decomp_lda_rows", "x_decomp_dijkstra_rows", "x_decomp_barycenter_rows", "x_decomp_als_rows", "x_decomp_absmax_sign", "x_decomp_qr_r", "x_decomp_geqrf", "x_decomp_orgqr", "x_decomp_als_cg_rows",
            # lane py-decomp-nbrs (2026-09-28): fast_mcd, LDA online and the MDS / Isomap moves
            "x_decomp_mcd", "x_decomp_lda_online", "x_decomp_gather", "x_decomp_scatter", "x_decomp_triu_nonzero",
            "x_decomp_argsort_f32", "x_decomp_iso_order",
            # lane hr2-graph-embed (2026-10-02): the Isomap / LLE graph builds as cells
            "x_decomp_graph_knn", "x_decomp_graph_knn_dense", "x_decomp_graph_radius", "x_decomp_graph_lle_iw",
            "x_decomp_graph_components", "x_decomp_graph_join", "x_decomp_graph_dijkstra",
            "x_decomp_numeric_mode", "x_decomp_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the decomp expansion lane's CPU route (lane/algos-decomp).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {"x-decomp-umap-options": "UMAP options", "x-decomp-pca-randomized": "PCA/TruncatedSVD randomized", "x-decomp-als": "AlternatingLeastSquares", "x-decomp-robust-cov": "MinCovDet", "x-decomp-manifold": "Isomap", "x-decomp-lda": "LatentDirichletAllocation", "x-decomp-sparse-pca": "SparsePCA", "x-decomp-dict-learning": "DictionaryLearning", "x-decomp-pls": "PLSRegression", "x-decomp-lstsq-rsvd": "lstsq", "x-decomp-lu": "lu_factor", "x-decomp-spectral-rbf": "SpectralEmbedding", "x-decomp-factor-analysis": "FactorAnalysis", "x-decomp-fastica": "FastICA", "x-decomp-nmf": "NMF", "x-decomp-ipca": "IncrementalPCA", "x-decomp-grp": "GaussianRandomProjection",
                       "x-decomp-srp": "SparseRandomProjection"}
PUBLIC_PENDING_LANES = {}
