"""THE DECOMP LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

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
        training_lanes=("x-decomp-ipca", "x-decomp-grp", "x-decomp-srp", "x-decomp-nmf", "x-decomp-fastica", "x-decomp-factor-analysis", "x-decomp-spectral-rbf", "x-decomp-lu", "x-decomp-lstsq-rsvd", "x-decomp-pls", "x-decomp-dict-learning", "x-decomp-sparse-pca", "x-decomp-lda", "x-decomp-manifold", "x-decomp-robust-cov", "x-decomp-als", "x-decomp-pca-randomized"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("IncrementalPCA", "GaussianRandomProjection", "SparseRandomProjection", "NMF", "FastICA", "FactorAnalysis", "SpectralEmbedding", "lu_factor", "lstsq", "PLSRegression", "PLSCanonical", "CCA", "DictionaryLearning", "MiniBatchDictionaryLearning", "SparsePCA", "MiniBatchSparsePCA", "LatentDirichletAllocation", "Isomap", "MDS", "ClassicalMDS", "LocallyLinearEmbedding", "MinCovDet", "EllipticEnvelope", "AlternatingLeastSquares", "SparseCoder"),
        display="the decomposition and linear algebra expansion",
        host_modules=("x_decomp/host.mojo", "x_decomp/cells.mojo", "x_decomp/api.mojo", "x_decomp/exec_trait.mojo"),
        exports=(
            "x_decomp_host_numeric_mode", "x_decomp_host_vendor", "x_decomp_host_column", "x_decomp_host_sabotage",
            "x_decomp_gemm", "x_decomp_ew", "x_decomp_colsum", "x_decomp_rowsum", "x_decomp_sqdist",
            "x_decomp_rand", "x_decomp_lu", "x_decomp_lu_solve", "x_decomp_chol", "x_decomp_eigh", "x_decomp_cd_rows", "x_decomp_orth", "x_decomp_svd", "x_decomp_lasso_rows", "x_decomp_omp_rows", "x_decomp_rand_gamma", "x_decomp_lda_rows", "x_decomp_dijkstra_rows", "x_decomp_barycenter_rows", "x_decomp_als_rows",
            "x_decomp_numeric_mode", "x_decomp_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the decomp expansion lane's CPU route (lane/algos-decomp).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {"x-decomp-pca-randomized": "PCA/TruncatedSVD randomized", "x-decomp-als": "AlternatingLeastSquares", "x-decomp-robust-cov": "MinCovDet", "x-decomp-manifold": "Isomap", "x-decomp-lda": "LatentDirichletAllocation", "x-decomp-sparse-pca": "SparsePCA", "x-decomp-dict-learning": "DictionaryLearning", "x-decomp-pls": "PLSRegression", "x-decomp-lstsq-rsvd": "lstsq", "x-decomp-lu": "lu_factor", "x-decomp-spectral-rbf": "SpectralEmbedding", "x-decomp-factor-analysis": "FactorAnalysis", "x-decomp-fastica": "FastICA", "x-decomp-nmf": "NMF", "x-decomp-ipca": "IncrementalPCA", "x-decomp-grp": "GaussianRandomProjection",
                       "x-decomp-srp": "SparseRandomProjection"}
PUBLIC_PENDING_LANES = {"x-decomp-pca-randomized": "no reference", "x-decomp-als": "no reference", "x-decomp-robust-cov": "no reference", "x-decomp-manifold": "no reference", "x-decomp-lda": "no reference", "x-decomp-sparse-pca": "no reference", "x-decomp-dict-learning": "no reference", "x-decomp-pls": "no reference", "x-decomp-lstsq-rsvd": "no reference", "x-decomp-lu": "no reference", "x-decomp-spectral-rbf": "no reference", "x-decomp-factor-analysis": "no reference", "x-decomp-fastica": "no reference", "x-decomp-nmf": "no reference", "x-decomp-ipca": "no reference", "x-decomp-grp": "no reference", "x-decomp-srp": "no reference"}
