"""THE NEIGHBORS LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `neighbors` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The `exports` tuple is written by x_neighbors/gen.py from its op table.
"""
GPU_BINDINGS = ("_mojolearn_x_neighbors",)
FAMILIES = (
    dict(
        family="x_neighbors",
        binding="_mojolearn_x_neighbors_host",
        routes="_mojolearn_x_neighbors",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "x-neighbors-lof", "x-neighbors-nearest-centroid", "x-neighbors-ocsvm",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=("LocalOutlierFactor", "NearestCentroid", "OneClassSVM"),
        display="the neighbors + kernel expansion lane",
        host_modules=("x_neighbors/items.mojo", "x_neighbors/host_ops.mojo", "x_neighbors/eigh.mojo"),
        exports=(
            # BEGIN GENERATED EXPORTS
            "x_neighbors_host_numeric_mode",
            "x_neighbors_host_vendor",
            "x_neighbors_host_column",
            "x_neighbors_host_sabotage",
            "xn_sqdist",
            "xn_nan_sqdist",
            "xn_l1dist",
            "xn_kernel",
            "xn_matmul",
            "xn_rowsum",
            "xn_colsum",
            "xn_unary",
            "xn_knn_select",
            "xn_group_mean",
            "xn_take_rows",
            "xn_take_cols",
            "xn_variance",
            "xn_ocsvm",
            "xn_lof_lrd",
            "xn_lof_score",
            "xn_kpca_center",
            "xn_scale_div",
            "xn_svd_flip",
            "xn_kpca_alpha_scale",
            "xn_nc_std",
            "xn_nc_shrink",
            "xn_nc_decision",
            "xn_softmax",
            "xn_eigh",
            "x_neighbors_numeric_mode",
            "x_neighbors_vendor",
            # END GENERATED EXPORTS
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the neighbors + kernel expansion lane's CPU route (x_neighbors/).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "x-neighbors-lof": "LocalOutlierFactor",
    "x-neighbors-nearest-centroid": "NearestCentroid",
    "x-neighbors-ocsvm": "OneClassSVM",
}
PUBLIC_PENDING_LANES = {
    "x-neighbors-lof": "no reference",
    "x-neighbors-nearest-centroid": "no reference",
    "x-neighbors-ocsvm": "no reference",
}
