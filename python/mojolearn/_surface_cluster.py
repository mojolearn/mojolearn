"""THE CLUSTER LANE'S HOST SURFACE FRAGMENT.

Owned by the `cluster` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_cluster",) once bindings/build_x_cluster.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_cluster", binding="_mojolearn_x_cluster_host",
                        routes="_mojolearn_x_cluster", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_cluster",)
FAMILIES = (
    dict(
        family="x_cluster",
        binding="_mojolearn_x_cluster_host",
        routes="_mojolearn_x_cluster",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("x-cluster-minibatch-kmeans", "x-cluster-bisecting-kmeans", "x-cluster-meanshift", "x-cluster-optics", "x-cluster-affinity-propagation", "x-cluster-bgmm",
                        "x-cluster-minibatch-options", "x-cluster-bisecting-options", "x-cluster-meanshift-binned", "x-cluster-optics-metrics", "x-cluster-ap-precomputed", "x-cluster-bgmm-inits", "x-cluster-bgmm-covtypes", "x-cluster-minibatch-partial", "x-cluster-gmm-options", "x-cluster-dbscan-metrics", "x-cluster-hdbscan-epsilon", "x-cluster-kmeans-init", "x-cluster-agglo-linkages", "x-cluster-agglo-connectivity", "x-cluster-spectral-affinities"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("MiniBatchKMeans", "BisectingKMeans", "MeanShift", "OPTICS", "AffinityPropagation", "BayesianGaussianMixture"),
        display="the cluster expansion lane (MiniBatchKMeans, BisectingKMeans, MeanShift, OPTICS, AffinityPropagation)",
        host_modules=(
            "x_cluster/host/host_ops.mojo", "x_cluster/bodies.mojo", "x_cluster/ops.mojo",
            "x_cluster/common.mojo", "x_cluster/entries.mojo", "x_cluster/out.mojo",
            "x_cluster/minibatch.mojo", "x_cluster/bisect.mojo", "x_cluster/meanshift.mojo", "x_cluster/optics.mojo", "x_cluster/affinity.mojo", "x_cluster/bgmm.mojo", "x_cluster/agglo.mojo", "x_cluster/spectral_assign.mojo", "x_cluster/post_bodies.mojo", "x_cluster/tree_cut.mojo",
        ),
        exports=(
            "x_cluster_host_numeric_mode", "x_cluster_host_vendor", "x_cluster_host_column",
            "x_cluster_host_sabotage", "x_cluster_call", "x_cluster_numeric_mode", "x_cluster_vendor", "x_cluster_py2mojo",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the cluster expansion lane's CPU route (x_cluster/, lane/algos-cluster).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {"x-cluster-minibatch-kmeans": "MiniBatchKMeans",
                       "x-cluster-bisecting-kmeans": "BisectingKMeans",
                       "x-cluster-meanshift": "MeanShift",
                       "x-cluster-optics": "OPTICS",
                       "x-cluster-affinity-propagation": "AffinityPropagation",
                       "x-cluster-bgmm": "BayesianGaussianMixture",
                       "x-cluster-minibatch-options": "MiniBatchKMeans options",
                       "x-cluster-bisecting-options": "BisectingKMeans options",
                       "x-cluster-meanshift-binned": "MeanShift bin seeding",
                       "x-cluster-optics-metrics": "OPTICS metrics",
                       "x-cluster-ap-precomputed": "AffinityPropagation precomputed",
                       "x-cluster-bgmm-inits": "BayesianGaussianMixture inits",
                       "x-cluster-bgmm-covtypes": "BayesianGaussianMixture covariance types",
                       "x-cluster-minibatch-partial": "MiniBatchKMeans partial_fit",
                       "x-cluster-gmm-options": "GaussianMixture options",
                       "x-cluster-dbscan-metrics": "DBSCAN cosine and precomputed metrics",
                       "x-cluster-hdbscan-epsilon": "HDBSCAN cluster_selection_epsilon",
                       "x-cluster-kmeans-init": "KMeans array and callable init",
                       "x-cluster-agglo-linkages": "AgglomerativeClustering linkages and metrics",
                       "x-cluster-agglo-connectivity": "AgglomerativeClustering connectivity",
                       "x-cluster-spectral-affinities": "SpectralClustering affinities and assign_labels"}
PUBLIC_PENDING_LANES = {}
