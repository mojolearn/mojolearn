"""THE CLUSTER LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

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
        training_lanes=("x-cluster-minibatch-kmeans", "x-cluster-bisecting-kmeans"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("MiniBatchKMeans", "BisectingKMeans"),
        display="the cluster expansion lane (MiniBatchKMeans, BisectingKMeans, MeanShift, OPTICS, AffinityPropagation)",
        host_modules=(
            "x_cluster/host/host_ops.mojo", "x_cluster/bodies.mojo", "x_cluster/ops.mojo",
            "x_cluster/common.mojo", "x_cluster/entries.mojo", "x_cluster/out.mojo",
            "x_cluster/minibatch.mojo", "x_cluster/bisect.mojo",
        ),
        exports=(
            "x_cluster_host_numeric_mode", "x_cluster_host_vendor", "x_cluster_host_column",
            "x_cluster_host_sabotage", "x_cluster_call", "x_cluster_numeric_mode", "x_cluster_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the cluster expansion lane's CPU route (x_cluster/, lane/algos-cluster).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {"x-cluster-minibatch-kmeans": "MiniBatchKMeans",
                       "x-cluster-bisecting-kmeans": "BisectingKMeans"}
PUBLIC_PENDING_LANES = {"x-cluster-minibatch-kmeans": "no reference",
                        "x-cluster-bisecting-kmeans": "no reference"}
