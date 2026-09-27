"""THE ANN LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `ann` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_ann",) once bindings/build_x_ann.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_ann", binding="_mojolearn_x_ann_host",
                        routes="_mojolearn_x_ann", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_ann",)
FAMILIES = (
    dict(
        family="x_ann",
        binding="_mojolearn_x_ann_host",
        routes="_mojolearn_x_ann",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("x-ann-ivf-pq", "x-ann-tsne"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("IVFPQIndex", "TSNE"),
        display="the ann lane (IVF-PQ, t-SNE)",
        host_modules=("x_ann/host/ivf_pq_host.mojo", "x_ann/host/tsne_host.mojo"),
        exports=(
            "x_ann_host_numeric_mode", "x_ann_host_vendor", "x_ann_host_column", "x_ann_host_sabotage",
            "x_ann_ivf_pq_build", "x_ann_ivf_pq_search", "x_ann_tsne_fit", "x_ann_numeric_mode", "x_ann_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the ann lane's CPU route (IVF-PQ build and search, t-SNE).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {"x-ann-ivf-pq": "the IVF-PQ index", "x-ann-tsne": "t-SNE"}
PUBLIC_PENDING_LANES = {"x-ann-ivf-pq": "no reference", "x-ann-tsne": "no reference"}
