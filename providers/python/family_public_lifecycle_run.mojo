# SPDX-License-Identifier: Apache-2.0
# RUN part of providers/python/family_public_lifecycle_pyapi.mojo: typed Mojo
# only (no Python imports). The lifecycle table of the connected family
# operations; the door applies it to the Python arguments.


struct FamilyLifecycle(Copyable, Movable):
    """How one public operation is checked. `method` is the public method it
    serves ("" for an operation outside this lifecycle: a property, the
    pickle state, a composed internal call)."""
    var method: String
    var requires_fit: Bool   # False for a stateless transformer (scikit-learn's requires_fit tag)
    var scan: Bool           # the X finite scan scikit-learn's validate_data makes
    var replace: Bool        # hand the owner the admitted dense float32 snapshot of X
    var graph: Bool          # an adjacency input: an (indptr, indices, n) tuple passes through
    var target: Int          # the position of y validated at an outermost fit (-1: none)
    var classifier: Bool     # discrete labels; public ClassLabels classes_ after fit
    var required: Bool       # fit refuses y=None
    var widen: Bool          # float32 result widened exactly to float64 for outside callers
    # The owner has device-X fold entries (lane a7d): a model-selection fold
    # whose store ids resolve reaches it as the DeviceRows itself (no host
    # gather here); its entry gathers the rows on the device.
    var device_x: Bool
    # The owner's first x_neighbors op tests a borrowed X behind its upload
    # (its status slot, lane a8b-device-x-neighbors-rest): the scan is
    # deferred to it; a cast X is still refused by its cast.
    var defer: Bool

    def __init__(out self, method: String, requires_fit: Bool = True, scan: Bool = False,
                 replace: Bool = True, graph: Bool = False, target: Int = -1,
                 classifier: Bool = False, required: Bool = False, widen: Bool = False,
                 device_x: Bool = False, defer: Bool = False):
        self.method = method
        self.requires_fit = requires_fit
        self.scan = scan
        self.replace = replace
        self.graph = graph
        self.target = target
        self.classifier = classifier
        self.required = required
        self.widen = widen
        self.device_x = device_x
        self.defer = defer


def family_lifecycle_spec(operation: String) -> FamilyLifecycle:
    """The lifecycle of a connected public operation. The positions are the
    owner argument tuple's: 0 the resolved binding, 1 X, then the rest."""
    # OneClassSVM (bindings/svc_ocsvm_native_door.mojo). sample_weight is not a target.
    if operation == "family_svc_ocsvm_fit":
        return FamilyLifecycle("fit", scan=True)
    if operation == "family_svc_ocsvm_fit_predict":
        return FamilyLifecycle("fit_predict", scan=True)
    if operation == "family_svc_ocsvm_score_samples":
        return FamilyLifecycle("score_samples", scan=True, widen=True)
    if operation == "family_svc_ocsvm_decision":
        return FamilyLifecycle("decision_function", scan=True, widen=True)
    if operation == "family_svc_ocsvm_predict":
        return FamilyLifecycle("predict", scan=True)
    # AdditiveChi2Sampler is stateless (transform needs no fit); a sparse
    # input keeps its sparse result, so X passes through unreplaced when sparse.
    if operation == "family_xn_semi_achi2_fit":
        return FamilyLifecycle("fit", scan=True)
    if operation == "family_xn_semi_achi2_transform":
        return FamilyLifecycle("transform", requires_fit=False, scan=True)
    if operation == "family_xn_semi_achi2_fit_transform":
        return FamilyLifecycle("fit_transform", scan=True)
    if operation == "family_xn_semi_schi2_fit":
        return FamilyLifecycle("fit", scan=True)
    if operation == "family_xn_semi_schi2_transform":
        return FamilyLifecycle("transform", scan=True)
    if operation == "family_xn_semi_schi2_fit_transform":
        return FamilyLifecycle("fit_transform", scan=True)
    # LabelPropagation / LabelSpreading: y (-1 marks unlabeled) at position 2.
    # Lane a8b-device-x-neighbors-rest: X's finite test rides behind the
    # owner's first op upload (bindings/neighbors_semi_native_door.mojo).
    if operation == "family_xn_semi_lp_fit":
        return FamilyLifecycle("fit", scan=True, target=2, classifier=True, required=True, defer=True)
    if operation == "family_xn_semi_lp_predict_proba":
        return FamilyLifecycle("predict_proba", scan=True, defer=True)
    if operation == "family_xn_semi_lp_predict":
        return FamilyLifecycle("predict", scan=True, defer=True)
    if operation == "family_xn_semi_lp_score":
        return FamilyLifecycle("score", scan=True, defer=True)
    # PageRank / Louvain take an adjacency (dense, scipy.sparse or a CSR tuple).
    if operation == "family_xn_graph_pagerank_fit" or operation == "family_xn_graph_louvain_fit":
        return FamilyLifecycle("fit", replace=False, graph=True)
    if operation == "family_xn_graph_louvain_fit_predict":
        return FamilyLifecycle("fit_predict", replace=False, graph=True)
    # SVGP (predict_f / predict_y are latent-function queries outside this table).
    if operation == "family_xn_graph_svgp_fit":
        return FamilyLifecycle("fit", target=2, required=True)
    if operation == "family_xn_graph_svgp_predict":
        return FamilyLifecycle("predict", widen=True)
    if operation == "family_xn_graph_svgp_score":
        return FamilyLifecycle("score")
    # KMeans (bindings/kmeans_native.mojo): a fit's X reaches the owner as the
    # caller passed it (array init receives the rows). Perf (lane
    # perf2-cluster): predict and transform take the admitted float32
    # snapshot, so X is admitted and cast once per query, not twice. Lane
    # a7d: no scan here; the owner's kernel entry returns X's finite status
    # word over its own upload, and a resident fold reaches the owner as the
    # DeviceRows for its device-X entries (`device_x`).
    if operation == "family_kmeans_fit":
        return FamilyLifecycle("fit", replace=False, device_x=True)
    if operation == "family_kmeans_fit_predict":
        return FamilyLifecycle("fit_predict", replace=False, device_x=True)
    if operation == "family_kmeans_fit_transform":
        return FamilyLifecycle("fit_transform", replace=False, device_x=True)
    if operation == "family_kmeans_predict":
        return FamilyLifecycle("predict", device_x=True)
    if operation == "family_kmeans_transform":
        return FamilyLifecycle("transform", device_x=True)
    # LinearSVC / LinearSVR (bindings/svc_linear_native_door.mojo, svm_mixture_native.mojo).
    if operation == "family_svc_linear_svc_fit":
        return FamilyLifecycle("fit", scan=True, replace=False, target=2, classifier=True, required=True)
    if operation == "family_svm_linear_svc_decision":
        return FamilyLifecycle("decision_function", scan=True, replace=False)
    if operation == "family_svc_linear_svc_predict":
        return FamilyLifecycle("predict", scan=True, replace=False)
    if operation == "family_svc_linear_svc_score":
        return FamilyLifecycle("score", scan=True, replace=False)
    if operation == "family_svc_linear_svr_fit":
        return FamilyLifecycle("fit", scan=True, replace=False, target=2, required=True)
    if operation == "family_svm_linear_svr_predict":
        return FamilyLifecycle("predict", scan=True, replace=False, widen=True)
    if operation == "family_svc_linear_svr_score":
        return FamilyLifecycle("score", scan=True, replace=False)
    return FamilyLifecycle("")


def kpca_lifecycle_spec(operation: String) -> FamilyLifecycle:
    """KernelPCA through the decomposition door (X at position 0)."""
    if operation == "fit":
        return FamilyLifecycle("fit", scan=True)
    if operation == "fit_transform":
        return FamilyLifecycle("fit_transform", scan=True)
    if operation == "transform":
        return FamilyLifecycle("transform", scan=True)
    return FamilyLifecycle("")
