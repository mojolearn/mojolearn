#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the member ABI's family table: `python3 tools/gen_member_families.py`.

One list, `FAMILIES`, is the record of every family number of the member ABI
(core/member_abi.mojo). From it this tool writes

  * `core/member_families.mojo`: the numbers, the names and the eight table
    slots (tier x vendor) of every family, for publication and resolution;
  * `bindings/member_boundary_door.mojo`: the one place a caller's UNPACK /
    PACK reaches a family's member boundary (admit, pack, state);
  * `providers/python/member_<family>_boundary_door.mojo` for a family that
    has none yet: the unpublished form (admit answers False). A file that
    exists is never rewritten: it belongs to the family's publication.

A number is never reused or renumbered: a new family is appended before the
spares, or takes a spare's place. Serial by nature: three small text files
from one list; nothing here is product code and nothing is timed.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# (constant, slot token, file stem, public classes, images that publish it)
FAMILIES = [
    ("LINEAR", "Linear", "linear", "LinearRegression Ridge", "estimators"),
    ("LOGISTIC", "Logistic", "logistic", "LogisticRegression QNRegressor", "estimators"),
    ("XLINEAR", "XLinear", "xlinear",
     "PoissonRegressor GammaRegressor TweedieRegressor HuberRegressor BayesianRidge ARDRegression Lars LassoLars "
     "QuantileRegressor RidgeClassifier RidgeCV LassoCV ElasticNetCV LogisticRegressionCV IsotonicRegression", "x_linear"),
    ("SGD", "Sgd", "sgd",
     "SGDClassifier SGDRegressor Perceptron PassiveAggressiveClassifier PassiveAggressiveRegressor SGDOneClassSVM", "x_linear"),
    ("CD", "Cd", "cd", "Lasso ElasticNet", "solver"),
    ("LINEAR_SVM", "LinearSvm", "linear_svm", "LinearSVC LinearSVR", "estimators"),
    ("KERNEL_SVM", "KernelSvm", "kernel_svm", "SVC SVR", "svm"),
    ("KERNEL_RIDGE", "KernelRidge", "kernel_ridge", "KernelRidge", "kernel_methods"),
    ("KERNEL_APPROX", "KernelApprox", "kernel_approx", "Nystroem RBFSampler", "kernel_methods"),
    ("NAIVE_BAYES", "NaiveBayes", "naive_bayes", "GaussianNB MultinomialNB BernoulliNB ComplementNB CategoricalNB", "x_prep"),
    ("DISCRIMINANT", "Discriminant", "discriminant", "LinearDiscriminantAnalysis QuadraticDiscriminantAnalysis", "x_prep"),
    ("NEIGHBORS", "Neighbors", "neighbors", "KNeighborsClassifier KNeighborsRegressor RadiusNeighbors NearestNeighbors", "neighbors"),
    ("XNEIGHBORS", "XNeighbors", "xneighbors",
     "NearestCentroid LocalOutlierFactor OneClassSVM KernelPCA PolynomialCountSketch AdditiveChi2Sampler "
     "SkewedChi2Sampler LabelPropagation LabelSpreading KNNImputer SVGP", "x_neighbors"),
    ("TREE", "Tree", "tree", "DecisionTreeClassifier DecisionTreeRegressor", "rf"),
    ("FOREST", "Forest", "forest", "RandomForestClassifier RandomForestRegressor", "rf"),
    ("EXTRA_TREES", "ExtraTrees", "extra_trees", "ExtraTreesClassifier ExtraTreesRegressor", "trees"),
    ("GBDT", "Gbdt", "gbdt", "GradientBoosting GradientBoostingClassifier GradientBoostingRegressor OrderedRMSE", "gbdt"),
    ("DART", "Dart", "dart", "DARTClassifier DARTRegressor", "x_trees"),
    ("ISOLATION_FOREST", "IsolationForest", "isolation_forest", "IsolationForest", "trees"),
    ("MLP", "Mlp", "mlp", "MLPClassifier MLPRegressor", "x_sequence"),
    ("SEQUENCE", "Sequence", "sequence",
     "LSTMRegressor LSTMClassifier GRURegressor GRUClassifier RNNRegressor RNNClassifier", "x_sequence"),
    ("CNN", "Cnn", "cnn", "CNNClassifier", "x_cnn"),
    ("PLS", "Pls", "pls", "PLSRegression PLSCanonical CCA", "x_decomp"),
    ("GP", "Gp", "gp", "GaussianProcessRegressor GaussianProcessClassifier", "gp"),
    ("KMEANS", "KMeans", "kmeans", "KMeans", "cluster"),
    ("XCLUSTER", "XCluster", "xcluster",
     "MiniBatchKMeans BisectingKMeans MeanShift OPTICS AffinityPropagation BayesianGaussianMixture", "x_cluster"),
    ("MIXTURE", "Mixture", "mixture", "GaussianMixture", "mixture"),
    ("DENSITY", "Density", "density", "KernelDensity DBSCAN", "estimators"),
    ("HDBSCAN", "Hdbscan", "hdbscan", "HDBSCAN", "hdbscan"),
    ("HIERARCHY", "Hierarchy", "hierarchy", "AgglomerativeClustering", "hierarchy"),
    ("SPECTRAL", "Spectral", "spectral", "SpectralClustering SpectralEmbedding", "spectral"),
    ("DECOMPOSITION", "Decomposition", "decomposition", "PCA TruncatedSVD", "estimators"),
    ("XDECOMP", "XDecomp", "xdecomp",
     "IncrementalPCA GaussianRandomProjection SparseRandomProjection NMF FastICA FactorAnalysis DictionaryLearning "
     "MiniBatchDictionaryLearning SparsePCA MiniBatchSparsePCA SparseCoder LatentDirichletAllocation Isomap MDS "
     "ClassicalMDS LocallyLinearEmbedding MinCovDet EllipticEnvelope", "x_decomp"),
    ("MANIFOLD", "Manifold", "manifold", "UMAP", "metrics"),
    ("SCALER", "Scaler", "scaler", "StandardScaler MinMaxScaler", "preprocessing"),
    ("PREP", "Prep", "prep",
     "RobustScaler MaxAbsScaler OrdinalEncoder OneHotEncoder TargetEncoder SimpleImputer IterativeImputer "
     "KBinsDiscretizer QuantileTransformer PowerTransformer Normalizer PolynomialFeatures SplineTransformer Binarizer",
     "x_prep"),
    ("SELECT", "Select", "select", "VarianceThreshold SelectKBest RFE", "x_prep"),
    ("FORECAST", "Forecast", "forecast", "ARIMA AutoARIMA ETS Theta (the forecasters)", "arima / x_sequence"),
    ("CHOLESKY", "Cholesky", "cholesky", "Cholesky", "cholesky"),
    ("ENSEMBLE", "Ensemble", "ensemble",
     "BaggingClassifier BaggingRegressor AdaBoostClassifier AdaBoostRegressor VotingClassifier VotingRegressor "
     "StackingClassifier StackingRegressor MultiOutputClassifier MultiOutputRegressor OneVsRestClassifier "
     "CalibratedClassifierCV RandomTreesEmbedding (a meta-estimator as the member of another)", "x_trees"),
    ("SEARCH", "Search", "search", "GridSearchCV RandomizedSearchCV and the halving searches as members", "model_selection"),
    ("PIPELINE", "Pipeline", "pipeline", "Pipeline and the column composers, when they exist", "-"),
    # Number 42 was SPARE0 until 2026-11-21 (nothing had published on it): TSNE lives in another image
    # than UMAP, and one family publishes from ONE image per (tier, vendor).
    ("TSNE", "Tsne", "tsne", "TSNE", "x_ann"),
    ("SPARE1", "Spare1", None, "(unassigned)", "-"),
    ("SPARE2", "Spare2", None, "(unassigned)", "-"),
    ("SPARE3", "Spare3", None, "(unassigned)", "-"),
]

KEYS = (("vendor == 0 and mode == 1", "IdenticalCPU"), ("vendor == 0 and mode == 0", "FastCPU"),
        ("vendor == 1 and mode == 1", "IdenticalMetal"), ("vendor == 1 and mode == 0", "FastMetal"),
        ("vendor == 2 and mode == 1", "IdenticalCUDA"), ("vendor == 2 and mode == 0", "FastCUDA"),
        ("vendor == 3 and mode == 1", "IdenticalHIP"), ("vendor == 3 and mode == 0", "FastHIP"))

FAMILIES_HEAD = '''# SPDX-License-Identifier: Apache-2.0
# GENERATED by tools/gen_member_families.py from its FAMILIES list. Do not edit
# by hand: change the list and run the tool.
"""The member ABI's family table (core/member_abi.mojo): every family number,
and for each the eight process-wide table slots (tier x vendor) its images
publish their member entry in and a caller resolves it from.

A family publishes one entry per image. Its slot name holds the family, the
tier and the vendor (`MojolearnMember<Family><Identical|Fast><CPU|Metal|
CUDA|HIP>V2`), so a CPU image and a GPU image loaded in one process hold two
slots. Only the slot a route asks for is ever created.

SERIAL BY NATURE: publication and resolution are metadata reads, once per
image and once per member.
"""
from core.entry_table import EntryRef, entry_ref_word, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall

#: The schema of a member entry and its operation count (the operations are
#: the MEMBER_OP_* codes of core/member_abi.mojo).
comptime MEMBER_ABI = "mojolearn.member.v2"
comptime MEMBER_OPERATION_COUNT = 16

'''

SLOT = '''

def member_family_name(family: Int) -> String:
    """The family's name in a refusal."""
%(names)s    return String("unknown")


def _member_slot[name: StaticString, publish: Int](mode: Int, vendor: Int, publisher: Int,
    entry: NativeDescriptorCall) raises -> EntryRef:
    comptime if publish == 1:
        var schemas: List[String] = [String(MEMBER_ABI)]
        var operations: List[Int] = [MEMBER_OPERATION_COUNT]
        var entries: List[NativeDescriptorCall] = [entry]
        entry_table_publish[name](mode, vendor, publisher, schemas, operations, entries)
        return EntryRef(entry, entry_ref_word(mode, vendor, MEMBER_OPERATION_COUNT))
    else:
        return entry_table_resolve[name](mode, vendor, 0, String(MEMBER_ABI), MEMBER_OPERATION_COUNT)


def member_table[family: Int, publish: Int](mode: Int, vendor: Int, publisher: Int,
    entry: NativeDescriptorCall) raises -> EntryRef:
    """The family's slot of the image (`mode` 1 identical / 0 fast, `vendor`
    0 CPU, 1 Metal, 2 CUDA, 3 HIP): published with `entry` (`publish` 1) or
    resolved (`publish` 0; `publisher` and `entry` are then not read)."""
    comptime assert family >= 0 and family < MEMBER_FAMILY_COUNT, "unknown member family"
    comptime assert publish == 0 or publish == 1, "member table: publish is 0 or 1"
    if mode < 0 or mode > 1 or vendor < 0 or vendor > 3:
        raise Error("member table: invalid tier/vendor key")
'''


def families_text():
    out = [FAMILIES_HEAD]
    for number, (const, _token, _stem, classes, _images) in enumerate(FAMILIES):
        out.append("comptime MEMBER_FAMILY_%s = %d\n" % (const, number))
    out.append("comptime MEMBER_FAMILY_COUNT = %d\n" % len(FAMILIES))
    names = "".join("    if family == MEMBER_FAMILY_%s:\n        return String(\"%s\")\n" % (c, t)
                    for c, t, _s, _k, _i in FAMILIES)
    out.append(SLOT % dict(names=names))
    for const, token, _stem, _classes, _images in FAMILIES:
        out.append("    comptime if family == MEMBER_FAMILY_%s:\n" % const)
        for test, key in KEYS:
            out.append("        if %s:\n" % test)
            out.append("            comptime NAME = \"MojolearnMember%s%sV2\"\n" % (token, key))
            out.append("            return _member_slot[NAME, publish](mode, vendor, publisher, entry)\n")
    out.append("    raise Error(\"member table: unknown family\")\n")
    out.append('''

def member_table_resolve(family: Int, mode: Int, vendor: Int, absent: NativeDescriptorCall) raises -> EntryRef:
    """`member_table[family, 0]` for a family number known only at run time
    (a caller that read it from the member's class). `absent` is any entry
    pointer; a resolution does not read it."""
''')
    for const, _token, _stem, _classes, _images in FAMILIES:
        out.append("    if family == MEMBER_FAMILY_%s:\n" % const)
        out.append("        return member_table[MEMBER_FAMILY_%s, 0](mode, vendor, 0, absent)\n" % const)
    out.append("    raise Error(\"member table: unknown family number \" + String(family))\n")
    return "".join(out)


STUB = '''# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the %(token)s family (%(classes)s): NOT PUBLISHED YET.

This file is the family's place in `bindings/member_boundary_door.mojo`. The
family's publication replaces the three bodies below (recipe:
MEMBER-ABI-V2.md, worked form: providers/python/member_linear_boundary_door.mojo)
and sets `_mojolearn_member = (MEMBER_FAMILY_%(const)s, kind)` on each class
it publishes. Until then no class carries the mark and a caller keeps the
member's own doors.
"""
from std.python import PythonObject
from bindings.member_owner_unpack import MemberOwner
from core.member_abi import MemberPlan


def %(stem)s_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this family has no member entry."""
    return False


def %(stem)s_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`."""
    raise Error("RuntimeError: mojolearn: the %(token)s family has not published its member entry")


def %(stem)s_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`."""
    raise Error("RuntimeError: mojolearn: the %(token)s family has not published its member entry")
'''

DOOR_HEAD = '''# SPDX-License-Identifier: Apache-2.0
# GENERATED by tools/gen_member_families.py from its FAMILIES list. Do not edit
# by hand: change the list and run the tool.
"""The one place a caller's UNPACK and PACK reach a family's member boundary
(core/member_abi.mojo; recipe: MEMBER-ABI-V2.md).

A class that runs as a native member carries `_mojolearn_member = (family,
kind)`; model selection's import-time plan holds it (`plan["member"]`,
python/mojolearn/_native_msel_dispatch.py). `member_admit` reads the mark,
has the family fill the member's route and parameter words and resolves the
family's entry: once per member, with every refusal here. Everything after
it is typed (`MemberPlan`, `MemberCall`) and runs with the GIL released.

SERIAL BY NATURE: one member's admission; a caller with many members admits
each once and runs them on its own workers.
"""
from std.python import PythonObject
from bindings.member_owner_unpack import MemberOwner, member_mark, member_labels_hold, member_labels_take
from bindings.py_balanced import py_setitem
from core.member_abi import MemberPlan, member_resolve_family
from core.member_families import (
%(consts)s)
%(imports)s

def member_admit(model: PythonObject, fields: PythonObject, plan: PythonObject, context: PythonObject,
    b: PythonObject) raises -> MemberOwner:
    """The native member of `model`, or an owner with `published` False when
    its class has no member entry (the caller then keeps the member's own
    doors). `plan` is the class's model-selection plan, `context` the
    member's family context (`plan["context"]`), `fields` the instance
    dictionary, `b` the builtins module the caller already admitted."""
    var owner = MemberOwner()
    var mark = member_mark(plan, b)
    if mark.family < 0:
        return owner^
    owner.plan.family = mark.family
    owner.plan.kind = mark.kind
    var filled = False
%(admit)s    if not filled:
        return owner^
    owner.plan.member = member_resolve_family(owner.plan.family, owner.plan.route)
    owner.published = True
    return owner^


def member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of member state `state` into the
    estimator `model` (`best_estimator_`, `estimators_`), as its own fit
    would have."""
%(pack)s    raise Error("RuntimeError: mojolearn: unknown member family number " + String(plan.family))


def member_pack_labeled(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int, labels: PythonObject) raises:
    """PACK of a CLASSIFIER member: `member_pack`, with the caller's class
    labels (the object the estimator's own fit publishes as its classes for
    the `y` the caller encoded: code `c` is `labels[c]`).

    The labels never enter the typed plan. They are held in the instance
    dictionary under the one reserved key while the family's pack runs
    (bindings/member_owner_unpack.mojo `member_labels_hold`). A family whose
    fitted estimator keeps the labels in its own form (a property over a
    private field, a label-name table) takes them with
    `member_labels_take(fields, b)` inside its pack and writes that form. A
    family that does not take them gets the plain attribute: `classes_` is
    written here, as the estimators that keep it as an instance field do.
    If the family's pack raises, the caller drops the estimator; the key is
    not removed from an estimator nobody will see."""
    member_labels_hold(fields, labels)
    member_pack(model, fields, context, b, plan, state)
    var left = member_labels_take(fields, b)
    if left is not None:
        py_setitem(fields, PythonObject("classes_"), left)


def member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of the
    fitted estimator `model` (a prefit member). The caller frees it. A prefit
    classifier's labels are its public `classes_`: the caller reads them at
    its own UNPACK and sets `plan.classes` before this call."""
%(state)s    raise Error("RuntimeError: mojolearn: unknown member family number " + String(plan.family))
'''


def door_text():
    live = [f for f in FAMILIES if f[2]]
    consts = "".join("    MEMBER_FAMILY_%s,\n" % f[0] for f in live)
    imports = "".join(
        "from providers.python.member_%s_boundary_door import %s_member_admit, %s_member_pack, %s_member_state\n"
        % (f[2], f[2], f[2], f[2]) for f in live)
    admit = "".join(
        "    %s owner.plan.family == MEMBER_FAMILY_%s:\n        filled = %s_member_admit(model, fields, mark.kind, context, b, owner)\n"
        % ("if" if i == 0 else "elif", f[0], f[2]) for i, f in enumerate(live))
    pack = "".join(
        "    if plan.family == MEMBER_FAMILY_%s:\n        %s_member_pack(model, fields, context, b, plan, state)\n        return\n"
        % (f[0], f[2]) for f in live)
    state = "".join(
        "    if plan.family == MEMBER_FAMILY_%s:\n        return %s_member_state(model, fields, context, b, plan)\n"
        % (f[0], f[2]) for f in live)
    return DOOR_HEAD % dict(consts=consts, imports=imports, admit=admit, pack=pack, state=state)


def table_text():
    rows = ["| number | constant | slot token | member file (RUN) | boundary file (UNPACK / PACK) | image | classes |",
            "|---|---|---|---|---|---|---|"]
    for number, (const, token, stem, classes, images) in enumerate(FAMILIES):
        rows.append("| %d | `MEMBER_FAMILY_%s` | `%s` | %s | %s | %s | %s |" % (
            number, const, token,
            "`core/%s_member.mojo`" % stem if stem else "-",
            "`providers/python/member_%s_boundary_door.mojo`" % stem if stem else "-", images, classes))
    return "\n".join(rows) + "\n"


def main(argv):
    if "--table" in argv:
        sys.stdout.write(table_text())
        return 0
    written = []
    for path, text in ((os.path.join(ROOT, "core", "member_families.mojo"), families_text()),
                       (os.path.join(ROOT, "bindings", "member_boundary_door.mojo"), door_text())):
        open(path, "w", encoding="utf-8").write(text)
        written.append(path)
    for const, token, stem, classes, _images in FAMILIES:
        if not stem:
            continue
        path = os.path.join(ROOT, "providers", "python", "member_%s_boundary_door.mojo" % stem)
        if os.path.exists(path):
            continue
        open(path, "w", encoding="utf-8").write(STUB % dict(const=const, token=token, stem=stem, classes=classes))
        written.append(path)
    print("families:", len(FAMILIES), "files written:", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
