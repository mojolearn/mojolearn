# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The set that answers a numeric-mode word: the one shared step of every
door's mode admission (lane fx-fast-selector).

`fast` is `identical` with named relaxations switched on. Off Apple no fast
set is built yet, and the selector decides once, at import, that the identical
set answers the name `fast` (python/mojolearn/_backend.py, `_FAST_ANSWER`).
A door admits the user's word as before (`fast`, `identical`, or its refusal
with its own wording) and then calls `answer_mode_word` (here) or
`answer_mode_object` (bindings/mode_answer_door.mojo, for a door that keys its
tables by the Python word), which replace an admitted `fast` by `identical`
where that decision holds. Everything after the admission (the mode-keyed
tables, the compiled-mode checks, the fitted-mode comparisons, the branches)
then sees the word of the set that runs. The estimator's own `numeric_mode`
attribute is never written: it reads back what the user passed.

Cost. On a macOS build the helpers return at compile time, so Apple is
unchanged. Elsewhere an admitted `identical` pays one comparison of the
admitted word; only the word `fast` reads the process-wide answer (one
named-slot read and one mask test), which `connect_mode_answer` stored once
at import. No Python lookup, lock, allocation or string build.

Why the answer is not stored beside each estimator's word: the word is
written by plain Python attribute stores (the constructors, `set_params`,
`clone`, a user's own assignment), which may not run package logic, so a
stored effective word could only be trusted after comparing it with the
user's word on every call: a second instance-dictionary read on every call
of every mode, against one slot read on the `fast` word alone.

Families (lane fx-fast-selector2). The answer is held per door family, as one
bit of `ModeAnswer.families`. A family's bit is set only when EVERY public
operation of that family (fit, predict, score, transform, explain, partial
fit, save, load, parameter read-back) answers the word the same way; the set
of such families is the one constant `MODE_FAMILIES_ANSWERED` below, which
names the shared doors a family passes (RESOLVER, DECOMP_KIT, METADATA,
PROGRAM) beside the family itself. A family
whose bit is clear keeps the uniform refusal of an install with no fast set
in all its operations: its doors' calls here leave the word `fast` alone, so
no estimator fits under `fast` and then refuses in its next call. Turning a
family on is one edit of that constant, made when the built-install test
(python/mojolearn/tests/test_fast_selector.py, BuiltInstallFamilies) shows
every operation of the family the same under `fast`.

Not connected (a native test, or an install that carries a fast set): no
family is answered and `fast` names the fast set as before.

Build kind. The test here is `CompilationTarget.is_macos()`, not
`POLICY_BUILD` of checks/mode_policy.mojo: these helpers compile into the
host frontend, whose own build is always the identical contract and says
nothing about which GPU set the install carries. The question a door asks is
whether the platform has its own fast set (Apple) or the selector decided at
import. Inside a tier binary the build kind is `POLICY_BUILD`.

A leaf module with no Python in it (std and core/shared_slot only), so that
core/linear_metadata.mojo and the doors can all import it.
"""

from std.sys.info import CompilationTarget
from core.shared_slot import SharedSlot


# One bit per door family. A new family takes the next free bit; the bits are
# process-internal (never saved), so they may be renumbered with the slot's
# version.
comptime MODE_FAMILY_FOREST = 1 << 0        # trees_public_forest_door, trees_public_score_door
comptime MODE_FAMILY_IFOREST = 1 << 1       # trees_public_iforest_door
comptime MODE_FAMILY_GBDT = 1 << 2          # trees_public_gbdt_door, trees_public_gbdt_fit_door
comptime MODE_FAMILY_SCALER = 1 << 3        # scaler_native_door, scaler_lifecycle_native_door, minmax_native_door
comptime MODE_FAMILY_PCA = 1 << 4           # pca_descriptor_boundary_door, pca_metadata_unpack
comptime MODE_FAMILY_IPCA = 1 << 5          # ipca_descriptor_boundary_door
comptime MODE_FAMILY_NMF = 1 << 6           # nmf_descriptor_boundary_door
comptime MODE_FAMILY_TSVD = 1 << 7          # tsvd_native_door
comptime MODE_FAMILY_KPCA = 1 << 8          # kpca_native_door
comptime MODE_FAMILY_PLS = 1 << 9           # pls_native_door
comptime MODE_FAMILY_DECOMP_KIT = 1 << 10   # decomp_kit_native_door, decomp_owner_common_unpack
comptime MODE_FAMILY_LINEAR = 1 << 11       # core/linear_metadata, x_linear, linear_score, classical_labels, qn_boundary
comptime MODE_FAMILY_KERNEL = 1 << 12       # kernel_methods_native_door
comptime MODE_FAMILY_NEIGHBORS = 1 << 13    # neighbor_query_native_door
comptime MODE_FAMILY_SPECTRAL = 1 << 14     # spectral_umap_routes_pack
comptime MODE_FAMILY_TSA = 1 << 15          # tsa_arima, tsa_hw, autoarima_lifecycle
comptime MODE_FAMILY_NEURAL = 1 << 16       # neural_block, sequential, sequence_optimizer, embedding, training_metadata, arena, transformer_decode, autograd frontends
comptime MODE_FAMILY_XCNN = 1 << 17         # xcnn_common_door
comptime MODE_FAMILY_XSEQ = 1 << 18         # xseq_native_common_door, xseq_score_native_door
comptime MODE_FAMILY_RESOLVER = 1 << 19     # estimator_binding_pyapi_door (shared by every family that hands it a raw word)
comptime MODE_FAMILY_METADATA = 1 << 20     # mode_metadata_native_door (shared: numeric_mode_used / vendor_used of every estimator; the neural model catalog answers under NEURAL)
comptime MODE_FAMILY_MSEL = 1 << 21         # msel_scorer_native_door
comptime MODE_FAMILY_LINALG = 1 << 22       # linalg_ops_resample_door
comptime MODE_FAMILY_PROGRAM = 1 << 23      # arena_prog_native_door (the shared x_prep / x_metrics program runner: every classifier's accuracy score passes it)
comptime MODE_FAMILY_MIXTURE = 1 << 24      # mixture_native_door, svm_mixture_native_door (GaussianMixture's compiled-mode check)
comptime MODE_FAMILY_GP = 1 << 25           # gp_common_native_pack (the GP estimators' compiled-mode check)
comptime MODE_FAMILY_PREP = 1 << 26         # prep_classify_common_door (cls_requested_mode: the x_prep classifiers that take numeric_mode)
comptime MODE_FAMILY_CLUSTER = 1 << 27      # cluster_fit_a_native_door `_hdbscan_mode` (HDBSCAN's compiled-mode check)
comptime MODE_FAMILIES_ALL = (1 << 28) - 1

# THE ONE DECISION of which families answer `fast` by the identical set where
# the selector decided so. Every family not named here refuses `fast` in all
# its operations, as an install with no fast set always has.
#
# Evidence (lane fx-fast-per-estimator, 2026-11-19; the built-install trial on
# wheel 7f5eac7eab, CPU runner, runs 44627498380 and 44628153265 under
# ~/mojolearn-evidence/measure-candidate/build-7f5eac7eab/fast-trial/):
# - whole in the EXACT trial 0x83ffe (every public operation, pickle and
#   save/load equal to identical): GBDT, IFOREST, SCALER, PCA, IPCA, NMF,
#   TSVD, KPCA, PLS, LINEAR, KERNEL, with the shared doors they pass,
#   DECOMP_KIT and RESOLVER (without those two KERNEL and PLS refuse at fit).
# - whole in the every-family trial and open in one door that this lane
#   closed, so they are on from that trial plus the edit and the next built
#   wheel's trial must confirm them: NEIGHBORS with PROGRAM (the classifier's
#   accuracy score through the shared program runner), FOREST with METADATA
#   (the archive save compared the raw word with the fitted one; the mode
#   metadata door compares them too), MIXTURE (GaussianMixture's
#   compiled-mode check) and PREP (GaussianNB's new parameter).
# - lane fx-fast-rest, 2026-11-20 (the every-family and exact 0x5983fff
#   trials on the CPU wheel of b59aff2e57, under
#   ~/mojolearn-evidence/measure-candidate/build-b59aff2e57/fast-trial/):
#   every estimator answered except four, each closed in its door:
#   GP (GaussianProcessRegressor answered under the every-family mask;
#   GaussianProcessClassifier refused only because the trial's earlier
#   pickle wrote `__slotnames__` into _BinaryLaplace's snapshotted namespace,
#   python/mojolearn/_gpc_impl.py), SPECTRAL (TSNE answered; UMAP's
#   `_umap_check_mode` compared the raw word with the binary's key and now
#   answers it, bindings/mani_umap_native_door.mojo), CLUSTER (HDBSCAN's
#   compiled-mode check, the MIXTURE form) and MSEL (the estimator-call
#   binding table and the parallel-CV identical-only check now answer the
#   word, as the scorer door did). These four are on from that trial plus the
#   edits; the next built wheel's exact trial with this mask (0xfb87fff)
#   confirms them, and a family that shows one `mixed` row there is taken
#   out here (one line).
# Off: TSA, NEURAL, XCNN, XSEQ, LINALG (doors that still read the raw word;
# lists in the lanes' RESULT files).
# A family is added only from a trial run with no `mixed` row.
comptime MODE_FAMILIES_ANSWERED = (
    MODE_FAMILY_FOREST | MODE_FAMILY_IFOREST | MODE_FAMILY_GBDT | MODE_FAMILY_SCALER | MODE_FAMILY_PCA
    | MODE_FAMILY_IPCA | MODE_FAMILY_NMF | MODE_FAMILY_TSVD | MODE_FAMILY_KPCA | MODE_FAMILY_PLS
    | MODE_FAMILY_DECOMP_KIT | MODE_FAMILY_LINEAR | MODE_FAMILY_KERNEL | MODE_FAMILY_NEIGHBORS
    | MODE_FAMILY_RESOLVER | MODE_FAMILY_METADATA | MODE_FAMILY_PROGRAM | MODE_FAMILY_MIXTURE | MODE_FAMILY_PREP
    | MODE_FAMILY_GP | MODE_FAMILY_SPECTRAL | MODE_FAMILY_MSEL | MODE_FAMILY_CLUSTER
)


struct ModeAnswer(Defaultable, Movable):
    # Bit f set: the identical set answers the name `fast` for family f in
    # this process.
    var families: Int

    def __init__(out self):
        self.families = 0


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever ModeAnswer's layout changes.
comptime MODE_ANSWER = SharedSlot[ModeAnswer, "MojolearnModeAnswerV2", 2]


def connect_mode_answer(families: Int) raises:
    """Store the selector's decision, once, at package import: the families
    whose word `fast` the identical set answers (0: none)."""
    MODE_ANSWER.get_or_create_ptr()[].families = families


@always_inline
def fast_is_answered_by_identical(family: Int) raises -> Bool:
    """Whether the identical set answers the word `fast` for `family` in this
    process."""
    comptime if CompilationTarget.is_macos():
        return False
    return (MODE_ANSWER.get_or_create_ptr()[].families & family) != 0


@always_inline
def answer_mode_word(mut word: String, family: Int) raises:
    """`word` is an admitted mode word of a door of `family`. Replace `fast`
    by `identical` where the identical set answers it for that family."""
    comptime if CompilationTarget.is_macos():
        return
    if word == "fast":
        if (MODE_ANSWER.get_or_create_ptr()[].families & family) != 0:
            word = "identical"


@always_inline
def word_runs_identical(word: String, family: Int) raises -> Bool:
    """Whether an admitted mode word of a door of `family` runs the identical
    set: `identical`, or `fast` where the identical set answers it for that
    family (lane fx-fast-mask). For a door whose check or route is "the word
    is `identical`" and that holds the word as text; an `identical` word pays
    the one comparison it paid before."""
    if word == "identical":
        return True
    comptime if CompilationTarget.is_macos():
        return False
    return word == "fast" and (MODE_ANSWER.get_or_create_ptr()[].families & family) != 0
