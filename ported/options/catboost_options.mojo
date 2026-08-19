"""Training options, under CatBoost's own names.

MIRRORS `catboost/private/libs/options/`, principally
`catboost_options.h`, `boosting_options.h` and `oblivious_tree_options.h`. Their spellings are kept exactly,
including the ones this port does not honor yet, because a name that differs
from CatBoost's is a name somebody has to translate every time they read
their docs against our source.

**Every option carries an `honored` note.** An option that exists and is
ignored is worse than one that is absent: absent fails loudly, ignored fails
silently, and this repository has already spent a day on machinery that was
present and unreachable.
"""


# --- grow_policy -----------------------------------------------------------
comptime GROW_SYMMETRIC = 0
comptime GROW_DEPTHWISE = 1
comptime GROW_LOSSGUIDE = 2


def grow_policy_name(p: Int) -> String:
    if p == GROW_SYMMETRIC:
        return String("SymmetricTree")
    if p == GROW_DEPTHWISE:
        return String("Depthwise")
    return String("Lossguide")


# --- determinism -----------------------------------------------------------
#
# NO CATBOOST COUNTERPART. They ship one GPU backend and accept whatever
# their float atomics do, so the question does not arise for them. It arises
# here because this port targets Metal, CUDA and HIP from one source.
#
# THREE LEVELS, A LADDER RATHER THAN TWO SWITCHES, because across-device
# determinism strictly implies within-device determinism and two independent
# booleans would let a caller ask for the combination that cannot exist.

comptime DETERMINISM_OFF = 0
"""No guarantee. Every row of the kernel matrix is read from the device's own
column and histograms flush through whatever the vendor's fastest path is.
On NVIDIA and AMD that is a float atomic, so the last bits move between two
runs of the SAME fit on the SAME device."""

comptime DETERMINISM_DEVICE = 1
"""Same fit, same device, same model, every run. The flush becomes a
fixed-point integer accumulator, whose addition is associative, so the answer
does not depend on which block lands first. Scheduling rows stay on the
device's column, so this costs only the flush.

**Free on Apple**, because Metal has no float atomic add and the integer
flush is the only path that exists there. NVIDIA and AMD give something up
for it."""

comptime DETERMINISM_CROSS_DEVICE = 2
"""Same fit, ANY supported GPU, same model. Adds the numeric rows that differ
between vendors to what `DEVICE` already pins: the replication factor and the
reduction width move to `COLUMN_BIT_IDENTICAL`, so the reduction tree has one
shape everywhere.

This is the level that costs something real and it is not paid by Apple:
Apple's lane width already equals the pinned 32 and its shared-memory budget
IS the safe column, so the bit-identical column and the apple column
coincide. NVIDIA gives up a 768-thread block for 512, and AMD additionally
gives up its 64-wide wavefront."""


def determinism_name(d: Int) -> String:
    if d == DETERMINISM_OFF:
        return String("off")
    if d == DETERMINISM_DEVICE:
        return String("device")
    return String("cross_device")


@fieldwise_init
struct CatBoostOptions(Copyable, Movable):
    """Their option names, their defaults, and what this port honors."""

    var depth: Int
    """`depth`. CatBoost's default is 6. HONORED: `run_tree`'s `max_depth`."""

    var grow_policy: Int
    """`grow_policy`. Default SymmetricTree. HONORED for SymmetricTree only;
    Depthwise and Lossguide are not ported and `check()` refuses them rather
    than silently growing a symmetric tree under another name."""

    var max_leaves: Int
    """`max_leaves`. Default 31. NOT HONORED: meaningful only under
    Lossguide, which is not ported. Refused when set with SymmetricTree,
    which is what CatBoost does."""

    var min_data_in_leaf: Int
    """`min_data_in_leaf`. Default 1. NOT HONORED YET: the score kernel has
    no minimum-count test. Recorded rather than silently ignored."""

    var l2_leaf_reg: Float32
    """`l2_leaf_reg`. Default 3.0. HONORED: the score kernel's `lambda_l2`
    and the leaf estimator's `l2`. NOTE the port currently passes 1.0 at both
    call sites rather than this default, which is a wiring gap."""

    var border_count: Int
    """`border_count`. Default 254. NOT HONORED: quantization is not ported,
    so the port consumes fold counts a caller has already produced. This is
    the option that would drive them."""

    var leaf_estimation_iterations: Int
    """`leaf_estimation_iterations`. Default 1 for most objectives. HONORED
    only at 1: the leaf estimator is the single Newton step. `check()`
    refuses anything larger."""

    var random_strength: Float32
    """`random_strength`. **CatBoost's default is 1.0 and OURS IS 0.0**, one
    of only two places this port's default differs from theirs.

    Not a preference: no score noise is applied here, so 1.0 would be a
    default that `check()` refuses, and defaults that fail their own
    validation are how a library ships an unusable out-of-the-box
    configuration. 0.0 is the value that describes what actually happens."""

    var rsm: Float32
    """`rsm`, feature sampling rate. Default 1.0. NOT HONORED: every feature
    is scored every level. Refused below 1.0."""

    var determinism: Int
    """NO CATBOOST COUNTERPART. See the three constants above. Default
    `DETERMINISM_DEVICE`, because on Apple it is free and a library that
    returns a different model on a rerun should have to be asked for that."""

    @staticmethod
    def default() -> Self:
        """CatBoost's defaults, with two deliberate departures.

        `random_strength` is 0.0 rather than 1.0 because the feature is not
        ported and 1.0 would fail `check()`. `determinism` has no CatBoost
        counterpart and defaults to `device`, which is free on Apple.

        Everything else is theirs: depth 6, SymmetricTree, max_leaves 31,
        min_data_in_leaf 1, l2_leaf_reg 3.0, border_count 254,
        leaf_estimation_iterations 1, rsm 1.0.
        """
        return Self(
            6, GROW_SYMMETRIC, 31, 1, 3.0, 254, 1, 0.0, 1.0,
            DETERMINISM_DEVICE,
        )

    def check(self) raises:
        """Refuse what is not honored, by name.

        The rule this enforces: an option that is present and ignored is
        worse than one that is absent, because absent fails loudly and
        ignored fails silently.
        """
        if self.grow_policy != GROW_SYMMETRIC:
            raise Error(
                "grow_policy="
                + grow_policy_name(self.grow_policy)
                + " is not ported; only SymmetricTree is implemented, and"
                " growing a symmetric tree under another policy's name would"
                " be a silently wrong model"
            )
        if self.depth < 1 or self.depth > 16:
            raise Error("depth must be in [1, 16]; got " + String(self.depth))
        if self.leaf_estimation_iterations != 1:
            raise Error(
                "leaf_estimation_iterations="
                + String(self.leaf_estimation_iterations)
                + " is not ported; the leaf estimator is a single Newton"
                " step and the descent loop is not written"
            )
        if self.random_strength != 0.0:
            raise Error(
                "random_strength is not ported; no score noise is applied,"
                " so set it to 0.0 rather than believing scores were"
                " randomized"
            )
        if self.rsm != 1.0:
            raise Error(
                "rsm is not ported; every feature is scored at every level,"
                " so set it to 1.0"
            )
        if self.min_data_in_leaf != 1:
            raise Error(
                "min_data_in_leaf is not ported; the score kernel applies no"
                " minimum-count test"
            )
        if self.determinism < DETERMINISM_OFF or (
            self.determinism > DETERMINISM_CROSS_DEVICE
        ):
            raise Error("determinism must be off, device or cross_device")
