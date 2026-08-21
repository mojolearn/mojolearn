"""The Random Forest estimator surface: parameters, metrics, the forest,
and the host inference path.

MIRRORS, at rapidsai/cuml `v26.08.00`
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`), read-only at
`~/CascadeProjects/upstream/cuml-v26.08.00`:

  * `cpp/include/cuml/ensemble/randomforest.hpp` -- `RF_type`,
    `task_category`, `RF_metrics`, `RF_params`, `RandomForestMetaData`,
    and the `fit` / `predict` / `score` / `set_rf_params` C-API
  * `cpp/src/randomforest/randomforest.cu`  -- the metrics setters,
    `validity_check`, `set_rf_params`, `compute_feature_importances`
  * `cpp/src/randomforest/randomforest.cuh` -- `class RandomForest`:
    `error_checking`, `fit`, `predict`, `score`
  * `python/cuml/cuml/ensemble/randomforest_common.pyx` and
    `randomforestclassifier.py` / `randomforestregressor.py` -- the
    defaults a user actually gets, and `compute_max_features`

A note on that last bullet, because the brief that commissioned this
file expected `.pyx`: at THIS PIN the two estimator modules are plain
Python, `randomforestclassifier.py` (232 lines of `__init__` at :209)
and `randomforestregressor.py` (`__init__` at :153). Only the shared
base and the C++ bindings are still `.pyx`
(`randomforest_common.pyx`). Any citation of
`randomforestclassifier.pyx:NNN` in this repository predates the pin and
does not resolve; one such citation is corrected below.

## THE DEFAULT TABLE

The C++ header's defaults and the Python estimator's defaults are not
the same thing, and the Python ones are what ship. Both are recorded.
C++ defaults are the default arguments of `set_tree_params`
(`decisiontree.hpp:82-90`) -- `set_rf_params`
(`randomforest.hpp:231-244`) itself has NO defaults, so its parameters'
"C++ default" is whatever `set_tree_params` supplies for the tree half
and nothing at all for the forest half.

| param | C++ default | Python default | THIS PORT TAKES | citation |
|---|---|---|---|---|
| `n_trees` / `n_estimators` | none (required) | 100 | 100 | `randomforest_common.pyx:314` |
| `max_depth` | **-1** | **None -> INT32_MAX** | **INT32_MAX** | `decisiontree.hpp:82`; `randomforest_common.pyx:317, 480-481` |
| `max_leaves` | -1 | -1 | -1 | `decisiontree.hpp:83`; `randomforest_common.pyx:318` |
| `max_features` | **1.0f** | **`'sqrt'` (clf) / `1.0` (reg)** | **`'sqrt'` clf, `1.0` reg** | `decisiontree.hpp:84`; `randomforestclassifier.py:218`, `randomforestregressor.py:162` |
| `max_n_bins` / `n_bins` | 128 | 128 | 128 | `decisiontree.hpp:85`; `randomforest_common.pyx:320` |
| `min_samples_leaf` | 1 | 1 | 1 | `decisiontree.hpp:86`; `randomforest_common.pyx:321` |
| `min_samples_split` | 2 | 2 | 2 | `decisiontree.hpp:87`; `randomforest_common.pyx:322` |
| `min_impurity_decrease` | 0.0f | 0.0 | 0.0 | `decisiontree.hpp:88` and `:54`; `randomforest_common.pyx:323` |
| `bootstrap` | none (required) | True | True | `randomforest_common.pyx:315` |
| `max_samples` | none (required) | 1.0 | 1.0 | `randomforest_common.pyx:316` |
| `split_criterion` | **CRITERION_END** | **`'gini'` (clf) / `'mse'` (reg)** | **GINI clf, MSE reg** | `decisiontree.hpp:89`; `randomforestclassifier.py:213`, `randomforestregressor.py:157` |
| `max_batch_size` | 4096 | 4096 | 4096 | `decisiontree.hpp:90`; `randomforest_common.pyx:324` |
| `n_streams` | none (required) | 4 | **1** | `randomforest_common.pyx:326`; DEVIATION 117 |
| `random_state` / `seed` | none (required) | **None -> 0** | 0 | `randomforest_common.pyx:325, 517-519` |

**FOUR DISAGREEMENTS, flagged:**

1. `max_depth`. The header says -1 and `validity_check` would REFUSE it
   (`ASSERT(params.max_depth >= 0)`, `decisiontree.cu:19`) -- the C++
   default is a value the C++ validator rejects, so nothing can be
   fitted through `set_tree_params()`'s own default. Python never sends
   -1: `None` becomes `np.iinfo(np.int32).max`
   (`randomforest_common.pyx:480-481`). This port takes INT32_MAX.
2. `max_features`. 1.0f in the header; `'sqrt'` for the CLASSIFIER and
   `1.0` for the REGRESSOR in Python. The two estimators disagree with
   each other, so there is no single Python default -- the classifier
   default is a strict subset of columns and the regressor default is
   all of them. Both are carried below as
   `default_rf_params_classifier` / `default_rf_params_regressor`.
   Their own docstring records that `'sqrt'` replaced `'auto'`
   (`randomforestclassifier.py:88`).
3. `split_criterion`. CRITERION_END in the header, which is a SENTINEL
   resolved later to GINI for an integer label type and MSE otherwise
   (`decisiontree.cuh:251-256`). Python sends the resolved string
   directly. Same outcome by two routes; this port carries
   CRITERION_END in `RF_params` and resolves it at the same place they
   do.
4. `n_streams`. 4 in Python, 1 here. DEVIATION 117.

`random_state=None` mapping to seed 0 (`randomforest_common.pyx:517-519`)
is worth reading twice: unseeded does not mean randomly seeded. Seed 0
is a fixed seed, so an unseeded cuML forest is already deterministic in
its RNG stream; what makes it vary run to run is the reduction order
inside their kernels, not the seed.

## THE PREDICT PATH

`RandomForest::predict` (`randomforest.cuh:382-436`) is host code that
copies the input off the device, walks every tree on the CPU, and
copies the answer back. Their production inference path is elsewhere
(treelite -> nvForest; the in-repo FIL is deprecated,
`randomforest_common.pyx:373-388`) and is NOT ported -- see
DEVIATION 119. What is ported is exactly this traversal, and every
decision in it is cited:

  * Per ROW, a `row_prediction` vector of `num_outputs` is
    ZERO-INITIALIZED (`randomforest.cuh:403`), then every tree ADDS into
    it (`decisiontree.cuh:387` is `+=`). There is no per-tree buffer.
  * The loop bound is `this->rf_params.n_trees`
    (`randomforest.cuh:404`), NOT `forest->trees.size()`.
  * `num_outputs` is read from TREE 0 only
    (`randomforest.cuh:403, 411, 414, 420`).
  * The sum is divided by `n_trees` BEFORE the argmax
    (`randomforest.cuh:414-416`).
  * **CLASSIFIER aggregation**: argmax over the averaged per-class
    vector, initialized `best_class = 0`, `best_prob = 0.0`, updated on
    STRICT `>` (`randomforest.cuh:417-427`). Two consequences that a
    reimplementation gets wrong: a tie between two classes keeps the
    LOWER class index, and if no class has a positive average the answer
    is class 0 by fallthrough, never an error.
  * **REGRESSOR aggregation**: `row_prediction[0]`, i.e. the mean of the
    per-tree leaf values (`randomforest.cuh:429`). Outputs 1..n are not
    read even if `num_outputs > 1`.
  * The per-tree walk itself is `decisiontree.mojo`: `<=` goes LEFT,
    equality goes LEFT, right child is `left + 1`.

## MAX_FEATURES IS A TRUNCATION KNIFE-EDGE

`compute_max_features` (`randomforest_common.pyx:156-174`) returns a
RATIO, in float64, which Cython narrows to a `float`
(`randomforest_common.pyx:516`), which their builder turns back into a
column count with

    max(1, IdxT(params.max_features * n_cols))        builder.cuh:240

-- a C++ `int` cast, i.e. TRUNCATION toward zero, not a round and not a
ceil. So for `'sqrt'` at `n_cols = 16` the chain is
`sqrt(16.0)/16.0 = 0.25` (float64) -> `0.25f` -> `0.25f * 16 = 4.0f` ->
4 columns; and a value one ULP low anywhere in that chain gives
3.9999998 -> **3 columns**. The ratio's last bit is load-bearing.

This is why `compute_max_features` below calls libm's `log2` through
`external_call` instead of `std.math.log2`. This repository has already
measured `std.math.log` at ~5e-8 ABSOLUTE error against libm's 1e-12,
and has already had that error silently re-decide plateau ties in a
ported algorithm. `sqrt` is left to `std.math` because IEEE-754 requires
sqrt to be correctly rounded and libm cannot differ; `log2` has no such
requirement.

**MEASURED, so the choice is not left as an assumption.**
`mojo_only/predict_check.mojo` compares `std.math.log2` against libm's
`log2` at every integer `n_cols` from 2 to 4096 and finds **4051
bit-level disagreements out of 4095, the first at n_cols = 3** -- so the
scar generalizes from `log` to `log2` and the two functions are not the
same function. It ALSO finds that **none of those 4051 disagreements
changes the resulting column count** at any of those sizes, so the libm
call is currently a defence with nothing yet proven to defend against.
Both halves of that are recorded because the second half is the one a
later reader would otherwise assume in the wrong direction.

================= DEVIATION BLOCK (whole file) =================

DEVIATION 117. `n_streams` is not ported and cannot be. THEIRS runs the
per-tree loop under `#pragma omp parallel for num_threads(n_streams)`
with each OpenMP thread owning one CUDA stream from the handle's stream
pool (`randomforest.cuh:336-341`), giving cross-tree parallelism. OURS
has no streams on Metal, so the loop is serial and `n_streams` is
clamped to 1: `set_rf_params` below keeps their clamp
(`randomforest.cu:584`) and `RF_params.check()` refuses any other value
by name rather than accepting it and ignoring it.

THE PRICE, and it is close to zero on OUTPUT for a reason that is in
their source rather than in an argument:

  * Their OWN non-OpenMP build already produces exactly this. They ship
    `#else / #define omp_get_thread_num() 0 / #define
    omp_get_max_threads() 1` (`randomforest.cuh:38-43`), and
    `set_rf_params` computes `min(cfg_n_streams,
    omp_get_max_threads())` (`randomforest.cu:584`). Compile cuML
    without OpenMP and `n_streams` is 1 no matter what the user passed.
    The clamp below is theirs, not ours.
  * **Nothing about the OUTPUT depends on stream count, because their
    RNG is a pure function rather than a stream.** Both draws are keyed
    by hash, not by draw order: the per-tree row sample is
    `rs = fnv1a32(fnv1a32(basis, seed), tree_id)`
    (`randomforest.cuh:120-122`, under their own comment "Hash these
    together so per-tree row samples are uncorrelated",
    `randomforest.cuh:119`), and the per-node feature sample is
    `fnv1a32_hash(seed, treeid, nodeid)`
    (`kernels/builder_kernels.cuh:88`). Tree 7's rows and node 12's
    columns are the same values whether 1 stream or 8 built them.
  * So what `n_streams` buys is WALL-CLOCK ONLY, and this round takes no
    timing measurement of any kind, so no number is claimed here in
    either direction.

**A CORRECTION THAT IS PART OF THIS RESULT.** `ensemble/PLAN.md` and
`decisiontree/batched_levelalgo/random_utils.mojo` both justify this
deviation by citing "cuML's own docs say `n_streams=1` for
reproducibility (`randomforestclassifier.pyx:182`)". **That sentence is
not in cuML at this pin.** The file is `randomforestclassifier.py`, not
`.pyx`; its `n_streams` documentation is two lines at
`randomforestclassifier.py:94-95` and reads, in full, "Number of
parallel streams used for forest building." A search for
`reproduc|deterministic` across `python/cuml/cuml/ensemble/`,
`cpp/src/randomforest/` and `cpp/src/decisiontree/` returns no such
guidance for RF at all. The deviation survives -- the two bullets above
are stronger evidence than a docstring would have been, and the `#else`
branch is decisive -- but the citation is wrong and must be struck
wherever it appears. Neither of those two files belongs to this lane,
so neither was edited; this is the report.

DEVIATION 119. What sits behind this estimator that is NOT PORTED YET,
each refused by name at its own call site rather than quietly missing.

(a) `fit`. LARGELY CLOSED. `RandomForest::fit`
(`randomforest.cuh:286-370`) is ported as the free function `fit_forest`
below, and `detail::RowSampler` (`:62-226`) as `RowSampler`;
`computeQuantiles` and the builder both exist now. A forest trains, and
`ensemble/mojo_only/forest_check.mojo` fits one and predicts it back.

THREE THINGS ARE STILL OUT, each raising by name rather than behaving
like a neighbouring arm:

  * BOOTSTRAP. `RowSampler::sample`'s default arm is
    `raft::random::uniformInt` under `GenPhilox` (`:140-142`), and that
    generator is being ported bit-exactly against a compiled RAFT oracle.
    Until it lands, only `bootstrap=False` fits -- their
    `thrust::sequence` arm (`:155-157`). PRICE: no bagging, so a forest
    of identical trees unless `max_features < 1.0` supplies the only
    remaining source of per-tree variation. That is a real statistical
    difference and it is why the arm raises rather than silently using
    the identity.
  * WEIGHTED BOOTSTRAP (`:125-138`) and ZERO-WEIGHT REMOVAL (`:144-154`).
    Both need `sample_weight`, which this port does not accept, and the
    first also needs a float64 prefix scan this device cannot run.
  * The METHOD `RandomForest.fit` versus the free function; see its
    docstring.

(b) The treelite export surface -- `build_treelite_tree`
(`decisiontree.cuh:154-230`), `build_treelite_forest` and `fit_treelite`
(`randomforest.hpp:126-128, 188-201`), and the `as_treelite` /
`as_nvforest` / `as_fil` Python inference path
(`randomforest_common.pyx:361-424`) -- is NOT PORTED and is not planned.
PRICE: none for correctness and a real one for scope. cuML's PRODUCTION
inference is that path, not the host walk ported here; their host walk
is what the C-API `predict()` uses and what their own tests exercise,
but it is not what a cuML user timing inference is running. Any
inference comparison against cuML must say which of the two it ran.

(c) The device-pointer boundary is gone. THEIRS asserts that `input`
and `predictions` are DEVICE pointers and that the two agree
(`error_checking`, `randomforest.cuh:235-251`, via `DT::is_dev_ptr`,
`decisiontree.cuh:45-55`), copies the whole input to the host
(`raft::update_host`, `randomforest.cuh:396`), predicts, and copies back
(`raft::update_device`, `:433`). OURS takes host `List`s and returns
into a host `List`; the two `is_dev_ptr` asserts have no counterpart and
the two copies do not happen. The `n_rows > 0` / `n_cols > 0` asserts
(`randomforest.cuh:240-241`) ARE kept, and length checks stand in for
the pointer checks. PRICE: an API-shape difference only -- their own
implementation does the whole traversal on the host, so the copies were
never part of the computation. When a device evaluator is decided on,
this is the boundary it moves.

(d) `double`, resolved per site. Every `double` in this file is HOST
side and stays Float64: `RF_metrics.mean_abs_error`,
`.mean_squared_error`, `.median_abs_error` (`randomforest.hpp:36-38`);
the `accumulated_importances` / `finite_importances` /
`infinite_importances` vectors and the `contribution` product in
`compute_feature_importances` (`randomforest.cu:804, 806-807, 823-824`);
and the abs/squared-difference sums in `score`'s regression arm. None of
them reaches a kernel. `RF_metrics.accuracy` is `float`
(`randomforest.hpp:34`) and is Float32, including the division that
produces it (`correctly_predicted * 1.0f / n`,
raft `stats/detail/scores.cuh:110`).

(e) Two summation-ORDER differences inside `score`'s regression arm, and
one allocation bug of theirs that is not reproduced. THEIRS accumulates
`|diff|` and `diff*diff` with `raft::myAtomicAdd` on doubles into shared
memory and then into global (raft `stats/detail/scores.cuh:129-140`), so
the addition order is not fixed and the low bits of their
`mean_abs_error` / `mean_squared_error` are not reproducible run to run
on the same input. OURS sums sequentially in row order, which is
deterministic and is one of the orders theirs could produce. PRICE: our
two means can differ from any given run of theirs in the last bits; the
MEDIAN, which is a selection and not a sum, is bit-identical. Separately,
their `rmm::device_uvector<double> abs_diffs_array(array_size, ...)` uses
`array_size = n * sizeof(double)` (raft `scores.cuh:158-160`) -- an
element count multiplied by a byte size, so they allocate 8x what they
need. That is a harmless over-allocation and is not reproduced.

(f) Three spellings. `std::vector<std::shared_ptr<TreeMetaDataNode>>`
(`randomforest.hpp:104`) becomes a `List` of values -- shared ownership
with no sharing, since nothing else holds a tree. `void print(const
RF_metrics)` (`randomforest.hpp:50`) becomes `print_metrics`, because
`print` is a Mojo builtin and shadowing it in a library module would be
a trap for every later reader. Their four `RandomForestClassifierF/D` /
`RandomForestRegressorF/D` typedefs (`randomforest.hpp:142-143,
248-249`) become one parametric type. No value changes in any of the
three.
=================================================================
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from ensemble.decisiontree.batched_levelalgo.bins import Bin
from ensemble.decisiontree.batched_levelalgo.builder import Builder
from ensemble.decisiontree.batched_levelalgo.objectives import ObjectiveLike
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.quantiles import (
    compute_quantiles,
    Quantiles,
)
from ensemble.decisiontree.batched_levelalgo.random_utils import (
    fnv1a32_hash_seed_tree,
)
from ensemble.mojo_only.philox import RNG_STRIDE, launch_uniform_int
from std.ffi import external_call
from std.math import isfinite, isinf, sqrt

from ensemble.decisiontree.decisiontree import (
    CRITERION_END,
    GINI,
    MSE,
    DecisionTree,
    DecisionTreeParams,
    TreeMetaDataNode,
    criterion_name,
    set_tree_params,
)


# ---------------------------------------------------------------------------
# RF_type, task_category -- `randomforest.hpp:22-27`
# ---------------------------------------------------------------------------
# Unscoped C++ enums; the values are declaration order for `RF_type` and
# EXPLICIT for `task_category` (1 and 2, not 0 and 1).
comptime CLASSIFICATION: Int = 0
comptime REGRESSION: Int = 1

comptime REGRESSION_MODEL: Int = 1
comptime CLASSIFICATION_MODEL: Int = 2

# `randomforest_common.pyx:481` -- `np.iinfo(np.int32).max`, the value
# Python sends for `max_depth=None`.
comptime INT32_MAX: Int32 = 2147483647


# ---------------------------------------------------------------------------
# RF_metrics -- `randomforest.hpp:29-39`, `randomforest.cu:588-641`
# ---------------------------------------------------------------------------


@fieldwise_init
struct RF_metrics(ImplicitlyCopyable, Movable):
    """`ML::RF_metrics`, `randomforest.hpp:29-39`.

    One struct for both tasks, with the unused half filled with -1.0 by
    their own setters -- see `set_rf_metrics_classification`.
    """

    # `randomforest.hpp:30`
    var rf_type: Int
    # `randomforest.hpp:34` -- float, not double
    var accuracy: Float32
    # `randomforest.hpp:36`
    var mean_abs_error: Float64
    # `randomforest.hpp:37`
    var mean_squared_error: Float64
    # `randomforest.hpp:38`
    var median_abs_error: Float64


def set_all_rf_metrics(
    rf_type: Int,
    accuracy: Float32,
    mean_abs_error: Float64,
    mean_squared_error: Float64,
    median_abs_error: Float64,
) -> RF_metrics:
    """`ML::set_all_rf_metrics`, `randomforest.cu:588-601`."""
    return RF_metrics(
        rf_type=rf_type,
        accuracy=accuracy,
        mean_abs_error=mean_abs_error,
        mean_squared_error=mean_squared_error,
        median_abs_error=median_abs_error,
    )


def set_rf_metrics_classification(accuracy: Float32) -> RF_metrics:
    """`ML::set_rf_metrics_classification`, `randomforest.cu:608-611`.

    The three regression fields are set to -1.0, which is a SENTINEL and
    not a measurement -- a caller reading `mean_squared_error` off a
    classification result gets -1.0, not zero and not an error.
    """
    return set_all_rf_metrics(CLASSIFICATION, accuracy, -1.0, -1.0, -1.0)


def set_rf_metrics_regression(
    mean_abs_error: Float64,
    mean_squared_error: Float64,
    median_abs_error: Float64,
) -> RF_metrics:
    """`ML::set_rf_metrics_regression`, `randomforest.cu:620-626`.

    `accuracy` is set to -1.0 for the same reason.
    """
    return set_all_rf_metrics(
        REGRESSION,
        -1.0,
        mean_abs_error,
        mean_squared_error,
        median_abs_error,
    )


def print_metrics(rf_metrics: RF_metrics):
    """`ML::print(const RF_metrics)`, `randomforest.cu:633-641`.

    Renamed; DEVIATION 119f. Theirs logs at DEBUG level through
    `CUML_LOG_DEBUG` and this port has no logger, so it prints.
    """
    if rf_metrics.rf_type == CLASSIFICATION:
        print("Accuracy:", rf_metrics.accuracy)
    elif rf_metrics.rf_type == REGRESSION:
        print("Mean Absolute Error:", rf_metrics.mean_abs_error)
        print("Mean Squared Error:", rf_metrics.mean_squared_error)
        print("Median Absolute Error:", rf_metrics.median_abs_error)


# ---------------------------------------------------------------------------
# RF_params -- `randomforest.hpp:52-86`, `randomforest.cu:790-838`
# ---------------------------------------------------------------------------


@fieldwise_init
struct RF_params(ImplicitlyCopyable, Movable):
    """`ML::RF_params`, `randomforest.hpp:52-86`.

    Their six members in their order. Note that their comment block for
    `tree_params` is misplaced in the header (`randomforest.hpp:72-78`:
    the "Decision tree training hyper parameter struct" comment sits
    above `seed`, and `tree_params` at `:85` is undocumented) -- kept as
    a curiosity, not copied.
    """

    # `randomforest.hpp:56` -- number of decision trees in the forest
    var n_trees: Int32
    # `randomforest.hpp:67` -- if false the whole dataset builds every tree
    var bootstrap: Bool
    # `randomforest.hpp:71` -- ratio of rows used per tree; IGNORED and
    # forcibly reset to 1.0 when bootstrap is false (`randomforest.cuh:304-308`)
    var max_samples: Float32
    # `randomforest.hpp:78` -- random seed, uint64_t
    var seed: UInt64
    # `randomforest.hpp:84` -- see DEVIATION 117
    var n_streams: Int32
    # `randomforest.hpp:85`
    var tree_params: DecisionTreeParams

    def validity_check(self) raises:
        """`ML::validity_check(const RF_params)`, `randomforest.cu:794-801`.

        Their three asserts, in their order, with their messages. Note
        that this does NOT recurse into `tree_params` -- the tree half is
        validated separately by `DT::validity_check` when
        `set_tree_params` runs (`decisiontree.cu:71`).
        """
        if not (self.n_trees > 0):
            raise Error("Invalid n_trees " + String(self.n_trees))
        if not ((self.max_samples > 0) and (self.max_samples <= 1.0)):
            raise Error(
                "max_samples value "
                + String(self.max_samples)
                + " outside permitted (0, 1] range"
            )
        if not (self.n_streams > 0):
            raise Error("Invalid n_streams " + String(self.n_streams))

    def check(self) raises:
        """Refuse what is not honored, by name.

        An option that is present and ignored is worse than one that is
        absent: absent fails loudly, ignored fails silently.

        `n_streams` is the one field a caller can set today and get
        something other than what they asked for. `set_rf_params` clamps
        it the way THEIR non-OpenMP build clamps it
        (`randomforest.cu:584` with `randomforest.cuh:41-42`), but a
        caller who builds `RF_params` field by field goes around that
        clamp, so it is refused here.

        The TRAINING fields are a different question with a different
        answer: `fit` is not ported at all, so `DecisionTreeParams
        .check_fit_supported()` refuses every one of them together
        rather than letting any single one look honored. Calling `check`
        does NOT imply this port can fit.
        """
        if self.n_streams != 1:
            raise Error(
                "n_streams="
                + String(self.n_streams)
                + " is not honored; this port builds trees serially"
                " because Metal has no streams, so anything but 1 would"
                " be silently ignored. cuML's own non-OpenMP build does"
                " exactly the same thing -- randomforest.cuh:41-42"
                " defines omp_get_max_threads() to 1 and"
                " randomforest.cu:584 takes min(cfg_n_streams,"
                " omp_get_max_threads()). Nothing about the OUTPUT"
                " changes: their per-tree and per-node RNG is a pure"
                " hash of (seed, treeid, nodeid), not a stream"
                " (randomforest.cuh:120-122,"
                " kernels/builder_kernels.cuh:88). See DEVIATION 117."
            )
        self.tree_params.check()
        self.validity_check()

    def check_fit_supported(self) raises:
        """Training is NOT PORTED YET; DEVIATION 119a. Names the four
        pieces their `fit` needs and this repository does not have."""
        raise Error(
            "RandomForest.fit is NOT PORTED YET. RandomForest::fit"
            " (randomforest.cuh:286-370) needs four things that do not"
            " exist here: DT::computeQuantiles (randomforest.cuh:318),"
            " detail::RowSampler (randomforest.cuh:63-226), the four"
            " Builder<ObjectiveT> instantiations"
            " (decisiontree.cuh:259-332), and a CUDA stream pool"
            " (randomforest.cuh:336-339). The first three are"
            " batched_levelalgo/quantiles.mojo, an unwritten row"
            " sampler and batched_levelalgo/builder.mojo; the fourth"
            " does not exist on Metal (DEVIATION 117). RF_params"
            " n_trees, bootstrap, max_samples and seed are therefore"
            " also unhonored, along with every DecisionTreeParams"
            " field."
        )


def set_rf_params(
    max_depth: Int32,
    max_leaves: Int32,
    max_features: Float32,
    max_n_bins: Int32,
    min_samples_leaf: Int32,
    min_samples_split: Int32,
    min_impurity_decrease: Float32,
    bootstrap: Bool,
    n_trees: Int32,
    max_samples: Float32,
    seed: UInt64,
    split_criterion: Int,
    cfg_n_streams: Int32,
    max_batch_size: Int32,
) raises -> RF_params:
    """`ML::set_rf_params`, declared `randomforest.hpp:231-244`, defined
    `randomforest.cu:803-838`.

    THIS FUNCTION HAS NO DEFAULT ARGUMENTS in their header. Every value
    is supplied by the caller, and the caller that matters is
    `randomforest_common.pyx:539-554`. That is why the default table at
    the top of this file exists.

    Their body, line for line (`randomforest.cu:818-837`):
      * build `tree_params` through `DT::set_tree_params`, which
        validates the tree half
      * copy n_trees / bootstrap / max_samples / seed
      * `n_streams = min(cfg_n_streams, omp_get_max_threads())` -- and
        `omp_get_max_threads()` is 1 in a build without OpenMP, which
        this one is (`randomforest.cuh:41-42`). DEVIATION 117.
      * clamp `n_streams` down to `n_trees` if there are fewer trees
        than streams (`randomforest.cu:585`)
      * `validity_check`
    """
    var tree_params = DecisionTreeParams(
        max_depth=max_depth,
        max_leaves=max_leaves,
        max_features=max_features,
        max_n_bins=max_n_bins,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        split_criterion=split_criterion,
        min_impurity_decrease=min_impurity_decrease,
        max_batch_size=max_batch_size,
    )
    set_tree_params(
        tree_params,
        cfg_max_depth=max_depth,
        cfg_max_leaves=max_leaves,
        cfg_max_features=max_features,
        cfg_max_n_bins=max_n_bins,
        cfg_min_samples_leaf=min_samples_leaf,
        cfg_min_samples_split=min_samples_split,
        cfg_min_impurity_decrease=min_impurity_decrease,
        cfg_split_criterion=split_criterion,
        cfg_max_batch_size=max_batch_size,
    )
    # `randomforest.cu:584` -- min(cfg_n_streams, omp_get_max_threads()),
    # and `randomforest.cuh:42` makes omp_get_max_threads() 1 without
    # OpenMP. Their clamp, their `#else` branch, our only arm.
    comptime OMP_GET_MAX_THREADS: Int32 = 1
    var n_streams = cfg_n_streams
    if OMP_GET_MAX_THREADS < n_streams:
        n_streams = OMP_GET_MAX_THREADS
    # `randomforest.cu:585`
    if n_trees < n_streams:
        n_streams = n_trees
    var rf_params = RF_params(
        n_trees=n_trees,
        bootstrap=bootstrap,
        max_samples=max_samples,
        seed=seed,
        n_streams=n_streams,
        tree_params=tree_params,
    )
    rf_params.validity_check()
    return rf_params


# ---------------------------------------------------------------------------
# max_features -- `randomforest_common.pyx:156-174`
# ---------------------------------------------------------------------------
# Their Python accepts an int, a float, None, 'sqrt' or 'log2'. Mojo has
# no such union, so the five arms are five entry points with their
# names. Every one returns a RATIO in float64, which their Cython then
# narrows to float32 (`randomforest_common.pyx:516`) -- so callers here
# narrow at the same place, not earlier.

comptime MAX_FEATURES_SQRT: Int = 0
comptime MAX_FEATURES_LOG2: Int = 1
comptime MAX_FEATURES_NONE: Int = 2


def compute_max_features_sqrt(n_cols: Int) -> Float64:
    """`randomforest_common.pyx:164-165` -- `math.sqrt(n_cols) / n_cols`.

    `sqrt` is left to `std.math`: IEEE-754 requires it correctly
    rounded, so libm cannot disagree.
    """
    return sqrt(Float64(n_cols)) / Float64(n_cols)


def compute_max_features_log2(n_cols: Int) -> Float64:
    """`randomforest_common.pyx:166-167` -- `math.log2(n_cols) / n_cols`.

    Through libm, NOT `std.math.log2`. See the knife-edge note in the
    module docstring: this ratio is multiplied back by `n_cols` and
    TRUNCATED to an int column count (`builder.cuh:240`), and this
    repository has measured `std.math.log` at ~5e-8 absolute error
    against libm -- enough to turn 4.0 into 3.9999998 and take a column
    away. `math.log2` in CPython is libm's `log2`, so libm's is the
    oracle.

    Measured this round: the two disagree at the bit level on 4051 of
    the 4095 integer inputs in [2, 4096], first at 3, and on none of
    them does the disagreement reach the column count. See the module
    docstring.
    """
    return (
        external_call["log2", Float64](Float64(n_cols)) / Float64(n_cols)
    )


def compute_max_features_int(max_features: Int, n_cols: Int) -> Float64:
    """`randomforest_common.pyx:160-161` -- `max_features / n_cols`.

    Python's `/` is true division even on two ints, so this is float64.
    """
    return Float64(max_features) / Float64(n_cols)


def compute_max_features_float(max_features: Float64) -> Float64:
    """`randomforest_common.pyx:162-163` -- returned unchanged."""
    return max_features


def compute_max_features_none() -> Float64:
    """`randomforest_common.pyx:168-169` -- 1.0."""
    return 1.0


def compute_max_features(kind: Int, n_cols: Int) raises -> Float64:
    """The three STRING/None arms behind one dispatch, so a caller that
    holds `'sqrt'` as a value can pass it through."""
    if kind == MAX_FEATURES_SQRT:
        return compute_max_features_sqrt(n_cols)
    if kind == MAX_FEATURES_LOG2:
        return compute_max_features_log2(n_cols)
    if kind == MAX_FEATURES_NONE:
        return compute_max_features_none()
    raise Error(
        "Expected `max_features` to be an int, float, None, or one of"
        " ['sqrt', 'log2']. Got "
        + String(kind)
        + " instead."
    )


def n_sampled_cols(max_features: Float32, n_cols: Int) -> Int32:
    """`builder.cuh:240` -- `max(1, IdxT(params.max_features * n_cols))`.

    Not in the estimator surface; transcribed HERE because it is the
    consumer that makes `max_features`'s last bit matter, and a reader
    of the table above needs to see the truncation to understand it. The
    multiply is float32 (`float * int` promotes the int) and the cast
    truncates toward zero.
    """
    var k = Int32(max_features * Float32(n_cols))
    if k < 1:
        return 1
    return k


# ---------------------------------------------------------------------------
# Shipped defaults -- what a Python user actually gets
# ---------------------------------------------------------------------------


def default_rf_params_classifier(n_cols: Int) raises -> RF_params:
    """`RandomForestClassifier.__init__`, `randomforestclassifier.py:209-230`,
    marshalled through `randomforest_common.pyx:477-554`.

    `max_features='sqrt'` needs `n_cols`, which is why this takes one --
    their Python needs it too (`randomforest_common.pyx:516`).
    `split_criterion='gini'` maps to GINI
    (`randomforest_common.pyx:105-106`). `n_streams` is 4 in their
    default and is passed as 4 here so their clamp is the thing that
    reduces it, not a value this port quietly substituted.
    """
    return set_rf_params(
        max_depth=INT32_MAX,
        max_leaves=-1,
        max_features=Float32(
            compute_max_features(MAX_FEATURES_SQRT, n_cols)
        ),
        max_n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        bootstrap=True,
        n_trees=100,
        max_samples=1.0,
        seed=0,
        split_criterion=GINI,
        cfg_n_streams=4,
        max_batch_size=4096,
    )


def default_rf_params_regressor() raises -> RF_params:
    """`RandomForestRegressor.__init__`, `randomforestregressor.py:153-172`.

    Differs from the classifier in exactly two places:
    `max_features=1.0` (`randomforestregressor.py:162`) and
    `split_criterion='mse'` -> MSE (`randomforestregressor.py:157`,
    `randomforest_common.pyx:109-110`). No `n_cols` is needed because
    1.0 is not a function of the column count.
    """
    return set_rf_params(
        max_depth=INT32_MAX,
        max_leaves=-1,
        max_features=Float32(compute_max_features_float(1.0)),
        max_n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        bootstrap=True,
        n_trees=100,
        max_samples=1.0,
        seed=0,
        split_criterion=MSE,
        cfg_n_streams=4,
        max_batch_size=4096,
    )


# ---------------------------------------------------------------------------
# RandomForestMetaData -- `randomforest.hpp:102-111`
# ---------------------------------------------------------------------------


@fieldwise_init
struct RandomForestMetaData[dtype: DType, label_dtype: DType](
    Copyable, Movable
):
    """`ML::RandomForestMetaData<T, L>`, `randomforest.hpp:102-111`.

    `L` is REAL here, unlike on `TreeMetaDataNode` -- it is the type
    `predict` writes (`randomforest.cuh:386`) and the type `score`
    compares (`randomforest.cuh:453`). Their four typedefs at
    `randomforest.hpp:142-143, 248-249` are (float,int), (double,int),
    (float,float), (double,double).

    `trees` is a `vector<shared_ptr<...>>` in their header; DEVIATION
    119f.
    """

    # `randomforest.hpp:104`
    var trees: List[TreeMetaDataNode[Self.dtype]]
    # `randomforest.hpp:105`
    var rf_params: RF_params
    # `randomforest.hpp:110` -- number of features in the training data,
    # `= 0` in their header and set by `fit` (`randomforest.cuh:334`)
    var n_features: Int32


# ---------------------------------------------------------------------------
# class RandomForest -- `randomforest.cuh:229-483`
# ---------------------------------------------------------------------------


@fieldwise_init
struct RandomForest[dtype: DType, label_dtype: DType](
    ImplicitlyCopyable, Movable
):
    """`ML::RandomForest<T, L>`, `randomforest.cuh:229-483`.

    Their two protected members and their constructor
    (`randomforest.cuh:232-233, 259-260`), whose `cfg_rf_type` defaults
    to `RF_type::CLASSIFICATION`.
    """

    # `randomforest.cuh:232`
    var rf_params: RF_params
    # `randomforest.cuh:233` -- 0 classification, 1 regression
    var rf_type: Int

    def error_checking(self, n_rows: Int, n_cols: Int) raises:
        """`RandomForest::error_checking`, `randomforest.cuh:235-251`.

        The two `is_dev_ptr` asserts (`:243-250`) and the
        `predictions != nullptr` assert (`:238`) have no counterpart;
        DEVIATION 119c. The two that remain are theirs verbatim.
        """
        if not (n_rows > 0):
            raise Error("Invalid n_rows " + String(n_rows))
        if not (n_cols > 0):
            raise Error("Invalid n_cols " + String(n_cols))

    def fit(self) raises:
        """`RandomForest::fit`, `randomforest.cuh:286-370`.

        THE FOREST LOOP IS PORTED, as the free function `fit_forest` below;
        this METHOD still raises, and the difference is not laziness.
        `RandomForest` is parameterized on `[dtype, label_dtype]`, matching
        their `RandomForest<T, L>`, but a fit also needs the BIN type, and
        the launchers underneath are overloaded on the concrete objective
        because Mojo traits are nominal (their DEVIATION 129a). So the
        entry point that works today is `fit_forest[label_dtype, BinT]`,
        which names the bin explicitly.

        Collapsing the two -- declaring the objective structs conformant to
        a trait the launchers can dispatch on -- deletes two adapters, six
        launcher overloads, `Builder`'s classification-only restriction AND
        this method's raise together. It is the highest-value cleanup left
        in this directory and is recorded in `ensemble/PLAN.md`.
        """
        self.rf_params.check_fit_supported()
        raise Error(
            "RandomForest.fit is not the ported entry point; call"
            " fit_forest[label_dtype, BinT](ctx, x, y, sample_weight,"
            " n_rows, n_cols, n_unique_labels, rf_params) instead. This"
            " method needs the bin type, which RandomForest[T, L] does not"
            " carry, and cannot dispatch generically until objectives.mojo"
            " declares an objective trait."
        )

    def predict(
        self,
        input: List[Scalar[Self.dtype]],
        n_rows: Int,
        n_cols: Int,
        mut predictions: List[Scalar[Self.label_dtype]],
        forest: RandomForestMetaData[Self.dtype, Self.label_dtype],
    ) raises:
        """`RandomForest::predict`, `randomforest.cuh:382-436`.

        Ported statement for statement. `input` is ROW-MAJOR
        (`randomforest.cuh:375`, and the indexing at `:407` proves it).
        """
        self.error_checking(n_rows, n_cols)
        # `randomforest.cu:482` -- the C-API wrapper's assert, hoisted
        # here because this port has no separate wrapper layer.
        if len(forest.trees) == 0:
            raise Error("Cannot predict! No trees in the forest.")
        if len(predictions) < n_rows:
            raise Error(
                "predictions holds "
                + String(len(predictions))
                + " values but n_rows is "
                + String(n_rows)
            )
        if len(input) < n_rows * n_cols:
            raise Error(
                "input holds "
                + String(len(input))
                + " values but n_rows * n_cols is "
                + String(n_rows * n_cols)
            )

        # `randomforest.cuh:399` -- row_size is n_cols, so the row stride
        # is the column count: ROW-MAJOR.
        var row_size = n_cols
        # `randomforest.cuh:403` -- num_outputs comes from TREE 0 and
        # from nowhere else.
        var num_outputs = Int(forest.trees[0].num_outputs)
        if num_outputs < 1:
            raise Error(
                "forest.trees[0].num_outputs is "
                + String(num_outputs)
                + "; randomforest.cuh:403 sizes every row's prediction"
                " buffer from it, so it must be >= 1"
            )
        # `randomforest.cuh:404` -- the loop bound is rf_params.n_trees,
        # NOT trees.size(). If a caller hands in a forest with fewer
        # trees than n_trees claims, theirs reads out of bounds; ours
        # refuses.
        var n_trees = Int(self.rf_params.n_trees)
        if len(forest.trees) < n_trees:
            raise Error(
                "rf_params.n_trees is "
                + String(n_trees)
                + " but the forest holds "
                + String(len(forest.trees))
                + " trees; randomforest.cuh:404 loops to n_trees"
            )

        for row_id in range(n_rows):
            # `randomforest.cuh:403` -- ZERO-INITIALIZED once per ROW,
            # and every tree ADDS into it (`decisiontree.cuh:387`).
            var row_prediction = List[Scalar[Self.dtype]]()
            for _ in range(num_outputs):
                row_prediction.append(0)
            for i in range(n_trees):
                # `randomforest.cuh:405-412` -- one row at a time, with
                # the pointer offset their `&h_input[row_id * row_size]`
                # produces.
                DecisionTree.predict(
                    forest.trees[i],
                    input,
                    1,
                    n_cols,
                    row_prediction,
                    num_outputs,
                    rows_offset=row_id * row_size,
                    preds_offset=0,
                )
            # `randomforest.cuh:414-416` -- divide by n_trees BEFORE the
            # argmax.
            for k in range(num_outputs):
                row_prediction[k] = row_prediction[k] / Scalar[Self.dtype](
                    n_trees
                )
            if self.rf_type == CLASSIFICATION:
                # `randomforest.cuh:417-427`. best_class starts at 0 and
                # best_prob at 0.0, and the update is STRICT `>`: a tie
                # keeps the LOWER class index, and an all-non-positive
                # row answers class 0 rather than failing.
                var best_class: Int = 0
                var best_prob = Scalar[Self.dtype](0.0)
                for k in range(num_outputs):
                    if row_prediction[k] > best_prob:
                        best_class = k
                        best_prob = row_prediction[k]
                predictions[row_id] = Scalar[Self.label_dtype](best_class)
            else:
                # `randomforest.cuh:429` -- output 0 only.
                predictions[row_id] = row_prediction[0].cast[
                    Self.label_dtype
                ]()

    @staticmethod
    def score(
        ref_labels: List[Scalar[Self.label_dtype]],
        n_rows: Int,
        predictions: List[Scalar[Self.label_dtype]],
        rf_type: Int = CLASSIFICATION,
    ) raises -> RF_metrics:
        """`RandomForest::score`, `randomforest.cuh:450-482`.

        Their body dispatches on `rf_type` to one of two raft
        primitives. Those live in a DIFFERENT checkout at a DIFFERENT
        pin -- rapidsai/raft `661a3b840c3300f95f053812a560c952c9d049a4`,
        `~/CascadeProjects/upstream/raft` -- and are cited as such:

          * classification: `raft::stats::accuracy`
            (`randomforest.cuh:461`), which is
            `stats/detail/scores.cuh:96-112`: count the elements where
            `predictions - ref == 0` and return
            `correctly_predicted * 1.0f / n` -- a FLOAT32 division of an
            unsigned long long by an int.
          * regression: `raft::stats::regression_metrics`
            (`randomforest.cuh:470-476`), which is
            `stats/detail/scores.cuh:158-217`: mean of `|diff|`, mean of
            `diff*diff`, and the median of the SORTED `|diff|` array,
            averaging the two middle elements when n is even
            (`scores.cuh:211-216`).

        Summation-order and allocation differences: DEVIATION 119e.
        """
        if not (n_rows > 0):
            raise Error("Invalid n_rows " + String(n_rows))
        if len(ref_labels) < n_rows or len(predictions) < n_rows:
            raise Error(
                "ref_labels/predictions must hold at least n_rows = "
                + String(n_rows)
                + " values"
            )
        if rf_type == CLASSIFICATION:
            # raft `scores.cuh:101-111`
            var correctly_predicted: Int = 0
            for i in range(n_rows):
                if predictions[i] - ref_labels[i] == 0:
                    correctly_predicted += 1
            var accuracy = (
                Float32(correctly_predicted) * Float32(1.0) / Float32(n_rows)
            )
            return set_rf_metrics_classification(accuracy)

        # raft `scores.cuh:127-141` (the sums) and `:203-216` (the median)
        var abs_diffs = List[Float64]()
        var abs_sum: Float64 = 0.0
        var sq_sum: Float64 = 0.0
        for i in range(n_rows):
            var diff = Float64(predictions[i]) - Float64(ref_labels[i])
            var abs_diff = diff
            if abs_diff < 0.0:
                abs_diff = -abs_diff
            abs_sum += abs_diff
            sq_sum += diff * diff
            abs_diffs.append(abs_diff)
        var mean_abs_error = abs_sum / Float64(n_rows)
        var mean_squared_error = sq_sum / Float64(n_rows)
        sort(abs_diffs)
        var middle = n_rows // 2
        var median_abs_error: Float64
        if n_rows % 2 == 1:
            median_abs_error = abs_diffs[middle]
        else:
            median_abs_error = (
                abs_diffs[middle] + abs_diffs[middle - 1]
            ) / 2.0
        return set_rf_metrics_regression(
            mean_abs_error, mean_squared_error, median_abs_error
        )


# ---------------------------------------------------------------------------
# compute_feature_importances -- `randomforest.cu:797-860`
# ---------------------------------------------------------------------------


def compute_feature_importances[
    dtype: DType, label_dtype: DType
](
    forest: RandomForestMetaData[dtype, label_dtype],
    mut importances: List[Scalar[dtype]],
) raises:
    """`ML::compute_feature_importances`, `randomforest.cu:799-860`.

    Pure host code over `sparsetree`, so it ports whole. Their structure,
    which is not the obvious one:

      * PER TREE, two parallel accumulators are built: `finite` (sum of
        `BestMetric() * InstanceCount()` over split nodes, and ONLY where
        that product is positive, `:832`) and `infinite` (a COUNT of
        split nodes whose product is +inf, `:834`).
      * If ANY node in that tree produced +inf, the infinite vector is
        used and the finite one is discarded entirely (`:838-839`). Their
        comment at `:826-829` explains why: normalizing inf/inf gives
        NaN, and the useful signal is then which features got +inf.
      * Each tree's chosen vector is normalized to sum 1 before being
        added to the forest accumulator (`:846-850`), so every tree
        contributes equally regardless of size.
      * Trees with an empty sparsetree or a non-positive root instance
        count are SKIPPED (`:815-820`) -- and skipped before the
        per-tree vectors are used, so they contribute nothing rather
        than zeros.
      * A forest whose total is not positive returns all zeros, not
        NaN (`:858-859`).

    DEVIATION 119d: the accumulators are `double` in their code and
    Float64 here; only the final store narrows to `T`.
    """
    # `randomforest.cu:801`
    if forest.n_features == 0:
        return
    var n_cols = Int(forest.n_features)
    if len(importances) < n_cols:
        raise Error(
            "importances holds "
            + String(len(importances))
            + " values but n_features is "
            + String(n_cols)
        )
    # `randomforest.cu:804`
    var accumulated_importances = List[Float64]()
    for _ in range(n_cols):
        accumulated_importances.append(0.0)

    for t in range(len(forest.trees)):
        # `randomforest.cu:806-808`
        var finite_importances = List[Float64]()
        var infinite_importances = List[Float64]()
        for _ in range(n_cols):
            finite_importances.append(0.0)
            infinite_importances.append(0.0)
        var has_infinite_importance = False

        # `randomforest.cu:810`
        if len(forest.trees[t].sparsetree) == 0:
            continue
        # `randomforest.cu:811`
        var root_sample_count = Int(
            forest.trees[t].sparsetree[0].InstanceCount()
        )
        # `randomforest.cu:813`
        if root_sample_count <= 0:
            continue

        # `randomforest.cu:820-837`
        for j in range(len(forest.trees[t].sparsetree)):
            var node = forest.trees[t].sparsetree[j]
            if not node.IsLeaf():
                var feature_id = Int(node.ColumnId())
                var contribution = Float64(node.BestMetric()) * Float64(
                    Int(node.InstanceCount())
                )
                if isfinite(contribution):
                    if contribution > 0.0:
                        finite_importances[feature_id] += contribution
                elif isinf(contribution) and contribution > 0.0:
                    infinite_importances[feature_id] += 1.0
                    has_infinite_importance = True

        # `randomforest.cu:838-839`
        var tree_sum: Float64 = 0.0
        for i in range(n_cols):
            if has_infinite_importance:
                tree_sum += infinite_importances[i]
            else:
                tree_sum += finite_importances[i]
        # `randomforest.cu:846-850`
        if tree_sum > 0:
            for i in range(n_cols):
                if has_infinite_importance:
                    accumulated_importances[i] += (
                        infinite_importances[i] / tree_sum
                    )
                else:
                    accumulated_importances[i] += (
                        finite_importances[i] / tree_sum
                    )

    # `randomforest.cu:853-866`
    var total: Float64 = 0.0
    for i in range(n_cols):
        total += accumulated_importances[i]
    if total > 0:
        for i in range(n_cols):
            importances[i] = Scalar[dtype](
                accumulated_importances[i] / total
            )
    else:
        for i in range(n_cols):
            importances[i] = 0


# ===========================================================================
# `detail::RowSampler`, `randomforest.cuh:62-226`, and the forest loop,
# `RandomForest::fit`, `randomforest.cuh:286-370`.
# ===========================================================================


struct RowSampler(Movable):
    """`ML::DT::detail::RowSampler`, `randomforest.cuh:62-226`.

    THEIR FOUR ARMS, and which one a default fit takes. `sample()`
    (`:110-165`) is a four-way branch and the ORDER of the tests is the
    dispatch:

      1. `use_weighted_bootstrap()` -- `bootstrap && sample_weight != nullptr`
         (`:214`). Draws `uniform<double>` into `[0, weight_sum)` and
         `thrust::upper_bound`s it against a prefix-summed weight CDF
         (`:125-138`).
      2. `bootstrap_` -- `raft::random::uniformInt<int>(..., 0, n_rows_)`
         (`:140-143`). **THIS IS THE DEFAULT**, because `bootstrap` defaults
         True and `sample_weight` defaults null.
      3. `sample_weight_ != nullptr` without bootstrap -- `thrust::copy_if`
         drops the zero-weight rows (`:144-154`).
      4. otherwise -- `thrust::sequence`, the identity (`:155-157`).

    Arms 1 and 3 need `sample_weight`, which this port does not accept yet
    (DEVIATION 100 carries the field as Float32 and nothing fills it), so
    they raise by name rather than silently behaving like arm 2 or 4.

    THE SEED CHAIN IS PER TREE AND IS NOT A STREAM (`:119-123`):

        rs = fnv1a32_basis
        rs = fnv1a32(rs, seed_)      // ONE round on the low 32 bits
        rs = fnv1a32(rs, tree_id)
        RngState(rs, GenPhilox)

    Their comment at `:119` says why: "Hash these together so per-tree row
    samples are uncorrelated." Tree 7's rows are a pure function of
    `(seed, 7)`, so they do not depend on how many streams built the forest,
    on the order the trees were built, or on anything else. That is what
    makes DEVIATION 117 free.

    Note `fnv1a32_hash_seed_tree` folds the uint64 seed in ONE round on its
    low 32 bits and DISCARDS the high half -- because this call site uses
    `fnv1a32` directly rather than `fnv1a32_combine`. That asymmetry with
    the per-node chain (which folds both halves) is theirs and is
    transcribed, not corrected.
    """

    var bootstrap: Bool
    var seed: UInt64
    var n_rows: Int
    var n_sampled_rows: Int
    var has_sample_weight: Bool
    # `:224` -- one `device_uvector<int>` per stream. One stream here.
    var selected_rows: DeviceBuffer[DType.int32]
    var h_rows: HostBuffer[DType.int32]

    def __init__(
        out self,
        ctx: DeviceContext,
        bootstrap: Bool,
        seed: UInt64,
        n_rows: Int,
        n_sampled_rows: Int,
        has_sample_weight: Bool = False,
    ) raises:
        """`:63-105`, their constructor, minus the weighted machinery."""
        self.bootstrap = bootstrap
        self.seed = seed
        self.n_rows = n_rows
        self.n_sampled_rows = n_sampled_rows
        self.has_sample_weight = has_sample_weight
        var n = n_sampled_rows if n_sampled_rows > 0 else 1
        self.selected_rows = ctx.enqueue_create_buffer[DType.int32](n)
        self.h_rows = ctx.enqueue_create_host_buffer[DType.int32](n)
        ctx.synchronize()

    def rng_seed_for(self, tree_id: Int32) -> UInt32:
        """`:120-123`, the per-tree seed, exposed so a check can hold it to
        the same value the sampler uses."""
        return fnv1a32_hash_seed_tree(self.seed, tree_id)

    def sample(mut self, ctx: DeviceContext, tree_id: Int32) raises:
        """`RowSampler::sample`, `:110-165`. Fills `selected_rows`.

        Their four-way dispatch, in their order. Arms 1 and 3 raise; see
        the struct docstring and DEVIATION 181.
        """
        if self.bootstrap and self.has_sample_weight:
            raise Error(
                "weighted bootstrap is NOT PORTED YET"
                " (randomforest.cuh:125-138: raft::random::uniform<double>"
                " into a thrust::upper_bound over a weight CDF). It needs"
                " sample_weight, which this port does not accept, and a"
                " float64 prefix scan, which this device cannot run."
            )
        if self.bootstrap:
            # `:140-142` -- THE DEFAULT ARM.
            #
            #   raft::random::uniformInt<int>(
            #       stream_resources, rng_state,
            #       selected_rows.data(), selected_rows.size(), 0, n_rows_);
            #
            # `rng_state` is a FRESH `RngState(rs, GenPhilox)` local, built
            # at `:123` and dead at the end of this call, so its
            # `base_subsequence` is 0 on every launch and the advance
            # `call_rng_kernel` performs writes into an object nobody reads
            # again. Hence no state is threaded here -- verified in RAFT
            # rather than assumed.
            launch_uniform_int(
                ctx,
                self.selected_rows,
                self.n_sampled_rows,
                Int32(0),
                Int32(self.n_rows),
                UInt64(Int(self.rng_seed_for(tree_id))),
            )
            ctx.synchronize()
            return
        if self.has_sample_weight:
            raise Error(
                "zero-weight row removal is NOT PORTED YET"
                " (randomforest.cuh:144-154: thrust::copy_if over"
                " NonzeroSampleWeight). It needs sample_weight, which this"
                " port does not accept."
            )
        # `:155-157` -- `thrust::sequence`, the identity. This is the arm
        # `bootstrap=False` takes, and it is the only one that runs today.
        var p = self.h_rows.unsafe_ptr()
        for i in range(self.n_sampled_rows):
            p.unsafe_store(i, Int32(i))
        ctx.enqueue_copy(
            dst_buf=self.selected_rows, src_ptr=self.h_rows.unsafe_ptr()
        )
        ctx.synchronize()

    @always_inline
    def rows_ptr(mut self) -> MutPointer[Int32, MutUntrackedOrigin]:
        return (
            self.selected_rows.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )


def n_sampled_rows_for(
    bootstrap: Bool, max_samples: Float32, n_rows: Int
) -> Int:
    """`randomforest.cuh:299-309`.

        if (bootstrap) n_sampled_rows = std::round(max_samples * n_rows);
        else           n_sampled_rows = n_rows;

    THEIR ROUND IS `std::round`, i.e. round-half-AWAY-from-zero, and it is
    written out here rather than delegated to a Mojo stdlib rounding mode:
    this repository has been bitten once by assuming a Mojo numeric matched
    libm. `max_samples` is a float32 and non-negative at this call site
    (`RF_params::validity_check` refuses otherwise), so `floor(x) + (frac
    >= 0.5)` IS round-half-away here.

    Their `else` branch also WARNS and overwrites `max_samples` to 1.0 when
    it was not 1.0 (`:301-306`) -- the caller does that, because it mutates
    the params.
    """
    if not bootstrap:
        return n_rows
    var x = Float64(max_samples) * Float64(n_rows)
    var f = Float64(Int(x))
    if x - f >= 0.5:
        return Int(f) + 1
    return Int(f)


def fit_forest[
    O: ObjectiveLike
](
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[O.LabelT],
    mut sample_weight: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_unique_labels: Int,
    mut rf_params: RF_params,
    objective: O,
    row_major: Bool = False,
) raises -> RandomForestMetaData[O.DataT, O.LabelT] where (
    O.DataT == DType.float32
):
    """`RandomForest::fit`, `randomforest.cuh:286-370`. THE FOREST LOOP.

    GENERIC OVER THE OBJECTIVE, as `Builder<ObjectiveT>` is, so a
    REGRESSION forest and a classification forest are the same code with a
    different `O`. That was not true until `objectives.mojo` declared
    `ObjectiveLike`.

    FLOAT32 FEATURES ONLY, and that is inherited rather than chosen here.
    cuML instantiates for `float` and `double` (their `-double.cu` TUs);
    this port has no float64 on device at all, so `computeQuantiles` is
    float32-only upstream of this function and the `double` instantiations
    are declined where the bins are (DEVIATION 114). The LABEL type is
    `O.LabelT` and is free -- Int32 for classification, Float32 for
    regression.

    Their body, in their order, with the two things that are easy to get
    wrong called out:

    1. **THE QUANTILES ARE COMPUTED ONCE FOR THE WHOLE FOREST** (`:317-325`),
       before the tree loop and outside it, from the FULL dataset and the
       FOREST's seed -- not per tree and not from the tree's bootstrap
       sample. Every tree in the forest bins against the same edges. A port
       that recomputed them per tree would produce a defensible-looking
       forest that is not theirs, and nothing downstream would say so.
       Note also the hard-coded `4` at `:322`: `oversampling_factor` is a
       literal at this call site, never the signature default.

    2. **`max_samples` IS IGNORED AND OVERWRITTEN when bootstrap is off**
       (`:301-306`), with a warning. Theirs mutates `rf_params`, so a caller
       that reads it back afterwards sees 1.0; ours does the same, which is
       why `rf_params` is `mut`.

    The `#pragma omp parallel for num_threads(n_streams)` at `:337` is
    DEVIATION 117: one stream here, so the loop is serial, and no output bit
    depends on it because every tree's rows and every node's columns are
    pure hashes of `(seed, tree_id)` and `(seed, tree_id, node_id)`.

    Their `DT::DecisionTree::fit` call (`:353-366`) passes the FOREST seed
    and the tree INDEX separately; the per-tree hashing happens below, in
    `RowSampler` and in the builder's feature sampler. Passing an
    already-hashed seed down would double-hash and produce a different
    forest.
    """
    if n_rows <= 0:
        raise Error("Invalid n_rows " + String(n_rows))
    if n_cols <= 0:
        raise Error("Invalid n_cols " + String(n_cols))
    rf_params.validity_check()
    rf_params.check()

    # `:298-309`
    if not rf_params.bootstrap and rf_params.max_samples != Float32(1.0):
        print(
            "WARN: If bootstrap sampling is disabled, max_samples value is"
            " ignored and whole dataset is used for building each tree"
        )
        rf_params.max_samples = Float32(1.0)
    var n_sampled = n_sampled_rows_for(
        rf_params.bootstrap, rf_params.max_samples, n_rows
    )
    if n_sampled <= 0:
        raise Error(
            "max_samples "
            + String(rf_params.max_samples)
            + " x n_rows "
            + String(n_rows)
            + " rounds to "
            + String(n_sampled)
            + " sampled rows; a tree needs at least one"
        )

    # `:317-325` -- ONCE, for the whole forest, with their literal 4.
    var qr = compute_quantiles(
        ctx,
        x,
        Int(rf_params.tree_params.max_n_bins),
        n_rows,
        n_cols,
        4,
        rf_params.seed,
        row_major,
    )
    # `rebind` because `O.DataT` and `DType.float32` are EQUAL by this
    # function's `where` clause but not syntactically the same expression,
    # so the pointer types do not unify on their own. This is rebind's
    # documented job and it reinterprets nothing at runtime.
    var quantiles = Quantiles[O.DataT](
        rebind[MutPointer[Scalar[O.DataT], MutUntrackedOrigin]](
            qr.quantiles_array.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        ),
        qr.n_bins_array.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
    )

    var sampler = RowSampler(
        ctx, rf_params.bootstrap, rf_params.seed, n_rows, n_sampled
    )

    var forest = RandomForestMetaData[O.DataT, O.LabelT](
        List[TreeMetaDataNode[O.DataT]](),
        rf_params,
        Int32(n_cols),
    )

    # `:337-367` -- their tree loop, serial here (DEVIATION 117).
    for i in range(Int(rf_params.n_trees)):
        sampler.sample(ctx, Int32(i))
        var dataset = DatasetView[O.DataT, O.LabelT](
            rebind[MutPointer[Scalar[O.DataT], MutUntrackedOrigin]](
                x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            ),
            y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            sample_weight.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(n_rows),
            Int64(n_cols),
            # `builder.cuh:238-239` -- row_major picks the strides.
            Int64(n_cols) if row_major else Int64(1),
            Int64(1) if row_major else Int64(n_rows),
            Int32(n_sampled),
            Int32(n_cols),
            sampler.rows_ptr(),
            Int32(n_unique_labels),
            False,
        )
        var builder = Builder[O](
            ctx,
            rf_params.tree_params,
            Int32(i),
            rf_params.seed,
            n_sampled,
            n_cols,
            Int32(n_unique_labels),
            objective.copy(),
        )
        var tree = builder.train(ctx, dataset, quantiles)
        tree.treeid = Int32(i)
        forest.trees.append(tree^)
        _ = builder^

    ctx.synchronize()
    # Mojo frees a value at its LAST USE, and every buffer above reached a
    # kernel as a raw pointer. These uses keep them alive past the final
    # synchronize. Measured hazard, not a precaution.
    _ = qr^
    _ = sampler^
    return forest^
