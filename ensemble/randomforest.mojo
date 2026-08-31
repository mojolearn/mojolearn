# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
| `n_streams` | none (required) | 4 | 4 | `randomforest_common.pyx:326`; DEVIATION 117, PORTED as K-way pipelining |
| `random_state` / `seed` | none (required) | **None -> 0** | 0 | `randomforest_common.pyx:325, 517-519` |

**THREE DISAGREEMENTS, flagged (a fourth -- n_streams clamped to 1 --
resolved when DEVIATION 117 was ported):**

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
   directly. Same outcome by two routes. This port carries CRITERION_END
   in `RF_params` and resolves it in `Builder`'s constructor -- theirs
   does it in `DecisionTree::fit`, which sits between their
   `RandomForest::fit` and that constructor and dispatches the objective
   FAMILY on the same integer-label test. Ours has the family fixed by
   `O` already, so only the value is left to resolve, and the constructor
   is the first point that sees both `params` and `O.LabelT`.
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
`checks/predict_check.mojo` compares `std.math.log2` against libm's
`log2` at every integer `n_cols` from 2 to 4096 and finds **4051
bit-level disagreements out of 4095, the first at n_cols = 3** -- so the
scar generalizes from `log` to `log2` and the two functions are not the
same function. It ALSO finds that **none of those 4051 disagreements
changes the resulting column count** at any of those sizes, so the libm
call is currently a defence with nothing yet proven to defend against.
Both halves of that are recorded because the second half is the one a
later reader would otherwise assume in the wrong direction.

================= DEVIATION BLOCK (whole file) =================

DEVIATION 117, PORTED (2026-08-21 late; it stood as "not ported and
cannot be" until then). THEIRS runs the per-tree loop under `#pragma omp
parallel for num_threads(n_streams)` with each OpenMP thread owning one
CUDA stream from the handle's stream pool (`randomforest.cuh:336-341`) --
cross-tree parallelism, and their Python default is n_streams=4
(`randomforestclassifier.py:94`), so the SHIPPED cuML overlaps four
trees. Metal has one queue and this port one host thread, so the same
overlap is expressed as K-WAY PIPELINING in `fit_forest`: K =
`n_streams` trees in flight, each suspended at `doSplit`'s two sync
points (`builder.cuh:479-481`, `:501-502`), one synchronize per cycle
serving all of them, each consume step immediately enqueueing that
tree's next phase. The state machine lives in `builder.mojo`
(`BatchState`/`TreeState`, `begin_*`/`advance_*`); `train` and
`do_split` are its serial K=1 drive, operation for operation.

WHAT MAKES THE OVERLAP FREE ON OUTPUT, from their source rather than an
argument: **their RNG is a pure function rather than a stream.** Both
draws are keyed by hash, not by draw order: the per-tree row sample is
`rs = fnv1a32(fnv1a32(basis, seed), tree_id)` (`randomforest.cuh:120-122`,
under their own comment "Hash these together so per-tree row samples are
uncorrelated"), and the per-node feature sample is
`fnv1a32_hash(seed, treeid, nodeid)` (`kernels/builder_kernels.cuh:88`).
Tree 7's rows and node 12's columns are the same values whether 1 stream
or 8 built them. Each in-flight tree owns its whole `Builder` workspace
(DEVIATION 313, pool-of-K) and its `selected_rows_` slot (their
per-stream vector, ported); the only shared device objects are
read-only. GATED: `fingerprint_probe.mojo` holds K=1 and K=4 to
bit-identical forests on five configs, and the sabotage that aliases
every slot to row buffer 0 moves every K4 line while the K1 lines stand
-- so the gate watches the concurrency, not just the totals.

Their non-OpenMP `#else` build (`randomforest.cuh:38-43`,
`randomforest.cu:584`) pins n_streams to 1; the serial port mirrored
that build and the old text here defended it. Porting the parallel
design supersedes it: only their `:585` clamp (streams <= trees)
survives in `set_rf_params`.

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
`ensemble/checks/forest_check.mojo` fits one and predicts it back.

ALL FOUR `RowSampler` ARMS RUN. This block used to say three of them
were "still out, each raising by name"; that stopped being true and the
sentence stayed, which is worse than never having written it.

  * BOOTSTRAP (`:140-142`, `raft::random::uniformInt` under `GenPhilox`)
    is ported and held to a compiled RAFT oracle per cell.
    **BUT THE ROWS IT DRAWS DEPEND ON THE GPU'S SM COUNT.** RAFT's
    launch config is `n_blocks = 4 * getMultiProcessorCount()`
    (`rng_impl.cuh:64-74`) and the generator strides by
    `gridDim.x * blockDim.x`, so which row each thread draws is a
    function of the DEVICE. This port pins the stride to 4 x 108 x 256
    (DEVIATION 184 in `philox.mojo`), which is a 108-SM part -- A100 or
    A30. On any other NVIDIA card cuML itself draws different rows, for
    every tree. So "tree 7's rows are a pure function of (seed, 7)" is
    true of the STREAM COUNT and false of the DEVICE, and the
    cross-checkable claim is only cross-checkable on one class of card.
  * WEIGHTED BOOTSTRAP (`:125-138`) and ZERO-WEIGHT REMOVAL (`:144-154`)
    both run. The CDF is a sequential HOST scan where theirs is
    `thrust::inclusive_scan` on device -- a different summation order,
    so `upper_bound` can land on a different row (DEVIATION 306).
  * `RowSampler::store_bootstrap_mask` (`:170-183`) IS still out, and
    with it the whole OOB feature. It does not raise, because there is
    no parameter to raise on: `bootstrap_masks` is simply absent.
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

from std.gpu import global_idx
from std.math import ceildiv as _ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz

from ensemble.decisiontree.batched_levelalgo.bins import Bin
from ensemble.decisiontree.batched_levelalgo.builder import (
    Builder,
    SplitStaging,
    TreeState,
    flush_splits_downloads,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    launch_bin_dataset,
)
from ensemble.flatnode import SparseTreeNode
from ensemble.decisiontree.batched_levelalgo.bins import BinScales
from ensemble.decisiontree.batched_levelalgo.objectives import ObjectiveLike
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.quantiles import (
    compute_quantiles,
    Quantiles,
)
from ensemble.decisiontree.batched_levelalgo.random_utils import (
    fnv1a32_hash_seed_tree,
)
from core.launch_log import log_launch
from ensemble.instruments import FitInstruments
from core.philox import (
    RNG_STRIDE,
    launch_uniform_int,
    uniform_double_host,
)
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

        The TRAINING fields used to be a different question, refused
        wholesale by a `check_fit_supported()` that no longer exists on
        this struct. They are honored now: `Builder` builds the objective
        from `params` at `builder.cuh:592-596`'s call site, so
        `min_samples_leaf`, `split_criterion` and `min_impurity_decrease`
        reach the device from here and nowhere else. `criteria_check`
        arm F holds them to it.
        """
        # `n_streams` IS honored since DEVIATION 117 was ported: the
        # forest loop pipelines that many trees over the one Metal queue,
        # mirroring their omp/stream pool (`randomforest.cuh:336-367`).
        # The refusal that stood here guarded the serial port and is gone
        # with it; no output bit depends on the value, because their
        # per-tree and per-node RNG is a pure hash of (seed, treeid[,
        # nodeid]) (`randomforest.cuh:120-122`,
        # `kernels/builder_kernels.cuh:88`) -- the fingerprint probe holds
        # K=1 and K=4 to identical forests.
        self.tree_params.check()
        self.validity_check()

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
      * `n_streams = min(cfg_n_streams, omp_get_max_threads())` -- the
        omp term models their host worker threads and drops out under
        the pipelined loop (DEVIATION 117, PORTED)
      * clamp `n_streams` down to `n_trees` if there are fewer trees
        than streams (`randomforest.cu:585`) -- KEPT
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
    # `randomforest.cu:584` -- min(cfg_n_streams, omp_get_max_threads()).
    # That clamp models the HOST WORKER THREADS their omp fan-out needs,
    # one per stream. The pipelined loop (DEVIATION 117) drives every
    # slot from one host thread, so there is no thread count to clamp to
    # and the omp term drops out; what survives is `randomforest.cu:585`,
    # the clamp to the tree count. (The old text here modeled their
    # non-OpenMP `#else` build, which pins n_streams to 1 -- that was the
    # serial port's story and it is gone with it.)
    var n_streams = cfg_n_streams
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


def min_samples_leaf_fraction(fraction: Float64, n_rows: Int) raises -> Int32:
    """`randomforest_common.pyx:520-523`, the FLOAT arm:

        min_samples_leaf = (
            self.min_samples_leaf if isinstance(self.min_samples_leaf, int)
            else math.ceil(self.min_samples_leaf * n_rows))

    A Python caller may pass `min_samples_leaf` as a fraction of the
    training set and get an absolute count back, resolved at fit time
    against `n_rows`. Their C++ `DecisionTreeParams::min_samples_leaf` is
    an `int` (`decisiontree.hpp:39`) and never sees the fraction.

    Mojo has no `int | float` parameter, so the two arms are two entry
    points rather than one: the integer arm IS just passing the Int32, and
    this is the other. `ceil`, not round and not truncate.
    """
    if not (fraction > 0.0):
        raise Error(
            "min_samples_leaf fraction must be positive, got "
            + String(fraction)
        )
    var x = fraction * Float64(n_rows)
    var f = Float64(Int(x))
    var c = Int(f) if x == f else Int(f) + 1
    return Int32(c)


def min_samples_split_fraction(
    fraction: Float64, n_rows: Int
) raises -> Int32:
    """`randomforest_common.pyx:524-527`, the FLOAT arm:

        min_samples_split = (
            self.min_samples_split if isinstance(..., int)
            else max(2, math.ceil(self.min_samples_split * n_rows)))

    Same shape as `min_samples_leaf_fraction` with `max(2, ...)` on top --
    their `validity_check` refuses anything below 2 (`decisiontree.cu:33`),
    so the floor is what keeps a small fraction from failing that check
    rather than a separate opinion about small splits.
    """
    var c = Int(min_samples_leaf_fraction(fraction, n_rows))
    return Int32(2) if c < 2 else Int32(c)


def check_random_seed(random_state: Int) raises -> UInt64:
    """`internals/validation.py:73-79`.

        if random_state < 0 or random_state >= 2**32:
            raise ValueError(
                f"Expected `0 <= random_state <= 2**32 - 1`, ...")

    WHY THIS MATTERS MORE THAN IT LOOKS. Their per-tree seed chain folds
    the user seed in ONE round of `fnv1a32` (`randomforest.cuh:121` calls
    `fnv1a32` directly, not `fnv1a32_combine`), so the high 32 bits are
    DISCARDED. That is lossless only because this check ran first and
    capped the seed below 2^32. A caller who reaches the C API directly
    silently loses half their seed, and this port reproduces that
    truncation faithfully (`random_utils.mojo:107-120`).

    So this is not a bounds check for its own sake -- it is the reason the
    truncation upstream is harmless, and a caller building `RF_params` by
    hand should run it. `RF_params.seed` stays a full UInt64 because their
    C++ field is a `uint64_t`; the restriction is the Python layer's.

    Their `random_state=None` arm draws a seed from numpy. Ours does not
    have one: `default_rf_params_*` uses 0, which is what
    `randomforest_common.pyx:517` maps None to. Unseeded does not mean
    randomly seeded on either side.
    """
    if random_state < 0:
        raise Error(
            "Expected `0 <= random_state <= 2**32 - 1`, got "
            + String(random_state)
        )
    if UInt64(random_state) >= UInt64(4294967296):
        raise Error(
            "Expected `0 <= random_state <= 2**32 - 1`, got "
            + String(random_state)
        )
    return UInt64(random_state)


def class_weight_uniform(n_classes: Int) -> List[Float64]:
    """`common/classification.py:70-72` -- the `class_weight is None` arm,
    `np.ones(n_classes, dtype=np.float64)`."""
    var w = List[Float64]()
    for _ in range(n_classes):
        w.append(Float64(1.0))
    return w^


def class_weight_balanced(
    n_classes: Int,
    y_ind: List[Int32],
    sample_weight: List[Float32] = List[Float32](),
) raises -> List[Float64]:
    """`common/classification.py:73-80` -- the `"balanced"` arm:

        counts  = cp.bincount(y_ind, weights=sample_weight)
        weights = (counts.sum() / (n_classes * counts)).astype(dtype)

    NOTE THE WEIGHTED BINCOUNT. `balanced_with_sample_weight` defaults
    True (`:15`), and their own comment says why: some cuml estimators
    "weren't doing this and we may need to maintain this bug for a bit."
    Random forest is not one of those -- `randomforestclassifier.py:279`
    takes the default -- so a class's count here is the SUM OF ITS
    SAMPLE WEIGHTS, not its row count.

    THE NARROWING IS PART OF THIS ARM AND NOT OF THE OTHERS. `.astype`
    rounds these weights to float32 before they are ever applied, while
    the uniform and explicit arms stay float64 until the `take` at `:97`.
    That asymmetry is theirs; it is transcribed rather than smoothed,
    because a weight is a multiplier on a histogram accumulation and the
    two orders do not round the same.

    A class with zero total weight divides by zero and yields `inf`,
    exactly as theirs does -- no guard on either side.
    """
    var counts = List[Float64]()
    for _ in range(n_classes):
        counts.append(Float64(0.0))
    var weighted = len(sample_weight) > 0
    for i in range(len(y_ind)):
        var k = Int(y_ind[i])
        if k < 0 or k >= n_classes:
            raise Error(
                "y_ind holds " + String(k) + ", outside [0, n_classes)"
            )
        counts[k] += Float64(sample_weight[i]) if weighted else Float64(1.0)
    var total = Float64(0.0)
    for k in range(n_classes):
        total += counts[k]
    var w = List[Float64]()
    for k in range(n_classes):
        var v = total / (Float64(n_classes) * counts[k])
        # `.astype(dtype)` -- dtype is float32 here
        w.append(Float64(Float32(v)))
    return w^


def class_weight_explicit(
    n_classes: Int, weights: List[Float64]
) raises -> List[Float64]:
    """`common/classification.py:81-91` -- the dict arm.

    Theirs looks each class up in a mapping and leaves the ones it does
    not find at 1.0, then raises if SOME classes were missing and the
    mapping's size does not account for the rest:

        if unweighted and (n_classes - len(unweighted)) != len(class_weight):
            raise ValueError(f"The classes, ..., are not in class_weight")

    A Mojo `Dict` keyed by an arbitrary label type has no counterpart here
    -- this port's labels are already the indices `y_ind` holds -- so the
    mapping becomes a full per-class vector and the partial-coverage error
    cannot arise. What CAN arise is the wrong length, which is refused.

    Their `'balanced_subsample'` is refused by name for random forest
    (`randomforestclassifier.py:173-176`); there is no string argument
    here to refuse, and no subsample arm to reach.
    """
    if len(weights) != n_classes:
        raise Error(
            "class_weight must hold one weight per class: got "
            + String(len(weights))
            + " for "
            + String(n_classes)
            + " classes"
        )
    return weights.copy()


def apply_class_weight(
    class_weight: List[Float64],
    y_ind: List[Int32],
    sample_weight: List[Float32] = List[Float32](),
) raises -> List[Float32]:
    """`common/classification.py:93-100`:

        if (weights != 1).any():
            if sample_weight is None:
                sample_weight = cp.asarray(weights, dtype).take(y_ind)
            else:
                sample_weight = sample_weight.copy()
                for ind, weight in enumerate(weights):
                    sample_weight[y_ind == ind] *= weight

    THE `any()` GUARD IS LOAD-BEARING, not an optimization. When every
    weight is exactly 1 theirs returns `sample_weight` UNCHANGED -- which,
    when none was passed, is None. Downstream that is the difference
    between a weighted fit and an unweighted one:
    `RowSampler::use_weighted_bootstrap` (`randomforest.cuh:214`) tests
    the pointer, not the values, so materialising a vector of ones would
    silently move a default fit onto the weighted-bootstrap arm. Ours
    returns an EMPTY list for the same reason, which is what
    `has_sample_weight` reads.
    """
    var any_non_unit = False
    for k in range(len(class_weight)):
        if class_weight[k] != Float64(1.0):
            any_non_unit = True
    if not any_non_unit:
        return sample_weight.copy()

    var out = List[Float32]()
    var weighted = len(sample_weight) > 0
    for i in range(len(y_ind)):
        var k = Int(y_ind[i])
        if k < 0 or k >= len(class_weight):
            raise Error(
                "y_ind holds " + String(k) + ", outside [0, n_classes)"
            )
        if weighted:
            # `:99` -- multiply the caller's weight IN PLACE
            out.append(sample_weight[i] * Float32(class_weight[k]))
        else:
            # `:97` -- take, so the weight is narrowed once, here
            out.append(Float32(class_weight[k]))
    return out^


def preprocess_labels(
    n_rows: Int, mut labels: List[Int32]
) raises -> Dict[Int32, Int32]:
    """`preprocess_labels`, `randomforest.cu:113-131`.

        for (int i = 0; i < n_rows; i++) {
          ret = labels_map.insert(pair<int,int>(labels[i], n_unique_labels));
          if (ret.second) { n_unique_labels += 1; }
          labels[i] = ret.first->second;   // IN-PLACE
        }

    THE MAPPING IS BY FIRST APPEARANCE, NOT BY SORTED ORDER, and that is
    easy to get wrong because `std::map` IS sorted -- but the dense index
    comes from `n_unique_labels`, which increments only when `insert`
    actually inserted, so it counts the order labels were FIRST SEEN.
    `[7, 3, 7, 3]` becomes `[0, 1, 0, 1]`, not `[1, 0, 1, 0]`.

    Their Python layer does its own label encoding and never calls this;
    it is a C-API helper, and it is ported because it is part of that
    surface, not because anything here needs it.

    `verbosity` is dropped -- this port has no logger (DEVIATION 119f).
    """
    if len(labels) < n_rows:
        raise Error(
            "labels holds "
            + String(len(labels))
            + " values but n_rows is "
            + String(n_rows)
        )
    var labels_map = Dict[Int32, Int32]()
    var n_unique_labels = Int32(0)
    for i in range(n_rows):
        var key = labels[i]
        if key in labels_map:
            labels[i] = labels_map[key]
        else:
            # `:124-125` -- insert succeeded, so this is a NEW label
            labels_map[key] = n_unique_labels
            labels[i] = n_unique_labels
            n_unique_labels += 1
    return labels_map^


def postprocess_labels(
    n_rows: Int, mut labels: List[Int32], labels_map: Dict[Int32, Int32]
) raises:
    """`postprocess_labels`, `randomforest.cu:140-161`.

        reverse_map.resize(labels_map.size());
        for (it = labels_map.begin(); it != labels_map.end(); it++)
          reverse_map[it->second] = it->first;
        for (int i = 0; i < n_rows; i++)
          labels[i] = reverse_map[labels[i]];

    The iteration order of their `std::map` does not matter here --
    `reverse_map` is INDEXED by the dense value, so any order fills the
    same array. That is why a Mojo `Dict`, which is not ordered, is a
    faithful stand-in for their `std::map` at this call site and would
    not be at a call site that walked it in key order.

    Theirs indexes `reverse_map[labels[i]]` with no bounds check, so a
    label outside the map reads out of bounds; ours refuses.
    """
    if len(labels) < n_rows:
        raise Error(
            "labels holds "
            + String(len(labels))
            + " values but n_rows is "
            + String(n_rows)
        )
    var n_unique = len(labels_map)
    var reverse_map = List[Int32]()
    for _ in range(n_unique):
        reverse_map.append(Int32(0))
    for entry in labels_map.items():
        reverse_map[Int(entry.value)] = entry.key
    for i in range(n_rows):
        var v = Int(labels[i])
        if v < 0 or v >= n_unique:
            raise Error(
                "label " + String(v) + " is outside the labels_map"
            )
        labels[i] = reverse_map[v]


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

    # --- OOB. NOT in their C++ struct ------------------------------------
    # These are Python ATTRIBUTES on the estimator, set by
    # `_compute_oob_score` (`randomforest_common.pyx:741-753`): the C++
    # side only ever fills the mask buffer. This port has no Python layer,
    # so they land on the thing a fit returns. DEVIATION 311.
    #
    # `has_oob` is False unless `fit_forest(oob_score=True)` ran, and the
    # three fields below are meaningless when it is False. Their absence
    # is `hasattr(self, 'oob_score_')` on the Python side
    # (`:276`, `:299`).
    var has_oob: Bool
    # `:748` / `:753` -- accuracy for a classifier, R^2 for a regressor.
    var oob_score_: Float64
    # `:742` -- classifier only: the averaged per-class OOB probabilities,
    # n_rows x num_outputs, row-major.
    var oob_decision_function_: List[Float64]
    # `:750` -- regressor only: the averaged OOB prediction per row.
    var oob_prediction_: List[Float64]

    def __init__(
        out self,
        var trees: List[TreeMetaDataNode[Self.dtype]],
        rf_params: RF_params,
        n_features: Int32,
    ):
        """Their three real members. The OOB fields default to absent,
        which is what `hasattr(self, 'oob_score_')` tests on the Python
        side (`randomforest_common.pyx:276`, `:299`) -- they exist only
        after `_compute_oob_score` has run."""
        self.trees = trees^
        self.rf_params = rf_params
        self.n_features = n_features
        self.has_oob = False
        self.oob_score_ = Float64(0.0)
        self.oob_decision_function_ = List[Float64]()
        self.oob_prediction_ = List[Float64]()


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

    def fit[
        O: ObjectiveLike
    ](
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        mut y: DeviceBuffer[O.LabelT],
        mut sample_weight: DeviceBuffer[DType.float32],
        n_rows: Int,
        n_cols: Int,
        n_unique_labels: Int,
        scales: BinScales = BinScales(1.0, 1.0),
        sample_weight_host: List[Float32] = List[Float32](),
        row_major: Bool = False,
    ) raises -> RandomForestMetaData[O.DataT, O.LabelT] where (
        O.DataT == DType.float32
    ):
        """`RandomForest::fit`, `randomforest.cuh:286-370`.

        THIS USED TO RAISE. The body is now `fit_forest`, which IS the port
        of their `:286-370` -- error checking, `n_sampled_rows`, quantiles
        once for the whole forest, the row sampler, and the per-tree loop.
        It is a free function because the objective type has to come from
        somewhere and `RandomForest[T, L]` carries only two of the three
        types a fit needs; this method is the estimator-shaped door onto it
        and takes the third as a parameter.

        `self.rf_params` is passed by reference and CAN BE MUTATED, exactly
        as theirs is: their `:301-306` overwrites `max_samples` to 1.0 with
        a warning when `bootstrap` is off. A caller that reads the params
        back afterwards sees the corrected value on both sides.
        """
        return fit_forest[O](
            ctx,
            x,
            y,
            sample_weight,
            n_rows,
            n_cols,
            n_unique_labels,
            self.rf_params,
            scales,
            sample_weight_host,
            row_major,
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

    def predict_proba(
        self,
        input: List[Scalar[Self.dtype]],
        n_rows: Int,
        n_cols: Int,
        mut probabilities: List[Scalar[Self.dtype]],
        forest: RandomForestMetaData[Self.dtype, Self.label_dtype],
    ) raises:
        """`RandomForestClassifier.predict_proba`,
        `randomforestclassifier.py:357-...`.

        THIS IS `predict` STOPPED ONE LINE EARLY. Their `predict`
        (`randomforest.cuh:403-427`) zero-inits a row vector, lets every
        tree ADD its leaf vector into it (`decisiontree.cuh:387`), divides
        by `n_trees` at `:414-416`, and only THEN takes the argmax at
        `:417-427`. The divided vector is the class-probability estimate,
        so `predict_proba` is that vector and `predict` is its argmax --
        which is why they can never disagree.

        Their Python routes this through nvForest rather than through
        `RandomForest::predict`, and nvForest is DEVIATION 119b's declined
        path; this walks the trees on the host instead. Same definition,
        same value, different executor -- the same substitution the OOB
        pass makes for its per-tree predictions (DEVIATION 311).

        CLASSIFIER ONLY, as theirs is: `predict_proba` is defined on
        `RandomForestClassifier` and not on the regressor.
        """
        if self.rf_type != CLASSIFICATION:
            raise Error(
                "predict_proba is defined on RandomForestClassifier only;"
                " cuML's regressor has no such method"
            )
        self.error_checking(n_rows, n_cols)
        if len(forest.trees) == 0:
            raise Error("Cannot predict! No trees in the forest.")
        var num_outputs = Int(forest.trees[0].num_outputs)
        if num_outputs < 1:
            raise Error(
                "forest.trees[0].num_outputs is " + String(num_outputs)
            )
        if len(probabilities) < n_rows * num_outputs:
            raise Error(
                "probabilities holds "
                + String(len(probabilities))
                + " values but n_rows * num_outputs is "
                + String(n_rows * num_outputs)
            )
        if len(input) < n_rows * n_cols:
            raise Error(
                "input holds "
                + String(len(input))
                + " values but n_rows * n_cols is "
                + String(n_rows * n_cols)
            )
        var n_trees = Int(self.rf_params.n_trees)
        if len(forest.trees) < n_trees:
            raise Error(
                "rf_params.n_trees is "
                + String(n_trees)
                + " but the forest holds "
                + String(len(forest.trees))
                + " trees"
            )

        for row_id in range(n_rows):
            # `randomforest.cuh:403`
            var row_prediction = List[Scalar[Self.dtype]]()
            for _ in range(num_outputs):
                row_prediction.append(0)
            # `:404-412`
            for i in range(n_trees):
                DecisionTree.predict(
                    forest.trees[i],
                    input,
                    1,
                    n_cols,
                    row_prediction,
                    num_outputs,
                    rows_offset=row_id * n_cols,
                    preds_offset=0,
                )
            # `:414-416` -- and STOP. No argmax.
            for k in range(num_outputs):
                probabilities[row_id * num_outputs + k] = row_prediction[
                    k
                ] / Scalar[Self.dtype](n_trees)

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
            # raft `scores.cuh:116` -- `double diff = predictions[i] -
            # ref_predictions[i];`. BOTH OPERANDS ARE `T`, which is `float`
            # for `RandomForestRegressorF`, so THE SUBTRACTION IS FLOAT32
            # and only its result is widened. Widening first is a silent
            # improvement on them: the two sums might be excused by the
            # atomic ordering already priced in DEVIATION 119e, but the
            # median is a SELECTION, so a different low bit here returns a
            # different element, not a differently-rounded one.
            var diff = Float64(predictions[i] - ref_labels[i])
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
# store_bootstrap_mask's two device ops -- `randomforest.cuh:170-183`
# ---------------------------------------------------------------------------
# Theirs is a `thrust::fill` followed by a `thrust::scatter` of a
# `constant_iterator(true)`. Two ops there, two kernels here, in their
# order: the fill must complete before the scatter, because a row drawn
# twice must stay `true` and a row never drawn must end `false`.


def ftz_features_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    n: Int64,
):
    """NOT THEIRS. IDENTITY_PATHS row 10 for the feature matrix, DEVIATION
    1942 (2026-08-29): under NUMERIC_IDENTICAL the device copy of X is
    flushed IN PLACE, once, before any kernel reads it, so every value a
    quantile sort, a `lower_bound` bin lookup, a split partition or the
    host predict walk ever sees is the same value on a flush-to-zero
    backend (Apple) and on a denormal-honoring one (CUDA, HIP). The
    quantiles are COPIES of X values, so flushing X here also flushes
    every `quesval` a tree stores. Launched only under the pin; under
    FAST the kernel is never enqueued."""
    var i = Int(global_idx.x)
    if Int64(i) < n:
        x[unsafe_offset=i] = ftz(x[unsafe_offset=i])


def bootstrap_mask_fill_kernel(
    masks: MutPointer[UInt8, MutAnyOrigin],
    offset: Int64,
    n_rows: Int32,
):
    """`:180` -- `thrust::fill(policy, tree_mask, tree_mask + n_rows_,
    false)`."""
    var i = Int(global_idx.x)
    if i < Int(n_rows):
        masks[unsafe_offset = Int(offset) + i] = UInt8(0)


def bootstrap_mask_scatter_kernel(
    masks: MutPointer[UInt8, MutAnyOrigin],
    offset: Int64,
    rows: MutPointer[Int32, MutAnyOrigin],
    n_selected: Int32,
    n_rows: Int32,
):
    """`:181-185`:

        thrust::scatter(policy,
                        thrust::make_constant_iterator(true),
                        thrust::make_constant_iterator(true) + selected_rows.size(),
                        selected_rows.data(),
                        tree_mask);

    A scatter of a CONSTANT is idempotent, so the duplicate row ids a
    bootstrap draw produces are not a race: every writer writes the same
    byte. That is why theirs needs no atomic and neither does this.

    The `n_rows` bound has no counterpart -- theirs would scatter out of
    bounds on a bad row id and this refuses to. It cannot fire: every arm
    of `sample` produces ids in `[0, n_rows)`.
    """
    var i = Int(global_idx.x)
    if i < Int(n_selected):
        var r = Int(rows[unsafe_offset=i])
        if r >= 0 and r < Int(n_rows):
            masks[unsafe_offset = Int(offset) + r] = UInt8(1)



def compute_oob_score[
    O: ObjectiveLike, sabotage: Int = 0
](
    ctx: DeviceContext,
    mut forest: RandomForestMetaData[O.DataT, O.LabelT],
    sampler: RowSampler,
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[O.LabelT],
    n_rows: Int,
    n_cols: Int,
    row_major: Bool,
) raises where O.DataT == DType.float32:
    """`RandomForestClassifier/Regressor._compute_oob_score`,
    `randomforest_common.pyx:695-753`.

    Their loop, verbatim:

        oob_predictions = cp.zeros(output_shape, dtype=cp.float64)
        oob_counts      = cp.zeros(n_samples, dtype=cp.int32)
        for tree_idx in range(self.n_estimators):
            in_bag_mask = bootstrap_masks_cp[tree_idx]
            oob_mask    = ~in_bag_mask
            oob_predictions[oob_mask] += per_tree_preds[oob_mask, tree_idx]
            oob_counts[oob_mask] += 1
        valid_oob = oob_counts > 0
        ...
        oob_predictions[valid_oob] /= oob_counts[valid_oob]

    THE MASK IS INVERTED HERE (`:718`) and nowhere else: the buffer holds
    IN-BAG, a tree scores only the rows it never saw, and forgetting the
    `~` would report a memorization score.

    ACCUMULATION IS FLOAT64 (`:711`) while the per-tree predictions are
    the model's own dtype -- so the widening happens on the way IN to the
    sum, once per tree per row, and the division at the end is float64.

    TWO DEVIATIONS, both in where the numbers come from rather than what
    is done to them:

      * PER-TREE PREDICTIONS. Theirs calls `nvforest_model
        .predict_per_tree(X)` (`:703`), the treelite/nvForest path this
        port declines (DEVIATION 119b). Ours walks each tree on the host
        with `DecisionTree.predict_one`, which is the same traversal
        their `decisiontree.cuh:370-389` defines -- `<=` goes left, right
        is `left_child_id + 1`, a leaf adds its whole `vector_leaf` row
        with `+=`. Same values, different executor.
      * THE WHOLE PASS IS ON THE HOST. Theirs is cupy on device. The
        accumulation order therefore differs from theirs: ours is
        tree-major then row, theirs is a vectorised add per tree. Both
        are float64 and both are sequential in the tree index, so the
        only float difference available is within a row's class vector,
        which is added in class order on both sides. DEVIATION 311.
    """
    var n_trees = len(forest.trees)
    if n_trees == 0:
        raise Error("cannot compute an OOB score for an empty forest")
    var num_outputs = Int(forest.trees[0].num_outputs)

    # The masks, back on the host. `:717` indexes them per tree.
    var hm = ctx.enqueue_create_host_buffer[DType.uint8](n_trees * n_rows)
    log_launch("xfer_oob_masks")
    ctx.enqueue_copy(dst_buf=hm, src_buf=sampler.bootstrap_masks)

    # X, row-major, because `predict_one` walks a row (`:366` does the
    # same pointer arithmetic). Training may have been column-major.
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n_rows * n_cols)
    log_launch("xfer_oob_x")
    ctx.enqueue_copy(dst_buf=hx, src_buf=x)

    var hy = ctx.enqueue_create_host_buffer[O.LabelT](n_rows)
    log_launch("xfer_oob_y")
    ctx.enqueue_copy(dst_buf=hy, src_buf=y)
    ctx.synchronize()

    var rows = List[Scalar[O.DataT]]()
    for r in range(n_rows):
        for c in range(n_cols):
            var src = r * n_cols + c if row_major else c * n_rows + r
            rows.append(
                rebind[Scalar[O.DataT]](hx.unsafe_ptr().unsafe_load(src))
            )

    # `:711-712`
    var oob_predictions = List[Float64]()
    for _ in range(n_rows * num_outputs):
        oob_predictions.append(Float64(0.0))
    var oob_counts = List[Int]()
    for _ in range(n_rows):
        oob_counts.append(0)

    var scratch = List[Scalar[O.DataT]]()
    for _ in range(num_outputs):
        scratch.append(Scalar[O.DataT](0))

    # `:715-722`
    for t in range(n_trees):
        for r in range(n_rows):
            # `:717-718` -- oob_mask = ~in_bag_mask
            #
            # CHECK HOOK. 1 drops the `~`, scoring each tree on the rows
            # it was TRAINED on. That is the failure this line exists to
            # prevent and it does not look like a failure -- it reports a
            # better number.
            var in_bag = hm.unsafe_ptr().unsafe_load(t * n_rows + r) != UInt8(
                0
            )
            comptime if sabotage == 1:
                if not in_bag:
                    continue
            else:
                if in_bag:
                    continue
            for k in range(num_outputs):
                scratch[k] = Scalar[O.DataT](0)
            DecisionTree.predict_one[O.DataT](
                rows, r * n_cols, forest.trees[t], scratch, 0, num_outputs
            )
            # `:721` -- += , and the widening happens here
            for k in range(num_outputs):
                oob_predictions[r * num_outputs + k] += Float64(scratch[k])
            # `:722`
            oob_counts[r] += 1

    # `:725-732`
    var n_valid = 0
    for r in range(n_rows):
        if oob_counts[r] > 0:
            n_valid += 1
    if n_valid != n_rows:
        print(
            "WARN: Some inputs do not have OOB scores. This probably means"
            " too few trees were used to compute any reliable OOB"
            " estimates."
        )
    if n_valid == 0:
        raise Error(
            "no sample was out of bag for any tree, so there is no OOB"
            " score to report"
        )

    # `:734-738`
    for r in range(n_rows):
        if oob_counts[r] > 0:
            var d = Float64(oob_counts[r])
            for k in range(num_outputs):
                oob_predictions[r * num_outputs + k] /= d

    forest.has_oob = True
    comptime if O.LabelT.is_integral():
        # `:741-748` -- argmax over the averaged class probabilities, then
        # accuracy. `cp.argmax` keeps the FIRST maximum on a tie, which is
        # what a strict `>` from index 0 does.
        forest.oob_decision_function_ = oob_predictions.copy()
        var correct = 0
        for r in range(n_rows):
            if oob_counts[r] <= 0:
                continue
            var best = 0
            var best_p = oob_predictions[r * num_outputs]
            for k in range(1, num_outputs):
                if oob_predictions[r * num_outputs + k] > best_p:
                    best_p = oob_predictions[r * num_outputs + k]
                    best = k
            if Int(hy.unsafe_ptr().unsafe_load(r)) == best:
                correct += 1
        # `_classification.py:102` -- float(cp.average(correct))
        forest.oob_score_ = Float64(correct) / Float64(n_valid)
    else:
        # `:750-753` -- r2_score over the valid rows only.
        forest.oob_prediction_ = oob_predictions.copy()
        var mean = Float64(0.0)
        for r in range(n_rows):
            if oob_counts[r] > 0:
                mean += Float64(hy.unsafe_ptr().unsafe_load(r))
        mean /= Float64(n_valid)
        # `metrics/regression.py:136-140`
        var numerator = Float64(0.0)
        var denominator = Float64(0.0)
        for r in range(n_rows):
            if oob_counts[r] <= 0:
                continue
            var yt = Float64(hy.unsafe_ptr().unsafe_load(r))
            var d1 = yt - oob_predictions[r * num_outputs]
            numerator += d1 * d1
            var d2 = yt - mean
            denominator += d2 * d2
        # `:145-157`, force_finite=True: numerator == 0 -> 1;
        # numerator != 0 and denominator == 0 -> 0; else 1 - num/den.
        if numerator == Float64(0.0):
            forest.oob_score_ = Float64(1.0)
        elif denominator == Float64(0.0):
            forest.oob_score_ = Float64(0.0)
        else:
            forest.oob_score_ = Float64(1.0) - numerator / denominator

    _ = hm^
    _ = hx^
    _ = hy^


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

    ALL FOUR ARMS RUN. This paragraph used to say arms 1 and 3 "raise by
    name" because `sample_weight` was not accepted; both halves stopped
    being true and the sentence stayed.

    AND EVERY ARM RECORDS ITS BOOTSTRAP MASK. `store_bootstrap_mask` is
    called at `:163` -- after the dispatch, outside the branch -- so the
    non-bootstrap arms record one too. Their Python refuses `oob_score`
    without `bootstrap` (`randomforest_common.pyx:498`), so those masks
    are never read; recording them anyway is theirs, and it is the reason
    the call site is one line rather than four.

    THE SEED CHAIN IS PER TREE AND IS NOT A STREAM (`:119-123`):

        rs = fnv1a32_basis
        rs = fnv1a32(rs, seed_)      // ONE round on the low 32 bits
        rs = fnv1a32(rs, tree_id)
        RngState(rs, GenPhilox)

    Their comment at `:119` says why: "Hash these together so per-tree row
    samples are uncorrelated." Tree 7's rows are a pure function of
    `(seed, 7)`, so they do not depend on how many streams built the forest,
    on the order the trees were built, or on anything else. That is what
    makes DEVIATION 117's pipelined overlap free on output.

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
    # `:151` -- their `selected_rows.resize(n_selected)`. The zero-weight
    # arm produces FEWER rows than `n_sampled_rows`, and the builder must
    # be told how many, or it reads past the live entries.
    var n_selected: Int
    # `:223` -- `rmm::device_uvector<double> sample_weight_cdf_`, and `:222`
    # `double sample_weight_sum_`. Float64 on the HOST; see DEVIATION 306.
    var weight_cdf: List[Float64]
    var weight_sum: Float64
    # `:224` -- `std::vector<rmm::device_uvector<int>> selected_rows_`,
    # ONE PER STREAM. The pipelined forest loop (DEVIATION 117) is their
    # stream pool expressed on one queue, so the slot dimension is ported
    # with it: each in-flight tree reads its own row buffer. `h_rows` is
    # DEVIATION 305's host staging and stays single -- the arms that use
    # it synchronize, so it is never live for two slots at once.
    var selected_rows_: List[DeviceBuffer[DType.int32]]
    var h_rows: HostBuffer[DType.int32]
    # `:70`, `:81` -- `bool* bootstrap_masks_`, an `n_trees x n_rows`
    # DEVICE buffer the CALLER owns; theirs asserts it is a device
    # pointer (`:82-83`) and treats null as "OOB not requested".
    # `MutPointer` is non-null, so the null test becomes a Bool, the same
    # substitution DEVIATION 100 made for `sample_weight`.
    var bootstrap_masks: DeviceBuffer[DType.uint8]
    var has_masks: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        bootstrap: Bool,
        seed: UInt64,
        n_rows: Int,
        n_sampled_rows: Int,
        has_sample_weight: Bool = False,
        n_trees_for_masks: Int = 0,
        n_slots: Int = 1,
    ) raises:
        """`:63-105`, their constructor.

        `n_trees_for_masks` is 0 when OOB was not asked for, which is
        their `bootstrap_masks == nullptr`. Theirs is allocated by the
        PYTHON layer (`randomforest_common.pyx:568`,
        `cp.zeros((n_estimators, n_rows), bool)`) and passed in; this port
        has no Python layer, so `fit_forest` allocates it and hands the
        size here -- the same place the `n_bins` clamp went, and for the
        same reason (DEVIATION 309).
        """
        self.bootstrap = bootstrap
        self.seed = seed
        self.n_rows = n_rows
        self.n_sampled_rows = n_sampled_rows
        self.has_sample_weight = has_sample_weight
        self.n_selected = n_sampled_rows
        self.weight_cdf = List[Float64]()
        self.weight_sum = Float64(0.0)
        var n = n_sampled_rows if n_sampled_rows > 0 else 1
        self.selected_rows_ = List[DeviceBuffer[DType.int32]]()
        for _ in range(n_slots if n_slots > 0 else 1):
            self.selected_rows_.append(
                ctx.enqueue_create_buffer[DType.int32](n)
            )
        self.h_rows = ctx.enqueue_create_host_buffer[DType.int32](n)
        self.has_masks = n_trees_for_masks > 0
        var m = n_trees_for_masks * n_rows if self.has_masks else 1
        self.bootstrap_masks = ctx.enqueue_create_buffer[DType.uint8](m)
        ctx.synchronize()

    def prepare_weights(
        mut self, ctx: DeviceContext, weights: List[Float32]
    ) raises:
        """`validate_sample_weight` (`:198-211`) + the `copy_if` of
        `:144-154`, both done once.

        THEIR TWO REFUSALS ARE KEPT AND ARE NOT ADVISORY.
        `InvalidSampleWeight` rejects anything non-finite or negative
        (`:202-208`), and `compute_sample_weight_sum` asserts the total is
        strictly positive (`:93-95`). A silently-accepted bad weight would
        train a forest on a row set nobody asked for, which is the same
        failure class as a wrong bootstrap.

        DEVIATION 305: theirs runs the `copy_if` on the DEVICE, inside
        `sample()`, once per tree. Ours runs it on the HOST, once per
        forest. The set is a function of `sample_weight` alone -- no tree
        id, no RNG -- so every tree gets the identical set either way; the
        difference is `n_trees - 1` redundant device passes, which is work
        rather than meaning. PRICE: one host pass over `n_rows` weights
        per forest, and the weights must be available on the host, which
        they are because the caller supplies them.
        """
        if len(weights) < self.n_rows:
            raise Error(
                "sample_weight holds "
                + String(len(weights))
                + " values but n_rows is "
                + String(self.n_rows)
            )
        var total = Float64(0.0)
        var kept = 0
        var p = self.h_rows.unsafe_ptr()
        for i in range(self.n_rows):
            var w = weights[i]
            # `:202-208` -- non-finite or negative is refused BY VALUE.
            if not (w == w):
                raise Error(
                    "sample_weight values must be finite and non-negative;"
                    " index " + String(i) + " is NaN"
                )
            if w < Float32(0.0):
                raise Error(
                    "sample_weight values must be finite and non-negative;"
                    " index " + String(i) + " is " + String(w)
                )
            total += Float64(w)
            if w != Float32(0.0):
                if kept < self.n_sampled_rows:
                    p.unsafe_store(kept, Int32(i))
                kept += 1
        if total <= 0.0:
            raise Error(
                "sample_weight values must contain at least one positive"
                " value (randomforest.cuh:93-95)"
            )
        self.n_selected = min(kept, self.n_sampled_rows)

        # `:84-89` -- the weighted-bootstrap CDF, an inclusive scan over the
        # weights, and `:91` takes the total from its LAST ELEMENT rather
        # than from a separate reduction. Kept: a separate sum could differ
        # in the last bits from the scan's running total, and their
        # `upper_bound` searches the SCAN.
        if self.bootstrap:
            self.weight_cdf = List[Float64]()
            var run = Float64(0.0)
            for i in range(self.n_rows):
                run += Float64(weights[i])
                self.weight_cdf.append(run)
            self.weight_sum = self.weight_cdf[self.n_rows - 1]

    def rng_seed_for(self, tree_id: Int32) -> UInt32:
        """`:120-123`, the per-tree seed, exposed so a check can hold it to
        the same value the sampler uses."""
        return fnv1a32_hash_seed_tree(self.seed, tree_id)

    def sample(
        mut self, ctx: DeviceContext, tree_id: Int32, slot: Int = 0
    ) raises:
        """`RowSampler::sample`, `:110-165`.

        Their body is the four-way dispatch followed by ONE call to
        `store_bootstrap_mask` at `:163`, outside the branch, so every arm
        records its mask -- including the two that are not bootstraps at
        all. That placement is the whole reason the mask is trustworthy,
        and it is why the dispatch is a separate method here: ours has an
        early `return` in each arm, so a call per arm would be four
        chances to forget one.
        """
        self._sample_rows(ctx, tree_id, slot)
        # `:163`
        self.store_bootstrap_mask(ctx, tree_id, slot)

    def store_bootstrap_mask(
        mut self, ctx: DeviceContext, tree_id: Int32, slot: Int = 0
    ) raises:
        """`RowSampler::store_bootstrap_mask`, `:170-183`.

            if (bootstrap_masks_ == nullptr) { return; }
            bool* tree_mask = bootstrap_masks_ + checked_mul(tree_id, n_rows_);
            thrust::fill(policy, tree_mask, tree_mask + n_rows_, false);
            thrust::scatter(policy, constant_iterator(true),
                            constant_iterator(true) + selected_rows.size(),
                            selected_rows.data(), tree_mask);

        THE MASK IS IN-BAG, NOT OUT-OF-BAG. `true` means the row was
        DRAWN for this tree; the OOB set is its complement, and their
        Python takes that complement at `randomforest_common.pyx:718`
        (`oob_mask = ~in_bag_mask`). Getting this backwards would score
        every tree on the rows it memorized and report a great number.

        Note it scatters `selected_rows.size()` entries, which for the
        zero-weight arm is the RESIZED count, not `n_sampled_rows`. Ours
        passes `n_selected` for the same reason.
        """
        if not self.has_masks:
            return
        # `:178` -- `checked_mul<std::size_t>(tree_id, n_rows_)`
        var offset = Int64(Int(tree_id)) * Int64(self.n_rows)
        log_launch("bootstrap_mask_fill")
        ctx.enqueue_function[bootstrap_mask_fill_kernel](
            self.bootstrap_masks.unsafe_ptr(),
            offset,
            Int32(self.n_rows),
            grid_dim=_ceildiv(self.n_rows, 256),
            block_dim=256,
        )
        if self.n_selected > 0:
            log_launch("bootstrap_mask_scatter")
            ctx.enqueue_function[bootstrap_mask_scatter_kernel](
                self.bootstrap_masks.unsafe_ptr(),
                offset,
                self.selected_rows_[slot].unsafe_ptr(),
                Int32(self.n_selected),
                Int32(self.n_rows),
                grid_dim=_ceildiv(self.n_selected, 256),
                block_dim=256,
            )
        # NO synchronize. Their `thrust::fill` + `thrust::scatter` run on
        # the stream and `sample()` returns without a sync (`:163-165`);
        # everything that reads the mask or `selected_rows` is enqueued on
        # this same queue afterwards, so ordering is the queue's. A sync
        # here was this port's own wait, with no counterpart in their
        # source -- HOST_AND_DEVICE.md's rule two says exactly this wait
        # is in scope to delete. Nothing host-side reads the mask before
        # `fit_forest`'s post-loop synchronize.

    def _sample_rows(
        mut self, ctx: DeviceContext, tree_id: Int32, slot: Int = 0
    ) raises:
        """`:112-161`, the four-way dispatch in their order. All four arms
        run; see the struct docstring."""
        if self.bootstrap and self.has_sample_weight:
            # `:125-138` -- "Draw bootstrap rows according to sample
            # weights."
            #
            #   raft::random::uniform<double>(res, rng, scratch.data(),
            #       scratch.size(), 0.0, sample_weight_sum_);
            #   thrust::upper_bound(policy, cdf.data(), cdf.data() + n_rows,
            #       scratch.begin(), scratch.end(), selected_rows.begin());
            #
            # `upper_bound` returns the index of the FIRST cdf entry
            # STRICTLY GREATER than the draw, which is what makes a row's
            # probability its own weight over the total. A `lower_bound`
            # here would hand every zero-weight row the mass of its
            # predecessor.
            if len(self.weight_cdf) < self.n_rows:
                raise Error(
                    "weighted bootstrap needs prepare_weights first"
                )
            var draws = uniform_double_host(
                UInt64(Int(self.rng_seed_for(tree_id))),
                UInt64(0),
                RNG_STRIDE,
                self.n_sampled_rows,
                Float64(0.0),
                self.weight_sum,
            )
            var p = self.h_rows.unsafe_ptr()
            for i in range(self.n_sampled_rows):
                # std::upper_bound over the cdf
                var lo = 0
                var hi = self.n_rows
                var d = draws[i]
                while lo < hi:
                    var mid = (lo + hi) // 2
                    if self.weight_cdf[mid] <= d:
                        lo = mid + 1
                    else:
                        hi = mid
                p.unsafe_store(i, Int32(lo))
            self.n_selected = self.n_sampled_rows
            log_launch("xfer_sampled_rows")
            ctx.enqueue_copy(
                dst_buf=self.selected_rows_[slot],
                src_ptr=self.h_rows.unsafe_ptr(),
            )
            ctx.synchronize()
            return
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
            self.n_selected = self.n_sampled_rows
            launch_uniform_int(
                ctx,
                self.selected_rows_[slot],
                self.n_sampled_rows,
                Int32(0),
                Int32(self.n_rows),
                UInt64(Int(self.rng_seed_for(tree_id))),
            )
            # NO synchronize -- theirs is `uniformInt` on the stream and
            # `sample()` returns with no sync; the builder's kernels are
            # enqueued after this on the same queue and read
            # `selected_rows` in order. The host never reads it. The
            # host-staging arms below DO keep their sync, because each
            # re-writes `h_rows` on the host next tree and the write must
            # not race the in-flight copy -- that wait is the price of
            # DEVIATION 305's host staging, not of their design.
            return
        if self.has_sample_weight:
            # `:144-154` -- `thrust::copy_if` over `NonzeroSampleWeight`,
            # dropping the zero-weight rows and keeping the rest IN ORDER.
            # Computed once in `prepare_weights` rather than per tree: it
            # reads only `sample_weight`, so it is the same set for every
            # tree, and theirs recomputing it per tree is work, not
            # meaning. DEVIATION 305.
            if self.n_selected <= 0:
                raise Error(
                    "sample_weight values must contain at least one"
                    " positive value (randomforest.cuh:94)"
                )
            log_launch("xfer_sampled_rows")
            ctx.enqueue_copy(
                dst_buf=self.selected_rows_[slot],
                src_ptr=self.h_rows.unsafe_ptr(),
            )
            ctx.synchronize()
            return
        # `:155-157` -- `thrust::sequence`, the identity. This is the arm
        # `bootstrap=False` takes, and it is the only one that runs today.
        self.n_selected = self.n_sampled_rows
        var p = self.h_rows.unsafe_ptr()
        for i in range(self.n_sampled_rows):
            p.unsafe_store(i, Int32(i))
        log_launch("xfer_sampled_rows")
        ctx.enqueue_copy(
            dst_buf=self.selected_rows_[slot],
            src_ptr=self.h_rows.unsafe_ptr(),
        )
        ctx.synchronize()

    @always_inline
    def rows_ptr(
        mut self, slot: Int = 0
    ) -> MutPointer[Int32, MutUntrackedOrigin]:
        return (
            self.selected_rows_[slot].unsafe_ptr()
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
    # `:302` -- `std::round(this->rf_params.max_samples * n_rows)`.
    # THE PRODUCT IS FLOAT32. `max_samples` is `float` (`randomforest.hpp:71`)
    # and `n_rows` is `int`, so C++'s usual arithmetic conversions promote
    # `n_rows` TO FLOAT and the multiply rounds there -- including the
    # int->float conversion of `n_rows` itself. Computing this in Float64
    # is not a harmless widening: above 2^24 float32 cannot hold every
    # integer, so at n_rows = 20_000_001 with max_samples = 1.0 theirs
    # gives 20_000_000 and a Float64 product gives 20_000_001. That is a
    # different bootstrap size, hence a different forest, on every tree.
    var x = Float32(max_samples) * Float32(n_rows)
    var f = Float32(Int(x))
    if x - f >= 0.5:
        return Int(f) + 1
    return Int(f)


def _record_tree[
    dt: DType
](mut instr: FitInstruments, tree: TreeMetaDataNode[dt], idx: Int) raises:
    """DEVIATION 401 -- a finished tree's two identity checkpoints.

    `treeN.nodes` is the tree STRUCTURE (colid, threshold bits, metric
    bits, left child, instance count -- the same fields the fingerprint
    probe folds, plus the metric), and `treeN.leaves` is `vector_leaf`.
    Both are host lists by the time a tree lands in the forest, hashed
    FIELD BY FIELD as u32 bit patterns -- never raw struct bytes (padding)
    and never decimal text (identity_trace rule 1). `idx` is the tree's
    FOREST index, a position in the algorithm, identical under any
    pipeline width K (DEVIATION 117's output-freedom is what makes that
    true)."""
    if not instr.trace.enabled:
        return
    var flat = List[UInt32]()
    for i in range(len(tree.sparsetree)):
        ref node = tree.sparsetree[i]
        flat.append(UInt32(Int(node.ColumnId()) & 0xFFFFFFFF))
        var qb = UInt64(node.QueryValue().to_bits())
        flat.append(UInt32(qb & 0xFFFFFFFF))
        flat.append(UInt32(qb >> 32))
        var mb = UInt64(node.BestMetric().to_bits())
        flat.append(UInt32(mb & 0xFFFFFFFF))
        flat.append(UInt32(mb >> 32))
        flat.append(UInt32(Int(node.LeftChildId()) & 0xFFFFFFFF))
        flat.append(UInt32(Int(node.InstanceCount()) & 0xFFFFFFFF))
    instr.trace.record_host(
        "tree" + String(idx) + ".nodes", flat.unsafe_ptr(), len(flat)
    )
    # `[[mojo-buffer-freed-at-last-use]]` -- keep the list past the hash.
    _ = flat^
    var leaves = List[UInt32]()
    for i in range(len(tree.vector_leaf)):
        var lb = UInt64(tree.vector_leaf[i].to_bits())
        leaves.append(UInt32(lb & 0xFFFFFFFF))
        leaves.append(UInt32(lb >> 32))
    instr.trace.record_host(
        "tree" + String(idx) + ".leaves", leaves.unsafe_ptr(), len(leaves)
    )
    _ = leaves^


def fit_forest[
    O: ObjectiveLike, oob_sabotage: Int = 0
](
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[O.LabelT],
    mut sample_weight: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_unique_labels: Int,
    mut rf_params: RF_params,
    scales: BinScales = BinScales(1.0, 1.0),
    sample_weight_host: List[Float32] = List[Float32](),
    row_major: Bool = False,
    oob_score: Bool = False,
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
    DEVIATION 117, PORTED: K = n_streams trees pipelined over the one
    queue (see the driver below), and no output bit depends on it because
    every tree's rows and every node's columns are pure hashes of
    `(seed, tree_id)` and `(seed, tree_id, node_id)`.

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

    # DEVIATION 401 (identity-trace checkpoints) and 402 (stage timers):
    # ONE instrument set per fit, read from the environment here and
    # threaded down by `mut` reference -- one sequence counter, so K
    # pipelined builders cannot interleave two traces into one file. Both
    # are off by default; see `ensemble/instruments.mojo`.
    var instr = FitInstruments()
    var t_fit = instr.times.start()
    if instr.trace.enabled:
        instr.trace.header(
            "fit_forest n_rows="
            + String(n_rows)
            + " n_cols="
            + String(n_cols)
            + " n_trees="
            + String(rf_params.n_trees)
            + " n_streams="
            + String(rf_params.n_streams)
            + " seed="
            + String(rf_params.seed)
            + " max_depth="
            + String(rf_params.tree_params.max_depth)
            + " max_n_bins="
            + String(rf_params.tree_params.max_n_bins)
            + " bootstrap="
            + String(rf_params.bootstrap)
        )

    # `randomforest_common.pyx:497-499` -- "Out of bag estimation only
    # available if bootstrap=True". Without a bootstrap every tree sees
    # every row, so no row is ever out of bag and the score would be
    # computed over an empty set. Refused rather than returned as a NaN.
    if oob_score and not rf_params.bootstrap:
        raise Error(
            "Out of bag estimation only available if bootstrap=True"
        )

    # `randomforest_common.pyx:529-536`. THE PYTHON LAYER CLAMPS n_bins TO
    # n_rows, with a warning, and it does it HERE -- inside `_fit_forest`,
    # where n_rows is finally known -- not in `__init__`. Their C++ side
    # never checks it: `validity_check` only bounds it to (0, 1024]
    # (`decisiontree.cu:26-27`), so a C-API caller keeps whatever they
    # passed. This port sits at the C-API shape but is the only door a
    # caller has, so the clamp lives here or nowhere, and without it the
    # quantile pass below asks for more bins than there are rows.
    if rf_params.tree_params.max_n_bins > Int32(n_rows):
        print(
            "WARN: The number of bins, `n_bins` is greater than the number"
            " of samples used for training. Changing `n_bins` to number of"
            " training samples."
        )
        rf_params.tree_params.max_n_bins = Int32(n_rows)

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

    # DEVIATION 1942, row 10: the feature matrix is flushed on the device
    # BEFORE the quantile pass, which is the first kernel that reads it.
    # See `ftz_features_kernel`. FAST never enqueues this.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        log_launch("ftz_features")
        ctx.enqueue_function[ftz_features_kernel](
            x.unsafe_ptr(),
            Int64(n_rows * n_cols),
            grid_dim=_ceildiv(n_rows * n_cols, 256),
            block_dim=256,
        )

    # `:317-325` -- ONCE, for the whole forest, with their literal 4.
    var t_stage = instr.times.start()
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
    instr.times.stop(ctx, "quantiles", t_stage)
    # DEVIATION 401 -- the forest's ONE quantile table. `nbins` is the
    # per-feature bin count, which is where the subnormal-flush divergence
    # (DEVIATION 123) lands FIRST on a cross-backend diff; `values` is the
    # full col-major table, whose tail past `nbins[col]` is the unique
    # pass's leftover -- a pure function of the input data, deterministic
    # per backend, so hashing the whole logical allocation is sound.
    if instr.trace.enabled:
        instr.trace.record_device(
            ctx, "forest.quantiles.nbins", qr.n_bins_array
        )
        instr.trace.record_device(
            ctx, "forest.quantiles.values", qr.quantiles_array
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

    # DEVIATION 314 -- the dataset is BINNED once, right here, where
    # their quantiles are computed once (`:317-325`). The histogram
    # kernel re-derives `lower_bound(quantiles[col], value)` per element
    # per LEVEL per TREE (`builder_kernels_impl.cuh:341`); that index is
    # a pure function of (row, col) once the quantiles exist, so storing
    # it as one uint8 per element makes every later lookup a 1-byte read
    # of the SAME index -- bit-identical by construction, and the launch
    # -log attribution measured that kernel at 85.3% of device time at
    # 500k x 50. uint8 caps the index at 255, so `max_n_bins > 256`
    # keeps the searching path; the buffer is a 1-byte dummy then.
    var use_bins = Int(rf_params.tree_params.max_n_bins) <= 256
    var d_bins = ctx.enqueue_create_buffer[DType.uint8](
        n_rows * n_cols if use_bins else 1
    )
    if use_bins:
        t_stage = instr.times.start()
        launch_bin_dataset(
            ctx,
            rebind[MutPointer[Scalar[O.DataT], MutUntrackedOrigin]](
                x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            ),
            d_bins,
            quantiles.quantiles_array,
            quantiles.n_bins_array,
            Int(rf_params.tree_params.max_n_bins),
            n_rows,
            n_cols,
            n_cols if row_major else 1,
            1 if row_major else n_rows,
        )
        instr.times.stop(ctx, "bin_dataset", t_stage)
        # DEVIATION 401 -- the pre-binned index matrix (DEVIATION 314),
        # one uint8 per (row, col), fully written by the kernel above.
        # Every level of every tree reads THIS; a divergence here explains
        # every histogram downstream of it.
        if instr.trace.enabled:
            instr.trace.record_device(ctx, "forest.binned", d_bins)

    var has_sw = len(sample_weight_host) > 0
    # DEVIATION 117, PORTED: cuML's shipped forest loop is
    # `#pragma omp parallel for num_threads(n_streams)` over trees with a
    # stream pool (`randomforest.cuh:336-367`), and their Python default
    # is n_streams=4 (`randomforestclassifier.py:94`). One Metal queue and
    # one host thread express the same overlap as K-WAY PIPELINING below:
    # K trees in flight, each suspended at its doSplit sync points, ONE
    # synchronize per cycle serving all of them. K=1 reproduces the serial
    # loop operation for operation.
    var k_streams = Int(rf_params.n_streams)
    if k_streams > Int(rf_params.n_trees):
        k_streams = Int(rf_params.n_trees)
    if k_streams < 1:
        k_streams = 1

    var sampler = RowSampler(
        ctx,
        rf_params.bootstrap,
        rf_params.seed,
        n_rows,
        n_sampled,
        has_sw,
        Int(rf_params.n_trees) if oob_score else 0,
        n_slots=k_streams,
    )
    if has_sw:
        sampler.prepare_weights(ctx, sample_weight_host)

    # `RowSampler::tree_sample_weight`, `randomforest.cuh:166-167`:
    #
    #     return bootstrap_ ? nullptr : sample_weight_;
    #
    # with their comment: "Use sample weights in impurity / objective
    # calculation only when bootstrapping is not enabled." THIS IS EASY TO
    # MISS AND CHANGES THE MODEL. When bootstrapping, the weights are
    # already expressed by DRAWING rows in proportion to them, so feeding
    # them to the objective as well would apply them twice. When not
    # bootstrapping, the objective is the only place they can act. A port
    # that always passed the weights down would double-count on the
    # default path and look merely "differently regularised".
    var objective_sees_weights = has_sw and not rf_params.bootstrap

    var forest = RandomForestMetaData[O.DataT, O.LabelT](
        List[TreeMetaDataNode[O.DataT]](),
        rf_params,
        Int32(n_cols),
    )

    # DEVIATION 313: ONE Builder for the whole forest, reset per tree.
    # Their per-tree Builder construction allocates from RMM's POOLED
    # resources, so it is pointer carving; a Metal buffer create per tree
    # is a driver cost their design never pays. See
    # `Builder.reset_for_tree` for the full argument. `builder_rows` is
    # what `sampler.n_selected` will hold in the loop -- constant across
    # one forest in every arm: three arms set it to `n_sampled`, and the
    # no-bootstrap weighted arm's value is fixed by `prepare_weights`
    # above. `reset_for_tree` re-checks it per tree and raises rather
    # than mis-sizing.
    var builder_rows = (
        sampler.n_selected if has_sw and not rf_params.bootstrap
        else n_sampled
    )
    # Pool-of-K: one Builder per pipeline slot, their stream pool's
    # per-stream workspaces.
    var builders = List[Builder[O]]()
    for _ in range(k_streams):
        builders.append(
            Builder[O](
                ctx,
                rf_params.tree_params,
                Int32(0),
                rf_params.seed,
                builder_rows,
                n_cols,
                Int32(n_unique_labels),
                scales,
            )
        )

    # DEVIATION 1908: ONE splits staging for all K slots, so the cycle
    # below reads every in-flight tree's split results back in a single
    # copy (`flush_splits_downloads`) instead of one host-priced enqueue
    # per slot. `builders[k]` owns slot k -- the flush indexes by that.
    var split_staging = SplitStaging(
        ctx, k_streams, builders[0].splits_capacity_bytes()
    )
    for k in range(k_streams):
        builders[k].adopt_shared_splits(split_staging, k)

    # `:337-367` -- their tree loop, K-WAY PIPELINED (DEVIATION 117; see
    # the block above `k_streams`). Trees finish out of order, so the
    # forest is preallocated and each tree lands at ITS index.
    var n_trees = Int(rf_params.n_trees)
    for _ in range(n_trees):
        forest.trees.append(
            TreeMetaDataNode[O.DataT](
                Int32(-1),
                Int32(0),
                Int32(0),
                Float64(0),
                List[Scalar[O.DataT]](),
                List[SparseTreeNode[O.DataT]](),
                Int32(0),
            )
        )

    var states = List[TreeState[O]]()
    var slot_tree = List[Int]()
    var next_tree = 0
    # PRIME: one tree per slot. A tree whose root is not expandable
    # (max_depth 0, min_samples_split too big) finishes inside
    # `begin_tree` and its slot immediately takes the next tree.
    for k in range(k_streams):
        if next_tree >= n_trees:
            break
        while next_tree < n_trees:
            t_stage = instr.times.start()
            sampler.sample(ctx, Int32(next_tree), k)
            instr.times.stop(ctx, "row_sampling", t_stage)
            # DEVIATION 401 -- the tree's sampled rows, the first per-tree
            # divergence point (a pure hash of (seed, tree_id), so K-free).
            if instr.trace.enabled:
                instr.trace.record_device(
                    ctx,
                    "tree" + String(next_tree) + ".rows",
                    sampler.selected_rows_[k],
                    sampler.n_selected,
                )
            builders[k].reset_for_tree(Int32(next_tree), sampler.n_selected)
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
                # `:151` -- the zero-weight arm selects FEWER rows than
                # `n_sampled_rows`; the builder must see the real count.
                Int32(sampler.n_selected),
                Int32(n_cols),
                sampler.rows_ptr(k),
                Int32(n_unique_labels),
                objective_sees_weights,
                # DEVIATION 314 -- shared read-only across all K slots.
                d_bins.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                use_bins,
            )
            var ts = builders[k].begin_tree(ctx, dataset, quantiles, instr)
            if ts.done:
                forest.trees[next_tree] = ts.tree.copy()
                _record_tree(instr, forest.trees[next_tree], next_tree)
                next_tree += 1
                continue
            states.append(ts^)
            slot_tree.append(next_tree)
            next_tree += 1
            break

    # THE CYCLE: one synchronize covers every in-flight tree's enqueued
    # phase; each consume step immediately enqueues that tree's next
    # phase (or its successor tree's first), so the queue is never empty
    # while work remains. This is their stream pool's overlap, minus the
    # host threads it never needed.
    var active = len(states)
    while active > 0:
        # DEVIATION 1908 -- every in-flight tree's pending splits
        # readback rides ONE copy. Enqueued here, after the previous
        # pass's kernels (the queue is in-order) and before the drain
        # that makes the host bytes readable; the first pass flushes the
        # prime loop's enqueues. When the loop exits, nothing is
        # pending: a slot records a pending count only by enqueueing a
        # phase, and a slot that enqueued one stays active.
        flush_splits_downloads(ctx, split_staging, builders)
        # DEVIATION 402 -- "device_wait" is the one drain that serves
        # every in-flight tree's enqueued phase; `stop_host` because the
        # queue is empty at the stamp by construction.
        t_stage = instr.times.start()
        ctx.synchronize()
        instr.times.stop_host("device_wait", t_stage)
        for k in range(len(states)):
            if slot_tree[k] < 0:
                continue
            if not builders[k].advance_tree(
                ctx, quantiles, states[k], instr
            ):
                continue
            while True:
                forest.trees[slot_tree[k]] = states[k].tree.copy()
                _record_tree(
                    instr, forest.trees[slot_tree[k]], slot_tree[k]
                )
                if next_tree >= n_trees:
                    slot_tree[k] = -1
                    active -= 1
                    break
                t_stage = instr.times.start()
                sampler.sample(ctx, Int32(next_tree), k)
                instr.times.stop(ctx, "row_sampling", t_stage)
                # DEVIATION 401 -- same checkpoint as the prime loop's.
                if instr.trace.enabled:
                    instr.trace.record_device(
                        ctx,
                        "tree" + String(next_tree) + ".rows",
                        sampler.selected_rows_[k],
                        sampler.n_selected,
                    )
                builders[k].reset_for_tree(
                    Int32(next_tree), sampler.n_selected
                )
                var dataset = DatasetView[O.DataT, O.LabelT](
                    rebind[MutPointer[Scalar[O.DataT], MutUntrackedOrigin]](
                        x.unsafe_ptr()
                        .unsafe_origin_cast[MutUntrackedOrigin]()
                    ),
                    y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    sample_weight.unsafe_ptr()
                    .unsafe_origin_cast[MutUntrackedOrigin](),
                    Int64(n_rows),
                    Int64(n_cols),
                    Int64(n_cols) if row_major else Int64(1),
                    Int64(1) if row_major else Int64(n_rows),
                    Int32(sampler.n_selected),
                    Int32(n_cols),
                    sampler.rows_ptr(k),
                    Int32(n_unique_labels),
                    objective_sees_weights,
                    # DEVIATION 314.
                    d_bins.unsafe_ptr().unsafe_origin_cast[
                        MutUntrackedOrigin
                    ](),
                    use_bins,
                )
                states[k] = builders[k].begin_tree(
                    ctx, dataset, quantiles, instr
                )
                slot_tree[k] = next_tree
                next_tree += 1
                if not states[k].done:
                    break

    ctx.synchronize()
    # Mojo frees a value at its LAST USE; the builders' buffers must
    # outlive every launch that read them, so they are released only
    # after the drain above. The shared splits staging (DEVIATION 1908)
    # is under the same rule: every slot's kernels wrote into it.
    _ = builders^
    _ = split_staging^

    # `randomforest_common.pyx:669-670` -- after the tree loop, and only
    # if it was asked for.
    if oob_score:
        t_stage = instr.times.start()
        compute_oob_score[O, oob_sabotage](
            ctx, forest, sampler, x, y, n_rows, n_cols, row_major
        )
        instr.times.stop(ctx, "oob", t_stage)

    # Mojo frees a value at its LAST USE, and every buffer above reached a
    # kernel as a raw pointer. These uses keep them alive past the final
    # synchronize. Measured hazard, not a precaution.
    _ = qr^
    _ = sampler^
    _ = d_bins^
    # DEVIATION 402 -- the stage table, printed only under
    # MOJOLEARN_STAGE_TIMES=1. `stop_host` because everything above has
    # drained.
    instr.times.stop_host("fit_total", t_fit)
    instr.times.report()
    return forest^
