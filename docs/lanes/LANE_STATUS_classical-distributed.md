# Classical distributed inference and fit

Branch: lane/classical-distributed. Worktree: classical-distributed.

## Forecast checkpoint

Public import: `from mojolearn.parallel_forecasting import predict_arima,
forecast_arima, predict_exponential_smoothing, forecast_exponential_smoothing`.

Whole series are sent to ordered persistent GPU workers. Each worker receives
only its series' fitted state; results preserve ARIMA series-major and
Holt-Winters time-major/single-series/index shapes. Uneven shards preserve every
component byte. Failures close the pool and leave the source estimator untouched.
ARIMA exogenous models are explicitly deferred. Holt-Winters in-sample arithmetic
is host arithmetic in the existing binding, so this GPU API explicitly refuses
in-sample ranges. No public parallel CPU promise is added.

Validation: 20 binding-free partition/assembly tests passed. These exercise
transported state, ordering, component slicing, uneven shards, original state
immutability, shape contracts, worker failure cleanup and invalid input. They
are not numerical GPU evidence. NVIDIA/AMD one/two/reversed GPU, native fault,
installed-wheel replay and verifier admission remain owed. No hardware rental
or qualification claim in this checkpoint.

Next: class-level GPC partition; IVF requires preserving global coarse probe
selection with disjoint candidate storage, not independently trained indexes.

## Class-level GPC checkpoint

Public import: `from mojolearn.parallel_gaussian_process import
fit_gaussian_process_classifier, predict_gaussian_process_classifier`.
One-vs-rest class problems use the unchanged binary solver. Predictions ship only
one class covariance state to each worker and fold class probabilities in the
original float64 order, including zero-sum handling, ties and binary signs.
Fit publishes complete state atomically after all classes return. Full training
input is replicated; one binary covariance still must fit on a device. This
addresses multiclass throughput, not a larger single binary fit.

Validation: combined forecasting/GPC suite: 27 passed, binding-free. GPC tests
compare the original fit and composition against scheduled binary tasks with
numeric and string labels, zero probabilities, exact ties, binary signs and
failing tasks. Native numerical, physical multi-GPU and installed replay are
still owed.

## IVF implementation constraint

Existing native saved-index search rejects partitions with fewer stored rows
than global centers (`ivf_validate_index_arrays`) and queries with fewer than k
candidates (`ivf_flat_search`). Keeping global coarse centers and zeroing other
lists therefore cannot safely compose distributed search. A correct new native
entry must preserve global probe selection, admit disjoint candidate storage,
return an explicit valid count for empty/short candidate shards, and merge on
squared-distance/original-ID keys before Euclidean rooting. Independently built
indexes or replicated full indexes would not meet the requested capacity scope.
