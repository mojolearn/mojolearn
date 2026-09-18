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
