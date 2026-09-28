# NumPy policy audit of consolidated Apple sources

The consolidated source at `3cc07eabf` still fails the strict shipped-Python
NumPy audit: **23 import sites in 19 files**. This is a release blocker under
the current policy that runtime estimators do not require NumPy. A successful
base import without NumPy, or installation of NumPy through `mojolearn[verify]`,
does not establish that the expanded estimator APIs are NumPy-free.

The policy was not relaxed for these failures. `numpy_errors` still permits
NumPy imports only in named independent verification modules. The dependency
audit permits lazy sklearn interoperability hooks and a narrowly scoped,
conditional `scipy.sparse` return adapter for `AdditiveChi2Sampler.transform`.
The latter converts an already-sparse caller input back into CSR; dense inputs
do not execute that import. The redundant direct NumPy import in that adapter
was removed by passing the float32 dtype directly to SciPy's constructor.

## Remaining conditional imports

All locations below are relative to `python/mojolearn/` at the audited commit.

| Location | Trigger and use | Why this cannot be ignored |
| --- | --- | --- |
| `_expansion_ann.py:26` | `_ann_mask`: construct/validate boolean filters and convert them to contiguous int32 | Executed by ANN filtering paths; even its `filter=None` branch creates a NumPy array. |
| `_expansion_ann.py:248` | `TSNE._init`: random/PCA/caller initialization arrays, PCG generator, and scaling | Seeded random streams and float32 rounding must be preserved by any replacement. |
| `_expansion_ann.py:273` | `TSNE.fit`: array preparation and output buffers | Required for this estimator, not merely optional caller interoperability. |
| `_expansion_ann.py:408` | `CagraIndex._search_filtered`: candidate masks, buffer conversion, selection/fill | Required for filtered searches. |
| `_expansion_ann.py:611` | `refine`: candidate conversion and float32 square root for Euclidean distances | Replacing arithmetic needs bitwise validation, including distance ordering and rounding. |
| `_expansion_cluster.py:57` | `_callable_init`: try NumPy conversion and `RandomState`, otherwise retain the original input/seed | It tolerates missing NumPy, but changes the callback's array/RNG contract depending on availability; the strict import audit still rejects it. |
| `_expansion_cnn.py:27` | `_np()` called throughout CNN layers/trainers for tensors, RNG, initialization, and arithmetic | This is a substantive runtime dependency. A lazy import does not make these operations NumPy-free. |
| `_expansion_neighbors.py:785` | `_random_state(None)`: use NumPy's global legacy generator | Integer seeds already use the package's `_LegacyRandomState`; explicit RNG objects are adapted directly. Replacing only `None` changes coupling to `numpy.random.seed` and must be documented and tested, not silently substituted. |
| `_ivf_impl.py:215` | `IVFIndex` filtered query: boolean validation and int32 mask conversion | The unfiltered query avoids it, but filtered queries require NumPy. |
| `resample.py:331` | `_take`: array-protocol inputs converted/indexed through NumPy | Lists avoid it. Array-protocol support remains a runtime path and may include package-owned arrays. |

## Feature-module imports

These modules import NumPy at module scope. The sequence export layer delays
loading their modules until the corresponding feature is requested, so they
need not break a base package import. Once requested, however, these features
require NumPy for arrays, buffers, packing, initialization or parameter updates:

| Module | Import line | Feature |
| --- | ---: | --- |
| `_x_sequence_autoarima.py` | 31 | Automatic ARIMA |
| `_x_sequence_croston.py` | 8 | Croston forecasting |
| `_x_sequence_ets.py` | 21 | Exponential smoothing |
| `_x_sequence_garch.py` | 15 | GARCH |
| `_x_sequence_mlp.py` | 15 | Sequence MLP training |
| `_x_sequence_moe.py` | 14 | Mixture-of-experts layers |
| `_x_sequence_norm.py` | 10 | Layer normalization |
| `_x_sequence_optim.py` | 14 | Optimizers that explicitly accept mutable NumPy float32 parameters |
| `_x_sequence_prophet.py` | 25 | Prophet-style forecasting |
| `_x_sequence_rnn.py` | 19 | RNN/GRU/LSTM training, including seeded initialization |
| `_x_sequence_stl.py` | 16 | STL decomposition |
| `_x_sequence_theta.py` | 14 | Theta forecasting |
| `_x_sequence_var.py` | 13 | Vector autoregression |

## Validation and scope

On the consolidated source, the metadata audit, wheel-audit regression tests
and split-wheel packaging tests passed: **57 tests and 19 subtests in 2.64s**.
Those tests validate the audit machinery and the narrow metadata/CSR repair;
they do not assert that every shipped module satisfies the NumPy policy.
The separate strict scan above explicitly fails and retains all 23 failures.
No GPU fit or new estimator arithmetic was executed for this audit.

The installed-wheel metadata check now permits only NumPy guarded by exactly
`extra == "verify"` with the extra declared, plus exact same-version GPU plugin
requirements when the installed core carries the split-package marker. Base
NumPy requirements, unrelated dependencies, broadened extra markers, missing
plugin pins, loose pins and duplicate pins remain rejected. The installed
qualification still requires NumPy to be absent from the environment.

Before claiming all runtime APIs are NumPy-free, replace these runtime paths
with validated package-owned implementations or make a deliberate documented
policy change for optional features. CNN/TSNE/RNN seeded initialization and
host arithmetic especially need explicit old/new bitwise comparisons. Merely
adding the files to the verification-import allowlist would misrepresent
estimator code as independent verification and is not an acceptable repair.
