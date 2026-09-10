# NumPy-free Python layer: the contract every module codes against

The 0.8.0 source uses the shared Array/buffer layer across estimators,
metrics, preprocessing, model selection and neural training. NumPy is absent
from runtime dependencies; it remains an optional test/diagnostic oracle.
Existing NumPy inputs still work through the buffer protocol. Outputs are
`mojolearn.Array`; use `numpy.asarray(result)` for a zero-copy NumPy view.
This is an API change from published 0.7.0 and requires new native builds.

GPU learners remain the product. Compiled host casting, layout conversion,
validation, checkpoint packing and row gathering are supporting operations,
not CPU implementations of learners. See
[the integration and performance roadmap](../../docs/lanes/NUMPY_FREE_RESIDUAL_2026-09-10.md)
for qualification still owed; removal of a dependency does not certify new
cross-vendor training results.

## Forest fit ownership (DEVIATION 2482)

RF and ExtraTrees fit export directly into caller-allocated `Array` buffers.
A temporary typed native handle owns the fitted model until export finishes;
the shared Python wrapper releases it even when allocation, export or validation
fails. The estimator retains the exported arrays without per-node Python objects
or a second packing pass. `MOJOLEARN_FOREST_EXPORT=legacy` and `verify` retain
same-fit diagnostic comparisons; the default is `into`. Metal same-fit model and
archive bytes are gated in all three modes. CUDA/HIP qualification remains owed;
the measured large-forest gain is export-only, not a whole-fit speed claim.

## `_buffer.py` (replaces `_arrays.py`; DEVIATION 2300)

```python
class Buf:                       # context manager around PyObject_GetBuffer/PyBuffer_Release
    addr: int; nbytes: int; itemsize: int; ndim: int
    shape: tuple[int, ...]; strides: tuple[int, ...] | None
    format: str                  # struct-style, e.g. 'f', 'd', 'i', 'l', 'q', 'B'
    readonly: bool; c_contiguous: bool; f_contiguous: bool
def view(obj, *, writable=False) -> Buf        # raises TypeError("mojolearn: <name> does not support the buffer protocol")
def addr(obj, *, name) -> int                  # writable buffer required (was _addr)
def addr_ro(obj, *, name) -> int               # any buffer (was _addr_ro)
def as_f32_c(obj, *, ndim, name) -> tuple[Array, bool]      # C-contiguous float32 Array + copied flag; converts from f64/i32/i64/lists; rejects other dtypes with the SAME messages _arrays.py used
def as_f32_colmajor(obj, *, name) -> tuple[Array, bool]     # column-major float32 (DEVIATION 1887 argument holds: f64->f32 is one round-to-nearest-even)
def as_i32_c(obj, *, ndim, name) -> tuple[Array, bool]
def as_i64_c(obj, *, ndim, name) -> tuple[Array, bool]
def as_f64_c(obj, *, ndim, name) -> tuple[Array, bool]
def empty(shape, dtype) -> Array               # dtype in {'<f4','<f8','<i4','<i8','<u4','<f2','<u1'}
def zeros(shape, dtype) -> Array
def full(shape, value, dtype) -> Array
def frombytes(raw: bytes, dtype, shape) -> Array    # copies
def all_finite(arr: Array) -> bool             # float32/float64 only; uses binding `_mojolearn.all_finite_f32/f64` when loadable, else a memoryview loop
```
Errors keep the exact wording of the messages they replace wherever a test asserts on them.

## `_array.py`: `class Array` (DEVIATION 2301)

Backing store: owned `array.array` or raw-allocated `ctypes` buffer, or a
borrowed buffer pinned for the lifetime of its views. Native converters fill
uninitialized owned storage before exposing an Array. Public surface, and nothing else:

- attributes: `shape` (tuple), `dtype` (str typestr `'<f4'`, `'<f8'`, `'<i4'`, `'<i8'`, `'<u4'`, `'<u1'`, `'<f2'`), `ndim`, `size`, `nbytes`, `itemsize`, `strides`, `order` ('C' or 'F'), `flags` (dict with C_CONTIGUOUS, F_CONTIGUOUS, WRITEABLE)
- `__array_interface__` (version 3: shape, typestr, data=(addr, readonly), strides, version) so `numpy.asarray(a)` is ZERO-COPY when a caller has NumPy
- `__buffer__(flags)` returning a memoryview (Python 3.12+) on Python 3.12+ where the requested layout can be represented; older Python uses `_buffer.view` explicitly
- `tobytes()`, `tolist()` (nested lists by shape), `copy()`, `astype(dtype)` (round-to-nearest-even for f64->f32; exact for widening), `reshape(shape)` (C order view over the same buffer), `ravel()`, `__len__`, `__iter__` (rows for ndim 2, scalars for ndim 1), `__getitem__` with int / slice / tuple of them (returns Python scalar for full indexing, Array otherwise, copying), `__eq__` elementwise -> Array of '<u1'
- `min()`, `max()`, `sum()` (Python float accumulation, DOCUMENTED as host reductions that are NOT part of any identity claim), `argmax()` first-max-wins
- `__repr__` like `Array(shape=(3, 2), dtype='<f4')`
- classmethod `from_buffer(obj)` zero-copy view over any buffer-protocol object (no ownership), classmethod `from_list(nested, dtype)`
No arithmetic operators, no broadcasting, no fancy indexing. Every public estimator returns `Array` where it returned `ndarray`; `classes_` becomes a Python list.

## `_serialize.py` (DEVIATION 2302)

Same public functions and the SAME BYTES ON DISK: a hand-written NPY v1.0 codec
(magic `\x93NUMPY`, version 1.0, ASCII header dict with `descr`, `fortran_order`,
`shape`, padded with spaces to a 64-byte-aligned total length ending in `\n`) inside
the existing deterministic zip (pinned 1980-01-01 timestamps, sorted members,
ZIP_STORED). `read_npz` returns a dict of `Array`; `exact()` keeps refusing dtype
mismatches with the same messages. A model saved by 0.6.x loads with 0.7 and vice
versa; a test proves both directions using NumPy from the `test` extra.

## New native helpers (DEVIATION 2303; bindings/_mojolearn.mojo, the base extension)

```
all_finite_f32(addr: Int, n: Int) -> Int     # 1 if every element is finite
all_finite_f64(addr: Int, n: Int) -> Int
column_mean_f64(x_addr: Int, rows: Int, cols: Int, out_addr: Int)   # sequential row-order float64 accumulation of a C-contiguous float32 [rows, cols] matrix; out is float64[cols]
```
`linear_model.py` uses `column_mean_f64` where it used `x.mean(axis=0, dtype=np.float64)`
(and the weighted variant is computed in Python from the same helper applied to the
weighted copy). That moves OLS/ridge centering off NumPy's blocked reduction and onto
a defined order; the OLS reference cards are RE-BASELINE OWED and must be rerun on all
three vendors (recorded in this file's owed list, not hidden).
`ensemble.py` routes the sigmoid of every tier through the existing `gbdt_sigmoid`
binding (fast/deterministic predict_proba bits move; IDENTICAL does not).
`randomforest.py` computes `max_features='log2'` from `math.log2` in float64 and then
passes the resulting integer feature count in the params slot if the kernel accepts a
count, else keeps the fraction and records DEVIATION 2304 with the boundary case.

## Rules
- Nothing imports numpy at module import time. `_verify.py` may import it lazily under a
  clear "diagnostics need numpy from the test extra" error.
- Python loops over big data are forbidden in fit/predict paths: any per-element scan
  goes to a native helper or is a memoryview.cast slice operation; label encoding and
  argmax over class counts are the only permitted Python loops (O(classes) or O(rows)
  for labels).
- Tests keep NumPy (optional `test` extra); `np.asarray(result)` must be zero-copy.
- Every changed function gets a DEVIATION number from its agent's range in a comment.
