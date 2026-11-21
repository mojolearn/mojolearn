# SPDX-License-Identifier: Apache-2.0
# UNPACK of `SVC.fit` (bindings/svc_kernel_native_door.mojo sequences the
# operation): the parameter checks and the settings record, the admitted X
# (host words, or a model-selection fold the device entry gathers), the one
# label encoding (classes, int32 codes, the binary machine's float32
# labels), the bounds' operands, the probability seed, and the typed plan
# RUN reads (svm/svc_kernel_fit_run.mojo SvcFitPlan). Every check of the old
# `_svc_fit` / `_as_labels` / `_c_rows` is here, in the same order and words.
# After RUN: the refusals of what the solve reported.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family, admit_family_copied, admitted_array
from core.admitted_input import admitted_cast_f32
from core.cast_in import CAST_KIND_SIGNED
from bindings.native_loaded_module_door import native_loaded_module
from bindings.label_admission_native_unpack import labels_primitive_kind
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.classical_arrays_door import classical_address
from bindings.classical_labels_door import classical_encode_labels
from bindings.family_common_unpack import FAMILY_F32P, family_repr
from bindings.family_common_pack import family_call
from bindings.svc_common_unpack import svc_field, svc_float, svc_shape_of
from bindings.svc_common_run import svc_round_f32
from bindings.device_rows_door_run import DoorRowsStore
from bindings.svm_keys_unpack import svm_keys
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from core.svc_native_api import SvcDrivers
from svm.svc_kernel_fit_run import SvcFitPlan, SvcFitResult, SVC_FIT_PAIR_LABELS
from bindings.svc_kernel_rows_unpack import _addr, _svm_fit_admit, _svm_fit_host_rows, _svm_fit_refuse, _svm_x_scan
from bindings.svc_kernel_params_unpack import svm_settings_unpack, svm_scale_gamma_admit, svm_weights_unpack

comptime _DEV636_Y = "mojolearn SVC: y contains a non-finite value (DEVIATION 636: a NaN or inf cannot be fitted; a computed NaN carries a vendor-specific payload and cannot sit in a hashed stage)"


struct SvcLabelsUnpacked(Movable):
    """`_svm_impl._as_labels`: the class list, the int32 class code of every
    row, and for two classes the float32 labels the solver reads (None for
    more) with the host's sorted label pair (`p0`, `p1`)."""
    var labels: PythonObject
    var classes: PythonObject
    var codes: PythonObject
    var p0: Float64
    var p1: Float64

    def __init__(out self, labels: PythonObject, classes: PythonObject, codes: PythonObject, p0: Float64, p1: Float64):
        self.labels = labels
        self.classes = classes
        self.codes = codes
        self.p0 = p0
        self.p1 = p1


def svc_labels_unpack(y: PythonObject, context: PythonObject) raises -> SvcLabelsUnpacked:
    # The one label encoder (classical_encode_labels; lane a2-admit-family
    # deleted the SVM's svc_encode_1d), after the 1-D test it made first.
    var shape = PythonObject(None)
    var classes = PythonObject(None)
    var codes = PythonObject(None)
    try:
        shape = svc_shape_of(y, context)
        if len(shape) == 1:
            var encoded = classical_encode_labels(y, context)
            classes = encoded.classes
            codes = encoded.codes
    except error:
        if String(error).find("NaN label") < 0:
            raise error
        raise Error(_DEV636_Y)
    if len(shape) != 1:
        raise Error("mojolearn SVC: y must be 1-D, got " + String(len(shape)) + "-D shape " + family_repr(shape))
    if len(classes) < 2:
        raise Error("mojolearn SVC: y has " + String(len(classes)) + " class; at least two are needed")
    if len(classes) > 2:
        return SvcLabelsUnpacked(PythonObject(None), classes, codes, 0.0, 1.0)
    var real = True
    for i in range(len(classes)):  # small-loop(classes: the two labels)
        if labels_primitive_kind(py_type(classes[i]), context) < 2:
            real = False
    if not real:
        # non-numeric labels: the solver sees each row's code, 0.0 or 1.0
        var n = Int(py=codes.size)
        # The int32 codes (0 or 1) as float32, one cast in row tasks into an
        # uninitialised Array (lane a2-admit-family; codes 0 and 1 convert
        # exactly).
        var f = admitted_array(Python.tuple(PythonObject(n)), "<f4", context)
        if n > 0:
            _ = admitted_cast_f32(Int(py=classical_address(codes)), 4, 0, n, 1, 4, CAST_KIND_SIGNED,
                                  FAMILY_F32P(unsafe_from_address=Int(py=classical_address(f))))
            _ = codes  # keep the Python object alive through the native call above
            _ = f  # keep the Python object alive through the native call above
        return SvcLabelsUnpacked(f, classes, codes, 0.0, 1.0)
    # The finiteness test rides in the admission (lane a2-admit-family: was a
    # second pass through the base binding's all_finite_f32).
    var f = admit_family(y, 1, "y", context, False, finite=True, refusal=_DEV636_Y).array
    return SvcLabelsUnpacked(f, classes, codes, svc_float(classes[0]), svc_float(classes[1]))


struct SvcFitUnpacked(Movable):
    """A fit: the plan, the drivers, the instance dictionary, X through its
    pin, the class list, and `owners`, a Python list that keeps the labels,
    the codes and the weights alive until PACK. `copied` is the
    `input_copied_` flag PACK publishes; `fold` the device fold of X, read by
    the refusal of a flagged X."""
    var fields: PythonObject
    var x_pin: ArrayPin
    var x_meta: ArrayMetadata
    var owners: PythonObject
    var classes: PythonObject
    var p0: Float64
    var p1: Float64
    var fold: DoorRowsStore
    var copied: Bool
    var plan: SvcFitPlan
    var drivers: SvcDrivers

    def __init__(out self, fields: PythonObject, var x_pin: ArrayPin, var x_meta: ArrayMetadata, owners: PythonObject,
                 classes: PythonObject, p0: Float64, p1: Float64, fold: DoorRowsStore, copied: Bool,
                 var plan: SvcFitPlan, drivers: SvcDrivers):
        self.fields = fields
        self.x_pin = x_pin^
        self.x_meta = x_meta^
        self.owners = owners
        self.classes = classes
        self.p0 = p0
        self.p1 = p1
        self.fold = fold
        self.copied = copied
        self.plan = plan^
        self.drivers = drivers.copy()


def svc_fit_unpack(model: PythonObject, args: PythonObject, context: PythonObject,
                   drivers: SvcDrivers) raises -> SvcFitUnpacked:
    """`SVC.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var fields = py_fields(model)
    var hp = svm_settings_unpack(model, fields, True, context)
    # Lane a8a: the GPU binding's binary fit takes X on the device (a fold
    # gathered there, host words copied once) and tests it there; every
    # other route tests the host words here (`_svm_x_scan`). One wording.
    var entry = drivers.has_device
    var fold = DoorRowsStore.none()
    var x_pin = ArrayPin()
    var x_meta = _svm_fit_admit(args[1], "SVC", entry, context, fold, x_pin)
    var nk = svm_keys()
    var copied = True if fold.active() else admit_family_copied(x_meta, args[1])
    var parsed = svc_labels_unpack(args[2], context)
    var binary = parsed.labels is not None
    var n_rows = x_meta.shape[0]
    var n_cols = x_meta.shape[1]
    var n_y = Int(py=parsed.labels.shape[0]) if binary else Int(py=parsed.codes.shape[0])
    if n_y != n_rows:
        raise Error("mojolearn SVC: y has " + String(n_y) + " entries, X has " + String(n_rows) + " rows")
    if hp.precomputed and n_cols != n_rows:
        raise Error("mojolearn SVC: kernel='precomputed' needs the square n x n "
                    + "kernel matrix as X, got " + String(n_rows) + " x " + String(n_cols))
    # The device-X route: one binary machine on a computed kernel. The
    # one-vs-one pairs and Platt's folds gather host rows of X, and a
    # precomputed X is gathered by rows and columns: those read host words.
    var device_x = entry and binary and not hp.precomputed and not hp.probability
    if fold.active() and not device_x:
        x_meta = _svm_fit_host_rows(args[1], context, x_pin)
        fold = DoorRowsStore.none()
    var plan = SvcFitPlan(hp.fit_operation(1.0, n_cols))
    if device_x:
        var resolved = hp.device_gamma(n_cols)
        plan.gamma = resolved[0]
        plan.scale_gamma = resolved[1]
    else:
        _svm_x_scan(x_meta, "SVC")
        svm_scale_gamma_admit(hp, drivers, x_meta)
        plan.gamma_kind = hp.gamma_kind
        plan.gamma = hp.gamma
    var weights = svm_weights_unpack(n_rows, args[3], py_get(fields, nk[].k_class_weight, PythonObject(None)),
                                     parsed.classes, parsed.codes, "SVC", context)
    var owners = Python.list()
    _ = py_call(owners.append, parsed.labels)
    _ = py_call(owners.append, parsed.codes)
    _ = py_call(owners.append, weights.owners)
    plan.x = x_meta.source.addr
    plan.rows = n_rows
    plan.cols = n_cols
    plan.cells = x_meta.count
    plan.base_id = fold.base_id
    plan.base_rows = fold.base_rows
    plan.index_id = fold.index_id
    plan.device_form = device_x
    plan.classes = len(parsed.classes)
    plan.labels = _addr(parsed.labels) if binary else 0
    plan.codes = _addr(parsed.codes)
    plan.weighted = weights.weighted
    plan.sample_weights = weights.sample_weights
    plan.class_weights = weights.class_weights
    plan.probability = hp.probability
    if hp.probability:
        # The key of probability's fold shuffle, an int in [0, 2**64) checked
        # by validation (None: 0), as two 32-bit halves.
        var rs = svc_field(fields, nk[].k__hp_random_state, "_hp_random_state")
        if rs is not None:
            var operator = native_loaded_module("operator")
            var low = Int(py=family_call(operator.and_, rs, PythonObject(0xFFFFFFFF)))
            var high = Int(py=family_call(operator.rshift, rs, PythonObject(32)))
            plan.seed = (UInt64(high) << UInt64(32)) | UInt64(low)
    return SvcFitUnpacked(fields, x_pin^, x_meta^, owners, parsed.classes, parsed.p0, parsed.p1, fold, copied, plan^,
                          drivers)


def svc_fit_refused(unpacked: SvcFitUnpacked, result: SvcFitResult) raises:
    """After the solves: a fit X the device entry flagged, in admit_x's
    wording ('Input X contains NaN or infinity (SVC).'); then THE LABEL PAIR
    IS CHECKED, NOT ASSUMED (the solver's own host sort)."""
    if unpacked.plan.classes > 2:
        if result.status == SVC_FIT_PAIR_LABELS:
            raise Error("RuntimeError: mojolearn SVC: the solver's label pair for classes ("
                        + family_repr(unpacked.classes[result.bad_i]) + ", " + family_repr(unpacked.classes[result.bad_j])
                        + ") is not (0.0, 1.0); the class mapping cannot be trusted")
        return
    if unpacked.plan.device_form:
        _svm_fit_refuse(Int(result.solved.info[5]), Int(result.solved.info[6]), unpacked.fold, "SVC")
    var label0 = svc_round_f32(result.solved.info[3])
    var label1 = svc_round_f32(result.solved.info[4])
    if label0 != svc_round_f32(unpacked.p0) or label1 != svc_round_f32(unpacked.p1):
        raise Error("RuntimeError: mojolearn SVC: the solver's label pair "
                    + "(" + family_repr(PythonObject(label0)) + ", " + family_repr(PythonObject(label1)) + ") does not match the "
                    + "host's sorted unique (" + family_repr(PythonObject(unpacked.p0)) + ", "
                    + family_repr(PythonObject(unpacked.p1)) + "); the class mapping cannot be trusted")
