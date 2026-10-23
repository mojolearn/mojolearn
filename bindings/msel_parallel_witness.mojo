# SPDX-License-Identifier: Apache-2.0
# Native completed-fold capture. The testing controller retains these actual
# bytes after the public operation; no scorer/capture callback enters a fold.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.family_common import family_call
from bindings.msel_estimator_native import msel_estimator_call, msel_estimator_plan
from bindings.classical_attributes import classical_check_layout
from core.sha256 import Sha256


def msel_parallel_binding_hash(context: PythonObject) raises -> PythonObject:
    var metadata = context["family_msel_parallel"]
    var saved = py_get(metadata, "binding_sha256")
    if saved is not None:
        return saved
    var path = metadata["binding_path"]
    if py_type(path) is not py_type(PythonObject("")):
        raise Error("ImportError: native CV capture has no connected GBDT artifact")
    var digest = Sha256()
    with open(String(path), "r") as stream:
        var bytes = stream.read_bytes()
        digest.update_list(bytes)
    var result = PythonObject(digest.hexdigest())
    py_setitem(metadata, "binding_sha256", result)
    return result


def _artifacts(context: PythonObject) raises -> PythonObject:
    var metadata = context["family_msel_parallel"]
    if metadata["artifact_hashes"] is not None:
        return metadata["artifact_hashes"]
    var result = Python.dict()
    var paths = metadata["artifact_paths"]
    for key in [String("cv"), String("trees"), String("archive"), String("gbdt")]:  # small-loop(four connected compiled owners)
        var path = py_get(paths, key)
        if py_type(path) is not py_type(PythonObject("")):
            raise Error("ImportError: CV capture is missing the actual " + key + " owner artifact")
        var hash = Sha256()
        with open(String(path), "r") as stream:
            var bytes = stream.read_bytes()
            hash.update_list(bytes)
        py_setitem(result, key, hash.hexdigest())
    py_setitem(metadata, "artifact_hashes", result)
    return result


def msel_parallel_capture(model: PythonObject, x: PythonObject, y: PythonObject,
                          score: PythonObject, inventory: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var metadata = context["family_msel_parallel"]
    if py_type(metadata["binding_vendor"]) is not py_type(b.len) or py_type(metadata["binding_mode"]) is not py_type(b.len):
        raise Error("TypeError: CV capture requires native vendor and numeric-mode witnesses")
    if String(py_call(metadata["binding_vendor"])) != String(metadata["vendor"]) or Int(py=py_call(metadata["binding_mode"])) != 1:
        raise Error("RuntimeError: CV capture requires a matching identical GBDT binding")
    var plan = msel_estimator_plan(model, context)
    var fields = py_call(plan["fields_get"], model, py_type(model))
    var learner = fields["_learner_"]
    var fixed = plan["context"]["cv_fixed_owner"]
    if py_type(learner) is not metadata["gbdt_class"]:
        raise Error("TypeError: CV capture requires its connected fitted GBDT learner")
    classical_check_layout(learner, fixed["learner_admission"])
    var learned = py_call(fixed["learner_fields_get"], learner, py_type(learner))
    var archive_entry = metadata["archive_entry"]
    var trees_entry = metadata["gbdt_entry"]
    if py_type(archive_entry) is not py_type(b.len) or py_type(trees_entry) is not py_type(b.len):
        raise Error("TypeError: native CV capture requires compiled archive and tree entries")
    var stream = py_call(metadata["bytesio_type"])
    var archive = PythonObject(None)
    var restored = PythonObject(None)
    try:
        _ = family_call(archive_entry, learner, "archive_save", Python.tuple(stream), metadata["archive_context"])
        archive = py_call(metadata["bytesio_getvalue"], stream)
        _ = py_call(metadata["bytesio_seek"], stream, 0)
        restored = family_call(archive_entry, metadata["gbdt_class"], "host_archive_load",
                               Python.tuple(stream), metadata["archive_context"])
    finally:
        _ = py_call(metadata["bytesio_close"], stream)
    var result = Python.dict()
    py_setitem(result, "X_test", x)
    py_setitem(result, "y_test", y)
    py_setitem(result, "archive", archive)
    py_setitem(result, "model_text", learned["model_"])
    py_setitem(result, "predict", msel_estimator_call(model, "predict", Python.tuple(x), context))
    py_setitem(result, "raw_predict", family_call(trees_entry, learner, "trees_gbdt_predict", Python.tuple(x), metadata["gbdt_context"]))
    py_setitem(result, "reload_predict", family_call(trees_entry, restored, "trees_gbdt_predict", Python.tuple(x), metadata["gbdt_context"]))
    py_setitem(result, "loss_curve", learned["loss_curve_"])
    py_setitem(result, "score", score)
    py_setitem(result, "inventory", inventory)
    py_setitem(result, "binding_sha256", msel_parallel_binding_hash(context))
    py_setitem(result, "binding_path", metadata["binding_path"])
    py_setitem(result, "selected_arch", metadata["selected_arch"])
    py_setitem(result, "selected_arch_how", metadata["selected_arch_how"])
    py_setitem(result, "binding_vendor", metadata["vendor"])
    py_setitem(result, "binding_mode", 1)
    py_setitem(result, "native_artifacts", _artifacts(context))
    return result
