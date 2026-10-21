# SPDX-License-Identifier: Apache-2.0
# Native checkpoint and verification-sidecar writer. The complete save,
# including seeded inputs, model replay and consumed output digests, is one
# native operation. It uses the existing model-verification wire format.
from std.collections.string import chr
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session
from autograd.program_arrays import output_array
from autograd.program_pyapi import ops_program_execute_py
from autograd.model_checkpoint import ops_save_program_py
from autograd.model_verify import verify_digest_py, verify_file_sha256_py, verify_model_info_py, verify_fill_py
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.vendor import COMPILED_VENDOR
from bindings.native_json import native_path
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _quote(text: String) -> String:
    var result = text.replace("\\", "\\\\").replace("\"", "\\\"")
    var digits = "0123456789abcdef"
    for k in range(32):  # small-loop(32: JSON control-character alphabet): text encoding, no tensor data
        result = result.replace(chr(k), "\\u00" + String(digits[byte=k // 16]) + String(digits[byte=k % 16]))
    return "\"" + result + "\""


def _array_entry(a: PythonObject, name: String, with_dtype: Bool) raises -> String:
    var result = "{\"name\":" + _quote(name) + ",\"shape\":["
    for k in range(len(a.shape)):  # small-loop(shape: tensor rank): result metadata
        if k > 0:
            result += ","
        result += String(Int(py=a.shape[k]))
    result += "]"
    if with_dtype:
        # ndarray dtype.__str__/name enter NumPy Python helpers. Saved
        # verification inputs are native int32 ids or float32 values.
        result += ",\"dtype\":" + _quote("int32" if String(a.dtype.kind) == "i" else "float32")
    var digest = verify_digest_py(a.__array_interface__["data"][0], a.nbytes)
    return result + ",\"digest\":" + _quote(String(digest)) + "}"


def ops_program_save_native_py[E: Exec & Movable & Deinitable, KIND: Int](
    program: PythonObject, path: PythonObject,
) raises -> PythonObject:
    var s = ops_session[E, KIND](Int(py=program._s._id))
    var p = s[].program(Int(py=program._id))
    var normalized = native_path(path)
    var file = String(py=normalized)
    _ = ops_save_program_py[E, KIND](program._s._id, program._id, normalized)
    var arrays = Python.list()
    var input_json = "["
    var b = Python.import_module("builtins")
    var have_inputs = Bool(py=b.hasattr(program, "inputs")) and len(program.inputs) == p.n_in
    var have_outputs = Bool(py=b.hasattr(program, "outputs")) and len(program.outputs) == len(p.outs)
    for k in range(p.n_in):  # driver-loop: over verification input owners
        var a = output_array(p.in_shapes[k], "int32" if p.in_kinds[k] == 1 else "float32")
        if p.in_kinds[k] != 1:
            _ = verify_fill_py(a.__array_interface__["data"][0], a.size, PythonObject(0), PythonObject(k))
        # bytearray zero-initialization supplies the established valid id 0.
        arrays.append(a)
        if k > 0:
            input_json += ","
        var name = String(program.inputs[k][0]) if have_inputs else "input" + String(k)
        input_json += _array_entry(a, name, True)
    input_json += "]"
    var outs = ops_program_execute_py[E, KIND](program._s._id, program._id, program._s._id,
                                               arrays, PythonObject(False), PythonObject(False), PythonObject(False))
    var output_json = "["
    for k in range(len(p.outs)):  # driver-loop: over verification output digests
        if k > 0:
            output_json += ","
        var a = outs if len(p.outs) == 1 else outs[k]
        var name = String(program.outputs[k][0]) if have_outputs else "output" + String(k)
        output_json += _array_entry(a, name, False)
    output_json += "]"
    var sha = verify_file_sha256_py(normalized)
    var model = verify_model_info_py(normalized)
    var basename = file.split("/")
    var package = Python.import_module("mojolearn._version")
    var uname = Python.import_module("os").uname()
    var vendor = String("cpu") if TARGET_COLUMN == COLUMN_CPU else String(COMPILED_VENDOR)
    var mode = "identical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "fast"
    var report = "{\"format\":\"mojolearn-model-verify/1\",\"install\":{\"mojolearn_version\":"
    report += _quote(String(package.__version__)) + ",\"numeric_mode\":" + _quote(mode)
    report += ",\"vendor\":" + _quote(vendor) + ",\"device\":" + _quote(vendor + " native executor")
    report += ",\"platform\":" + _quote(String(uname[0]) + " " + String(uname[4])) + "},\"model\":{\"file\":"
    report += _quote(String(basename[len(basename) - 1])) + ",\"kind\":" + _quote(String(model[0]))
    report += ",\"sha256\":" + _quote(String(sha[0])) + ",\"bytes\":" + String(Int(py=sha[1]))
    report += ",\"description_hash\":" + _quote(String(model[1])) + "},\"input_source\":\"seeded\",\"seed\":0,\"inputs\":"
    report += input_json + ",\"outputs\":" + output_json + "}\n"
    with open(file + ".verify.json", "w") as fh:
        fh.write(report)
    return PythonObject(None)
