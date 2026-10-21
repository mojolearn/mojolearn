# SPDX-License-Identifier: Apache-2.0
# Native checkpoint and verification-sidecar writer. The complete save,
# including seeded inputs, model replay and consumed output digests, is one
# native operation. It uses the existing model-verification wire format.
# The Python door (program_save_pyapi.mojo) passes the names and the
# install's version and platform strings.
from std.collections.string import chr
from sequence.exec_trait import Exec
from core.cast_in import CastIn, CAST_KIND_FLOAT, CAST_KIND_SIGNED
from autograd.ops_api import ops_session
from autograd.program import PROGRAM_IN_IDS, same_shape, shape_text
from autograd.program_io import HostArray, ProgramInput
from autograd.program_entries import program_execute_prepare, program_execute_run
from autograd.model_checkpoint import ops_save_program
from autograd.model_verify import verify_digest, verify_file_sha256, verify_model_info, verify_fill
from autograd.rankn import ShapeN
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _quote(text: String) -> String:
    var result = text.replace("\\", "\\\\").replace("\"", "\\\"")
    var digits = "0123456789abcdef"
    for k in range(32):  # small-loop(32: JSON control-character alphabet): text encoding, no tensor data
        result = result.replace(chr(k), "\\u00" + String(digits[byte=k // 16]) + String(digits[byte=k % 16]))
    return "\"" + result + "\""


def _array_entry(a: HostArray, name: String, dtype: String) raises -> String:
    """One verification-report array entry; `dtype` empty leaves it out."""
    var result = "{\"name\":" + _quote(name) + ",\"shape\":["
    for k in range(a.shape.ndim):  # small-loop(shape: tensor rank): result metadata
        if k > 0:
            result += ","
        result += String(a.shape.d[k])
    result += "]"
    if dtype.byte_length() > 0:
        result += ",\"dtype\":" + _quote(dtype)
    var digest = verify_digest(Int(a.data.unsafe_ptr()), a.shape.count() * 4)
    return result + ",\"digest\":" + _quote(digest) + "}"


def _seeded_input(a: HostArray, ids: Bool, what: String) raises -> ProgramInput:
    """The program input of a seeded verification array, described as the
    C-order float32 (or int32 ids) array of its shape it stands for."""
    var nd = a.shape.ndim
    if nd > 8:
        raise Error("ValueError: tensor rank must be at most 8")
    if a.shape.count() == 0:
        raise Error("ValueError: input must not be empty")
    var dims = List[Int]()
    if nd == 0:
        dims.append(1)
        dims.append(1)
    elif nd == 1:
        dims.append(a.shape.d[0])
        dims.append(1)
    else:
        for k in range(nd):  # small-loop(nd: input tensor rank): shape metadata, not array elements
            dims.append(a.shape.d[k])
    var sh = ShapeN.of(dims)
    var src = CastIn(Int(a.data.unsafe_ptr()), CAST_KIND_SIGNED if ids else CAST_KIND_FLOAT, 4,
                     sh.rows(), sh.cols(), sh.cols() * 4, 4)
    src.check(what)
    return ProgramInput(src, sh, List[Float64](), List[Int64]())


def program_save_verified[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, file: String, input_names: List[String], output_names: List[String],
    version: String, platform: String,
) raises:
    """Save program `pid` to `file` and write `file`.verify.json: seeded
    inputs, one replay, the inputs' and outputs' digests. `input_names` and
    `output_names` are empty to use input<k> and output<k>."""
    var s = ops_session[E, KIND](sid)
    var p = s[].program(pid)
    _ = ops_save_program[E, KIND](sid, pid, file)
    var arrays = List[HostArray]()
    var input_json = String("[")
    var have_inputs = len(input_names) == p.n_in
    var have_outputs = len(output_names) == len(p.outs)
    for k in range(p.n_in):  # driver-loop: over verification input owners
        var a = HostArray(p.in_shapes[k])
        if p.in_kinds[k] != 1:
            _ = verify_fill(Int(a.data.unsafe_ptr()), a.shape.count(), 0, k)
        # Zero initialization supplies the established valid id 0.
        if k > 0:
            input_json += ","
        var name = input_names[k].copy() if have_inputs else "input" + String(k)
        input_json += _array_entry(a, name, "int32" if p.in_kinds[k] == 1 else "float32")
        arrays.append(a^)
    input_json += "]"
    var plan = program_execute_prepare[E, KIND](sid, pid, sid, False, False)
    var inputs = List[ProgramInput]()
    for k in range(plan.program.n_in):  # driver-loop: over verification input descriptors
        var ids = plan.program.in_kinds[k] == PROGRAM_IN_IDS
        var item = _seeded_input(arrays[k], ids, "numeric program input")
        if not same_shape(item.shape, plan.program.in_shapes[k]):
            raise Error("ValueError: input " + String(k) + " must keep its captured shape "
                        + shape_text(plan.program.in_shapes[k]) + "; got " + shape_text(item.shape))
        inputs.append(item^)
    var r = program_execute_run[E, KIND](sid, plan, inputs^, False)
    _ = arrays^  # the seeded inputs are read by the uploads above
    var output_json = String("[")
    for k in range(len(p.outs)):  # driver-loop: over verification output digests
        if k > 0:
            output_json += ","
        var name = output_names[k].copy() if have_outputs else "output" + String(k)
        output_json += _array_entry(r.outputs[k], name, "")
    output_json += "]"
    var sha = verify_file_sha256(file)
    var model = verify_model_info(file)
    var basename = file.split("/")
    var vendor = String("cpu") if TARGET_COLUMN == COLUMN_CPU else String(COMPILED_VENDOR)
    var mode = "identical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "fast"
    var report = String("{\"format\":\"mojolearn-model-verify/1\",\"install\":{\"mojolearn_version\":")
    report += _quote(version) + ",\"numeric_mode\":" + _quote(mode)
    report += ",\"vendor\":" + _quote(vendor) + ",\"device\":" + _quote(vendor + " native executor")
    report += ",\"platform\":" + _quote(platform) + "},\"model\":{\"file\":"
    report += _quote(String(basename[len(basename) - 1])) + ",\"kind\":" + _quote(model[0])
    report += ",\"sha256\":" + _quote(sha[0]) + ",\"bytes\":" + String(sha[1])
    report += ",\"description_hash\":" + _quote(model[1]) + "},\"input_source\":\"seeded\",\"seed\":0,\"inputs\":"
    report += input_json + ",\"outputs\":" + output_json + "}\n"
    with open(file + ".verify.json", "w") as fh:
        fh.write(report)
