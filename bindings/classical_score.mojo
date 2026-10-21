# SPDX-License-Identifier: Apache-2.0
# The exact existing single-output _Reg R2 program, assembled and consumed
# natively. The grouped sums and binary64 finishing remain x_metrics kernels.
from std.python import Python, PythonObject
from std.memory import memcpy, bitcast
from bindings.classical_arrays import classical_array, classical_host_f32
from bindings.classical_ridge import classical_linear_predict
from bindings.sequential_arrays import sequential_address


struct ClassicalMetricProgram(Movable):
    var words: List[Int32]
    var size: Int

    def __init__(out self):
        self.words = List[Int32]()
        self.size = 0

    def alloc(mut self, count: Int) raises -> Int:
        if count < 0 or count > 2147483647 - self.size:
            raise Error("mojolearn metrics: the problem exceeds the Int32 arena bound")
        var off = self.size
        self.size += count
        return off

    def stage(mut self, op: Int, total: Int, params: PythonObject) raises:
        if len(params) > 14:
            raise Error("classical score: too many native stage parameters")
        self.words.append(Int32(op))
        self.words.append(Int32(total))
        for k in range(14):  # driver-loop: over the fixed metric stage descriptor fields
            self.words.append(Int32(Int(py=params[k])) if k < len(params) else Int32(0))

    def sum(mut self, key: Int, source: Int, rows: Int) raises -> Int:
        var off = self.alloc(2)
        var order = self.alloc(rows)
        var out = self.alloc(1)
        self.stage(0, 1, Python.list(key, rows, 1, off, order))
        self.stage(1, 1, Python.list(off, order, source, 1, -1, out, 1))
        return out

    def finish(mut self, sums: Int, rows: Int, divide: Bool, narrow: Bool) raises -> Tuple[Int, Int]:
        var wide = self.alloc(2)
        var nearest = self.alloc(1) if narrow else -1
        self.stage(60, 1, Python.list(0, 1, sums, -1, rows, wide, nearest, Int(divide)))
        return (wide, nearest)


def classical_linear_score(model: PythonObject, args: PythonObject, entries: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var prediction = classical_linear_predict(model, args[0], entries, context)
    var target = classical_host_f32(args[1], 1, "y", context)
    var rows = Int(py=prediction.size)
    if Int(py=target.size) != rows:
        raise Error("mojolearn score: y_true and y_pred have different shapes")
    # Preserve the current score contract: predict uses the estimator's mode,
    # while _r2_sums_of(..., None) uses the process-default metrics binding.
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var metrics = context["classical_metrics"][mode]
    var bi = Python.import_module("builtins")
    if bi.type(metrics) is bi.str:
        raise Error("ImportError: " + String(metrics))
    if String(metrics["x_metrics_vendor"]()) != String(context["vendor"]):
        raise Error("RuntimeError: score vendor differs from selected execution")
    if Int(py=metrics["x_metrics_numeric_mode"]()) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: score mode differs from process default")
    var program = ClassicalMetricProgram()
    var y = program.alloc(rows)
    var p = program.alloc(rows)
    var scalar = program.alloc(1)
    var key = program.alloc(rows)
    program.stage(2, rows, Python.list(0, 0, key, 1, 2))
    var ys = program.sum(key, y, rows)
    var mean = program.finish(ys, rows, True, True)
    var term = program.alloc(rows)
    program.stage(3, rows, Python.list(y, p, term, 1, 0, scalar, 0))
    var sums = program.sum(key, term, rows)
    var numerator = program.finish(sums, rows, False, False)[0]
    var centered = program.alloc(rows)
    program.stage(3, rows, Python.list(y, mean[1], centered, 1, 0, scalar, 1))
    sums = program.sum(key, centered, rows)
    var denominator = program.finish(sums, rows, False, False)[0]
    var arena = classical_array(program.size, 0, context)
    var address = Int(py=sequential_address(arena))
    var dest = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    var truth = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(target)))
    var predicted = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(prediction)))
    memcpy(dest=dest + y, src=truth, count=rows)
    memcpy(dest=dest + p, src=predicted, count=rows)
    var inputs = List[Int32](length=3, fill=0)
    inputs[1] = Int32(2 * rows + 1)
    inputs[2] = -1
    var outputs = List[Int32](length=8, fill=0)
    outputs[0] = Int32(numerator)
    outputs[1] = Int32(numerator + 2)
    outputs[2] = -1
    outputs[3] = 1
    outputs[4] = Int32(denominator)
    outputs[5] = Int32(denominator + 2)
    outputs[6] = -1
    outputs[7] = 1
    var stages = len(program.words) // 16
    var words_address = Int(program.words.unsafe_ptr())
    var outputs_address = Int(outputs.unsafe_ptr())
    var setting = context["environment"].get(PythonObject("MOJOLEARN_ARENA_RANGES").encode("ascii"), PythonObject("1").encode("ascii"))
    if bi.type(setting) is not bi.bytes:
        raise Error("TypeError: arena environment setting must be native bytes")
    var ranges = String(setting.decode()).strip() != "0"
    if ranges and metrics.get("x_metrics_run_ranges") is not None:
        _ = metrics["x_metrics_run_ranges"](address, words_address,
            Python.tuple(program.size, stages, 1, 2), Int(inputs.unsafe_ptr()), outputs_address)
    elif metrics.get("x_metrics_run_out") is not None:
        _ = metrics["x_metrics_run_out"](address, program.size, words_address, stages, outputs_address, 2)
    else:
        _ = metrics["x_metrics_run"](address, program.size, words_address, stages)
    # Decode two little-endian double words without assuming 8-byte alignment.
    var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)
    var numerator_bits = UInt64(words[numerator]) | (UInt64(words[numerator + 1]) << UInt64(32))
    var denominator_bits = UInt64(words[denominator]) | (UInt64(words[denominator + 1]) << UInt64(32))
    var num = bitcast[DType.float64](numerator_bits)
    var den = bitcast[DType.float64](denominator_bits)
    var result = Float64(0.0) if den == 0 else Float64(1.0) - num / den
    _ = arena^
    _ = target^
    _ = prediction^
    _ = program^
    _ = inputs^
    _ = outputs^
    return PythonObject(result)
