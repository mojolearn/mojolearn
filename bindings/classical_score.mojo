# SPDX-License-Identifier: Apache-2.0
# The exact existing single-output _Reg R2 program, assembled and consumed
# natively. The grouped sums and binary64 finishing remain x_metrics kernels.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from std.memory import memcpy, bitcast
from bindings.classical_arrays import classical_array, classical_host_f32, classical_address
from bindings.py_typed_call import py_call_typed
from bindings.sequential_description import sequential_fields
from bindings.classical_validate import classical_check_x, classical_check_width, classical_estimator_name


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


struct ClassicalR2Layout(Movable):
    """The `_Reg` / `_r2_sums_of(..., None)` R^2 program over `rows`
    predictions: its stages and the arena words of y, the prediction and the
    two binary64 sums it reads back."""
    var program: ClassicalMetricProgram
    var y: Int
    var p: Int
    var numerator: Int
    var denominator: Int

    def __init__(out self, rows: Int) raises:
        # The allocation and stage order of the Python `_Reg` program.
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
        self.program = program^
        self.y = y
        self.p = p
        self.numerator = numerator
        self.denominator = denominator


def classical_linear_score(model: PythonObject, args: PythonObject, entries: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    """LinearRegression / Ridge score: `classical_linear_predict`'s checks and
    `ols_predict` kernel, whose prediction is written straight into the R^2
    program's arena words (no separate prediction array and no host copy
    between the two), then the same R^2 program on the same words."""
    var fields = sequential_fields(model)
    if py_get(fields, "coef_") is None:
        raise Error("ValueError: call fit before predict")
    var x = classical_check_x(args[0], classical_estimator_name(model), context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    classical_check_width(cols, Int(py=fields["n_features_in_"]), classical_estimator_name(model))
    var coef = classical_host_f32(fields["coef_"], 1, "coef_", context)
    if Int(py=coef.size) != cols:
        raise Error("coef_ length differs from fitted feature count")
    var layout = ClassicalR2Layout(rows)
    var arena = classical_array(layout.program.size, 0, context)
    var address = Int(py=classical_address(arena))
    _ = arena  # keep the Python object alive through the native call above
    _ = py_call_typed(entries["ols_predict"], classical_address(x), classical_address(coef),
                      PythonObject(address + 4 * layout.p), Python.list(rows, cols, fields["intercept_"]))
    _ = coef  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = arena  # keep the Python object alive through the native call above
    _ = x^
    _ = coef^
    return _r2_finish(layout^, arena, rows, args[1], context)


def classical_r2_score(prediction: PythonObject, target_source: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """R^2 of a float32 prediction against the target cast once to float32,
    by the x_metrics program `_Reg` / `_r2_sums_of(..., None)` stages: 0.0
    for a constant target."""
    var rows = Int(py=prediction.size)
    var layout = ClassicalR2Layout(rows)
    var arena = classical_array(layout.program.size, 0, context)
    var dest = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(arena)))
    _ = arena  # keep the Python object alive through the native call above
    var predicted = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(prediction)))
    memcpy(dest=dest + layout.p, src=predicted, count=rows)
    _ = arena  # keep the Python object alive through the native call above
    return _r2_finish(layout^, arena, rows, target_source, context)


def _r2_finish(var layout: ClassicalR2Layout, arena: PythonObject, rows: Int, target_source: PythonObject,
               context: PythonObject) raises -> PythonObject:
    # The prediction is already in the arena at `layout.p`.
    var target = classical_host_f32(target_source, 1, "y", context)
    if Int(py=target.size) != rows:
        raise Error("mojolearn score: y_true and y_pred have different shapes (" + String(Int(py=target.size))
                    + ", 1) and (" + String(rows) + ", 1)")
    # Preserve the current score contract: predict uses the estimator's mode,
    # while _r2_sums_of(..., None) uses the process-default metrics binding.
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var metrics = context["classical_metrics"][mode]
    var bi = Python.import_module("builtins")
    if py_type(metrics) is bi.str:
        raise Error("ImportError: " + String(metrics))
    if String(metrics["x_metrics_vendor"]()) != String(py_get(context, "classical_metric_vendor", context["vendor"])):
        raise Error("RuntimeError: score vendor differs from selected execution")
    if Int(py=metrics["x_metrics_numeric_mode"]()) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: score mode differs from process default")
    var size = layout.program.size
    var numerator = layout.numerator
    var denominator = layout.denominator
    var address = Int(py=classical_address(arena))
    var dest = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    var truth = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(target)))
    _ = target  # keep the Python object alive through the native call above
    memcpy(dest=dest + layout.y, src=truth, count=rows)
    _ = target  # keep the Python object alive through the native call above
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
    var stages = len(layout.program.words) // 16
    var words_address = Int(layout.program.words.unsafe_ptr())
    var outputs_address = Int(outputs.unsafe_ptr())
    var setting = py_get(context["environment"], py_call(PythonObject("MOJOLEARN_ARENA_RANGES").encode, "ascii"), py_call(PythonObject("1").encode, "ascii"))
    if py_type(setting) is not bi.bytes:
        raise Error("TypeError: arena environment setting must be native bytes")
    var setting_text = String(setting.decode())
    var ranges = String(setting_text.strip()) != "0"
    if ranges and py_get(metrics, "x_metrics_run_ranges") is not None:
        _ = py_call(metrics["x_metrics_run_ranges"], address, words_address,
            Python.tuple(size, stages, 1, 2), Int(inputs.unsafe_ptr()), outputs_address)
    elif py_get(metrics, "x_metrics_run_out") is not None:
        _ = py_call(metrics["x_metrics_run_out"], address, size, words_address, stages, outputs_address, 2)
    else:
        _ = py_call(metrics["x_metrics_run"], address, size, words_address, stages)
    # Decode two little-endian double words without assuming 8-byte alignment.
    var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)
    var numerator_bits = UInt64(words[numerator]) | (UInt64(words[numerator + 1]) << UInt64(32))
    var denominator_bits = UInt64(words[denominator]) | (UInt64(words[denominator + 1]) << UInt64(32))
    var num = bitcast[DType.float64](numerator_bits)
    var den = bitcast[DType.float64](denominator_bits)
    var result = Float64(0.0) if den == 0 else Float64(1.0) - num / den
    _ = target^
    _ = layout^
    _ = inputs^
    _ = outputs^
    return PythonObject(result)
