# SPDX-License-Identifier: Apache-2.0
"""The existing single-output R² program, represented entirely as native words."""

struct LinearMetricProgram(Movable):
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

    def stage(mut self, op: Int, total: Int, params: List[Int]) raises:
        # Native stage words: the descriptor fields arrive as a native List,
        # not a Python list read back one Int(py=) conversion at a time.
        if len(params) > 14:
            raise Error("classical score: too many native stage parameters")
        self.words.append(Int32(op))
        self.words.append(Int32(total))
        for k in range(14):  # driver-loop: over the fixed metric stage descriptor fields
            self.words.append(Int32(params[k]) if k < len(params) else Int32(0))

    def sum(mut self, key: Int, source: Int, rows: Int) raises -> Int:
        var off = self.alloc(2)
        var order = self.alloc(rows)
        var out = self.alloc(1)
        self.stage(0, 1, [key, rows, 1, off, order])
        self.stage(1, 1, [off, order, source, 1, -1, out, 1])
        return out

    def finish(mut self, sums: Int, rows: Int, divide: Bool, narrow: Bool) raises -> Tuple[Int, Int]:
        var wide = self.alloc(2)
        var nearest = self.alloc(1) if narrow else -1
        self.stage(60, 1, [0, 1, sums, -1, rows, wide, nearest, Int(divide)])
        return (wide, nearest)


struct LinearR2Layout(Movable):
    """The `_Reg` / `_r2_sums_of(..., None)` R^2 program over `rows`
    predictions: its stages and the arena words of y, the prediction and the
    two binary64 sums it reads back."""
    var program: LinearMetricProgram
    var y: Int
    var p: Int
    var numerator: Int
    var denominator: Int

    def __init__(out self, rows: Int) raises:
        # The allocation and stage order of the Python `_Reg` program.
        var program = LinearMetricProgram()
        var y = program.alloc(rows)
        var p = program.alloc(rows)
        var scalar = program.alloc(1)
        var key = program.alloc(rows)
        program.stage(2, rows, [0, 0, key, 1, 2])
        var ys = program.sum(key, y, rows)
        var mean = program.finish(ys, rows, True, True)
        var term = program.alloc(rows)
        program.stage(3, rows, [y, p, term, 1, 0, scalar, 0])
        var sums = program.sum(key, term, rows)
        var numerator = program.finish(sums, rows, False, False)[0]
        var centered = program.alloc(rows)
        program.stage(3, rows, [y, mean[1], centered, 1, 0, scalar, 1])
        sums = program.sum(key, centered, rows)
        var denominator = program.finish(sums, rows, False, False)[0]
        self.program = program^
        self.y = y
        self.p = p
        self.numerator = numerator
        self.denominator = denominator

