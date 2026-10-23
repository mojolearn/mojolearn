# SPDX-License-Identifier: Apache-2.0
# The gradient-boosting model text's tree inspection records
# (`GradientBoosting.get_tree_leaf_counts` / `get_leaf_values`): the
# `trees`, `tree`/`ntree` and `leaf` lines read the way
# `ensemble.GradientBoosting._tree_metadata` read them, with its refusals
# and wording. Float leaf values come only from their IEEE bits, widened
# exactly to float64. No Python; the shell (bindings/trees_public_gbdt.mojo)
# publishes the lists as Arrays.
from std.memory import bitcast


@fieldwise_init
struct GBDTTreeMetadata(Movable):
    var counts: List[UInt32]
    var values: List[Float64]


def _is_space(byte: UInt8) -> Bool:
    # str.split(): ASCII whitespace (the model text is ASCII).
    return (byte == 32 or byte == 9 or byte == 10 or byte == 11 or byte == 12 or byte == 13
        or byte == 28 or byte == 29 or byte == 30 or byte == 31)


def _fields(line: String) -> List[String]:
    var out = List[String]()
    var bytes = line.as_bytes()
    var start = -1
    for i in range(len(bytes)):
        if _is_space(bytes[i]):
            if start >= 0:
                out.append(String(line[byte=start:i]))
                start = -1
        elif start < 0:
            start = i
    if start >= 0:
        out.append(String(line[byte=start:len(bytes)]))
    return out^


def _field(fields: List[String], index: Int) raises -> String:
    if index >= len(fields):
        raise Error("IndexError: list index out of range")
    return fields[index]


def _int(token: String, base: Int = 10) raises -> Int:
    """Python's `int(token, base)` for the decimal and hex words the text
    holds (a sign, digits, surrounding whitespace already split off)."""
    var bytes = token.as_bytes()
    var n = len(bytes)
    var i = 0
    var negative = False
    if n > 0 and (bytes[0] == 43 or bytes[0] == 45):
        negative = bytes[0] == 45
        i = 1
    if i >= n:
        raise Error("invalid literal for int() with base " + String(base) + ": '" + token + "'")
    var value = 0
    while i < n:
        var c = Int(bytes[i])
        var digit = -1
        if c >= 48 and c <= 57:
            digit = c - 48
        elif base == 16 and c >= 97 and c <= 102:
            digit = c - 87
        elif base == 16 and c >= 65 and c <= 70:
            digit = c - 55
        elif c == 95 and i > 0 and i + 1 < n:
            i += 1
            continue
        if digit < 0 or digit >= base:
            raise Error("invalid literal for int() with base " + String(base) + ": '" + token + "'")
        if value > (9223372036854775807 - digit) // base:
            # Larger than any count or bit pattern the checks accept.
            value = 9223372036854775807
        else:
            value = value * base + digit
        i += 1
    return -value if negative else value


def gbdt_tree_metadata(text: String) raises -> GBDTTreeMetadata:
    """The per-tree leaf counts and the leaf values, tree-major then
    leaf-major (`_tree_metadata`'s walk and refusals)."""
    var counts = List[Int]()
    var dims = List[Int]()
    var leaves = List[List[UInt32]]()
    var declared = -1
    var have_declared = False
    var bytes = text.as_bytes()
    var start = 0
    var n = len(bytes)
    while start <= n:
        var end = start
        while end < n and bytes[end] != 10 and bytes[end] != 13:
            end += 1
        var line = String(text[byte=start:end])
        if end < n and bytes[end] == 13 and end + 1 < n and bytes[end + 1] == 10:
            end += 1
        start = end + 1
        var fields = _fields(line)
        if len(fields) == 0:
            if end >= n:
                break
            continue
        var head = fields[0]
        if head == "trees":
            declared = _int(_field(fields, 1))
            have_declared = True
        elif head == "tree" or head == "ntree":
            if _int(_field(fields, 1)) != len(counts):
                raise Error("mojolearn: invalid tree order in model metadata")
            var size = _int(_field(fields, 3))
            var dim = _int(_field(fields, 5))
            if size < 0 or dim < 1 or (head == "tree" and size > 31):
                raise Error("mojolearn: invalid tree dimensions in model metadata")
            counts.append((1 << size) if head == "tree" else size + 1)
            dims.append(dim)
            leaves.append(List[UInt32]())
        elif head == "leaf":
            var tree = _int(_field(fields, 1))
            var index = _int(_field(fields, 2))
            if tree < 0 or tree >= len(leaves) or index != len(leaves[tree]):
                raise Error("mojolearn: invalid leaf order in model metadata")
            var word = _field(fields, 3)
            var slash = word.find("/")
            if slash < 0:
                raise Error("IndexError: list index out of range")
            var rest = String(word[byte=slash + 1:word.byte_length()])
            var second = rest.find("/")
            var hex = rest if second < 0 else String(rest[byte=0:second])
            var bits = _int(hex, 16)
            if bits < 0 or bits > 0xFFFFFFFF:
                raise Error("mojolearn: invalid leaf bits in model metadata")
            leaves[tree].append(UInt32(bits))
        if end >= n:
            break
    if not have_declared or declared != len(counts):
        raise Error("mojolearn: incomplete tree metadata")
    for t in range(len(counts)):
        if len(leaves[t]) != counts[t] * dims[t]:
            raise Error("mojolearn: incomplete tree metadata")
    var out_counts = List[UInt32]()
    var values = List[Float64]()
    for t in range(len(counts)):
        out_counts.append(UInt32(counts[t]))
        for i in range(len(leaves[t])):
            values.append(Float64(bitcast[DType.float32](leaves[t][i])))
    return GBDTTreeMetadata(out_counts^, values^)
