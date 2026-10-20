# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ONNX model reader in Mojo: the .onnx file (and its external data
beside it) into the import graph the lowering reads (docs/plans/MODEL_IMPORT.md
section 7), the same graph `importers/onnx.py` built in Python before.

Step 1 of the native import boundary. `onnx_read(path)` reads and checks the
model and keeps the graph under a handle: the value table (graph inputs and
outputs with dtypes and shapes), the node table in topological order with
their attributes, the initializer table (each a dtype, a shape and a byte
range of the model file, of an external data file or of the reader's own
small buffer for constants written as typed fields), and the problems found.
`onnx_info` hands the tables to Python as words, strings and floats; Python
wraps them in its GraphIR classes and does no parsing. `onnx_resolve`
(autograd/import_resolve.mojo) resolves every value's static shape on the
same handle.

The checks and their texts are the Python reader's: an opset older than 13,
a node outside the default ONNX domain, an unsupported dtype, a subgraph or
sparse attribute, an external data location outside the model's directory,
a size mismatch, an undefined input, a cycle; every problem is listed, in
graph order.

Word layout of `onnx_info` (all int64; a string is an index into the
strings list, -1 for none; a float an index into the floats list):
  header   MAGIC, VERSION, status, then
           status 0: opset, n_buffers, buffers (kind, path, nbytes) ...,
                     n_inputs, values ..., n_outputs, values ...,
                     n_nodes, nodes ..., n_initializers, tensors ...
           status 5: n_problems, problems (name, op, reason) ...
           status 1-4: message
  value    name, dtype, rank, rank x (is_str, value or string)
  tensor   name, dtype, rank, dims ..., buffer, start, end (a byte range)
  node     op, domain, name, n_in, inputs ..., n_out, outputs ...,
           n_attr, attributes ...
  attr     name, kind, payload: 1 float -> float; 2 int -> value;
           3 string -> string; 4 tensor -> tensor; 6 floats -> first,
           end (a range of floats); 7 ints -> count, values ...; 8 strings -> count,
           strings ...; 9 tensors -> count, tensors ...
dtype: 0 f32, 1 f16, 2 bf16, 3 f64, 4 i64, 5 i32, 6 bool. Buffer kinds:
0 the model file, 1 the reader's own bytes (`onnx_copy_blob`), 2 an
external data file (its resolved path)."""
# cpu-route: model file parsing. Reading a model file is the explicit input
# step before the runtime: names, dims, attributes and byte ranges, on the
# host; tensor element values are never computed on (integer and bool
# tensors are decoded to integers, the description the shape folding reads).
from std.ffi import _Global
from std.memory import bitcast, memcpy
from std.os.path import isfile, realpath
from std.python import Python, PythonObject

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from autograd.import_proto import (
    PB_FIXED32, PB_FIXED64, PB_LEN, PB_UTF8_PREFIX, PB_VARINT, PbField, PbFields, pb_f32,
    pb_packed_ints, pb_utf8,
)

comptime ONNX_READ_MAGIC = 0x4F4E5852  # "ONXR"
comptime ONNX_READ_VERSION = 1

comptime ONNX_OK = 0
comptime ONNX_NOT_A_MODEL = 1
comptime ONNX_CANNOT_OPEN = 2
comptime ONNX_BAD_TEXT = 3
comptime ONNX_BAD_VALUE = 4
comptime ONNX_PROBLEMS = 5

comptime OPSET_MIN = 13

# IR dtypes
comptime DT_F32 = 0
comptime DT_F16 = 1
comptime DT_BF16 = 2
comptime DT_F64 = 3
comptime DT_I64 = 4
comptime DT_I32 = 5
comptime DT_BOOL = 6

# IR attribute kinds (AttributeProto.AttributeType)
comptime AK_FLOAT = 1
comptime AK_INT = 2
comptime AK_STRING = 3
comptime AK_TENSOR = 4
comptime AK_FLOATS = 6
comptime AK_INTS = 7
comptime AK_STRINGS = 8
comptime AK_TENSORS = 9

comptime BUF_MODEL = 0
comptime BUF_BLOB = 1

# An error with this prefix is fatal for the whole read (the Python reader
# raised it out of `read` instead of listing it as a problem).
comptime FATAL_PREFIX = "fatal: "
comptime FATAL_LEN = 7  # bytes of FATAL_PREFIX

comptime SUPPORTED_TYPES = "float32, float16, bfloat16, float64, int64, int32 and bool"


def is_fatal(msg: String) -> Bool:
    return msg.startswith(FATAL_PREFIX)


def fatal_text(msg: String) -> String:
    """The message of a fatal error without its prefix."""
    return substr(msg, FATAL_LEN, msg.byte_length())


def substr(s: String, a: Int, b: Int) -> String:
    """The bytes [a, b) of s as a String (cut on code point boundaries)."""
    var bs = s.as_bytes()
    return String(StringSlice(unsafe_from_utf8=bs[a:b]))


# ------------------------------------------------------------------ dtypes --

def ir_dtype(code: Int) -> Int:
    """The IR dtype of an ONNX TensorProto.DataType code, -1 when unsupported."""
    if code == 1:
        return DT_F32
    if code == 10:
        return DT_F16
    if code == 16:
        return DT_BF16
    if code == 11:
        return DT_F64
    if code == 7:
        return DT_I64
    if code == 6:
        return DT_I32
    if code == 9:
        return DT_BOOL
    return -1


def dtype_name(dt: Int) -> String:
    if dt == DT_F32:
        return "f32"
    if dt == DT_F16:
        return "f16"
    if dt == DT_BF16:
        return "bf16"
    if dt == DT_F64:
        return "f64"
    if dt == DT_I64:
        return "i64"
    if dt == DT_I32:
        return "i32"
    return "bool"


def itemsize(dt: Int) -> Int:
    if dt == DT_F32 or dt == DT_I32:
        return 4
    if dt == DT_F64 or dt == DT_I64:
        return 8
    if dt == DT_BOOL:
        return 1
    return 2


def is_int_dtype(dt: Int) -> Bool:
    return dt == DT_I64 or dt == DT_I32 or dt == DT_BOOL


def onnx_dtype_name(code: Int) -> String:
    """The readable name of an ONNX DataType code (importers/_graph.py onnx_dtype_name)."""
    var dt = ir_dtype(code)
    if dt >= 0:
        return dtype_name(dt)
    if code == 0:
        return "undefined"
    if code == 2:
        return "uint8"
    if code == 3:
        return "int8"
    if code == 4:
        return "uint16"
    if code == 5:
        return "int16"
    if code == 8:
        return "string"
    if code == 12:
        return "uint32"
    if code == 13:
        return "uint64"
    if code == 14:
        return "complex64"
    if code == 15:
        return "complex128"
    if code == 17:
        return "float8e4m3fn"
    if code == 18:
        return "float8e4m3fnuz"
    if code == 19:
        return "float8e5m2"
    if code == 20:
        return "float8e5m2fnuz"
    if code == 21:
        return "uint4"
    if code == 22:
        return "int4"
    if code == 23:
        return "float4e2m1"
    return "data type " + String(code)


# -------------------------------------------------------------- formatting --
# Python's str() of the tuples and lists the messages print.

def fmt_tuple(v: List[Int]) -> String:
    if len(v) == 0:
        return "()"
    var s = String("(")
    for i in range(len(v)):  # cpu-route: model file parsing (one per axis of a message)
        if i > 0:
            s += ", "
        s += String(v[i])
    if len(v) == 1:
        s += ","
    return s + ")"


def fmt_list(v: List[Int]) -> String:
    var s = String("[")
    for i in range(len(v)):  # cpu-route: model file parsing (one per entry of a message)
        if i > 0:
            s += ", "
        s += String(v[i])
    return s + "]"


def _hexdigit(v: UInt8) -> UInt8:
    return v + 0x30 if v < 10 else v + 0x57


def py_repr(s: String) -> String:
    """Python's repr() of a str (quotes chosen and escapes as Python does
    for the ASCII and C1 control characters)."""
    var b = s.as_bytes()
    var has_sq = False
    var has_dq = False
    for i in range(len(b)):  # cpu-route: model file parsing (the bytes of one name)
        if b[i] == 0x27:
            has_sq = True
        elif b[i] == 0x22:
            has_dq = True
    var q = UInt8(0x22) if (has_sq and not has_dq) else UInt8(0x27)
    var out = List[UInt8]()
    out.append(q)
    var i = 0
    while i < len(b):  # cpu-route: model file parsing (the bytes of one name)
        var c = b[i]
        if c == q or c == 0x5C:
            out.append(0x5C)
            out.append(c)
        elif c == 0x0A:
            out.append(0x5C)
            out.append(0x6E)
        elif c == 0x0D:
            out.append(0x5C)
            out.append(0x72)
        elif c == 0x09:
            out.append(0x5C)
            out.append(0x74)
        elif c < 0x20 or c == 0x7F:
            out.append(0x5C)
            out.append(0x78)
            out.append(_hexdigit(c >> 4))
            out.append(_hexdigit(c & 15))
        elif c == 0xC2 and i + 1 < len(b) and ((b[i + 1] >= 0x80 and b[i + 1] <= 0xA0) or b[i + 1] == 0xAD):
            var d = b[i + 1]
            out.append(0x5C)
            out.append(0x78)
            out.append(_hexdigit(d >> 4))
            out.append(_hexdigit(d & 15))
            i += 1
        else:
            out.append(c)
        i += 1
    out.append(q)
    return String(StringSlice(unsafe_from_utf8=Span(out)))


# ------------------------------------------------------------ string table --

struct StrTab(Movable):
    """Every string of the graph once; a string is its index. Index 0 is ""."""
    var items: List[String]
    var index: Dict[String, Int]

    def __init__(out self):
        self.items = List[String]()
        self.index = Dict[String, Int]()
        _ = self.intern("")

    def intern(mut self, s: String) -> Int:
        var got = self.index.get(s)
        if got:
            return got.value()
        var k = len(self.items)
        self.items.append(s)
        self.index[s] = k
        return k

    def find(self, s: String) -> Int:
        """The index of s, -1 when the graph has no such string."""
        var got = self.index.get(s)
        if got:
            return got.value()
        return -1

    def text(self, k: Int) -> String:
        return self.items[k]


comptime S_EMPTY = 0


# -------------------------------------------------------- the import graph --

@fieldwise_init
struct TRec(Copyable, Movable):
    """An initializer or a constant: name, dtype, static shape and its bytes
    as a range of buffer `buf`; `ints` holds the row-major integers of an
    integer or bool tensor (the description the shape folding reads)."""
    var name: Int
    var dtype: Int
    var dims: List[Int]
    var buf: Int
    var off: Int
    var nbytes: Int
    var ints: List[Int]

    def numel(self) -> Int:
        var n = 1
        for i in range(len(self.dims)):  # cpu-route: model file parsing (one per axis)
            n *= self.dims[i]
        return n


@fieldwise_init
struct AttrVal(Copyable, Movable):
    """A node attribute as the lowering reads it (kind AK_*)."""
    var name: Int
    var kind: Int
    var f: Float64
    var i: Int
    var s: Int
    var floats: List[Float64]
    var ints: List[Int]
    var strs: List[Int]
    var tensors: List[TRec]

    @staticmethod
    def make(name: Int, kind: Int) -> AttrVal:
        return AttrVal(name, kind, Float64(0), 0, S_EMPTY, List[Float64](), List[Int](), List[Int](),
                       List[TRec]())


@fieldwise_init
struct GNode(Copyable, Movable):
    """One operation in ONNX's vocabulary; inputs hold "" for an absent
    optional input. Attributes in file order; a repeated name's last value
    is its value (a dict's assignment)."""
    var op: Int
    var domain: Int
    var name: Int
    var inputs: List[Int]
    var outputs: List[Int]
    var attrs: List[AttrVal]


@fieldwise_init
struct GValue(Copyable, Movable):
    """A graph input or output: name, dtype and dims; dim_str[i] >= 0 names
    a symbolic axis (its string), else dims[i] is the size."""
    var name: Int
    var dtype: Int
    var dims: List[Int]
    var dim_str: List[Int]


@fieldwise_init
struct Problem(Copyable, Movable):
    var name: Int
    var op: Int  # -1: a value, not a node
    var reason: Int


struct OnnxGraph(Movable):
    """The import graph of one model file, kept under a handle."""
    var st: StrTab
    var status: Int
    var message: String
    var opset: Int
    var buf_kind: List[Int]
    var buf_path: List[Int]
    var buf_size: List[Int]
    var blob: List[UInt8]
    var inputs: List[GValue]
    var outputs: List[GValue]
    var nodes: List[GNode]
    var inits: List[TRec]
    var init_slot: Dict[Int, Int]
    var problems: List[Problem]

    def __init__(out self):
        self.st = StrTab()
        self.status = ONNX_OK
        self.message = String("")
        self.opset = 0
        self.buf_kind = List[Int]()
        self.buf_path = List[Int]()
        self.buf_size = List[Int]()
        self.blob = List[UInt8]()
        self.inputs = List[GValue]()
        self.outputs = List[GValue]()
        self.nodes = List[GNode]()
        self.inits = List[TRec]()
        self.init_slot = Dict[Int, Int]()
        self.problems = List[Problem]()

    def problem(mut self, name: Int, op: Int, reason: String):
        var k = self.st.intern(reason)
        self.problems.append(Problem(name, op, k))

    def set_init(mut self, var t: TRec):
        """inits[t.name] = t: a repeated name keeps its first place and takes the new value."""
        var got = self.init_slot.get(t.name)
        if got:
            self.inits[got.value()] = t^
        else:
            self.init_slot[t.name] = len(self.inits)
            self.inits.append(t^)

    def init_of(self, name: Int) -> Int:
        """The slot of initializer `name`, -1 when none."""
        var got = self.init_slot.get(name)
        if got:
            return got.value()
        return -1


# ------------------------------------------------------ raw wire messages --
# The messages as the wire gives them (importers' former _onnx_proto.py
# dicts), decoded in full before any check, so a malformed file is refused
# before anything else is said about it.

@fieldwise_init
struct RawTensor(Copyable, Movable):
    var name: String
    var data_type: Int
    var dims: List[Int]
    var has_raw: Bool
    var raw_off: Int
    var raw_n: Int
    var typed: List[PbField]  # occurrences of fields 4 5 6 7 10 11, in order
    var ext_keys: List[String]
    var ext_vals: List[String]
    var data_location: Int
    var segment: Bool

    @staticmethod
    def empty() -> RawTensor:
        return RawTensor(String(""), 0, List[Int](), False, 0, 0, List[PbField](), List[String](),
                         List[String](), 0, False)

    def has_typed(self, num: Int) -> Bool:
        for i in range(len(self.typed)):  # cpu-route: model file parsing (field occurrences)
            if self.typed[i].num == num:
                return True
        return False

    def ext(self, key: String, mut found: Bool) -> String:
        found = False
        for i in range(len(self.ext_keys)):  # cpu-route: model file parsing (external data entries)
            if self.ext_keys[i] == key:
                found = True
                return self.ext_vals[i]
        return String("")


@fieldwise_init
struct RawAttr(Copyable, Movable):
    var name: String
    var type: Int
    var ref_attr: Bool
    var has_f: Bool
    var f: Float32
    var has_i: Bool
    var i: Int
    var has_s: Bool
    var s_off: Int
    var s_n: Int
    var t: List[RawTensor]  # zero or one
    var floats: List[Float32]
    var ints: List[Int]
    var str_off: List[Int]
    var str_n: List[Int]
    var tensors: List[RawTensor]
    var has_graph: Bool
    var has_sparse: Bool
    var has_type_proto: Bool

    @staticmethod
    def empty() -> RawAttr:
        return RawAttr(String(""), 0, False, False, Float32(0), False, 0, False, 0, 0, List[RawTensor](),
                       List[Float32](), List[Int](), List[Int](), List[Int](), List[RawTensor](), False, False,
                       False)


@fieldwise_init
struct RawNode(Copyable, Movable):
    var op: String
    var domain: String
    var name: String
    var inputs: List[String]
    var outputs: List[String]
    var attrs: List[RawAttr]


# TypeProto kinds
comptime TK_UNKNOWN = 0
comptime TK_TENSOR = 1
comptime TK_SEQUENCE = 2
comptime TK_MAP = 3
comptime TK_OPTIONAL = 4
comptime TK_SPARSE = 5


def type_kind_name(k: Int) -> String:
    if k == TK_TENSOR:
        return "tensor"
    if k == TK_SEQUENCE:
        return "sequence"
    if k == TK_MAP:
        return "map"
    if k == TK_OPTIONAL:
        return "optional"
    if k == TK_SPARSE:
        return "sparse"
    return "unknown"


@fieldwise_init
struct RawVI(Copyable, Movable):
    var name: String
    var has_type: Bool
    var kind: Int
    var elem_type: Int
    var has_shape: Bool
    var dim_kind: List[Int]  # 0 neither, 1 dim_value, 2 dim_param
    var dim_val: List[Int]
    var dim_par: List[String]

    @staticmethod
    def empty() -> RawVI:
        return RawVI(String(""), False, TK_UNKNOWN, 0, False, List[Int](), List[Int](), List[String]())


@fieldwise_init
struct RawGraph(Copyable, Movable):
    var nodes: List[RawNode]
    var inits: List[RawTensor]
    var sparse: Int
    var inputs: List[RawVI]
    var outputs: List[RawVI]


@fieldwise_init
struct RawModel(Movable):
    var opset_domain: List[String]
    var opset_version: List[Int]
    var has_graph: Bool
    var graph: List[RawGraph]  # zero or one (the last graph field)


def _string_entry(buf: List[UInt8], f: PbField, mut k: String, mut v: String) raises:
    """StringStringEntryProto -> (key, value)."""
    k = String("")
    v = String("")
    var it = PbFields(f.off, f.off + f.n)
    var g = PbField.empty()
    while it.next(buf, g):  # cpu-route: model file parsing (message fields)
        if g.num == 1 and g.wt == PB_LEN:
            k = pb_utf8(buf, g.off, g.n, False)
        elif g.num == 2 and g.wt == PB_LEN:
            v = pb_utf8(buf, g.off, g.n, False)


def _raw_tensor(buf: List[UInt8], start: Int, end: Int) raises -> RawTensor:
    """TensorProto."""
    var t = RawTensor.empty()
    var it = PbFields(start, end)
    var f = PbField.empty()
    while it.next(buf, f):  # cpu-route: model file parsing (message fields)
        if f.num == 1:
            pb_packed_ints(buf, f, t.dims)
        elif f.num == 2 and f.wt == PB_VARINT:
            t.data_type = f.ival()
        elif f.num == 3:
            t.segment = True
        elif f.num == 8 and f.wt == PB_LEN:
            t.name = pb_utf8(buf, f.off, f.n, False)
        elif f.num == 9 and f.wt == PB_LEN:
            t.has_raw = True
            t.raw_off = f.off
            t.raw_n = f.n
        elif f.num == 4 or f.num == 5 or f.num == 6 or f.num == 7 or f.num == 10 or f.num == 11:
            t.typed.append(f)
        elif f.num == 13 and f.wt == PB_LEN:
            var k = String("")
            var v = String("")
            _string_entry(buf, f, k, v)
            var put = False
            for i in range(len(t.ext_keys)):  # cpu-route: model file parsing (external data entries)
                if t.ext_keys[i] == k:
                    t.ext_vals[i] = v
                    put = True
                    break
            if not put:
                t.ext_keys.append(k)
                t.ext_vals.append(v)
        elif f.num == 14 and f.wt == PB_VARINT:
            t.data_location = f.ival()
    return t^


def _raw_shape(buf: List[UInt8], f: PbField, mut vi: RawVI) raises:
    """TensorShapeProto: one dim per field 1 (dim_value, dim_param or neither)."""
    var it = PbFields(f.off, f.off + f.n)
    var g = PbField.empty()
    while it.next(buf, g):  # cpu-route: model file parsing (one field per axis)
        if g.num == 1 and g.wt == PB_LEN:
            var kind = 0
            var val = 0
            var par = String("")
            var it2 = PbFields(g.off, g.off + g.n)
            var h = PbField.empty()
            while it2.next(buf, h):  # cpu-route: model file parsing (message fields)
                if h.num == 1 and h.wt == PB_VARINT:
                    kind = 1
                    val = h.ival()
                elif h.num == 2 and h.wt == PB_LEN:
                    kind = 2
                    par = pb_utf8(buf, h.off, h.n, False)
            vi.dim_kind.append(kind)
            vi.dim_val.append(val)
            vi.dim_par.append(par)


def _raw_type(buf: List[UInt8], f: PbField, mut vi: RawVI) raises:
    """TypeProto: the kind, and for a tensor its elem_type and shape."""
    vi.has_type = True
    vi.kind = TK_UNKNOWN
    vi.elem_type = 0
    vi.has_shape = False
    vi.dim_kind = List[Int]()
    vi.dim_val = List[Int]()
    vi.dim_par = List[String]()
    var it = PbFields(f.off, f.off + f.n)
    var g = PbField.empty()
    while it.next(buf, g):  # cpu-route: model file parsing (message fields)
        if g.num == 1 and g.wt == PB_LEN:
            vi.kind = TK_TENSOR
            var it2 = PbFields(g.off, g.off + g.n)
            var h = PbField.empty()
            while it2.next(buf, h):  # cpu-route: model file parsing (message fields)
                if h.num == 1 and h.wt == PB_VARINT:
                    vi.elem_type = h.ival()
                elif h.num == 2 and h.wt == PB_LEN:
                    vi.has_shape = True
                    vi.dim_kind = List[Int]()
                    vi.dim_val = List[Int]()
                    vi.dim_par = List[String]()
                    _raw_shape(buf, h, vi)
        elif g.num == 4:
            vi.kind = TK_SEQUENCE
        elif g.num == 5:
            vi.kind = TK_MAP
        elif g.num == 9:
            vi.kind = TK_OPTIONAL
        elif g.num == 8:
            vi.kind = TK_SPARSE


def _raw_value_info(buf: List[UInt8], f: PbField) raises -> RawVI:
    """ValueInfoProto."""
    var vi = RawVI.empty()
    var it = PbFields(f.off, f.off + f.n)
    var g = PbField.empty()
    while it.next(buf, g):  # cpu-route: model file parsing (message fields)
        if g.num == 1 and g.wt == PB_LEN:
            vi.name = pb_utf8(buf, g.off, g.n, False)
        elif g.num == 2 and g.wt == PB_LEN:
            _raw_type(buf, g, vi)
    return vi^


def _raw_attr(buf: List[UInt8], f: PbField) raises -> RawAttr:
    """AttributeProto."""
    var a = RawAttr.empty()
    var it = PbFields(f.off, f.off + f.n)
    var g = PbField.empty()
    while it.next(buf, g):  # cpu-route: model file parsing (message fields)
        if g.num == 1 and g.wt == PB_LEN:
            a.name = pb_utf8(buf, g.off, g.n, False)
        elif g.num == 20 and g.wt == PB_VARINT:
            a.type = g.ival()
        elif g.num == 21 and g.wt == PB_LEN:
            a.ref_attr = pb_utf8(buf, g.off, g.n, False).byte_length() > 0
        elif g.num == 2 and g.wt == PB_FIXED32:
            a.has_f = True
            a.f = pb_f32(buf, g.off)
        elif g.num == 3 and g.wt == PB_VARINT:
            a.has_i = True
            a.i = g.ival()
        elif g.num == 4 and g.wt == PB_LEN:
            a.has_s = True
            a.s_off = g.off
            a.s_n = g.n
        elif g.num == 5 and g.wt == PB_LEN:
            a.t = List[RawTensor]()
            a.t.append(_raw_tensor(buf, g.off, g.off + g.n))
        elif g.num == 7:
            if g.wt == PB_FIXED32:
                a.floats.append(pb_f32(buf, g.off))
            elif g.wt == PB_LEN:
                if g.n % 4 != 0:
                    raise Error("packed floats not a multiple of 4 bytes")
                for k in range(g.n // 4):  # cpu-route: model file parsing (an attribute's float list)
                    a.floats.append(pb_f32(buf, g.off + 4 * k))
        elif g.num == 8:
            pb_packed_ints(buf, g, a.ints)
        elif g.num == 9 and g.wt == PB_LEN:
            a.str_off.append(g.off)
            a.str_n.append(g.n)
        elif g.num == 10 and g.wt == PB_LEN:
            a.tensors.append(_raw_tensor(buf, g.off, g.off + g.n))
        elif g.num == 6 or g.num == 11:
            a.has_graph = True
        elif g.num == 22 or g.num == 23:
            a.has_sparse = True
        elif g.num == 14 or g.num == 15:
            a.has_type_proto = True
    return a^


def _raw_node(buf: List[UInt8], f: PbField) raises -> RawNode:
    """NodeProto."""
    var n = RawNode(String(""), String(""), String(""), List[String](), List[String](), List[RawAttr]())
    var it = PbFields(f.off, f.off + f.n)
    var g = PbField.empty()
    while it.next(buf, g):  # cpu-route: model file parsing (message fields)
        if g.num == 1 and g.wt == PB_LEN:
            n.inputs.append(pb_utf8(buf, g.off, g.n, False))
        elif g.num == 2 and g.wt == PB_LEN:
            n.outputs.append(pb_utf8(buf, g.off, g.n, False))
        elif g.num == 3 and g.wt == PB_LEN:
            n.name = pb_utf8(buf, g.off, g.n, False)
        elif g.num == 4 and g.wt == PB_LEN:
            n.op = pb_utf8(buf, g.off, g.n, False)
        elif g.num == 7 and g.wt == PB_LEN:
            n.domain = pb_utf8(buf, g.off, g.n, False)
        elif g.num == 5 and g.wt == PB_LEN:
            n.attrs.append(_raw_attr(buf, g))
    return n^


def _raw_graph(buf: List[UInt8], f: PbField) raises -> RawGraph:
    """GraphProto (value_info is decoded, so a malformed one is refused, and dropped)."""
    var gr = RawGraph(List[RawNode](), List[RawTensor](), 0, List[RawVI](), List[RawVI]())
    var it = PbFields(f.off, f.off + f.n)
    var g = PbField.empty()
    while it.next(buf, g):  # cpu-route: model file parsing (message fields)
        if g.wt != PB_LEN:
            continue
        if g.num == 1:
            gr.nodes.append(_raw_node(buf, g))
        elif g.num == 2:
            _ = pb_utf8(buf, g.off, g.n, False)
        elif g.num == 5:
            gr.inits.append(_raw_tensor(buf, g.off, g.off + g.n))
        elif g.num == 15:
            gr.sparse += 1
        elif g.num == 11:
            gr.inputs.append(_raw_value_info(buf, g))
        elif g.num == 12:
            gr.outputs.append(_raw_value_info(buf, g))
        elif g.num == 13:
            _ = _raw_value_info(buf, g)
    return gr^


def _raw_model(buf: List[UInt8]) raises -> RawModel:
    """ModelProto."""
    var m = RawModel(List[String](), List[Int](), False, List[RawGraph]())
    var it = PbFields(0, len(buf))
    var g = PbField.empty()
    while it.next(buf, g):  # cpu-route: model file parsing (message fields)
        if g.num == 8 and g.wt == PB_LEN:
            var dom = String("")
            var ver = 0
            var it2 = PbFields(g.off, g.off + g.n)
            var h = PbField.empty()
            while it2.next(buf, h):  # cpu-route: model file parsing (message fields)
                if h.num == 1 and h.wt == PB_LEN:
                    dom = pb_utf8(buf, h.off, h.n, False)
                elif h.num == 2 and h.wt == PB_VARINT:
                    ver = h.ival()
            m.opset_domain.append(dom)
            m.opset_version.append(ver)
        elif g.num == 7 and g.wt == PB_LEN:
            m.graph = List[RawGraph]()
            m.graph.append(_raw_graph(buf, g))
            m.has_graph = True
    return m^


# ----------------------------------------------------------- the reading --

def _dirname(p: String) -> String:
    """os.path.dirname."""
    var b = p.as_bytes()
    var k = -1
    for i in range(len(b)):  # cpu-route: model file parsing (the bytes of one path)
        if b[i] == 0x2F:
            k = i
    if k < 0:
        return String("")
    var j = k
    while j > 0 and b[j - 1] == 0x2F:  # cpu-route: model file parsing (trailing separators of one path)
        j -= 1
    return substr(p, 0, j if j > 0 else k + 1)


def _basename(p: String) -> String:
    """os.path.basename."""
    var b = p.as_bytes()
    var k = -1
    for i in range(len(b)):  # cpu-route: model file parsing (the bytes of one path)
        if b[i] == 0x2F:
            k = i
    return substr(p, k + 1, len(b))


def _normpath(p: String) -> String:
    """os.path.normpath of an absolute path: "." and ".." components resolved
    lexically, repeated separators dropped."""
    var parts = List[String]()
    var b = p.as_bytes()
    var start = 0
    for i in range(len(b) + 1):  # cpu-route: model file parsing (the bytes of one path)
        if i == len(b) or b[i] == 0x2F:
            if i > start:
                var c = substr(p, start, i)
                if c == "..":
                    if len(parts) > 0:
                        _ = parts.pop()
                elif c != ".":
                    parts.append(c)
            start = i + 1
    var out = String("")
    for i in range(len(parts)):  # cpu-route: model file parsing (the components of one path)
        out += "/" + parts[i]
    return out if out.byte_length() > 0 else String("/")


def _realpath_or_norm(p: String) -> String:
    """os.path.realpath: the resolved path, or for a path that does not
    exist (realpath refuses it) the lexically normalized one."""
    try:
        return realpath(p)
    except:
        return _normpath(p)


def _parse_py_int(s: String, mut ok: Bool) -> Int:
    """Python's int() of a decimal string: surrounding whitespace, a sign,
    digits with single underscores between them."""
    var b = s.as_bytes()
    var i = 0
    var n = len(b)
    while i < n and (b[i] == 0x20 or (b[i] >= 0x09 and b[i] <= 0x0D)):  # cpu-route: model file parsing (one string)
        i += 1
    while n > i and (b[n - 1] == 0x20 or (b[n - 1] >= 0x09 and b[n - 1] <= 0x0D)):  # cpu-route: model file parsing (one string)
        n -= 1
    var neg = False
    if i < n and (b[i] == 0x2B or b[i] == 0x2D):
        neg = b[i] == 0x2D
        i += 1
    ok = False
    var v = 0
    var prev_digit = False
    while i < n:  # cpu-route: model file parsing (one string)
        var c = b[i]
        if c >= 0x30 and c <= 0x39:
            v = v * 10 + Int(c - 0x30)
            prev_digit = True
            ok = True
        elif c == 0x5F and prev_digit and i + 1 < n and b[i + 1] >= 0x30 and b[i + 1] <= 0x39:
            prev_digit = False
        else:
            ok = False
            return 0
        i += 1
    return -v if neg else v


comptime _BytePtr = UnsafePointer[UInt8, MutUntrackedOrigin]


struct _Reader(Movable):
    """One read: the model's bytes, the external data files beside it (read
    once each) and the graph being built."""
    var model: List[UInt8]
    var base: String
    var ext_paths: List[String]
    var ext_data: List[List[UInt8]]
    var ext_bufs: List[Int]
    var g: OnnxGraph

    def __init__(out self, var model: List[UInt8], base: String):
        self.model = model^
        self.base = base
        self.ext_paths = List[String]()
        self.ext_data = List[List[UInt8]]()
        self.ext_bufs = List[Int]()
        self.g = OnnxGraph()

    def take_graph(deinit self) -> OnnxGraph:
        return self.g^

    def ext_slice(mut self, t: RawTensor, nbytes: Int, mut buf: Int, mut off: Int) raises:
        """The external data range of `t` (importers' former _External.slice)."""
        var found = False
        var loc = t.ext("location", found)
        if loc.byte_length() == 0:
            raise Error("external data without a location")
        if loc.startswith("/"):
            raise Error("external data location '" + loc + "' is an absolute path (it must sit beside the model)")
        var full = _realpath_or_norm(self.base + "/" + loc)
        if not (full == self.base or full.startswith(self.base + "/") or self.base == "/"):
            raise Error("external data location '" + loc + "' is outside the model's directory")
        var k = -1
        for i in range(len(self.ext_paths)):  # cpu-route: model file parsing (the external data files)
            if self.ext_paths[i] == full:
                k = i
        if k < 0:
            if not isfile(full):
                raise Error("external data file '" + loc + "' not found beside the model")
            var body: List[UInt8]
            with open(full, "r") as fh:
                body = fh.read_bytes()
            k = len(self.ext_paths)
            self.ext_paths.append(full)
            self.ext_bufs.append(len(self.g.buf_kind))
            self.g.buf_kind.append(2)
            var pk = self.g.st.intern(full)
            self.g.buf_path.append(pk)
            self.g.buf_size.append(len(body))
            self.ext_data.append(body^)
        var size = len(self.ext_data[k])
        var have_off = False
        var off_s = t.ext("offset", have_off)
        var o = 0
        var ok = True
        if off_s.byte_length() > 0:
            o = _parse_py_int(off_s, ok)
        var length = nbytes
        var have_len = False
        var len_s = t.ext("length", have_len)
        if ok and len_s.byte_length() > 0:
            length = _parse_py_int(len_s, ok)
        if not ok:
            raise Error("external data offset/length of '" + loc + "' are not integers")
        if o < 0 or length != nbytes or o + length > size:
            raise Error("external data '" + loc + "' [offset " + String(o) + ", length " + String(length)
                        + "] does not hold the " + String(nbytes) + " bytes the tensor needs (file size "
                        + String(size) + ")")
        buf = self.ext_bufs[k]
        off = o

    def bytes_of(self, buf: Int) -> _BytePtr:
        """The first byte of buffer `buf` (the model, the blob or an external file)."""
        if buf == BUF_MODEL:
            return _BytePtr(unsafe_from_address=Int(self.model.unsafe_ptr()))
        if buf == BUF_BLOB:
            return _BytePtr(unsafe_from_address=Int(self.g.blob.unsafe_ptr()))
        var k = 0
        for i in range(len(self.ext_bufs)):  # cpu-route: model file parsing (the external data files)
            if self.ext_bufs[i] == buf:
                k = i
        return _BytePtr(unsafe_from_address=Int(self.ext_data[k].unsafe_ptr()))

    def decode_ints(self, mut t: TRec):
        """The row-major integers of an integer or bool tensor."""
        if not is_int_dtype(t.dtype):
            return
        var n = t.numel()
        var w = itemsize(t.dtype)
        t.ints = List[Int](capacity=n)
        if n == 0:
            return
        var p = self.bytes_of(t.buf)
        for e in range(n):  # cpu-route: model file parsing (integer description tensor)
            var u = UInt64(0)
            for k in range(w):  # cpu-route: model file parsing (bytes of one integer)
                u |= UInt64(p[t.off + e * w + k]) << UInt64(8 * k)
            if t.dtype == DT_I64:
                t.ints.append(Int(bitcast[DType.int64](u)))
            elif t.dtype == DT_I32:
                t.ints.append(Int(bitcast[DType.int32](u.cast[DType.uint32]())))
            else:
                t.ints.append(Int(u))

    def blob_put(mut self, v: UInt64, nb: Int):
        for k in range(nb):  # cpu-route: model file parsing (bytes of one constant word)
            self.g.blob.append(UInt8((v >> UInt64(8 * k)) & UInt64(0xFF)))

    def typed_bytes(mut self, t: RawTensor, dt: Int, mut buf: Int, mut off: Int, mut n: Int) raises -> Bool:
        """The raw bytes of a tensor whose data sits in a typed repeated field;
        False when none applies. Exporters write weights as raw_data; the typed
        fields carry small constants, re-encoded to the raw layout bit for bit."""
        var num = -1
        if dt == DT_F32 and t.has_typed(4):
            num = 4
        elif dt == DT_F64 and t.has_typed(10):
            num = 10
        if num > 0:
            var count = 0
            var first = -1
            for i in range(len(t.typed)):  # cpu-route: model file parsing (field occurrences)
                if t.typed[i].num == num:
                    if t.typed[i].wt == PB_VARINT:
                        raise Error(String(FATAL_PREFIX) + "float data of tensor '" + t.name + "' in a varint field")
                    count += 1
                    if first < 0:
                        first = i
            if count == 1:
                buf = BUF_MODEL
                off = t.typed[first].off
                n = t.typed[first].n
                return True
            buf = BUF_BLOB
            off = len(self.g.blob)
            for i in range(len(t.typed)):  # cpu-route: model file parsing (field occurrences)
                if t.typed[i].num == num:
                    for k in range(t.typed[i].n):  # cpu-route: model file parsing (bytes of one typed field)
                        var byte = self.model[t.typed[i].off + k]
                        self.g.blob.append(byte)
            n = len(self.g.blob) - off
            return True
        var inum = -1
        if (dt == DT_F16 or dt == DT_BF16 or dt == DT_I32 or dt == DT_BOOL) and t.has_typed(5):
            inum = 5
        elif dt == DT_I64 and t.has_typed(7):
            inum = 7
        if inum < 0:
            return False
        var vals = List[Int]()
        for i in range(len(t.typed)):  # cpu-route: model file parsing (field occurrences)
            if t.typed[i].num == inum:
                try:
                    pb_packed_ints(self.model, t.typed[i], vals)
                except e:
                    raise Error(String(FATAL_PREFIX) + String(e))
        buf = BUF_BLOB
        off = len(self.g.blob)
        for i in range(len(vals)):  # cpu-route: model file parsing (a typed constant's words)
            var v = vals[i]
            if dt == DT_F16 or dt == DT_BF16:
                self.blob_put(UInt64(v & 0xFFFF), 2)
            elif dt == DT_I32:
                if v < -2147483648 or v > 2147483647:
                    raise Error(String(FATAL_PREFIX) + "'i' format requires -2147483648 <= number <= 2147483647")
                self.blob_put(bitcast[DType.uint64](Int64(v)), 4)
            elif dt == DT_BOOL:
                if v < 0 or v > 255:
                    raise Error(String(FATAL_PREFIX) + "ubyte format requires 0 <= number <= 255")
                self.blob_put(UInt64(v), 1)
            else:
                self.blob_put(bitcast[DType.uint64](Int64(v)), 8)
        n = len(self.g.blob) - off
        return True

    def tensor_info(mut self, t: RawTensor, name_in: String) raises -> TRec:
        """A TensorProto as a TRec; raises with the reason."""
        var name = name_in if name_in.byte_length() > 0 else t.name
        var dt = ir_dtype(t.data_type)
        if dt < 0:
            raise Error(onnx_dtype_name(t.data_type) + " tensors are not supported (" + SUPPORTED_TYPES + " are)")
        if t.segment:
            raise Error("segmented tensors are not supported")
        for i in range(len(t.dims)):  # cpu-route: model file parsing (one per axis)
            if t.dims[i] < 0:
                raise Error("negative dimension in " + fmt_tuple(t.dims))
        var numel = 1
        for i in range(len(t.dims)):  # cpu-route: model file parsing (one per axis)
            numel *= t.dims[i]
        var nbytes = numel * itemsize(dt)
        var buf = BUF_MODEL
        var off = 0
        var n = 0
        if t.data_location == 1:
            self.ext_slice(t, nbytes, buf, off)
            n = nbytes
        elif t.has_raw:
            off = t.raw_off
            n = t.raw_n
        else:
            if not self.typed_bytes(t, dt, buf, off, n):
                if numel != 0:
                    raise Error("has no data")
                buf = BUF_BLOB
                off = len(self.g.blob)
                n = 0
        if n != nbytes:
            raise Error("holds " + String(n) + " bytes, its shape " + fmt_tuple(t.dims) + " of " + dtype_name(dt)
                        + " needs " + String(nbytes))
        var rec = TRec(self.g.st.intern(name), dt, t.dims.copy(), buf, off, n, List[Int]())
        self.decode_ints(rec)
        return rec^

    def attr(mut self, a: RawAttr) raises -> AttrVal:
        """An AttributeProto as an attribute value; raises with the reason."""
        if a.ref_attr:
            raise Error("attribute '" + a.name + "' refers to a function attribute")
        if a.has_graph:
            raise Error("attribute '" + a.name + "' holds a subgraph (control flow is not supported)")
        if a.has_sparse:
            raise Error("attribute '" + a.name + "' holds a sparse tensor")
        if a.has_type_proto:
            raise Error("attribute '" + a.name + "' holds a type")
        var ty = a.type
        if ty == 0:  # writers older than IR 3 leave the type out: infer it from the field present
            if a.has_f:
                ty = AK_FLOAT
            elif a.has_i:
                ty = AK_INT
            elif a.has_s:
                ty = AK_STRING
            elif len(a.t) > 0:
                ty = AK_TENSOR
            elif len(a.floats) > 0:
                ty = AK_FLOATS
            elif len(a.ints) > 0:
                ty = AK_INTS
            elif len(a.str_off) > 0:
                ty = AK_STRINGS
            elif len(a.tensors) > 0:
                ty = AK_TENSORS
        var v = AttrVal.make(self.g.st.intern(a.name), ty)
        if ty == AK_FLOAT:
            v.f = Float64(a.f) if a.has_f else Float64(0)
        elif ty == AK_INT:
            v.i = a.i if a.has_i else 0
        elif ty == AK_STRING:
            v.s = self.g.st.intern(pb_utf8(self.model, a.s_off, a.s_n, True)) if a.has_s else S_EMPTY
        elif ty == AK_TENSOR:
            if len(a.t) == 0:
                raise Error("attribute '" + a.name + "' has no tensor")
            var nm = a.t[0].name if a.t[0].name.byte_length() > 0 else a.name
            v.tensors.append(self.tensor_info(a.t[0], nm))
        elif ty == AK_FLOATS:
            for i in range(len(a.floats)):  # cpu-route: model file parsing (attribute list)
                v.floats.append(Float64(a.floats[i]))
        elif ty == AK_INTS:
            v.ints = a.ints.copy()
        elif ty == AK_STRINGS:
            for i in range(len(a.str_off)):  # cpu-route: model file parsing (attribute list)
                v.strs.append(self.g.st.intern(pb_utf8(self.model, a.str_off[i], a.str_n[i], True)))
        elif ty == AK_TENSORS:
            for i in range(len(a.tensors)):  # cpu-route: model file parsing (attribute list)
                var nm = a.tensors[i].name if a.tensors[i].name.byte_length() > 0 else a.name
                v.tensors.append(self.tensor_info(a.tensors[i], nm))
        else:
            raise Error("attribute '" + a.name + "' has an unsupported type " + String(ty))
        return v^

    def constant_tensor(mut self, name: Int, attrs: List[AttrVal]) raises -> TRec:
        """The tensor a Constant node holds, from whichever value attribute it sets."""
        var iv = _last_attr(self.g.st, attrs, "value")
        if iv >= 0 and attrs[iv].kind == AK_TENSOR:
            var t = attrs[iv].tensors[0].copy()
            t.name = name
            return t^
        iv = _last_attr(self.g.st, attrs, "value_float")
        if iv >= 0:
            var f: Float32
            if attrs[iv].kind == AK_FLOAT:
                f = Float32(attrs[iv].f)
            elif attrs[iv].kind == AK_INT:
                f = Float32(Float64(attrs[iv].i))
            else:
                raise Error(String(FATAL_PREFIX) + "required argument is not a float")
            var off = len(self.g.blob)
            self.blob_put(UInt64(bitcast[DType.uint32](f)), 4)
            return TRec(name, DT_F32, List[Int](), BUF_BLOB, off, 4, List[Int]())
        iv = _last_attr(self.g.st, attrs, "value_floats")
        if iv >= 0:
            var off = len(self.g.blob)
            var cnt = 0
            if attrs[iv].kind == AK_FLOATS:
                cnt = len(attrs[iv].floats)
                for k in range(cnt):  # cpu-route: model file parsing (a constant's attribute list)
                    self.blob_put(UInt64(bitcast[DType.uint32](Float32(attrs[iv].floats[k]))), 4)
            elif attrs[iv].kind == AK_INTS:
                cnt = len(attrs[iv].ints)
                for k in range(cnt):  # cpu-route: model file parsing (a constant's attribute list)
                    self.blob_put(UInt64(bitcast[DType.uint32](Float32(Float64(attrs[iv].ints[k])))), 4)
            else:
                raise Error(String(FATAL_PREFIX) + "required argument is not a float")
            var dims = List[Int]()
            dims.append(cnt)
            return TRec(name, DT_F32, dims^, BUF_BLOB, off, 4 * cnt, List[Int]())
        iv = _last_attr(self.g.st, attrs, "value_int")
        if iv >= 0:
            if attrs[iv].kind != AK_INT:
                raise Error(String(FATAL_PREFIX) + "required argument is not an integer")
            var off = len(self.g.blob)
            self.blob_put(bitcast[DType.uint64](Int64(attrs[iv].i)), 8)
            var ints = List[Int]()
            ints.append(attrs[iv].i)
            return TRec(name, DT_I64, List[Int](), BUF_BLOB, off, 8, ints^)
        iv = _last_attr(self.g.st, attrs, "value_ints")
        if iv >= 0:
            if attrs[iv].kind != AK_INTS:
                raise Error(String(FATAL_PREFIX) + "required argument is not an integer")
            var off = len(self.g.blob)
            var cnt = len(attrs[iv].ints)
            for k in range(cnt):  # cpu-route: model file parsing (a constant's attribute list)
                self.blob_put(bitcast[DType.uint64](Int64(attrs[iv].ints[k])), 8)
            var dims = List[Int]()
            dims.append(cnt)
            return TRec(name, DT_I64, dims^, BUF_BLOB, off, 8 * cnt, attrs[iv].ints.copy())
        raise Error("a Constant without a numeric value (string and sparse constants are not supported)")

    def value(mut self, vi: RawVI, what: String) raises -> GValue:
        """A graph input or output; raises with the reason."""
        if not vi.has_type or vi.kind != TK_TENSOR:
            var kind = String("no type") if not vi.has_type else "a " + type_kind_name(vi.kind) + " type"
            raise Error(what + " has " + kind + " (only tensors are supported)")
        var dt = ir_dtype(vi.elem_type)
        if dt < 0:
            raise Error(what + " is " + onnx_dtype_name(vi.elem_type) + " (" + SUPPORTED_TYPES + " are supported)")
        if not vi.has_shape:
            raise Error(what + " has no shape (its rank must be known)")
        var v = GValue(self.g.st.intern(vi.name), dt, List[Int](), List[Int]())
        for i in range(len(vi.dim_kind)):  # cpu-route: model file parsing (one per axis)
            if vi.dim_kind[i] == 1:
                v.dims.append(vi.dim_val[i])
                v.dim_str.append(-1)
            elif vi.dim_kind[i] == 2:
                v.dims.append(0)
                v.dim_str.append(self.g.st.intern(vi.dim_par[i]))
            else:
                v.dims.append(0)
                v.dim_str.append(self.g.st.intern("?" + String(i)))
        return v^


def _last_attr(st: StrTab, attrs: List[AttrVal], name: String) -> Int:
    """The index of attribute `name`'s value (its last occurrence), -1 when absent."""
    var k = st.find(name)
    if k < 0:
        return -1
    var i = len(attrs) - 1
    while i >= 0:  # cpu-route: model file parsing (node attributes)
        if attrs[i].name == k:
            return i
        i -= 1
    return -1


def _heap_push(mut h: List[Int], v: Int):
    h.append(v)
    var i = len(h) - 1
    while i > 0:  # cpu-route: model file parsing (heap of ready graph nodes)
        var p = (i - 1) // 2
        if h[p] <= h[i]:
            break
        var t = h[p]
        h[p] = h[i]
        h[i] = t
        i = p


def _heap_pop(mut h: List[Int]) -> Int:
    var top = h[0]
    var last = h.pop()
    if len(h) > 0:
        h[0] = last
        var i = 0
        var n = len(h)
        while True:  # cpu-route: model file parsing (heap of ready graph nodes)
            var l = 2 * i + 1
            var r = l + 1
            var m = i
            if l < n and h[l] < h[m]:
                m = l
            if r < n and h[r] < h[m]:
                m = r
            if m == i:
                break
            var t = h[m]
            h[m] = h[i]
            h[i] = t
            i = m
    return top


def _toposort(mut g: OnnxGraph, nodes: List[GNode], available: Dict[Int, Bool]) -> List[GNode]:
    """Nodes in a topological order (the smallest ready index first, so a
    graph already sorted keeps its order); problems for redefined outputs,
    undefined inputs and cycles."""
    var producer = Dict[Int, Int]()
    for i in range(len(nodes)):  # cpu-route: model file parsing (graph nodes)
        for k in range(len(nodes[i].outputs)):  # cpu-route: model file parsing (node outputs)
            var o = nodes[i].outputs[k]
            if o == S_EMPTY:
                continue
            if o in producer or o in available:
                g.problem(nodes[i].name, nodes[i].op, "output '" + g.st.text(o) + "' is already defined")
            if o not in producer:
                producer[o] = i
    var deps = List[Int]()
    var users = List[List[Int]]()
    for _ in range(len(nodes)):  # cpu-route: model file parsing (graph nodes)
        users.append(List[Int]())
    for i in range(len(nodes)):  # cpu-route: model file parsing (graph nodes)
        var d = List[Int]()
        for k in range(len(nodes[i].inputs)):  # cpu-route: model file parsing (node inputs)
            var x = nodes[i].inputs[k]
            if x == S_EMPTY or x in available:
                continue
            var p = producer.get(x)
            if not p:
                g.problem(nodes[i].name, nodes[i].op,
                          "input '" + g.st.text(x) + "' is not produced by any node, initializer or input")
            elif p.value() != i:
                var seen = False
                for j in range(len(d)):  # cpu-route: model file parsing (a node's producers)
                    if d[j] == p.value():
                        seen = True
                if not seen:
                    d.append(p.value())
        deps.append(len(d))
        for j in range(len(d)):  # cpu-route: model file parsing (a node's producers)
            users[d[j]].append(i)
    var ready = List[Int]()
    for i in range(len(nodes)):  # cpu-route: model file parsing (graph nodes)
        if deps[i] == 0:
            _heap_push(ready, i)
    var order = List[GNode]()
    var placed = List[Bool](length=len(nodes), fill=False)
    while len(ready) > 0:  # cpu-route: model file parsing (graph nodes)
        var i = _heap_pop(ready)
        order.append(nodes[i].copy())
        placed[i] = True
        for k in range(len(users[i])):  # cpu-route: model file parsing (node users)
            var u = users[i][k]
            deps[u] -= 1
            if deps[u] == 0:
                _heap_push(ready, u)
    if len(order) != len(nodes):
        for i in range(len(nodes)):  # cpu-route: model file parsing (graph nodes)
            if not placed[i]:
                g.problem(nodes[i].name, nodes[i].op, "is part of a cycle or depends on a node that is")
    return order^


def _is_default_domain(d: String) -> Bool:
    return d.byte_length() == 0 or d == "ai.onnx"


def _read_graph(mut r: _Reader, path: String, m: RawModel) raises:
    """onnx.py read() after the wire decode: the checks and the graph."""
    var src = r.g.st.intern(_basename(path))
    var have_opset = False
    var opset = 0
    for i in range(len(m.opset_domain)):  # cpu-route: model file parsing (opset entries)
        if _is_default_domain(m.opset_domain[i]):
            opset = m.opset_version[i]
            have_opset = True
    if not have_opset:
        r.g.problem(src, -1, "declares no opset for the default ONNX domain")
        opset = 0
    elif opset < OPSET_MIN:
        r.g.problem(src, -1, "opset " + String(opset) + " (opset " + String(OPSET_MIN)
                    + " or later is needed; convert the model once with onnx.version_converter)")
    r.g.opset = opset
    var dead = Dict[Int, Bool]()  # refused initializers, inputs and Constants: defined, so users are not reported twice
    ref gr = m.graph[0]
    for i in range(len(gr.inits)):  # cpu-route: model file parsing (initializers)
        try:
            var t = r.tensor_info(gr.inits[i], gr.inits[i].name)
            r.g.set_init(t^)
        except e:
            var msg = String(e)
            if is_fatal(msg):
                raise Error(msg)
            var nm = r.g.st.intern(gr.inits[i].name)
            r.g.problem(nm, -1, msg)
            dead[nm] = True
    if gr.sparse > 0:
        r.g.problem(src, -1, String(gr.sparse) + " sparse initializers (not supported)")
    var nodes = List[GNode]()
    for i in range(len(gr.nodes)):  # cpu-route: model file parsing (graph nodes)
        ref nd = gr.nodes[i]
        var op = r.g.st.intern(nd.op)
        var name = r.g.st.intern(nd.name if nd.name.byte_length() > 0 else nd.op + "_" + String(i))
        var ins = List[Int]()
        for k in range(len(nd.inputs)):  # cpu-route: model file parsing (node inputs)
            ins.append(r.g.st.intern(nd.inputs[k]))
        var outs = List[Int]()
        for k in range(len(nd.outputs)):  # cpu-route: model file parsing (node outputs)
            outs.append(r.g.st.intern(nd.outputs[k]))
        if not _is_default_domain(nd.domain):
            r.g.problem(name, op, "domain '" + nd.domain + "' is not supported (only the default ONNX domain is)")
            nodes.append(GNode(op, r.g.st.intern(nd.domain), name, ins^, outs^, List[AttrVal]()))
            continue
        var attrs = List[AttrVal]()
        var bad = False
        for k in range(len(nd.attrs)):  # cpu-route: model file parsing (node attributes)
            try:
                attrs.append(r.attr(nd.attrs[k]))
            except e:
                var msg = String(e)
                if is_fatal(msg):
                    raise Error(msg)
                r.g.problem(name, op, msg)
                bad = True
        if nd.op == "Constant":
            if len(outs) != 1 or outs[0] == S_EMPTY:
                r.g.problem(name, op, "a Constant needs one output")
                continue
            try:
                if bad:
                    raise Error("its value cannot be read")
                var t = r.constant_tensor(outs[0], attrs)
                r.g.set_init(t^)
            except e:
                var msg = String(e)
                if is_fatal(msg):
                    raise Error(msg)
                r.g.problem(name, op, msg)
                dead[outs[0]] = True
            continue
        nodes.append(GNode(op, S_EMPTY, name, ins^, outs^, attrs^))
    var inames = Dict[Int, Bool]()
    for i in range(len(gr.inputs)):  # cpu-route: model file parsing (graph inputs)
        var nm = r.g.st.intern(gr.inputs[i].name)
        if r.g.init_of(nm) >= 0 or nm in dead:
            continue  # an initializer listed as an input (IR 3 style) stays a constant
        try:
            var v = r.value(gr.inputs[i], "input")
            r.g.inputs.append(v^)
            inames[nm] = True
        except e:
            r.g.problem(nm, -1, String(e))
            dead[nm] = True
    for i in range(len(gr.outputs)):  # cpu-route: model file parsing (graph outputs)
        try:
            var v = r.value(gr.outputs[i], "output")
            r.g.outputs.append(v^)
        except e:
            r.g.problem(r.g.st.intern(gr.outputs[i].name), -1, String(e))
    var available = Dict[Int, Bool]()
    for i in range(len(r.g.inits)):  # cpu-route: model file parsing (initializers)
        available[r.g.inits[i].name] = True
    for e in dead.items():  # cpu-route: model file parsing (refused names)
        available[e.key] = True
    for e in inames.items():  # cpu-route: model file parsing (graph inputs)
        available[e.key] = True
    r.g.nodes = _toposort(r.g, nodes, available)
    var produced = available.copy()
    for i in range(len(nodes)):  # cpu-route: model file parsing (graph nodes)
        for k in range(len(nodes[i].outputs)):  # cpu-route: model file parsing (node outputs)
            produced[nodes[i].outputs[k]] = True
    for i in range(len(r.g.outputs)):  # cpu-route: model file parsing (graph outputs)
        var nm = r.g.outputs[i].name
        if nm not in produced:
            r.g.problem(nm, -1, "output is not produced by any node")
    if len(r.g.problems) > 0:
        r.g.status = ONNX_PROBLEMS


def onnx_read_file(path: String) -> OnnxGraph:
    """Read and check the ONNX model at `path`. Never raises: the status,
    the message and the problems say what went wrong."""
    var body: List[UInt8]
    try:
        with open(path, "r") as fh:
            body = fh.read_bytes()
    except e:
        var g = OnnxGraph()
        g.status = ONNX_CANNOT_OPEN
        g.message = String(e)
        return g^
    var size = len(body)
    var base = String(".")
    try:
        var d = _dirname(path)
        base = realpath(d if d.byte_length() > 0 else ".")
    except e:
        var g = OnnxGraph()
        g.status = ONNX_CANNOT_OPEN
        g.message = String(e)
        return g^
    var r = _Reader(body^, base)
    r.g.buf_kind.append(0)
    var pk = r.g.st.intern(path)
    r.g.buf_path.append(pk)
    r.g.buf_size.append(size)
    r.g.buf_kind.append(1)
    r.g.buf_path.append(-1)
    r.g.buf_size.append(0)
    var m: RawModel
    try:
        m = _raw_model(r.model)
    except e:
        var msg = String(e)
        var g = OnnxGraph()
        if msg.startswith(PB_UTF8_PREFIX):
            g.status = ONNX_BAD_TEXT
        else:
            g.status = ONNX_NOT_A_MODEL
        g.message = msg
        return g^
    if not m.has_graph:
        var g = OnnxGraph()
        g.status = ONNX_NOT_A_MODEL
        g.message = String("it has no graph")
        return g^
    try:
        _read_graph(r, path, m)
    except e:
        var g = OnnxGraph()
        g.status = ONNX_BAD_VALUE
        g.message = fatal_text(String(e)) if is_fatal(String(e)) else String(e)
        return g^
    r.g.buf_size[BUF_BLOB] = len(r.g.blob)
    # the model and external file bytes are freed here; Python maps the files itself
    return r^.take_graph()


# ------------------------------------------------------------- the handles --

struct _OnnxStore(Defaultable, Movable):
    var ids: List[Int]
    var addr: List[Int]
    var next_id: Int

    def __init__(out self):
        self.ids = List[Int]()
        self.addr = List[Int]()
        self.next_id = 1


comptime _ONNX_NAME = "MojoOnnxReaderGraphsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoOnnxReaderGraphsFast"
comptime _ONNX_GRAPHS = _Global[StorageType=_OnnxStore, name=_ONNX_NAME, init_fn=_OnnxStore.__init__]
comptime OnnxPtr = UnsafePointer[OnnxGraph, MutUntrackedOrigin]


def onnx_graph(h: Int) raises -> OnnxPtr:
    """The graph of handle `h`."""
    var s = _ONNX_GRAPHS.get_or_create_ptr()
    for i in range(len(s[].ids)):  # small-loop(ids: open model graphs): a handle lookup, no data
        if s[].ids[i] == h:
            return OnnxPtr(unsafe_from_address=s[].addr[i])
    raise Error("onnx: model graph " + String(h) + " is not open")


def _w_tensor(w: PythonObject, t: TRec) raises:
    w.append(PythonObject(t.name))
    w.append(PythonObject(t.dtype))
    w.append(PythonObject(len(t.dims)))
    for i in range(len(t.dims)):  # cpu-route: model file parsing (one per axis)
        w.append(PythonObject(t.dims[i]))
    w.append(PythonObject(t.buf))
    w.append(PythonObject(t.off))
    w.append(PythonObject(t.off + t.nbytes))


def _w_value(w: PythonObject, v: GValue) raises:
    w.append(PythonObject(v.name))
    w.append(PythonObject(v.dtype))
    w.append(PythonObject(len(v.dims)))
    for i in range(len(v.dims)):  # cpu-route: model file parsing (one per axis)
        if v.dim_str[i] >= 0:
            w.append(PythonObject(1))
            w.append(PythonObject(v.dim_str[i]))
        else:
            w.append(PythonObject(0))
            w.append(PythonObject(v.dims[i]))


def onnx_read_py(path: PythonObject) raises -> PythonObject:
    """Read the ONNX model at `path`; returns its handle (>= 1)."""
    var file = String(py=path)
    var p = alloc[OnnxGraph](1)
    p.init_pointee_move(onnx_read_file(file))
    var s = _ONNX_GRAPHS.get_or_create_ptr()
    var h = s[].next_id
    s[].next_id += 1
    s[].ids.append(h)
    s[].addr.append(Int(p))
    return PythonObject(h)


def onnx_info_py(h: PythonObject) raises -> PythonObject:
    """[words, strings, floats] of handle h (the layout in this module's docstring)."""
    var g = onnx_graph(Int(py=h))
    var w = Python.list()
    var floats = Python.list()
    var nf = 0
    w.append(PythonObject(ONNX_READ_MAGIC))
    w.append(PythonObject(ONNX_READ_VERSION))
    w.append(PythonObject(g[].status))
    if g[].status == ONNX_PROBLEMS:
        w.append(PythonObject(len(g[].problems)))
        for i in range(len(g[].problems)):  # cpu-route: model file parsing (the problems found)
            w.append(PythonObject(g[].problems[i].name))
            w.append(PythonObject(g[].problems[i].op))
            w.append(PythonObject(g[].problems[i].reason))
    elif g[].status != ONNX_OK:
        w.append(PythonObject(g[].st.intern(g[].message)))
    else:
        w.append(PythonObject(g[].opset))
        w.append(PythonObject(len(g[].buf_kind)))
        for i in range(len(g[].buf_kind)):  # cpu-route: model file parsing (the model's files)
            w.append(PythonObject(g[].buf_kind[i]))
            w.append(PythonObject(g[].buf_path[i]))
            w.append(PythonObject(g[].buf_size[i]))
        w.append(PythonObject(len(g[].inputs)))
        for i in range(len(g[].inputs)):  # cpu-route: model file parsing (graph inputs)
            _w_value(w, g[].inputs[i])
        w.append(PythonObject(len(g[].outputs)))
        for i in range(len(g[].outputs)):  # cpu-route: model file parsing (graph outputs)
            _w_value(w, g[].outputs[i])
        w.append(PythonObject(len(g[].nodes)))
        for i in range(len(g[].nodes)):  # cpu-route: model file parsing (graph nodes)
            ref n = g[].nodes[i]
            w.append(PythonObject(n.op))
            w.append(PythonObject(n.domain))
            w.append(PythonObject(n.name))
            w.append(PythonObject(len(n.inputs)))
            for k in range(len(n.inputs)):  # cpu-route: model file parsing (node inputs)
                w.append(PythonObject(n.inputs[k]))
            w.append(PythonObject(len(n.outputs)))
            for k in range(len(n.outputs)):  # cpu-route: model file parsing (node outputs)
                w.append(PythonObject(n.outputs[k]))
            w.append(PythonObject(len(n.attrs)))
            for k in range(len(n.attrs)):  # cpu-route: model file parsing (node attributes)
                ref a = n.attrs[k]
                w.append(PythonObject(a.name))
                w.append(PythonObject(a.kind))
                if a.kind == AK_FLOAT:
                    w.append(PythonObject(nf))
                    floats.append(PythonObject(a.f))
                    nf += 1
                elif a.kind == AK_INT:
                    w.append(PythonObject(a.i))
                elif a.kind == AK_STRING:
                    w.append(PythonObject(a.s))
                elif a.kind == AK_TENSOR:
                    _w_tensor(w, a.tensors[0])
                elif a.kind == AK_FLOATS:
                    w.append(PythonObject(nf))
                    for j in range(len(a.floats)):  # cpu-route: model file parsing (attribute list)
                        floats.append(PythonObject(a.floats[j]))
                        nf += 1
                    w.append(PythonObject(nf))
                elif a.kind == AK_INTS:
                    w.append(PythonObject(len(a.ints)))
                    for j in range(len(a.ints)):  # cpu-route: model file parsing (attribute list)
                        w.append(PythonObject(a.ints[j]))
                elif a.kind == AK_STRINGS:
                    w.append(PythonObject(len(a.strs)))
                    for j in range(len(a.strs)):  # cpu-route: model file parsing (attribute list)
                        w.append(PythonObject(a.strs[j]))
                else:
                    w.append(PythonObject(len(a.tensors)))
                    for j in range(len(a.tensors)):  # cpu-route: model file parsing (attribute list)
                        _w_tensor(w, a.tensors[j])
        w.append(PythonObject(len(g[].inits)))
        for i in range(len(g[].inits)):  # cpu-route: model file parsing (initializers)
            _w_tensor(w, g[].inits[i])
    var strs = Python.list()
    for i in range(len(g[].st.items)):  # cpu-route: model file parsing (the graph's strings)
        strs.append(PythonObject(g[].st.items[i]))
    var out = Python.list()
    out.append(w)
    out.append(strs)
    out.append(floats)
    return out


def onnx_copy_blob_py(h: PythonObject, addr: PythonObject) raises -> PythonObject:
    """Copy the reader's own bytes (buffer 1) to `addr` (the caller's
    buffer of that size) and release them. Returns the byte count."""
    var g = onnx_graph(Int(py=h))
    var n = len(g[].blob)
    if n > 0:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=addr)),
               src=g[].blob.unsafe_ptr(), count=n)
    g[].blob = List[UInt8]()
    return PythonObject(n)


def onnx_close_py(h: PythonObject) raises -> PythonObject:
    """Release handle h."""
    var hid = Int(py=h)
    var s = _ONNX_GRAPHS.get_or_create_ptr()
    for i in range(len(s[].ids)):  # small-loop(ids: open model graphs): a handle lookup, no data
        if s[].ids[i] == hid:
            var p = OnnxPtr(unsafe_from_address=s[].addr[i])
            _ = s[].ids.pop(i)
            _ = s[].addr.pop(i)
            p.destroy_pointee()
            p.free()
            return PythonObject(0)
    return PythonObject(0)
