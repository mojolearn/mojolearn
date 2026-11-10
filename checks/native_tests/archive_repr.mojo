# SPDX-License-Identifier: Apache-2.0
"""Typed archive error-name graphs: display, aliases, cycles and first failure."""
from core.host_parallel import host_parallelize
from core.native_archive_repr import NativeArchiveReprNode, archive_repr_format


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _display(nodes: List[NativeArchiveReprNode], expected: String) raises:
    var result = archive_repr_format(nodes)
    _require(result.failure == -1 and result.text == expected, "native archive graph display")


def _empty(unit: Int) raises:
    var nodes = List[NativeArchiveReprNode]()
    nodes.append(NativeArchiveReprNode(unit + 2))
    var expected: List[String] = ["[]", "()", "{}", "set()", "frozenset()"]
    _display(nodes, expected[unit])


def _single(unit: Int) raises:
    var nodes = List[NativeArchiveReprNode]()
    nodes.append(NativeArchiveReprNode(unit + 2))
    nodes.append(NativeArchiveReprNode(text="1"))
    if unit == 2:
        nodes.append(NativeArchiveReprNode(text="'a'"))
        nodes[0].children = [2, 1]
    else:
        nodes[0].children = [1]
    var expected: List[String] = ["[1]", "(1,)", "{'a': 1}", "{1}", "frozenset({1})"]
    _display(nodes, expected[unit])


def _graph(unit: Int) raises:
    var nodes = List[NativeArchiveReprNode]()
    nodes.append(NativeArchiveReprNode(2))
    nodes.append(NativeArchiveReprNode(2))
    nodes.append(NativeArchiveReprNode(1, "'//a/b'"))
    nodes[2].pure_path = True
    nodes.append(NativeArchiveReprNode(text="2j"))
    var expected = String("")
    if unit == 0:
        nodes[0].children = [1, 1]
        nodes[1].children = [2, 3]
        expected = "[[PurePosixPath('//a/b'), 2j], [PurePosixPath('//a/b'), 2j]]"
    elif unit == 1:
        nodes[0].children = [2, 0]
        expected = "[PurePosixPath('//a/b'), [...]]"
    elif unit == 2:
        nodes[0].kind = 3
        nodes[0].children = [1]
        nodes[1].children = [2, 0]
        expected = "([PurePosixPath('//a/b'), (...)],)"
    else:
        nodes[0].kind = 4
        nodes[1].kind = 0
        nodes[1].text = "'path'"
        nodes[3].text = "'cycle'"
        nodes[0].children = [1, 2, 3, 0]
        expected = "{'path': PurePosixPath('//a/b'), 'cycle': {...}}"
    _display(nodes, expected)


def _file(unit: Int) raises:
    var kind = 8 + min(unit, 3)
    if unit == 4:
        kind = 8
    var nodes = List[NativeArchiveReprNode]()
    nodes.append(NativeArchiveReprNode(kind, "'rb'"))
    nodes[0].closefd = unit != 4
    nodes[0].children = [1]
    nodes.append(NativeArchiveReprNode(1, "'/a/b'"))
    var expected: List[String] = [
        "<_io.FileIO name=PosixPath('/a/b') mode='rb' closefd=True>",
        "<_io.BufferedReader name=PosixPath('/a/b')>",
        "<_io.BufferedWriter name=PosixPath('/a/b')>",
        "<_io.BufferedRandom name=PosixPath('/a/b')>",
        "<_io.FileIO name=PosixPath('/a/b') mode='rb' closefd=False>",
    ]
    _display(nodes, expected[unit])


def _file_cycle(unit: Int) raises:
    var nodes = List[NativeArchiveReprNode]()
    nodes.append(NativeArchiveReprNode(8 + unit, "'rb'"))
    nodes[0].children = [1]
    nodes.append(NativeArchiveReprNode(2))
    nodes[1].children = [2, 0]
    nodes.append(NativeArchiveReprNode(1, "'a'"))
    var names: List[String] = ["FileIO", "BufferedReader", "BufferedWriter", "BufferedRandom"]
    var refused = False
    try:
        _ = archive_repr_format(nodes)
    except problem:
        refused = True
        _require(String(problem) == "RuntimeError: reentrant call inside _io." + names[unit] + ".__repr__", "native file cycle refusal")
    _require(refused, "native file cycle must refuse")


def _failure_order(unit: Int) raises:
    var nodes = List[NativeArchiveReprNode]()
    nodes.append(NativeArchiveReprNode(2))
    nodes.append(NativeArchiveReprNode())
    nodes[1].failure = 7
    nodes.append(NativeArchiveReprNode(8, "'rb'"))
    nodes[2].children = [2]
    nodes[0].children = [1, 2]
    if unit == 1:
        nodes[0].children = [2, 1]
        var refused = False
        try:
            _ = archive_repr_format(nodes)
        except problem:
            refused = True
            _require(String(problem) == "RuntimeError: reentrant call inside _io.FileIO.__repr__", "cycle precedes later captured scalar failure")
        _require(refused, "first cycle must refuse")
        return
    elif unit == 2:
        nodes[2].kind = 0
        nodes[2].failure = 3
        nodes[0].children = [1, 2]
    var result = archive_repr_format(nodes)
    _require(result.failure == 7, "first scalar failure follows display order")


def _deep(unit: Int) raises:
    var depth = 64 if unit == 0 else (256 if unit == 1 else 4096)
    var nodes = List[NativeArchiveReprNode]()
    for i in range(depth):
        nodes.append(NativeArchiveReprNode(2))
        nodes[i].children.append(i + 1)
    nodes.append(NativeArchiveReprNode(1, "'a'"))
    var result = archive_repr_format(nodes)
    var middle = String("PosixPath('a')")
    var bytes = result.text.as_bytes()
    _require(result.failure == -1 and len(bytes) == 2 * depth + len(middle.as_bytes()), "deep graph display size")
    for i in range(depth):
        _require(bytes[i] == 91 and bytes[len(bytes) - 1 - i] == 93, "deep graph delimiters")
    for i in range(len(middle.as_bytes())):
        _require(bytes[depth + i] == middle.as_bytes()[i], "deep graph leaf")


def _invalid(unit: Int) raises:
    var nodes = List[NativeArchiveReprNode]()
    var expected = String("archive representation requires a root node")
    if unit == 1:
        nodes.append(NativeArchiveReprNode(2))
        nodes[0].children = [1]
        expected = "archive representation child is outside its graph"
    elif unit == 2:
        nodes.append(NativeArchiveReprNode(4))
        nodes[0].children = [0]
        expected = "archive representation dictionary has an unmatched key"
    elif unit == 3:
        nodes.append(NativeArchiveReprNode(8))
        expected = "archive representation file requires one name"
    elif unit == 4:
        nodes.append(NativeArchiveReprNode(99))
        expected = "archive representation node has an unknown native kind"
    var refused = False
    try:
        _ = archive_repr_format(nodes)
    except problem:
        refused = True
        _require(String(problem) == expected, "invalid native graph refusal")
    _require(refused, "invalid native graph must refuse")


def main() raises:
    comptime FIXTURES = 34
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 5:
                _empty(unit)
            elif unit < 10:
                _single(unit - 5)
            elif unit < 14:
                _graph(unit - 10)
            elif unit < 19:
                _file(unit - 14)
            elif unit < 23:
                _file_cycle(unit - 19)
            elif unit < 26:
                _failure_order(unit - 23)
            elif unit < 29:
                _deep(unit - 26)
            else:
                _invalid(unit - 29)
        except problem:
            print("FAIL archive_repr fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "archive representation failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS archive_repr:", FIXTURES, "native fixtures, parallel")
