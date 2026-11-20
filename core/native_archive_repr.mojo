# SPDX-License-Identifier: Apache-2.0
"""Render an admitted path/error-name graph using native nodes and tasks.

Child order, aliases and active-container cycles determine the display. One
graph therefore has a dependent traversal; independent graph fixtures use
the host pool. Python owners and captured native exceptions stay at UNPACK.
"""
from std.collections import Dict


struct NativeArchiveReprNode(Movable):
    var kind: Int
    var text: String
    var children: List[Int]
    var pure_path: Bool
    var closefd: Bool
    var failure: Int

    def __init__(out self, kind: Int = 0, text: String = ""):
        self.kind = kind
        self.text = text
        self.children = List[Int]()
        self.pure_path = False
        self.closefd = True
        self.failure = -1


struct NativeArchiveReprResult(Movable):
    var text: String
    var failure: Int

    def __init__(out self, var text: String, failure: Int = -1):
        self.text = text^
        self.failure = failure


struct _ArchiveReprTask(Movable):
    var operation: Int
    var node: Int
    var text: String

    def __init__(out self, operation: Int, node: Int = 0, text: String = ""):
        self.operation = operation
        self.node = node
        self.text = text


def archive_repr_format(nodes: List[NativeArchiveReprNode]) raises -> NativeArchiveReprResult:
    if len(nodes) == 0:
        raise Error("archive representation requires a root node")
    var pending = List[_ArchiveReprTask]()
    pending.append(_ArchiveReprTask(0))
    var active = Dict[Int, Bool]()
    var result = String()
    while len(pending):
        var task = pending.pop()
        if task.operation == 1:
            result += task.text
            continue
        var index = task.node
        if index < 0 or index >= len(nodes):
            raise Error("archive representation child is outside its graph")
        if task.operation == 2:
            _ = active.pop(index)
            continue
        if nodes[index].failure >= 0:
            return NativeArchiveReprResult(result^, nodes[index].failure)
        var kind = nodes[index].kind
        if kind == 0 or kind == 7:
            result += nodes[index].text
        elif kind == 1:
            result += "PurePosixPath(" if nodes[index].pure_path else "PosixPath("
            result += nodes[index].text
            result += ")"
        elif kind >= 2 and kind <= 6:
            if index in active:
                result += "[...]" if kind == 2 else ("(...)" if kind == 3 else ("{...}" if kind == 4 else ("set(...)" if kind == 5 else "frozenset(...)")))
                continue
            var size = len(nodes[index].children)
            if kind == 4 and size % 2:
                raise Error("archive representation dictionary has an unmatched key")
            if size == 0:
                result += "[]" if kind == 2 else ("()" if kind == 3 else ("{}" if kind == 4 else ("set()" if kind == 5 else "frozenset()")))
                continue
            active[index] = True
            pending.append(_ArchiveReprTask(2, index))
            pending.append(_ArchiveReprTask(1, text="]" if kind == 2 else (")" if kind == 3 else ("})" if kind == 6 else "}"))))
            if kind == 3 and size == 1:
                pending.append(_ArchiveReprTask(1, text=","))
            for i in range(size - 1, -1, -1):
                pending.append(_ArchiveReprTask(0, nodes[index].children[i]))
                if i:
                    pending.append(_ArchiveReprTask(1, text=": " if kind == 4 and i % 2 else ", "))
            result += "[" if kind == 2 else ("(" if kind == 3 else ("frozenset({" if kind == 6 else "{"))
        elif kind >= 8 and kind <= 11:
            if len(nodes[index].children) != 1:
                raise Error("archive representation file requires one name")
            var label = String("FileIO" if kind == 8 else ("BufferedReader" if kind == 9 else ("BufferedWriter" if kind == 10 else "BufferedRandom")))
            if index in active:
                raise Error("RuntimeError: reentrant call inside _io." + label + ".__repr__")
            active[index] = True
            pending.append(_ArchiveReprTask(2, index))
            var suffix = String(">")
            if kind == 8:
                suffix = " mode=" + nodes[index].text + " closefd=" + ("True" if nodes[index].closefd else "False") + ">"
            pending.append(_ArchiveReprTask(1, text=suffix))
            pending.append(_ArchiveReprTask(0, nodes[index].children[0]))
            result += "<_io." + label + " name="
        else:
            raise Error("archive representation node has an unknown native kind")
    return NativeArchiveReprResult(result^)
