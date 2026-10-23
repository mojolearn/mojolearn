"""Conservative literal-operation specialization for source dependency scans.

This is a text analyzer, not a Mojo parser or an execution engine. Callers must
provide independently proven string arguments, and include those arguments in
their reach-cache keys. Unsupported syntax keeps the original source. Unknown
predicates retain both branches; class/schema dispatch is deliberately unknown.
No public export or verifier lane is named here.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import io
import re
import tokenize
from typing import Mapping


Strings = frozenset[str]


@dataclass(frozen=True)
class Call:
    name: str
    arguments: tuple[Strings | None, ...]
    keywords: tuple[tuple[str | None, Strings | None], ...]
    line: int
    unpacking: bool = False


@dataclass(frozen=True)
class Reach:
    text: str
    calls: tuple[Call, ...]
    supported: bool
    reason: str = ""


@dataclass
class _Statement:
    code: str
    line: int
    end: int
    indent: int
    children: list['_Statement'] = field(default_factory=list)

    @property
    def last(self):
        return self.children[-1].last if self.children else self.end


@dataclass
class _State:
    strings: dict[str, Strings]
    packed: dict[str, dict[int, Strings]]
    constructors: frozenset[str] = frozenset()

    def copy(self):
        return _State(dict(self.strings), {k: dict(v) for k, v in self.packed.items()}, self.constructors)


def _statements(text):
    lines = text.splitlines()
    rows, tokens, start = [], [], None
    ignored = {tokenize.INDENT, tokenize.DEDENT, tokenize.NL,
               tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.ERRORTOKEN and not token.string.isspace():
            raise ValueError('unsupported lexical token')
        if token.type == tokenize.NEWLINE:
            if tokens:
                prefix = lines[start - 1]
                indent = len(prefix) - len(prefix.lstrip(' '))
                if '\t' in prefix[:len(prefix) - len(prefix.lstrip())]:
                    raise ValueError('tab indentation')
                rows.append(_Statement(' '.join(tokens), start, token.end[0], indent))
            tokens, start = [], None
        elif token.type not in ignored:
            if start is None:
                start = token.start[0]
            tokens.append(token.string)
    if tokens:
        raise ValueError('unterminated statement')
    roots, stack = [], []
    for row in rows:
        while stack and row.indent <= stack[-1].indent:
            stack.pop()
        if stack:
            if not stack[-1].code.endswith(':'):
                raise ValueError('indented statement without a block')
            stack[-1].children.append(row)
        else:
            if row.indent:
                raise ValueError('unattached indentation')
            roots.append(row)
        stack.append(row)
    return roots


def _expression(text):
    try:
        return ast.parse(text, mode='eval').body
    except (SyntaxError, ValueError):
        return None


def _strings(node, state):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return frozenset((node.value,))
    if isinstance(node, ast.Name):
        return state.strings.get(node.id)
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int):
        return state.packed.get(node.value.id, {}).get(node.slice.value)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in state.constructors and len(node.args) == 1
            and not node.keywords):
        return _strings(node.args[0], state)
    return None


def _truth(node, state):
    """Three-valued result; only exact string operations are evaluated."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _truth(node.operand, state)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_truth(value, state) for value in node.values]
        if isinstance(node.op, ast.And):
            return False if False in values else (True if all(v is True for v in values) else None)
        if isinstance(node.op, ast.Or):
            return True if True in values else (False if all(v is False for v in values) else None)
    if (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))):
        left, right = _strings(node.left, state), _strings(node.comparators[0], state)
        if left is not None and right is not None:
            outcomes = {(a == b) if isinstance(node.ops[0], ast.Eq) else (a != b)
                        for a in left for b in right}
            return next(iter(outcomes)) if len(outcomes) == 1 else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'startswith' and len(node.args) == 1 and not node.keywords):
        values, prefixes = _strings(node.func.value, state), _strings(node.args[0], state)
        if values is not None and prefixes is not None:
            outcomes = {v.startswith(p) for v in values for p in prefixes}
            return next(iter(outcomes)) if len(outcomes) == 1 else None
    return None


def _pure_predicate(node, state):
    """Do not discard an unknown call evaluated before a known-false operand."""
    if isinstance(node, ast.BoolOp):
        return all(_pure_predicate(n, state) for n in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _pure_predicate(node.operand, state)
    if (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))):
        return _strings(node.left, state) is not None and _strings(node.comparators[0], state) is not None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'startswith' and len(node.args) == 1 and not node.keywords):
        return _strings(node.func.value, state) is not None and _strings(node.args[0], state) is not None
    return False


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return base + '.' + node.attr if base else None
    if isinstance(node, ast.Subscript):  # Mojo generic call foo[T](...).
        return _call_name(node.value)
    return None


def _join(states):
    if not states:
        return _State({}, {})
    common = set.intersection(*(set(s.strings) for s in states))
    strings = {key: frozenset.union(*(s.strings[key] for s in states)) for key in common}
    packed = {key: value for key, value in states[0].packed.items()
              if all(s.packed.get(key) == value for s in states[1:])}
    constructors = frozenset.intersection(*(s.constructors for s in states))
    return _State(strings, packed, constructors)


def _parameter_names(code):
    """Read a plain function header; complicated generic headers stay broad."""
    match = re.match(r'^(?:def|fn)\s+\w+\s*\(', code)
    if match is None:
        raise ValueError('unsupported function header')
    parameters, current, depth = [], [], 0
    for token in tokenize.generate_tokens(io.StringIO(code[match.end():]).readline):
        value = token.string
        if value == ')' and depth == 0:
            if current:
                parameters.append(current)
            break
        if value == ',' and depth == 0:
            parameters.append(current); current = []
            continue
        if value in ('(', '[', '{'):
            depth += 1
        elif value in (')', ']', '}'):
            depth -= 1
        current.append(value)
    names = []
    for parameter in parameters:
        if ':' not in parameter:
            raise ValueError('untyped function parameter')
        prefix = parameter[:parameter.index(':')]
        if not prefix or not re.fullmatch(r'[A-Za-z_]\w*', prefix[-1]):
            raise ValueError('unsupported parameter declaration')
        names.append(prefix[-1])
    return tuple(names)


def parameter_names(text: str) -> tuple[str, ...] | None:
    """Ordered parameters of one supported plain function, else unknown."""
    try:
        roots = _statements(text)
        functions = [r for r in roots if re.match(r'^(?:def|fn)\s+\w+', r.code)]
        if len(functions) != 1:
            return None
        return _parameter_names(functions[0].code)
    except (ValueError, SyntaxError, tokenize.TokenError, IndentationError, IndexError):
        return None


def _constructor_binding(rows, constructors):
    """A locally rebound constructor invalidates the caller's import proof.

    Inspect all blocks before specialization: a loop/ref/alias binding must
    not reintroduce constructor trust after an uncertain control-flow merge.
    Ambiguous assignment targets are refused rather than interpreted.
    """
    for row in rows:
        code = row.code
        candidates = []
        declaration = re.match(r'^(?:var|let|ref|alias|comptime)\s+(.+)', code)
        if declaration:
            candidates.append(declaration.group(1).split('=', 1)[0].split(':', 1)[0])
        assignment = re.match(r'^(.*?)\s*(?:[+*/%&|^-]=|=(?!=))', code)
        if assignment and not re.match(r'^(?:if|elif|while|return|raise)\b', code):
            candidates.append(assignment.group(1).split(':', 1)[0])
        loop = re.search(r'\bfor\s+(.+?)\s+in\b', code)
        if loop:
            candidates.append(loop.group(1))
        candidates.extend(re.findall(r'\bas\s+([A-Za-z_]\w*)', code))
        definition = re.match(r'^(?:def|fn|struct|trait)\s+([A-Za-z_]\w*)', code)
        if definition:
            candidates.append(definition.group(1))
        if re.match(r'^(?:from|import)\b', code):
            candidates.append(code)  # Local import resolution is not proven here.
        if any(set(re.findall(r'\b[A-Za-z_]\w*\b', value)) & constructors for value in candidates):
            return True
        if _constructor_binding(row.children, constructors):
            return True
    return False


def specialize_function(text: str, *, parameters: Mapping[str, Strings] | None = None,
                        packed_parameters: Mapping[str, Mapping[int, Strings]] | None = None,
                        constructors: frozenset[str] = frozenset()) -> Reach:
    """Retain source reachable for proven literal string parameters.

    ``packed_parameters={'args_ptr': {1: frozenset({'ols_fit'})}}`` models
    a proven CPython packed argument. Its facts become visible only through
    ``PythonObject(from_borrowed=args_ptr)`` and constant tuple indexing.
    Direct callees instead receive ``parameters={'operation': ...}``.

    Results preserve original line numbers. ``calls`` carries facts for
    explicit call arguments, including String/PythonObject literal wrappers;
    unknown expressions and argument unpacking must stay broad downstream.
    This does not resolve a cached function object or a class/schema table.
    ``constructors`` may contain String/PythonObject only after the caller
    proves those names resolve to the standard native constructors in this
    file; the default assumes no such name-resolution proof.
    """
    try:
        roots = _statements(text)
        functions = [r for r in roots if re.match(r'^(?:def|fn)\s+\w+', r.code)]
        if len(functions) != 1 or any(r.children and r is not functions[0] for r in roots):
            raise ValueError('expected one function, not overloads or a type body')
        function = functions[0]
        if not function.children or any(re.match(r'^(?:def|fn|struct|trait)\b', r.code)
                                       for r in function.children):
            raise ValueError('empty or nested definition')
        names = set(_parameter_names(function.code))
        if set(constructors) - {'String', 'PythonObject'} or names & constructors:
            raise ValueError('shadowed string constructors')
        if _constructor_binding(function.children, constructors):
            raise ValueError('locally bound string constructors')
        if not set(parameters or {}) <= names or not set(packed_parameters or {}) <= names:
            raise ValueError('facts do not name function header parameters')
        state = _State({k: frozenset(v) for k,v in (parameters or {}).items()},
                       {k: {i: frozenset(v) for i,v in values.items()}
                        for k,values in (packed_parameters or {}).items()}, frozenset(constructors))
        if any(not values or any(not isinstance(v, str) for v in values)
               for values in list(state.strings.values()) + [v for p in state.packed.values() for v in p.values()]):
            raise ValueError('string facts must be nonempty sets of strings')
        lines = text.splitlines(keepends=True)
        retained = [True] * len(lines)
        calls = []

        def discard(row):
            for i in range(row.line - 1, row.last):
                retained[i] = False

        def scan(node, current, line):
            if node is None:
                current.strings.clear()
                current.packed.clear()
                return
            invalidate = set()
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                name = _call_name(call.func)
                if name:
                    argument_state = current.copy()
                    # An earlier argument can mutate a value read by a later
                    # argument. Do not forward its pre-call value downstream.
                    for nested in (n for n in ast.walk(call) if isinstance(n, ast.Call) and n is not call):
                        if _call_name(nested.func) in current.constructors or _pure_predicate(nested, current):
                            continue
                        for referenced in (n.id for n in ast.walk(nested) if isinstance(n, ast.Name)):
                            argument_state.strings.pop(referenced, None)
                            argument_state.packed.pop(referenced, None)
                    calls.append(Call(name, tuple(_strings(a, argument_state) for a in call.args),
                                      tuple((k.arg, _strings(k.value, argument_state)) for k in call.keywords),
                                      line, any(isinstance(a, ast.Starred) for a in call.args)
                                      or any(k.arg is None for k in call.keywords)))
                if name in current.constructors or _pure_predicate(call, current):
                    continue
                invalidate.update(n.id for n in ast.walk(call) if isinstance(n, ast.Name))
            for name in invalidate:
                current.strings.pop(name, None)
                current.packed.pop(name, None)

        def statement(row, current):
            code = row.code
            if row.children:
                # A try body starts once with these facts; exceptions remain
                # retained as separate unknown blocks. Loops/with/defer/etc.
                # may mutate or repeat, so facts are not narrowed inside them.
                if code == 'try :':
                    inside = current.copy()
                else:
                    scan(_expression(code.removesuffix(':')), current, row.line)
                    inside = _State({}, {}, current.constructors)
                sequence(row.children, inside)
                return _State({}, {}, current.constructors), False
            code = re.sub(r'^(?:var|let)\s+', '', code)
            if re.match(r'^(?:ref|alias)\b', code):
                current.strings.clear()
                current.packed.clear()
                return current, False
            terminal = bool(re.match(r'^(?:return|raise|break|continue)\b', code))
            if code in ('return', 'raise', 'break', 'continue', 'pass'):
                return current, terminal
            expression = re.sub(r'^(?:return|raise)\s+', '', code) if terminal else code
            assigned = re.match(r'^([A-Za-z_]\w*)\s*(?::[^=]+)?=\s*(.*)$', expression)
            target = assigned.group(1) if assigned else None
            if target in current.constructors:
                raise ValueError('shadowed string constructors')
            node = _expression(assigned.group(2) if assigned else expression)
            if node is None:
                # Some Python tokenizer versions classify unknown Mojo-like
                # punctuation (for example !) as OP rather than ERRORTOKEN.
                # A leaf statement we cannot parse is not a supported scan.
                raise ValueError('unsupported statement expression')
            values = _strings(node, current)
            packed = None
            if isinstance(node, ast.Name) and node.id in current.packed:
                # An untracked alias could mutate the same packed object.
                current.packed.clear()
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'PythonObject' and 'PythonObject' in current.constructors
                    and not node.args and len(node.keywords) == 1
                    and node.keywords[0].arg == 'from_borrowed' and isinstance(node.keywords[0].value, ast.Name)):
                packed = current.packed.get(node.keywords[0].value.id)
            scan(node, current, row.line)
            if target:
                current.strings.pop(target, None)
                current.packed.pop(target, None)
                if values is not None:
                    current.strings[target] = values
                if packed is not None:
                    current.packed[target] = dict(packed)
            return current, terminal

        def sequence(rows, current):
            index = 0
            while index < len(rows):
                row = rows[index]
                if re.match(r'^if\s', row.code) and row.children:
                    chain = [row]
                    index += 1
                    while index < len(rows) and re.match(r'^(?:elif\s|else\s*:)', rows[index].code):
                        chain.append(rows[index]); index += 1
                    alive, remaining = [], current.copy()
                    for branch in chain:
                        if remaining is None:
                            discard(branch); continue
                        if re.match(r'^else\s*:', branch.code):
                            result, stops = sequence(branch.children, remaining.copy())
                            if not stops:
                                alive.append(result)
                            remaining = None
                            continue
                        condition = _expression(re.sub(r'^(?:if|elif)\s+', '', branch.code).removesuffix(':').strip())
                        known = _truth(condition, remaining) if _pure_predicate(condition, remaining) else None
                        if known is False:
                            discard(branch); continue
                        scan(condition, remaining, branch.line)
                        result, stops = sequence(branch.children, remaining.copy())
                        if not stops:
                            alive.append(result)
                        if known is True:
                            remaining = None
                    if remaining is not None:
                        alive.append(remaining)
                    if not alive:
                        for later in rows[index:]:
                            discard(later)
                        return _State({}, {}), True
                    current = _join(alive)
                    continue
                current, stops = statement(row, current)
                index += 1
                if stops:
                    for later in rows[index:]:
                        discard(later)
                    return current, True
            return current, False

        sequence(function.children, state)
        result = ''.join(line if keep else ('\n' if line.endswith('\n') else '')
                         for line,keep in zip(lines, retained))
        return Reach(result, tuple(calls), True)
    except (ValueError, SyntaxError, tokenize.TokenError, IndentationError, IndexError):
        return Reach(text, (), False, 'unsupported syntax or unproven parameter facts; retain original source')
