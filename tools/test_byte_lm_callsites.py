#!/usr/bin/env python3
"""Every in-file call must match the signature it calls, checked by the parser.

WHY THIS FILE EXISTS, AND WHAT IT COST. DEVIATION 2682 threaded a `shape`
argument through the byte LM capture harness and its verifiers, which changed
the arity of ten functions. One call site was missed, on a continuation line
inside a dict literal, and Python does not notice a missing positional argument
until the line executes. That line executes only at the very END of a capture,
after the device has done all of its work, so the miss was discovered on a
rented GPU that had already computed the step correctly and written its arrays.

`py_compile` cannot see this. Nor can a normal import. The parser can, and that
costs nothing, so it runs here and in CI.

The check is deliberately narrow. It resolves only calls to functions DEFINED IN
THE SAME FILE and reached by a bare name, which is exactly the class that a
local refactor breaks, and it makes no attempt to follow attributes, aliases or
imports, because a wrong answer from a checker is worse than no checker.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The shape-threaded harness and its verifiers, plus the shape module itself.
FILES = (
    'tools/byte_lm_real_text_capture.py',
    'tools/byte_lm_gradient_oracle.py',
    'tools/byte_lm_state_compare.py',
    'tools/byte_lm_validation_admit.py',
    'tools/byte_lm_cpu_train_gate.py',
    'tools/byte_lm_shape.py',
)


def signatures(tree):
    """Name to (minimum, maximum, keyword names) for module-level functions.

    A maximum of None means the function takes *args and cannot be over-filled.
    Methods are included by name; calls to them go through an attribute and are
    therefore never resolved below, which is the conservative outcome."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        positional = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
        required = len(positional) - len(a.defaults)
        maximum = None if a.vararg is not None else len(positional)
        keywords = {p.arg for p in a.kwonlyargs} | set(positional)
        required_kw = {p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is None}
        out[node.name] = (required, maximum, keywords, required_kw, a.kwarg is not None)
    return out


def mismatches(path):
    """Every bare-name call in `path` that cannot satisfy its own signature."""
    tree = ast.parse((ROOT / path).read_text())
    sigs = signatures(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in sigs:
            continue
        required, maximum, keywords, required_kw, takes_kwargs = sigs[name]
        # A splat can supply any number, so it is never a miss.
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            continue
        if any(kw.arg is None for kw in node.keywords):
            continue
        given_positional = len(node.args)
        given_names = {kw.arg for kw in node.keywords}
        total = given_positional + len(given_names)
        if total < required:
            found.append((node.lineno, name, f'given {total}, needs at least {required}'))
            continue
        if maximum is not None and given_positional > maximum:
            found.append((node.lineno, name, f'{given_positional} positional, takes at most {maximum}'))
            continue
        unknown = given_names - keywords
        if unknown and not takes_kwargs:
            found.append((node.lineno, name, f'unknown keyword(s) {sorted(unknown)}'))
            continue
        # A keyword-only parameter with no default must be supplied by name.
        missing_kw = required_kw - given_names
        if missing_kw:
            found.append((node.lineno, name, f'missing keyword-only {sorted(missing_kw)}'))
    return found


@pytest.mark.parametrize('path', FILES)
def test_every_in_file_call_matches_its_signature(path):
    found = mismatches(path)
    assert not found, '\n'.join(f'{path}:{line}: {name}() {why}' for line, name, why in found)


def test_the_checker_catches_a_missing_positional_argument():
    """The checker is only worth having if it fails on the bug it is named for.
    This is the exact shape of the miss that cost a lease, a call on a
    continuation line inside a dict literal."""
    source = '''
def state_signature(state, shape):
    return (state, shape)

def main(state, shape):
    return dict(a=1,
        initial_state=state_signature(state), final=state_signature(state, shape))
'''
    tree = ast.parse(source)
    sigs = signatures(tree)
    assert sigs['state_signature'][0] == 2
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'state_signature']
    short = [n for n in calls if len(n.args) + len(n.keywords) < 2]
    assert len(short) == 1


def test_a_keyword_only_argument_is_not_reported_as_missing():
    """`load_foreign_checkpoint(cls, source, destination, *, resident=False)`
    called with three positionals and `resident=` is correct. An earlier version
    of this scan counted only positional parameters and called that a defect,
    which is the failure mode of a checker nobody can trust."""
    source = '''
def load_foreign_checkpoint(cls, source, destination, *, resident=False):
    return cls

def main():
    return load_foreign_checkpoint(int, 'a', 'b', resident=True)
'''
    tree = ast.parse(source)
    sigs = signatures(tree)
    required, maximum, keywords, required_kw, _ = sigs['load_foreign_checkpoint']
    assert (required, maximum) == (3, 3)
    assert 'resident' in keywords and not required_kw
