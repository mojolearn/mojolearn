"""Classify tests without importing (and accidentally executing) module gates."""
import ast
from pathlib import Path


def inventory(root):
    suites = {'pytest': [], 'gates': []}
    for path in sorted(Path(root).glob('test_*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        has_tests = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith('test_')
            or isinstance(node, ast.ClassDef) and (
                node.name.startswith('Test') or any(
                    isinstance(base, ast.Attribute) and base.attr == 'TestCase'
                    or isinstance(base, ast.Name) and base.id == 'TestCase'
                    for base in node.bases))
            for node in tree.body)
        suites['pytest' if has_tests else 'gates'].append(path)
    return suites
