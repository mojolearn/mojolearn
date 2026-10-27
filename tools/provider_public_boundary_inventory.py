#!/usr/bin/env python3
"""Source census of candidate public call bodies; never runtime qualification.

Include private implementation modules and base classes: public reexports and
inherited methods can reach their bodies. Record bases/decorators so these are
not silently treated as fully resolved public APIs. The manifest is for planning
and locating actual method bodies, not an exemption list or a Python-free claim.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

METADATA = {'shape', 'ndim', 'size', 'dtype', 'itemsize', 'strides', 'data_ptr', '__array_interface__'}
NATIVE_NAMES = {'_NATIVE', '_NATIVE_CALL', '_NATIVE_OWNER', '_NATIVE_DOOR'}
PROTOCOLS = {'__init__', '__call__', '__getstate__', '__setstate__', '__len__', '__getitem__',
             '__array__', '__array_interface__', '__iter__', '__enter__', '__exit__', '__sklearn_tags__'}

def describe(source, filename):
    tree = ast.parse(source, filename=filename)
    records = []
    owners = [(None, [], [], [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])]
    owners += [(c.name, [ast.unparse(b) for b in c.bases], [ast.unparse(d) for d in c.decorator_list],
                [n for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
               for c in tree.body if isinstance(c, ast.ClassDef)]
    for owner, bases, class_decorators, methods in owners:
        for method in methods:
            if method.name.startswith('_') and method.name not in PROTOCOLS:
                continue
            nodes = [n for statement in method.body for n in ast.walk(statement)]
            calls = [n for n in nodes if isinstance(n, ast.Call)]
            loops = [n for n in nodes if isinstance(n, (ast.For, ast.AsyncFor, ast.While, ast.comprehension))]
            names = [ast.unparse(n.func) for n in calls]
            entries = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id in NATIVE_NAMES]
            operations = [arg.value for call in entries for arg in call.args
                          if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            state = 'requires_manual_or_transitive_audit'
            if method.name == '__sklearn_tags__':
                state = 'allowed_sklearn_tag_metadata_only'
            elif len(calls) == 1 and len(entries) == 1 and not loops:
                state = 'single_known_native_call_body'
            metadata = sorted({ast.unparse(n) for n in nodes if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and n.attr in METADATA})
            loaders = sorted({name for name in names if name.endswith(('._bind', '.binding', '.load_set'))})
            records.append(dict(path=filename, line=method.lineno, owner=owner, name=method.name,
                                bases=bases, class_decorators=class_decorators,
                                decorators=[ast.unparse(d) for d in method.decorator_list],
                                body_classification=state, operations=operations,
                                call_count=len(calls), loop_count=len(loops), loader_calls=loaders,
                                metadata_expressions=metadata, calls=names))
    return records


def build(root):
    records, source_files, failures = [], [], []
    for path in sorted((root / 'python/mojolearn').rglob('*.py')):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        source_files.append(dict(path=relative, sha256=hashlib.sha256(raw).hexdigest()))
        try:
            records.extend(describe(raw.decode('utf-8'), relative))
        except (SyntaxError, UnicodeError) as error:
            failures.append(dict(path=relative, error=str(error)))
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    summary = dict(files=len(source_files), candidate_bodies=len(records),
                   classifications=dict(Counter(r['body_classification'] for r in records)),
                   bodies_with_loader_calls=sum(bool(r['loader_calls']) for r in records),
                   bodies_with_metadata_expressions=sum(bool(r['metadata_expressions']) for r in records),
                   bodies_with_decorators=sum(bool(r['decorators']) for r in records),
                   unresolved_parse_failures=len(failures))
    return dict(schema=1, kind='source_census_not_runtime_evidence', commit=commit,
                source_files=source_files, summary=summary, parse_failures=failures, records=records,
                limitations=['Candidate bodies include private modules/base classes; public reexports and inheritance require resolution.',
                             'One native call body does not audit descriptors, constructors, native transitive callees, or ownership.',
                             'Metadata syntax and calls are candidates for inspection, not proof of runtime execution or a violation.',
                             'Compiled/runtime-generated methods and callbacks may be absent from AST; this is not exhaustive runtime coverage.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], sort_keys=True))
    return int(bool(report['parse_failures']))

if __name__ == '__main__':
    raise SystemExit(main())
