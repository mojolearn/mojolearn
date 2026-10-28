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
NATIVE_NAMES = {'_NATIVE', '_NATIVE_CALL', '_NATIVE_OWNER', '_NATIVE_DOOR', '_XSEQ', '_LR_CALL', '_QN_CALL', '_ARRAY_NATIVE'}
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



LIFECYCLE_METHODS = {'fit','partial_fit','fit_predict','fit_transform','_fit_with_tree_start',
    'predict','predict_proba','predict_log_proba','decision_function','transform',
    'inverse_transform','score','score_samples','kneighbors','sample'}


def class_graph(sources):
    """Resolve source inheritance/imports; preserve unknown bases explicitly."""
    classes={}
    for filename,source in sources.items():
        module=filename.removeprefix('python/').removesuffix('.py').replace('/','.')
        if module.endswith('.__init__'):module=module[:-9]
        try:tree=ast.parse(source)
        except SyntaxError:continue
        imports={}
        for node in tree.body:
            if isinstance(node,ast.ImportFrom):
                prefix=module.split('.')[:-node.level] if node.level else []
                target='.'.join(prefix+([node.module] if node.module else []))
                for alias in node.names:imports[alias.asname or alias.name]=target+'.'+alias.name
            elif isinstance(node,ast.Import):
                for alias in node.names:imports[alias.asname or alias.name]=alias.name
        for node in tree.body:
            if not isinstance(node,ast.ClassDef):continue
            bases=[]
            for base in node.bases:
                name=ast.unparse(base);head,_,tail=name.partition('.')
                if head in imports:bases.append(imports[head]+('.'+tail if tail else ''))
                elif name in ('object','Exception','ValueError','AttributeError','list','dict','tuple'):bases.append('builtins.'+name)
                else:bases.append(module+'.'+name)
            marker=None
            for member in node.body:
                if isinstance(member,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_mojolearn_native_public' for t in member.targets):
                    if isinstance(member.value,ast.Constant) and type(member.value.value) is bool:marker=member.value.value
            classes[module+'.'+node.name]=dict(path=filename,line=node.lineno,bases=bases,marker=marker,
                methods={m.name:m.lineno for m in node.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef))})
    return classes


def lifecycle_candidates(sources):
    classes=class_graph(sources);cache={}
    def mro(name,active=()):
        if name in cache:return cache[name]
        if name in active:return [name],['inheritance cycle: '+name]
        cls=classes.get(name)
        if cls is None:return [name],([] if name.startswith('builtins.') else [name])
        orders=[];unknown=[]
        for base in cls['bases']:
            order,unresolved=mro(base,active+(name,));orders.append(list(order));unknown.extend(unresolved)
        orders.append(list(cls['bases']));result=[name]
        while any(orders):
            orders=[o for o in orders if o]
            candidate=next((o[0] for o in orders if not any(o[0] in other[1:] for other in orders)),None)
            if candidate is None:
                unknown.append('unresolved C3 order: '+name);break
            result.append(candidate)
            for order in orders:
                if order and order[0]==candidate:order.pop(0)
        cache[name]=(result,sorted(set(unknown)));return cache[name]
    def native_marker(order):
        return next((classes[c]['marker'] for c in order if c in classes and classes[c]['marker'] is not None),False)
    records=[]
    for name,cls in classes.items():
        order,unknown=mro(name)
        if 'mojolearn._mode.SklearnTagsMixin' not in order and 'mojolearn._mode.PublicOutputsMixin' not in order:continue
        marker=native_marker(order)
        for method in sorted(LIFECYCLE_METHODS):
            owner=next((c for c in order if c in classes and method in classes[c]['methods']),None)
            if owner is None:continue
            owner_order,_=mro(owner)
            # A marker on a derived class does not undo a Python wrapper
            # already installed while the defining base class was created.
            owner_has_hooks=('mojolearn._mode.SklearnTagsMixin' in owner_order or 'mojolearn._mode.PublicOutputsMixin' in owner_order)
            wrapped_owner=owner_has_hooks and not native_marker(owner_order)
            expected=not marker or wrapped_owner
            records.append(dict(class_name=name,method=method,definition_owner=owner,
                path=classes[owner]['path'],line=classes[owner]['methods'][method],
                inherited=owner!=name,native_marker=marker,
                defining_class_wrapper_risk=wrapped_owner,
                wrapper_assessment='python_lifecycle_wrappers_expected' if expected else 'native_lifecycle_unverified',
                unresolved_bases=unknown))
    return records

def build(root):
    records, source_files, failures, sources = [], [], [], {}
    for path in sorted((root / 'python/mojolearn').rglob('*.py')):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        source_files.append(dict(path=relative, sha256=hashlib.sha256(raw).hexdigest()))
        try:
            sources[relative]=raw.decode('utf-8')
            records.extend(describe(sources[relative], relative))
        except (SyntaxError, UnicodeError) as error:
            failures.append(dict(path=relative, error=str(error)))
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    lifecycle=lifecycle_candidates(sources)
    summary = dict(lifecycle_candidates=len(lifecycle),
                   expected_python_lifecycle_wrappers=sum(r["wrapper_assessment"]=="python_lifecycle_wrappers_expected" for r in lifecycle),
                   files=len(source_files), candidate_bodies=len(records),
                   classifications=dict(Counter(r['body_classification'] for r in records)),
                   bodies_with_loader_calls=sum(bool(r['loader_calls']) for r in records),
                   bodies_with_metadata_expressions=sum(bool(r['metadata_expressions']) for r in records),
                   bodies_with_decorators=sum(bool(r['decorators']) for r in records),
                   unresolved_parse_failures=len(failures))
    return dict(schema=1, kind='source_census_not_runtime_evidence', commit=commit,
                source_files=source_files, summary=summary, parse_failures=failures, records=records, lifecycle_candidates=lifecycle,
                limitations=['Candidate bodies include private modules/base classes; public reexports and inheritance require resolution.',
                             'One native call body does not audit descriptors, constructors, native transitive callees, or ownership.',
                             'Metadata syntax and calls are candidates for inspection, not proof of runtime execution or a violation.',
                             'Lifecycle wrapper findings are source predictions, not imported-class introspection; explicit installer calls and generated methods still need runtime audit.',
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
