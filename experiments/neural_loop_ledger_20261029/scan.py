#!/usr/bin/env python3
"""Source inventory only: reuse existing scanners without changing any policy gate."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import types


def git(*args):
    return subprocess.check_output(['git', *args])


def load_module(path, source):
    module = types.ModuleType(Path(path).stem + '_ledger')
    module.__file__ = str(Path.cwd() / path)
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


def indentation(line):
    return len(line) - len(line.lstrip())


def functions(lines):
    defs = []
    for i, line in enumerate(lines):
        match = re.match(r'^(\s*)(?:def|fn)\s+(\w+)', line)
        if match:
            defs.append(dict(start=i + 1, end=len(lines), name=match[2], indent=len(match[1])))
    for i, fn in enumerate(defs):
        for later in defs[i + 1:]:
            if later['indent'] <= fn['indent']:
                fn['end'] = later['start'] - 1
                break
    return defs


CPY = re.compile(r'\b(?:Python(?:Object)?|PyObjectPtr|py_\w+|getattr|setattr|hasattr)\b|\bpy\s*=')
PROVIDER = re.compile(r'\b(?:neural_descriptor_\w+|transformer_resident_\w+|generation_native_\w+|native_execution_\w+)\s*\(|py_call(?:_typed)?\(\s*(?:e|entries|native|parallel)\b')
CALL = re.compile(r'(?<![\w.])([A-Za-z_]\w*)\s*(?:\[[^\n]*?\])?\(')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    commit = git('rev-parse', args.tree + '^{commit}').decode().strip()
    scope_path = Path(__file__).with_name('scope.json')
    scope = json.loads(scope_path.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out / 'snapshot'
    names = scope['files'] + ['tools/hooks/no_compute_in_shells.py', 'tools/mojo_python_callback_audit.py']
    sources = {name: git('show', commit + ':' + name).decode() for name in names}
    for name, source in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    loops = load_module(names[-2], sources[names[-2]])
    callbacks = load_module(names[-1], sources[names[-1]])
    rows, callback_rows = [], []
    function_index = defaultdict(list)
    definitions = {}
    cleaned = {}
    for name in scope['files']:
        cleaned[name] = callbacks.code_lines(sources[name])
        definitions[name] = functions(cleaned[name])
        for fn in definitions[name]:
            function_index[fn['name']].append((name, fn))
        callback_rows.extend(callbacks.scan_file(snapshot / name, snapshot))
    callback_fields = ('file', 'line', 'function', 'op', 'receiver', 'origin', 'exact_admission', 'classification', 'owner', 'source')
    callbacks_by_file = defaultdict(list)
    for raw in callback_rows:
        row = dict(zip(callback_fields, raw))
        callbacks_by_file[row['file']].append(row)
    direct_callers = defaultdict(set)
    for name in scope['files']:
        for fn in definitions[name]:
            body = '\n'.join(cleaned[name][fn['start']:fn['end']])
            for target in set(CALL.findall(body)):
                if target in function_index:
                    direct_callers[target].add(name + ':' + str(fn['start']) + ':' + fn['name'])
    for name in scope['files']:
        source = sources[name]
        original_tag = loops.TAG
        default_hits = {n for n, kind, _ in loops.scan(source) if kind == 'loop'}
        # INVENTORY ONLY: include tagged loops that the required gate exempts.
        # No source or policy scanner file is modified.
        loops.TAG = re.compile(r'(?!)')
        hits = [hit for hit in loops.scan(source) if hit[1] == 'loop']
        loops.TAG = original_tag
        lines = cleaned[name]
        for number, _, code in hits:
            fn = next((fn for fn in reversed(definitions[name]) if fn['start'] <= number <= fn['end']), None)
            indent = indentation(lines[number - 1])
            end = number
            header_complete = lines[number - 1].rstrip().endswith(':')
            for i in range(number, len(lines)):
                if not lines[i].strip():
                    continue
                if not header_complete:
                    header_complete = lines[i].rstrip().endswith(':')
                    end = i + 1
                    continue
                if indentation(lines[i]) <= indent:
                    break
                end = i + 1
            body = '\n'.join(lines[number - 1:end])
            hook_hits = [row for row in callbacks_by_file[name] if number <= int(row['line']) <= end]
            metadata = bool(CPY.search(body))
            crossing = bool(PROVIDER.search(body))
            candidates = [row for row in hook_hits if row['classification'] == 'prohibited_user_object']
            kind = ('cpython_metadata_or_native_boundary' if metadata else
                    'native_orchestration_provider_crossing' if crossing else
                    'native_compute' if name in scope['native_compute_roots'] else
                    'native_orchestration_or_setup')
            rows.append(dict(id=name + ':' + str(number), file=name, line=number, end=end,
                function=fn['name'] if fn else '<module>', function_line=fn['start'] if fn else 0,
                code=code, gate_untagged=number in default_hits, primary_static_class=kind,
                local_cpython_syntax=metadata, local_provider_crossing=crossing,
                callback_candidate_count=len(candidates), callback_triage=hook_hits,
                direct_scoped_callers=sorted(direct_callers[fn['name']]) if fn else [],
                call_names=sorted(set(CALL.findall(body)))))
    (args.out / 'loops.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.out / 'callbacks.json').write_text(json.dumps([dict(zip(callback_fields, row)) for row in callback_rows], indent=2) + '\n')
    fields = ['id', 'function', 'function_line', 'end', 'primary_static_class', 'local_cpython_syntax', 'local_provider_crossing', 'callback_candidate_count', 'gate_untagged', 'code']
    with (args.out / 'loops.tsv').open('w') as output:
        output.write('\t'.join(fields) + '\n')
        for row in rows:
            output.write('\t'.join(str(row[field]).replace('\t', ' ') for field in fields) + '\n')
    summary = dict(tree=commit, scope=scope['scope'], files=len(scope['files']), loop_statements=len(rows),
        by_file=dict(sorted(Counter(row['file'] for row in rows).items())),
        by_static_class=dict(sorted(Counter(row['primary_static_class'] for row in rows).items())),
        loops_with_local_provider_crossings=sum(row['local_provider_crossing'] for row in rows),
        loops_with_callback_candidates=sum(bool(row['callback_candidate_count']) for row in rows),
        unique_callback_candidates=sum(row[7] == 'prohibited_user_object' for row in callback_rows),
        callback_triage_counts=dict(Counter(row[7] for row in callback_rows)),
        source_sha256={name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        inventory_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
        limitations=['Lexical loop statements only; comprehensions and compiler-generated loops excluded.',
            'Nested loops appear separately; body features include nested body syntax, so feature counts are not operation counts.',
            'Tagged loops included by disabling the imported scanner tag exemption only for this inventory.',
            'Local syntax classes are triage, not policy violations; native CPython is allowed.',
            'Scoped caller names are regex candidates, not a resolved call graph or execution-frequency measurement.',
            'Indirect native ABI targets and transitive callbacks require source review; absence of a scanner hit proves neither purity nor safety.',
            'No runtime, compiler, identity or performance verification performed by inventory.'])
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key:summary[key] for key in ['tree','files','loop_statements','by_static_class','loops_with_local_provider_crossings','unique_callback_candidates']}))


if __name__ == '__main__':
    main()
