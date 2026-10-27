"""Admit the 0.8.61 fixes from retained execution and final installed boundaries.

The original failed runs and original passing gates are immutable. This adapter
does not count the retained executions as executions of newly packed wheels.
Only the named repair components may change; every other executable member must
remain identical. New native linking/loading is checked by the installed boundary.
"""
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

import release_evidence as evidence
import release_gate_reuse as reuse

SOURCE = '24de487667cb2a6085cad686cd07b4824aec6593'
PYTHON = {'mojolearn/' + n for n in ('_backend.py', 'gpu_plugins.py', '_verify_graph.py', '_version.py')}
HOST = {'mojolearn/host/_mojolearn_' + n + '_host.so'
        for n in ('tokenizer', 'runtime_classical', 'x_sequence')}
DEVICE = {'mojolearn/' + p + '/identical/_mojolearn_x_sequence.so'
          for p in ('cuda_native/sm_89', 'cuda_ptx/sm_80', 'hip_native/gfx942', '')}
DEVICE.add('mojolearn/identical/_mojolearn_x_sequence.so')


def require(ok, reason):
    if not ok:
        raise ValueError('Qualified repair: ' + reason)


def full_report(doc, neural=False):
    expected, parts = (765, 2547) if neural else (3843, 11843)
    require(doc.get('verdict') == 'VERIFIED' and doc.get('exit') == 0
            and doc.get('native_provenance_complete') is True, 'incomplete native verifier report')
    counts = doc.get('counts', {})
    require(counts.get('IDENTICAL') == parts and all(counts.get(k) == 0
            for k in ('DIVERGENT', 'REFUSED', 'OWED')), 'failed or missing verifier parts')
    cells = doc.get('cells', [])
    actual = Counter(c.get('state') for c in cells)
    require(len(cells) == (6885 if neural else 34703)
            and all(actual.get(k, 0) == v for k, v in counts.items())
            and len({(c['lane'], c['fixture'], c['part']) for c in cells}) == len(cells),
            'raw verifier cell coverage differs from summary')
    execution = doc.get('execution', {})
    require(execution.get('total_cells') == execution.get('completed_cells') == expected
            and not any(execution.get(k) for k in ('batch_stops', 'crashed_cells', 'interrupted')),
            'incomplete verifier execution')
    require(doc.get('bindings'), 'missing executed native binding provenance')
    if not neural:
        cross = doc.get('cross_check', {})
        require(cross.get('passed') is True and cross.get('complete') is True
                and cross.get('differ') == 0, 'GPU/CPU comparison did not complete')


def comparison(doc, count):
    require(doc.get('verdict') == 'AGREE' and doc.get('exit') == 0 and doc.get('agree') == count,
            'cross-vendor comparison coverage differs')
    require(all(doc.get(k) == 0 for k in ('differ', 'incomparable', 'uncomputed'))
            and not doc.get('problems') and not doc.get('context_problems'), 'cross-vendor comparison failed')


def graph(doc):
    require(doc.get('exit_code') == 0 and len(doc.get('expected_cases', [])) == 31,
            'graph verifier did not complete')
    rows = doc.get('cases', {})
    require(isinstance(rows, dict) and len(rows) == 31 and set(rows) == set(doc['expected_cases']),
            'graph case inventory differs')
    require(all(r.get('status') == 'passed' for r in rows.values()), 'graph case failed or cannot run')
    controls = doc.get('comparator_controls', {})
    require(controls.get('same_report') == 'passed' and all(controls.get(k) == 'rejected'
            for k in ('changed_digest', 'failed_case', 'missing_case', 'wrong_mode')),
            'graph comparator negative controls did not pass')


def boundary(doc, candidate, reports):
    require(doc.get('schema') == 'mojolearn.release.repair-boundary.v1'
            and doc.get('status') == 'PASS' and doc.get('source_commit') == SOURCE
            and doc.get('version') == candidate['version'] and doc.get('vendor') == candidate['vendor'],
            'final installed boundary identifies another candidate or did not pass')
    require(set(doc['wheels'].values()) == set(candidate['wheels'].values()),
            'installed boundary did not install exactly the candidate wheels')
    # A failed controller attempt stays in the record. Only its subsequent
    # successful attempt qualifies; require retained logs for both.
    completed = {}
    for step in doc['steps']:
        log = reports['logs'][step['log']]
        require(log['sha256'] == step['log_sha256'], 'relocated boundary log differs')
        reuse.checked_ref(log)
        completed[step['name']] = step
    required = {'install', 'dependencies', 'self-test', 'repaired-public-state',
                'sequence-boundary', 'graph', 'installed-inventory', 'adapters'}
    require(required <= set(completed) and all(completed[n]['exit_code'] == 0 for n in required),
            'final installed boundary has missing or failed steps')
    public = evidence.read_text(reuse.checked_ref(reports['logs'][completed['repaired-public-state']['log']]))
    require(re.search(r'\b3 passed\b', public) and re.search(r'\b14 subtests passed\b', public),
            'model-state/tokenizer test coverage differs')
    sequence = reuse.read(reports['sequence'])
    require(sequence.get('verdict') == 'VERIFIED' and sequence.get('exit') == 0
            and sequence.get('native_provenance_complete') is True
            and sequence['execution']['completed_cells'] == sequence['execution']['total_cells'] == 27,
            'installed sequence boundary incomplete')
    require(all(sequence['counts'].get(k) == 0 for k in ('DIVERGENT', 'REFUSED', 'OWED')),
            'installed sequence boundary failed')
    graph(reuse.read(reports['graph']))
    self_test = evidence.read_text(reuse.checked_ref(reports['logs'][completed['self-test']['log']]))
    clean = re.search(r'untouched\s+([0-9a-f]+)\s+vs reference\s+([0-9a-f]+)\s+-> IDENTICAL', self_test)
    wrong = re.search(r'perturbed\s+([0-9a-f]+)\s+vs reference\s+([0-9a-f]+)\s+-> DIVERGENT', self_test)
    require(clean and wrong and clean[1] == clean[2] == wrong[2] and wrong[1] != wrong[2],
            'installed verifier self-test failed')
    import adapter_gate
    adapters = reuse.read(reports['adapters'])
    _, passed, why = adapter_gate.judge(adapters)
    require(passed, 'installed adapter boundary failed: ' + str(why))
    inventory = json.loads(evidence.read_text(reuse.checked_ref(reports['logs'][completed['installed-inventory']['log']])))
    require(inventory['version'] == candidate['version'] and inventory['source'] == SOURCE,
            'installed inventory version/source differs')
    return inventory, sequence


def wheel_delta(before, after, qualified):
    import release_incremental_gate as inc
    left = inc.contents(before, '0.8.60')
    right = inc.contents(after, '0.8.61')
    changed = []
    for name in sorted(set(left) | set(right)):
        if left.get(name) == right.get(name):
            continue
        changed.append(name)
        if name in qualified:
            require(name in PYTHON | HOST | DEVICE, 'unrecognized runtime repair: ' + name)
            require(name in right and hashlib.sha256(right[name]).hexdigest() == qualified[name],
                    'repair bytes differ from qualified component: ' + name)
        elif name.endswith(('.dist-info/RECORD', '/LINUX_PAYLOAD.json', '/BASE_LINUX_PAYLOAD.json',
                            '/NATIVE_PATCH.json', '/PTX_BASELINE.json')):
            continue
        elif name.endswith('/_mojolearn_merged.json'):
            require(name in left and name in right, 'unexpected merged index addition/removal')
            a, b = json.loads(left[name]), json.loads(right[name])
            for names in a['files'].values():
                if '_mojolearn_x_sequence' in names:
                    names.remove('_mojolearn_x_sequence')
            require(a == b, 'unrelated native dispatch change')
        elif name.endswith('/gpu_plugin.json'):
            a, b = json.loads(left[name].replace(b'0.8.60', b'0.8.61')), json.loads(right[name])
            a.pop('bundled_ptx', None); b.pop('bundled_ptx', None)
            require(a == b, 'unrelated plugin policy change')
        elif name == 'mojolearn/identity_columns/COMMIT':
            require(right[name].decode().strip() == SOURCE, 'package source differs')
        elif name.endswith(('.dist-info/METADATA', '/gpu_plugins.json')) or name == 'mojolearn/CITATION.cff':
            require(left[name].replace(b'0.8.60', b'0.8.61') == right[name], 'unrelated metadata change')
        else:
            raise ValueError('Qualified repair: unqualified payload change: ' + name)
    return changed


def validate(document, identity):
    import release_incremental_gate as inc
    require(identity['source_commit'] == SOURCE and identity['version'] == '0.8.61',
            'this adapter qualifies only the frozen 0.8.61 repair')
    old = reuse.read(document['retained_gate'])
    inc.validate(old, old['candidate'])
    require(old['candidate']['version'] == '0.8.60' and old['candidate']['vendor'] == identity['vendor'],
            'wrong original release/vendor')
    for key in ('correctness', 'autograd_counts', 'autograd_artifacts'):
        require(document[key] == old[key], 'retained execution was relabeled: ' + key)
    reports = document['repair_update']
    full = reuse.read(reports['full']); full_report(full)
    full_report(reuse.read(reports['neural']), neural=True)
    comparison(reuse.read(reports['full_compare']), 15695)
    comparison(reuse.read(reports['neural_compare']), 2547)
    graph(reuse.read(reports['graph']))
    observed = {r['sha256'] for r in full['bindings']}
    installed = None
    compile_only = {}
    if identity['vendor'] == 'nvidia':
        # Owner 2026-10-27: publish with retained verification, then exactly one
        # NVIDIA PTX/Mac comparison. Do not schedule another pre-publication GPU
        # campaign. The native sm_89 repair is compiled and audited here, never
        # presented as a fresh runtime pass. The already-executed PTX repair
        # and identical Linux host bytes supply their actual runtime evidence.
        require(reports.get('qualification_scope') == 'retained PTX runtime; native sm_89 compile-only',
                'NVIDIA admission must preserve its unexecuted native-image scope')
        prior_sequence = reuse.read(reports['ptx_sequence'])
        require(prior_sequence.get('verdict') == 'VERIFIED' and prior_sequence.get('exit') == 0
                and prior_sequence.get('native_provenance_complete') is True,
                'retained PTX repair boundary did not pass')
        require(all(prior_sequence['counts'].get(k) == 0 for k in ('DIVERGENT', 'REFUSED', 'OWED')),
                'retained PTX repair boundary has incomplete parts')
        observed.update(r['sha256'] for r in prior_sequence['bindings'])
        build = reports['native_compile']
        require(evidence.read_text(reuse.checked_ref(build['exit'])).strip() == '0',
                'native NVIDIA repair compilation failed')
        audit = reuse.read(build['audit'])
        require(audit.get('errors') == [] and audit.get('arches') == ['sm_89']
                and audit.get('ptx_modules') == 0 and audit.get('fatbins', 0) > 0,
                'native NVIDIA image audit failed')
        binary = reuse.checked_ref(build['binary'])
        sums = evidence.read_text(reuse.checked_ref(build['binary_sha256']))
        require(sums.split()[0] == evidence.sha256(binary), 'compiled native bytes differ')
        compile_only['mojolearn/cuda_native/sm_89/identical/_mojolearn_x_sequence.so'] = build['binary']['sha256']
    else:
        installed, sequence = boundary(reuse.read(reports['boundary']), identity, reports['boundary_reports'])
        observed.update(r['sha256'] for r in sequence['bindings'])
    qualified = {}
    require(set(document['wheels']) == set(identity['wheels']) == set(old['candidate']['wheels']),
            'wheel coverage differs')
    for project, row in document['wheels'].items():
        require(row['before']['sha256'] == old['candidate']['wheels'][project]
                and row['after']['sha256'] == identity['wheels'][project], 'wheel identities differ')
        assembly = reuse.read(row['assembly'])
        require(assembly['package_source_commit'] == SOURCE and assembly['after'] == row['after']
                and assembly['base'] == row['before'], 'assembly does not identify exact wheels')
        for name, component in assembly['components'].items():
            raw = evidence.read_bytes(reuse.checked_ref(component))
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == component['sha256'], 'retained repair bytes differ')
            if name.endswith('.so'):
                require(digest in observed or compile_only.get(name) == digest,
                        'repaired native component has no scoped runtime/compile evidence: ' + name)
                require(installed is None or installed['files'].get(name.removeprefix('mojolearn/')) == digest,
                        'installed repair differs: ' + name)
            qualified[name] = digest
        actual = wheel_delta(row['before'], row['after'], qualified)
        require(actual == row['changed'], 'candidate changes differ from the retained audit')
    return True
