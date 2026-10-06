"""Conservative, opt-in source-freeze attestation for retained comparisons.

This does not manufacture evidence, approve an attestation, or change a receipt.
Version 1 supports Linux native NVIDIA/AMD only. A reviewer must supply captured
execution inventories; reconstructing old deployed files from current Git is
not evidence. Missing inventories remain INCOMPLETE. See FREEZE_EQUIVALENCE.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

from six_lane_ab import comparable_compile_argv

SCHEMA = 'mojolearn.freeze-equivalence/1'
CLOSURE_SCHEMA = 'mojolearn.captured-execution-closure/1'
# These files perform registration/admission only, not a scored operation or
# capture. Adding to this list is a separate reviewed harness change.
CONTROL_FILES = frozenset({
    'tools/six_lane_full_variants.py', 'tools/six_lane_classification_variants.py',
    'tools/six_lane_register_full_tsvd.py', 'tools/six_lane_prepare_full_tsvd.py',
    'tools/six_lane_prepare_full_classification.py',
    'tools/six_lane_register_full_classification.py',
})
ROUTES = {'nvidia-native': ('nvidia', 'MOJOLEARN_COLUMN_NVIDIA', r'sm_[0-9]+'),
          'amd': ('amd', 'MOJOLEARN_COLUMN_AMD', r'gfx[0-9a-f]+')}
SECTIONS = ('harness', 'api', 'runtime', 'capture')


def require(ok, reason):
    if not ok:
        raise ValueError('freeze equivalence: ' + reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def is_sha(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def load_ref(ref, base):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str) and is_sha(ref.get('sha256')),
            'evidence requires path and SHA-256')
    path = (base / ref['path']).resolve()
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'evidence hash differs: ' + str(path))
    value = json.loads(raw)
    canonical(value)
    return value, path.parent


def reviewed_source_diff(repository, anchor, other, declared):
    require(is_sha(anchor, 40) and is_sha(other, 40), 'invalid source commit')
    cmd = ['git', '-C', str(repository), 'diff', '--name-only', '--no-renames', '-z', anchor, other, '--']
    result = subprocess.run(cmd, capture_output=True, check=True)
    changed = sorted(x for x in result.stdout.decode().split('\0') if x)
    require(changed == declared, 'reviewed changed-file inventory differs from Git objects')
    require(all(path in CONTROL_FILES or path.endswith('.md') or
                path.startswith(('docs/', 'experiments/')) and not path.endswith(('.py', '.mojo', '.c', '.h'))
                for path in changed), 'change reaches unapproved execution/source files')
    return changed


def compile_signature(record, artifact, column):
    vendor, backend_define, target_pattern = ROUTES[column]
    require(record.get('status') in ('COMPILED', 'REUSED') and record.get('returncode') == 0,
            'compile evidence is not successful')
    for key, artifact_key in (('artifact_sha256', 'sha256'), ('source_sha', 'numerical_source_sha'),
                              ('compiler_sha256', 'compiler_sha256'),
                              ('source_closure_sha256', 'source_closure_sha256'), ('defines', 'defines'), ('target', 'target')):
        require(record.get(key) == artifact.get(artifact_key) and record.get(key) is not None,
                'compile evidence differs from deployed artifact: ' + key)
    require(record.get('vendor') == vendor, 'compile vendor differs')
    files = record.get('source_files')
    require(isinstance(files, dict) and bool(files) and all(isinstance(k, str) and is_sha(v) for k, v in files.items()),
            'missing exact numerical source-file closure')
    require(is_sha(record.get('compiler_sha256')), 'missing compiler executable hash')
    binding = record.get('binding')
    require(isinstance(binding, str) and binding in files, 'binding absent from source closure')
    argv = comparable_compile_argv(record.get('argv', []), binding)
    require(argv and argv[0] == '<compiler>' and argv[-1] == '<source>/' + binding,
            'unsupported compile argv relocation')
    require(argv.count('--target-accelerator') == 1, 'native accelerator must be explicit')
    pos = argv.index('--target-accelerator')
    require(pos + 1 < len(argv) and re.fullmatch(target_pattern, argv[pos + 1]), 'unsupported native target')
    argv[pos + 1] = '<declared-native-backend>'
    pairs = [i for i in range(len(argv)-1) if argv[i] == '-D' and argv[i+1] == backend_define]
    require(len(pairs) == 1, 'backend define is missing or ambiguous')
    argv[pairs[0]+1] = '<declared-vendor-column>'
    return dict(binding=binding, source_files=files, compiler_sha256=record['compiler_sha256'], argv=argv)


def execution_signature(value, entry, receipt_sha):
    require(value.get('schema') == CLOSURE_SCHEMA and value.get('status') == 'CAPTURED',
            'missing captured execution closure')
    require(value.get('method') == 'observed_deployed_files' and value.get('complete') is True
            and value.get('missing') == [], 'execution closure is reconstructed or incomplete')
    require(value.get('source_sha') == entry['source_sha'] and value.get('receipt_sha256') == receipt_sha,
            'execution closure is not bound to selected original receipt')
    require(isinstance(value.get('capture_version'), str) and value['capture_version'], 'missing capture version')
    sections = value.get('sections')
    require(isinstance(sections, dict) and set(sections) == set(SECTIONS), 'incomplete execution sections')
    for section, files in sections.items():
        require(isinstance(files, dict) and files and all(isinstance(k, str) and is_sha(v) for k, v in files.items()),
                'missing exact ' + section + ' file inventory')
    # No runtime-library exception in v1: unequal vendor runtime inventories are
    # unproven, even if their package versions happen to match.
    return dict(capture_version=value['capture_version'], sections=sections)


def validate(ref, base, case, expected, column, selected_receipt_sha):
    att, location = load_ref(ref, base)
    require(att.get('schema') == SCHEMA and att.get('status') == 'REVIEWED', 'attestation is not reviewed')
    review = att.get('review')
    require(isinstance(review, dict) and all(isinstance(review.get(k), str) and review[k].strip()
            for k in ('reviewer', 'reviewed_at', 'rationale')), 'missing explicit reviewer attestation')
    require(att.get('case_id') == case['id'] and att.get('configuration_id') == case['configuration_id'],
            'attestation case/configuration differs')
    require(att.get('anchor_source_sha') == expected['source_sha'], 'anchor source differs')
    require(att.get('scope_sha256') == digest({k: v for k, v in expected.items() if k != 'source_sha'}),
            'pinned non-source scope differs')
    columns = att.get('columns')
    require(isinstance(columns, dict) and set(columns) == set(ROUTES) and column in columns,
            'v1 requires native NVIDIA and AMD evidence')
    repository = (location / att['repository']).resolve()
    signatures = {}
    for name, entry in columns.items():
        receipt, _ = load_ref(entry['receipt'], location)
        source = entry.get('source_sha')
        require(is_sha(source, 40) and receipt.get('source_sha') == source, 'original source differs')
        require(receipt.get('status') == 'MEASURED_FULL', 'selected attempt did not finish')
        if name == column:
            require(entry['receipt']['sha256'] == selected_receipt_sha, 'attestation selects another receipt')
        reviewed_source_diff(repository, expected['source_sha'], source, entry.get('reviewed_changed_files'))
        closure, _ = load_ref(entry['execution_closure'], location)
        signatures[name] = dict(execution=execution_signature(closure, entry, entry['receipt']['sha256']), arms={})
        for arm in ('A', 'B'):
            artifacts = receipt.get('workload', {}).get('artifact_provenance', {}).get(arm)
            refs = entry.get('compile_receipts', {}).get(arm)
            require(isinstance(artifacts, list) and artifacts and isinstance(refs, dict)
                    and set(refs) == {a['path'] for a in artifacts}, 'compile references do not cover deployed arm')
            records = []
            for artifact in artifacts:
                record, _ = load_ref(refs[artifact['path']], location)
                records.append(compile_signature(record, artifact, name))
            require(len({r['binding'] for r in records}) == len(records), 'duplicate binding closure')
            signatures[name]['arms'][arm] = sorted(records, key=lambda r: r['binding'])
    require(signatures['nvidia-native'] == signatures['amd'], 'numerical/compiler/flags/execution closures differ')
    return dict(path=str((base / ref['path']).resolve()), sha256=ref['sha256'], review=review,
                anchor_source_sha=expected['source_sha'], original_source_sha=columns[column]['source_sha'],
                scope='comparison_only', accepted=False, promoted=False)
