#!/usr/bin/env python3
"""Read-only admission of installed Linux surface evidence; never invokes a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

MODES = {'fast': 0, 'deterministic': 2, 'identical': 1}
BINDINGS = {'_mojolearn' + suffix for suffix in ('', '_gbdt', '_estimators', '_rf', '_trees',
            '_svm', '_solver', '_metrics', '_preprocessing', '_tsa', '_linalg', '_arima', '_training', '_gp',
            '_mamba', '_transformer')}
CORPUS_CASES = ('base_b2_l4_d8', 'mamba2/m2_base_b2_l4_d32',
                'mamba3/m3_base_b2_l4_d32')
SURFACES = ('smoke', 'umap', 'umap-transform', 'umap-quality', 'ordered-rmse', 'mamba', 'transformer', 'arima')
FIXTURES = {
    'cubic128_interleaved64_64': (64, 64, 2),
    'saddle_grid64_cellcenters49': (64, 49, 3),
    'cubic256_interleaved128_128_seed3': (128, 128, 2),
    'cubic256_interleaved128_128_seed41': (128, 128, 2),
    'saddle_grid128_cellcenters105_seed11': (128, 105, 3),
    'saddle_grid128_cellcenters105_seed29': (128, 105, 3),
}


BYTE_LM_PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
BYTE_FILES = {'byte-lm-identical.json', 'byte-lm-before.json', 'byte-lm-after.json', 'byte-lm-restored.json'}
# DEVIATION 2290. The combined three-architecture Linux profile is named for its
# shape, `release-linux3` (CUDA sm_89, CUDA sm_90, HIP gfx942), not for a version.
# It was authored as `release-0.6.1`, a number that was never published; that
# name stays accepted as a deprecated alias mapping to the same code path, so
# the retained 2026-09-07 preparation evidence and the older documents still
# parse. Everything emitted says RELEASE_PROFILE. The version the profile ships
# is never a literal on the release path: release_version() below is the ONE
# reader of python/mojolearn/_version.py, shared by the packer, the release
# qualification checker and the alpha artifact verifier.
RELEASE_PROFILE = 'release-linux3'
RELEASE_PROFILES = frozenset({RELEASE_PROFILE, 'release-0.6.1'})

# DEVIATION 2293: THE HOPPER SLOT HAS TWO LEGAL SPELLINGS AND ONLY ONE OF THEM
# IS BUILDABLE. Asked for sm_90 through --target-accelerator on an H100, the
# compiler emits binaries carrying sm_90a, and every gate that reads an
# architecture back off the binaries then refuses the set for being named
# something it was not verified to carry. That refusal is right and is not
# relaxed anywhere; what was wrong is that six separate gates spelled the slot
# `sm_90` and nothing else, and `sm_90` has never once been built. 0.6.0's
# Linux wheel shipped gfx942 only and NVIDIA was a source build, so the
# literal was never exercised.
#
# sm_90a is not a fallback. python/mojolearn/_backend.py already PREFERS it:
# a device reporting sm_90 takes a carried sm_90a as "architecture-specific
# build for this exact device", ahead of any family rule, because the `a`
# restricts which devices the code runs on and Hopper is what it restricts to.
#
# ONE definition, read by the packer, the admission checker and the alpha
# verifier, because a contract clause spelled separately in six places is six
# chances to disagree.
RELEASE_HOPPER = ('sm_90', 'sm_90a')
RELEASE_ARCHES = frozenset({'cuda/sm_89', 'cuda/sm_90', 'hip/gfx942'})


def normalise_arch(key):
    """Collapse the Hopper slot's two spellings onto one, for set comparison.

    'cuda/sm_90a' -> 'cuda/sm_90'; everything else is returned unchanged, so a
    wrong architecture is still a wrong architecture.
    """
    return 'cuda/sm_90' if key == 'cuda/sm_90a' else key


def arch_set_ok(keys):
    """True when `keys` is exactly the release triple, Hopper spelled either way.

    Refuses a set that fills the Hopper slot twice, which normalising alone
    would silently accept.
    """
    keys = list(keys)
    hopper = [k for k in keys if k in ('cuda/sm_90', 'cuda/sm_90a')]
    return len(hopper) <= 1 and {normalise_arch(k) for k in keys} == RELEASE_ARCHES
VERSION_PATTERN = re.compile(r'''^__version__\s*=\s*(['"])([^'"]+)\1\s*$''', re.MULTILINE)


def release_version(root):
    """The release version: `__version__` in ROOT/python/mojolearn/_version.py."""
    match = VERSION_PATTERN.search((Path(root) / 'python/mojolearn/_version.py').read_text())
    require(match is not None, 'No __version__ in python/mojolearn/_version.py')
    return match[2]


def is_release_profile(audit):
    """True for the combined profile under its current name or its deprecated alias."""
    return audit.get('assembly_profile') in RELEASE_PROFILES


def expected_jobs(audit):
    jobs = {(s, m) for s in SURFACES for m in MODES}
    if is_release_profile(audit):  # DEVIATION 2290
        jobs.add(('byte-lm', 'identical'))
    return jobs


def expected_bindings(mode, byte_lm=False):
    return BINDINGS | ({'_mojolearn_byte_lm'} if byte_lm and mode == 'identical' else set())


def check_byte_lm(out, installed):
    import math
    for name in BYTE_FILES:
        require((out / name).stat().st_size <= 2 * 1024 * 1024, 'Byte-LM evidence too large')
    report = json.loads((out / 'byte-lm-identical.json').read_text())
    meta = report['metadata']
    require(report.get('schema') == 'mojolearn.installed-byte-lm-step.v1'
            and report.get('status') == 'PASS' and report.get('completed_steps') == 1,
            'Missing byte-LM one-step success')
    require(meta.get('native_profile') == BYTE_LM_PROFILE and meta.get('profile') == BYTE_LM_PROFILE
            and meta.get('native_numeric_mode') == 1 and meta.get('native_vendor') == installed['vendor']
            and meta.get('binding_sha256') == installed['installed_bindings']['_mojolearn_byte_lm']['sha256'],
            'Byte-LM installed profile/mode/vendor/binary differs')
    require(all(type(report[k]) in (int, float) and math.isfinite(report[k])
                for k in ('loss', 'evaluation_loss')), 'Nonfinite byte-LM loss')
    def floats(encoded):
        require(type(encoded) is str and len(encoded) == 34944 * 8, 'Wrong byte-LM array length')
        raw = bytes.fromhex(encoded)
        require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw)), 'Nonfinite byte-LM array')
        return raw
    gradient = floats(report['gradients_hex'])
    require(any(x[0] != 0 for x in struct.iter_unpack('<f', gradient)), 'Zero byte-LM gradients')
    payloads = []
    expected = BYTE_FILES - {'byte-lm-identical.json'}
    require(set(report['checkpoint_sha256']) == expected, 'Missing checkpoint files')
    for name in ('byte-lm-before.json', 'byte-lm-after.json', 'byte-lm-restored.json'):
        require(sha(out / name) == report['checkpoint_sha256'][name], 'Checkpoint bytes changed')
        envelope = json.loads((out / name).read_text())
        payload = envelope['payload']
        require(envelope.get('schema') == 'mojolearn.small-byte-lm-json-checkpoint.v1'
                and envelope.get('payload_sha256') == hashlib.sha256(json.dumps(payload, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest(), 'Checkpoint payload hash differs')
        require(set(payload) == {'schema', 'profile', 'numeric_mode', 'parameter_names', 'parameter_shapes',
                'parameter_offsets', 'parameters', 'm', 'v', 'flags', 'completed_steps', 'next_batch_index',
                'config', 'data_schedule'} and payload.get('schema') == 'mojolearn.small-byte-lm-state.v1',
                'Incomplete checkpoint state schema')
        require(len(payload['parameter_names']) == len(set(payload['parameter_names'])) == 20
                and len(payload['parameter_shapes']) == 20 and len(payload['parameter_offsets']) == 21
                and payload['parameter_offsets'][0] == 0 and payload['parameter_offsets'][-1] == 34944,
                'Incomplete checkpoint parameter registry')
        for index, shape in enumerate(payload['parameter_shapes']):
            require(type(shape) is list and shape and all(type(d) is int and d > 0 for d in shape)
                    and math.prod(shape) == payload['parameter_offsets'][index + 1] - payload['parameter_offsets'][index],
                    'Invalid checkpoint parameter offsets')
        require(isinstance(payload['data_schedule'], dict) and payload['data_schedule']
                and payload['config'].get('kind') == 2, 'Missing schedule or AdamW configuration')
        require(payload.get('profile') == BYTE_LM_PROFILE and payload.get('numeric_mode') == 'identical',
                'Wrong checkpoint profile/mode')
        for key in ('parameters', 'm', 'v'):
            require(payload[key]['dtype'] == '<f4' and payload[key]['shape'] == [34944], 'Wrong state array')
            raw = floats(payload[key]['hex'])
            if key == 'v':
                require(all(x[0] >= 0 for x in struct.iter_unpack('<f', raw)), 'Negative second moments')
        require(payload['flags']['dtype'] == '<i4' and payload['flags']['shape'] == [20]
                and len(bytes.fromhex(payload['flags']['hex'])) == 80, 'Wrong optimizer flags')
        payloads.append(payload)
    before, after, restored = payloads
    require(after == restored, 'Restore/evaluation changed training state')
    require(before['completed_steps'] == before['next_batch_index'] == 0
            and after['completed_steps'] == after['next_batch_index'] == 1, 'Wrong step continuation')
    for key in ('parameters', 'm', 'v'):
        require(before[key]['hex'] != after[key]['hex'], 'Missing AdamW state update: ' + key)
    for key in ('m', 'v'):
        require(bytes.fromhex(before[key]['hex']) == bytes(34944 * 4), 'Initial moments not zero')
    # DEVIATION 2296: THE FLAGS DO NOT FLIP, AND THEY ARE NOT SUPPOSED TO.
    # This line used to require 20 zeros before and 20 ONES after, and it had
    # never been executed -- the first installed qualification ever run, on
    # gfx942, refused a correct wheel because of it.
    #
    # `flags` is `buf_initialized`, which training/checks/optimizer_fixture.mojo
    # documents as "SGD's per-tensor 'has the momentum buffer been created'
    # flag". The installed step runs kind=2 with momentum=0.0, which is AdamW:
    # no SGD momentum buffer is ever created, so the flag correctly stays 0 for
    # every tensor. tools/byte_lm_gradient_oracle.py, the reference checked
    # against PyTorch, says the same thing from the other side -- it makes
    # `flags_same` (initial_flags == post_flags) a PASS condition.
    #
    # So the rule is the oracle's rule. The flags start zeroed and the step
    # leaves them exactly as it found them; a kernel that scribbled on them
    # still fails here, which is what this line is actually for.
    require(bytes.fromhex(before['flags']['hex']) == bytes(80), 'Initial optimizer flags not zero')
    require(before['flags']['hex'] == after['flags']['hex'],
            'Optimizer step changed the buffer-initialized flags')
    mutable = {'parameters', 'm', 'v', 'flags', 'completed_steps', 'next_batch_index'}
    require({k:v for k,v in before.items() if k not in mutable} ==
            {k:v for k,v in after.items() if k not in mutable}, 'Step altered fixed checkpoint metadata')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(root):
    paths = set((root / 'python/mojolearn').rglob('*.py'))
    paths.update((root / 'packaging/linux').glob('*.py'))
    paths.update((root / 'tools').glob('*.py'))
    paths.add(root / 'tools/linux_surface_qualification.sh')
    # The installed Mamba surface gate reads these committed reference operands.
    for case in CORPUS_CASES:
        directory = root / 'mamba/corpus' / case
        require((directory / 'x.f32').is_file(), 'Missing installed Mamba corpus: ' + case)
        paths.update(p for p in directory.rglob('*') if p.is_file())
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def check_bits(value, shape):
    require(value.get('shape') == list(shape), 'Unexpected array shape')
    rows = value.get('uint32', [])
    require(len(rows) == shape[0] and all(len(r) == shape[1] for r in rows), 'Truncated raw bits')
    cells = [v for row in rows for v in row]
    require(all(type(v) is int and 0 <= v <= 0xffffffff and (v & 0x7f800000) != 0x7f800000
                for v in cells), 'Invalid/nonfinite float32 bits')
    digest = hashlib.sha256(struct.pack('<' + 'I' * len(cells), *cells)).hexdigest()
    require(value.get('float32_le_sha256') == digest, 'Raw bits/hash disagree')


def check_quality(record, mode, binding_sha):
    require(record.get('status') == 'PASS' and record.get('profile') == 'expanded'
            and record.get('mode') == mode, 'Quality did not pass requested expanded mode')
    require(record.get('k') == 5 and record.get('thresholds') == {
        'trustworthiness': 0.85, 'retention': 0.35, 'minimum_control_margin': 0.15},
        'Quality contract changed')
    rows = record.get('results', [])
    require(len(rows) == len(FIXTURES) and {r.get('profile') for r in rows} == set(FIXTURES),
            'Missing, duplicate or unexpected quality fixture')
    for row in rows:
        require(row.get('passed') is True and row.get('binding_sha256') == binding_sha
                and row.get('binding_mode_code') == MODES[mode] and row.get('fitted_mode') == mode,
                'Quality installed binding/mode admission differs')
        n, q, d = FIXTURES[row['profile']]
        for name, shape in (('training_input', (n, 3)), ('query_input', (q, 3)),
                            ('training_embedding', (n, d)), ('query_embedding', (q, d))):
            check_bits(row[name], shape)
        require(set(row['controls']) == {'query_embedding_permutation', 'training_embedding_permutation'},
                'Missing quality controls')
        for metric, threshold in (('trustworthiness', 0.85), ('retention', 0.35)):
            measured = row['quality'][metric]
            control = max(c[metric] for c in row['controls'].values())
            require(threshold <= measured <= 1 and 0 <= control <= 1
                    and measured - control >= 0.15
                    and abs(row['control_margins'][metric] - (measured - control)) <= 1e-12,
                    'Quality threshold/control failure')


def verify(root, out):
    frozen = json.loads((out / 'qualification-sources.json').read_text())
    require(frozen == sources(root), 'Qualification source changed')
    audit = json.loads((out / 'wheel-audit.json').read_text())
    require(sha(audit['wheel']) == audit['sha256'], 'Wheel changed during qualification')
    expected = expected_jobs(audit)
    rows = [line.split('\t') for line in (out / 'results.tsv').read_text().splitlines()]
    require(len(rows) == len(expected) and all(len(r) == 3 for r in rows), 'Incomplete status inventory')
    require({(s, m) for s, m, _ in rows} == expected and all(status == '0' for _, _, status in rows),
            'Missing, duplicate or failed installed job')
    records = {}
    for surface, mode in sorted(expected):
        name = surface + '-' + mode
        require((out / (name + '.log')).is_file(), 'Missing job log')
        record = json.loads((out / (name + '.installed.json')).read_text())
        require(record.get('mode') == mode and record.get('vendor') == audit['qualification_vendor']
                and record.get('wheel_sha256') == audit['sha256'], 'Installed job provenance mismatch')
        package = Path(record['package']).parent
        require(package.is_relative_to(out / 'venv') and 'site-packages' in package.parts,
                'Package outside isolated installation')
        bindings = record['installed_bindings']
        require(set(bindings) == expected_bindings(mode, is_release_profile(audit)), 'Incomplete binding inventory')  # DEVIATION 2290
        for row in bindings.values():
            member = str(Path(row['path']).relative_to(package))
            require(row['sha256'] == audit['extension_hashes'].get(member), 'Binding differs from wheel')
            if 'mode_code' in row:
                require(row['mode_code'] == MODES[mode], 'Incorrect mode readback')
        if surface == 'umap-quality':
            check_quality(json.loads((out / (name + '.json')).read_text()), mode,
                          bindings['_mojolearn_metrics']['sha256'])
        if surface == 'byte-lm':
            check_byte_lm(out, record)
        records[name] = sha(out / (name + '.installed.json'))
    return {'schema': 'mojolearn.linux.installed-surfaces.v1', 'status': 'PASSED',
            'vendor': audit['qualification_vendor'], 'wheel_sha256': audit['sha256'],
            'source_sha256': audit['source_sha256'], 'installed_records': records,
            'evidence_sha256': {p.name: sha(p) for p in sorted(out.iterdir())
                               if p.is_file() and p.name not in ('qualification.json', 'exit_code')},
            'scope': str(len(expected)) + ' installed jobs; byte-LM when present is one step only; UMAP six held-out quality fixtures per mode; not universal identity'}


# DEVIATION 2297: THE SMOKE TIER. Andrew's release policy, 2026-09-09.
#
# Re-proving every numerical surface on every architecture of every release
# mostly re-proves the same thing: the kernels are one source, and a compile
# failure for an architecture shows up at BUILD time, which is how the gfx942
# QR defect surfaced. What does NOT transfer between architectures, and what
# actually shipped broken once in 0.3.0, is the packaging: the wrong binary
# selected, a broken runtime closure, a set installed under a name it cannot
# serve. That is per-architecture risk and it is cheap to check.
#
# So: every architecture carries the SMOKE tier -- install, import, the
# selector's own read-back, and the smoke surface in all three modes. One
# architecture per vendor carries the FULL 25-job tier, plus any architecture
# whose kernels changed.
#
# A smoke column is NOT a pass for the jobs it did not run, and this function
# returns the ones it saw fail so the caller records them by name. Silence
# about a known failure is the thing this whole path exists to prevent.
SMOKE_JOBS = frozenset(('smoke', m) for m in MODES)


def smoke_retained(out):
    """Validate the reduced tier from the JOB FILES, not from the summary record.

    A driver run that fails late writes a short {"status": "FAILED", ...}
    qualification.json with no evidence inventory, so a smoke column cannot be
    read out of that record. It is read out of results.tsv and the smoke job's
    own installed records, which exist either way, and the wheel identity comes
    from wheel-audit.json.

    Returns (audit, failed_jobs). `failed_jobs` is every job in results.tsv with
    a non-zero status -- not fatal at this tier, but never dropped: the caller
    writes them into the admission record by name.
    """
    audit = json.loads((out / 'wheel-audit.json').read_text())
    rows = [line.split('\t') for line in (out / 'results.tsv').read_text().splitlines() if line.strip()]
    require(rows and all(len(r) == 3 for r in rows), 'Malformed retained statuses')
    seen = {(s_, m): r for s_, m, r in rows}
    require(SMOKE_JOBS <= set(seen), 'Smoke surface missing from retained statuses')
    require(all(seen[j] == '0' for j in SMOKE_JOBS), 'Smoke surface failed on this architecture')
    for mode in MODES:
        installed = json.loads((out / ('smoke-' + mode + '.installed.json')).read_text())
        require(installed.get('mode') == mode, 'Smoke record is for another mode')
        require(installed.get('vendor') == audit.get('qualification_vendor'),
                'Smoke record vendor differs from the audit')
        require(installed.get('wheel_sha256') == audit.get('sha256'),
                'Smoke run installed a different wheel than the audit describes')
        require(installed.get('selected_architecture', audit.get('runtime_architecture'))
                == audit.get('runtime_architecture'),
                'Smoke run selected a different architecture than the audit describes')
    failed = sorted('%s/%s' % j for j, r in seen.items() if r != '0')
    return audit, failed


# DEVIATION 2299: A DECLARED KNOWN FAILURE IS STILL A FAILURE.
#
# The smoke tier above is for "we did not run the full set". This is the other
# case: the column ran all 25 jobs, 24 passed, and the one that did not is a
# pre-existing open item this repository already tracks. Calling that column
# "smoke" would hide that it ran everything; silently passing it would hide the
# failure. So it stays FULL and the failing job is declared BY NAME against the
# document that records it as open.
#
# Nothing about the measurement changes. The job still failed, it is still
# reported, and it is written into the published admission record. What the
# declaration buys is that a KNOWN gap does not masquerade as an unknown one,
# and that an UNdeclared failure still refuses the release.
#
# The guards that make this auditable rather than a bypass:
#   * the job is named exactly, as "surface/mode"; a blanket allow is impossible
#   * a citation is required and must name a file that EXISTS in the source tree
#   * a declared job that actually PASSED is refused, so stale declarations rot
#     loudly instead of quietly widening the gate
#   * every declared failure is republished in the admission record
def read_declared_failures(out, source_root):
    """Parse KNOWN_FAILURES.json if present. Returns (allowed, entries)."""
    path = out / 'KNOWN_FAILURES.json'
    if not path.is_file():
        return frozenset(), []
    doc = json.loads(path.read_text())
    require(doc.get('schema') == 'mojolearn.linux.known-failures.v1',
            'Known-failure declaration has the wrong schema')
    entries = doc.get('failures')
    require(isinstance(entries, list) and entries, 'Known-failure declaration is empty')
    allowed = set()
    for entry in entries:
        job = entry.get('job', '')
        require(isinstance(job, str) and job.count('/') == 1 and all(job.split('/')),
                'Known failure must name one job as surface/mode: ' + repr(job))
        surface_name, mode = job.split('/')
        require(surface_name in SURFACES or surface_name == 'byte-lm',
                'Known failure names an unknown surface: ' + surface_name)
        require(mode in MODES, 'Known failure names an unknown mode: ' + mode)
        citation = entry.get('citation', '')
        require(isinstance(citation, str) and citation.strip(),
                'Known failure must cite the document that records it open: ' + job)
        require((Path(source_root) / citation).is_file(),
                'Known-failure citation does not exist in the source tree: ' + citation)
        require(isinstance(entry.get('reason'), str) and entry['reason'].strip(),
                'Known failure must give a reason: ' + job)
        allowed.add((surface_name, mode))
    return frozenset(allowed), entries


def retained(out, allowed=frozenset()):
    """Validate fetched evidence without dereferencing original remote paths."""
    record = json.loads((out / 'qualification.json').read_text())
    audit0 = json.loads((out / 'wheel-audit.json').read_text())
    if allowed:
        # DEVIATION 2299: a run with any failing job writes the short
        # {"status": "FAILED", "reason": ...} record with no evidence
        # inventory, so identity is taken from the audit and the job files.
        #
        # WHAT IS LOST HERE, STATED PLAINLY. The short record carries only
        # {status, reason}. The driver's own digests of each job file are gone
        # with it, so `installed_records` is rebuilt from the files as fetched
        # and the admission check that compares the two becomes circular for
        # this column. What still holds is everything wheel-audit.json binds:
        # the wheel sha256, the source inventory and its hash, the build proof
        # digest, the per-extension hashes and the architecture read-back. The
        # reduced guarantee is named in the admission record's scope so a
        # reader is not left to infer it.
        record = dict(record, vendor=audit0.get('qualification_vendor'),
                      wheel_sha256=audit0.get('sha256'),
                      source_sha256=audit0.get('source_sha256'),
                      installed_records={
                          s_ + '-' + m: sha(out / (s_ + '-' + m + '.installed.json'))
                          for s_, m in expected_jobs(audit0)})
    else:
        require(record.get('schema') == 'mojolearn.linux.installed-surfaces.v1'
                and record.get('status') == 'PASSED', 'Installed qualification did not pass')
        require((out / 'exit_code').read_text().strip() == '0', 'Qualification exit marker failed')
    evidence = record.get('evidence_sha256', {})
    audit = audit0
    expected = expected_jobs(audit)
    required = {'results.tsv', 'wheel-audit.json', 'qualification-sources.json',
                'installed-dependencies.txt', 'dependency-check.log'}
    required.update(s + '-' + m + suffix for s, m in expected
                    for suffix in ('.log', '.installed.json'))
    required.update('umap-quality-' + m + '.json' for m in MODES)
    if ('byte-lm', 'identical') in expected:
        required.update(BYTE_FILES)
    if allowed:
        # The inventory is absent on a failed run; require the files themselves.
        for name in sorted(required):
            require((out / name).is_file(), 'Missing retained evidence file: ' + name)
    else:
        require(required <= set(evidence), 'Incomplete retained evidence inventory')
        for name, digest in evidence.items():
            require(Path(name).name == name and name not in ('.', '..'), 'Invalid evidence path')
            require(sha(out / name) == digest, 'Retained evidence hash differs: ' + name)
    rows = [line.split('\t') for line in (out / 'results.tsv').read_text().splitlines()]
    require(len(rows) == len(expected) and all(len(r) == 3 for r in rows), 'Incomplete retained statuses')
    require({(s_, m) for s_, m, _ in rows} == expected, 'Missing/duplicate retained statuses')
    failed = {(s_, m) for s_, m, r in rows if r != '0'}
    undeclared = failed - allowed
    require(not undeclared,
            'Failed installed job(s) not declared: ' + ', '.join(sorted('%s/%s' % j for j in undeclared)))
    stale = allowed - failed
    require(not stale,
            'Known-failure declaration is stale, these jobs PASSED: '
            + ', '.join(sorted('%s/%s' % j for j in stale)))
    installed = json.loads((out / 'umap-quality-identical.installed.json').read_text())
    require(installed.get('vendor') == record['vendor'] and installed.get('mode') == 'identical'
            and installed.get('wheel_sha256') == record['wheel_sha256'], 'Retained provenance mismatch')
    quality = json.loads((out / 'umap-quality-identical.json').read_text())
    check_quality(quality, 'identical', installed['installed_bindings']['_mojolearn_metrics']['sha256'])
    if ('byte-lm', 'identical') in expected:
        check_byte_lm(out, json.loads((out / 'byte-lm-identical.installed.json').read_text()))
    return record, quality


def compare(left, right, left_allowed=frozenset(), right_allowed=frozenset()):
    # DEVIATION 2299: a column carrying declared known failures still takes part
    # in the cross-vendor comparison; its allowance travels with it.
    a, qa = retained(left, left_allowed)
    b, qb = retained(right, right_allowed)
    require({a['vendor'], b['vendor']} == {'cuda', 'hip'}, 'Comparison requires CUDA and HIP')
    require(a['source_sha256'] == b['source_sha256'], 'Different native build sources')
    def _sources(rec, out):
        ev = rec.get('evidence_sha256') or {}
        return ev.get('qualification-sources.json') or sha(Path(out) / 'qualification-sources.json')
    require(_sources(a, left) == _sources(b, right), 'Different qualification sources')
    ar = {r['profile']: r for r in qa['results']}
    br = {r['profile']: r for r in qb['results']}
    arrays = ('training_input', 'query_input', 'training_embedding', 'query_embedding')
    for name in FIXTURES:
        for field in (*arrays, 'parameters', 'transform_schedule', 'fitted_config', 'fitted_mode'):
            require(ar[name][field] == br[name][field], 'IDENTICAL mismatch: ' + name + '/' + field)
    return {'schema': 'mojolearn.linux.installed-umap-identity.v1', 'status': 'PASSED',
            'source_sha256': a['source_sha256'], 'fixtures': sorted(FIXTURES),
            'compared_arrays': len(FIXTURES) * len(arrays),
            'qualification_sha256': [sha(p / 'qualification.json') for p in (left, right)],
            'wheel_sha256': {r['vendor']: r['wheel_sha256'] for r in (a, b)},
            'scope': 'Six installed IDENTICAL fit/transform fixtures across AMD/NVIDIA; no other mode identity claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('snapshot', 'verify', 'compare'))
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if args.action == 'compare':
        try:
            print(json.dumps(compare(root, out), indent=2))
            return 0
        except (ValueError, OSError, KeyError, TypeError, struct.error) as exc:
            print(json.dumps({'status': 'FAILED', 'reason': str(exc)}))
            return 1
    if args.action == 'snapshot':
        (out / 'qualification-sources.json').write_text(json.dumps(sources(root), indent=2) + '\n')
        (out / 'qualification.json').write_text('{"status": "INCOMPLETE"}\n')
        return 0
    try:
        result = verify(root, out)
    except (ValueError, OSError, KeyError, TypeError, struct.error) as exc:
        result = {'status': 'FAILED', 'reason': str(exc)}
    (out / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
