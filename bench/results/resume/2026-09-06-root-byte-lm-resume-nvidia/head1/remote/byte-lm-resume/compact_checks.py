import os
from pathlib import Path
import sys

phase, repo_arg, base_arg, base_sha, out_arg, vendor, action, foreign_arg, foreign_sha, cp_sha = sys.argv[1:]
repo, out = Path(repo_arg), Path(out_arg)
sys.path.insert(0, str(repo / 'tools'))
from byte_lm_resume_handoff import (load_handoff, compatible_handoffs, compare_capture_hashes,
    compact_control, exclusive)
from byte_lm_state_compare import canonical, load_capture, parse, read, receipt, require, sha
from byte_lm_real_text_capture import source_inventory

base = load_handoff(Path(base_arg), base_sha, kind='baseline128', vendor=vendor)
reference = base['data']
require(sha(read(repo / 'tools/byte_lm_resume_handoff.py')) == reference['author_source_sha256'],
        'local handoff helper differs from root authoring source')
require(source_inventory() == reference['source'], 'numerical source differs from compact baseline')
binary = read(repo / 'python/mojolearn/identical/_mojolearn_byte_lm.so')
require(binary == read(base['directory'] / reference['binding_file']) and
        sha(binary) == reference['binding_sha256'], 'installed binding differs from retained baseline bytes')
head = None
if action == 'resume128':
    head = load_handoff(Path(foreign_arg), foreign_sha, kind='head64')
    require(head['data']['vendor'] != vendor and sha(head['checkpoint']) == cp_sha,
            'foreign vendor/checkpoint root pin differs')
    compatible_handoffs(base, head)

if phase == 'preflight':
    # Preserve offered compact witnesses and their original receipt-relative
    # files so fetched outputs carry the complete offered transfer chain.
    for label, bundle in (('baseline', base), ('foreign', head)):
        if bundle is None:
            continue
        for parent, _, files in os.walk(bundle['directory']):
            for name in files:
                path = Path(parent) / name
                exclusive(out / 'handoffs' / label / path.relative_to(bundle['directory']), read(path))
    if head is not None:
        exclusive(out / 'transferred-head64.checkpoint.json', head['checkpoint'])
    exclusive(out / 'preflight.json', canonical(dict(
        schema='mojolearn.byte-lm.compact-resume-preflight.v1', vendor=vendor, action=action,
        baseline_handoff_sha256=base_sha, foreign_handoff_sha256=foreign_sha or None,
        foreign_checkpoint_sha256=cp_sha or None, binding_sha256=sha(binary),
        baseline_summary_sha256=reference['summary_sha256'],
        foreign_summary_sha256=head['data']['summary_sha256'] if head is not None else None,
        campaign_source_sha256=sha(read(out / 'campaign-source.sh')),
        checks_source_sha256=sha(read(out / 'compact_checks.py')),
        identity_admitted=False, learning_admitted=False,
        boundary='Compact root handoff verified; remote hashes cannot admit a full baseline. '
                 'Fetch all new raw outputs and run final all-raw comparator locally.')))
else:
    before = parse(read(out / 'preflight.json'))
    require(before['baseline_handoff_sha256'] == base_sha and before['binding_sha256'] == sha(binary),
            'compact baseline changed during campaign')
    result = dict(schema='mojolearn.byte-lm.compact-resume-diagnostic.v1', phase=phase,
                  identity_admitted=False, learning_admitted=False,
                  boundary='Remote diagnostic only; final full raw baseline comparison remains mandatory')
    if head is not None:
        require(read(out / 'transferred-head64.checkpoint.json') == head['checkpoint'], 'transferred bytes changed')
    if phase == 'verify-head':
        leg = load_capture(out / 'head64', 'head64', vendor)
        compare_capture_hashes(leg, base)
        receipt(out / 'head64.receipt.json', leg['summary_sha256'], vendor)
        result['checkpoint_sha256'] = sha(leg['checkpoint'])
    elif phase in ('verify-resume', 'verify-control'):
        leg = load_capture(out / 'resume128', 'resume128', vendor)
        compare_capture_hashes(leg, base)
        require(leg['incoming'] == head['checkpoint'], 'resume did not consume actual foreign checkpoint')
        receipt(out / 'resume128.receipt.json', leg['summary_sha256'], vendor)
        result['terminal_checkpoint_sha256'] = sha(leg['checkpoint'])
        if phase == 'verify-control':
            control = load_capture(out / 'zero-moments65', 'zero-moments65', vendor)
            receipt(out / 'zero-moments65.receipt.json', control['summary_sha256'], vendor)
            result['control'] = compact_control(head, leg, control, base)
    else:
        raise ValueError('unknown compact check phase')
    exclusive(out / (phase + '.json'), canonical(result))
print('Compact checks complete; not a full-raw admission.')
