import json
import os
from pathlib import Path
import stat
import sys

phase, repo_arg, baseline_arg, out_arg, vendor, action, foreign_arg, expected_sha = sys.argv[1:]
repo, baseline, out = map(Path, (repo_arg, baseline_arg, out_arg))
sys.path.insert(0, str(repo / 'tools'))
from byte_lm_state_compare import (canonical, compatible, load_capture, parse, read,
    receipt, require, resume_control, same_steps, sha)
from byte_lm_real_text_capture import source_inventory
from byte_lm_validation_admit import admit


def exclusive(path, raw, readonly=False):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if readonly:
            os.fchmod(fd, 0o444)
    finally:
        os.close(fd)


base = load_capture(baseline / 'full128', 'continuous', vendor)
require(source_inventory() == base['source'], 'current numerical source differs from continuous baseline')
installed = repo / 'python/mojolearn/identical/_mojolearn_byte_lm.so'
installed_raw = read(installed)
retained_raw = read(baseline / 'bindings/_mojolearn_byte_lm.so')
require(installed_raw == retained_raw and sha(installed_raw) == base['runtime']['binding_sha256'],
        'installed/retained binding differs from continuous baseline; rebuilding is not allowed here')
head = None
if action == 'resume128':
    foreign = Path(foreign_arg)
    require(read(foreign / 'exit_code').strip() == b'0', 'foreign head campaign did not exit successfully')
    head = load_capture(foreign / 'head64', 'head64')
    require(head['runtime']['native_vendor'] != vendor, 'checkpoint must originate on the other vendor')
    receipt(foreign / 'head64.receipt.json', head['summary_sha256'], head['runtime']['native_vendor'])
    compatible(base, head)
    require(same_steps(base, head) == 64, 'foreign first64 steps differ from local continuous baseline')
    require(sha(head['checkpoint']) == expected_sha, 'foreign checkpoint differs from root-pinned transfer SHA')

if phase == 'preflight':
    admitted = admit(baseline)
    require(admitted['passed'] is True and admitted['vendor'] == vendor,
            'continuous baseline lacks single-vendor admission')
    exclusive(out / 'baseline-admission.json', canonical(admitted))
    (out / 'bindings').mkdir()
    exclusive(out / 'bindings/_mojolearn_byte_lm.so', installed_raw, readonly=True)
    witness = dict(schema='mojolearn.byte-lm.resume-preflight.v1', action=action, vendor=vendor,
        baseline_summary_sha256=base['summary_sha256'], binding_sha256=sha(installed_raw),
        source=base['source'], schedule=base['summary']['schedule'], config=base['metadata']['config'],
        python_executable=sys.executable, numeric_mode=os.environ.get('MOJOLEARN_NUMERIC_MODE'),
        campaign_source_sha256=sha(read(out / 'campaign-source.sh')),
        artifact_checks_sha256=sha(read(out / 'artifact_checks.py')),
        qualification='preflight only; model work and guard exits are still required')
    if head is not None:
        # Exclusive read-only copy of actual transferred bytes. Every model
        # call also makes its own sealed memfd and retains incoming bytes.
        exclusive(out / 'transferred-head64.checkpoint.json', head['checkpoint'], readonly=True)
        witness['transfer'] = dict(file='transferred-head64.checkpoint.json', bytes=len(head['checkpoint']),
            sha256=expected_sha, foreign_vendor=head['runtime']['native_vendor'],
            foreign_summary_sha256=head['summary_sha256'],
            foreign_receipt_sha256=sha(read(Path(foreign_arg) / 'head64.receipt.json')),
            copied_actual_checkpoint=True, publication='exclusive file, mode0444; per-model load is sealed')
    exclusive(out / 'preflight.json', canonical(witness))
elif phase in ('verify-head', 'verify-resume', 'verify-control'):
    before = parse(read(out / 'preflight.json'))
    require(before['baseline_summary_sha256'] == base['summary_sha256'] and
            before['binding_sha256'] == sha(installed_raw) and before['source'] == base['source'] and
            read(out / 'bindings/_mojolearn_byte_lm.so') == installed_raw,
            'baseline/source/binding changed during continuation')
    if head is not None:
        transferred = out / 'transferred-head64.checkpoint.json'
        require(read(transferred, limit=2 * 1024 * 1024) == head['checkpoint'] and
                stat.S_IMODE(transferred.stat().st_mode) == 0o444 and
                before['transfer']['sha256'] == expected_sha,
                'read-only transferred checkpoint changed')
    if phase == 'verify-head':
        leg = load_capture(out / 'head64', 'head64', vendor)
        compatible(base, leg)
        require(same_steps(base, leg) == 64, 'head trajectory differs from continuous')
        receipt(out / 'head64.receipt.json', leg['summary_sha256'], vendor)
        result = dict(head_checkpoint_sha256=sha(leg['checkpoint']), compared_steps=64)
    else:
        leg = load_capture(out / 'resume128', 'resume128', vendor)
        compatible(base, leg)
        require(same_steps(base, leg) == 64 and leg['checkpoint'] == base['checkpoint'] and
                leg['incoming'] == head['checkpoint'], 'resume trajectory/checkpoint chain differs')
        receipt(out / 'resume128.receipt.json', leg['summary_sha256'], vendor)
        result = dict(transferred_checkpoint_sha256=expected_sha, compared_steps=64,
                      terminal_checkpoint_sha256=sha(leg['checkpoint']))
        if phase == 'verify-control':
            control = load_capture(out / 'zero-moments65', 'zero-moments65', vendor)
            compatible(base, control)
            receipt(out / 'zero-moments65.receipt.json', control['summary_sha256'], vendor)
            result['control'] = resume_control(head, leg, control, base)
    result.update(schema='mojolearn.byte-lm.resume-local-check.v1', phase=phase,
                  claim='checked retained local continuation bytes; full cross-vendor admission remains separate')
    exclusive(out / (phase + '.json'), canonical(result))
else:
    raise ValueError('unknown artifact check phase')
print(json.dumps(dict(phase=phase, complete=True), sort_keys=True))
