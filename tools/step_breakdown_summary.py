#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 2630: breakdown.tsv and witnesses.tsv from a step breakdown leg.

Reads the run directories tools/step_breakdown_leg.sh writes under OUT:

  lean-shipped-<corpus>/result.json   shipped build, lean step, untimed
  lean-timers-<corpus>/result.json    timers build, switch OFF, untimed
  timing-<corpus>/result.json         timers build, --component-timing-steps N
  lean-shipped2-<corpus>/result.json  optional: the shipped build again, last

and writes, into OUT:

  breakdown.tsv   per corpus: every timed interval of the shipped step as a
                  TREE rooted at `envelope.native_call` (the Python wrapper's
                  whole native call). Each parent's children are the ticks
                  that partition it; `remainder:<parent>` is the parent minus
                  its children PER STEP, then the median (timer prints, host
                  work between ticks, anything no tick brackets), so the
                  leaves plus the remainders telescope to the envelope. Then
                  the leaves grouped by category, the GEMM call kinds (a
                  cross-cut that re-labels leaves, never added to the tree),
                  the per-step counts, and the lean medians with the timer
                  overhead (timers build switch OFF minus shipped, and the
                  switch-ON wall minus shipped).
  witnesses.tsv   per corpus and step: whether the shipped lean run, the timers
                  lean run and the switch-ON run have the same loss, gradient,
                  parameter, moment and flag hashes. `bits_identical` is True
                  only when every present pair is equal.

Every value is ms per step, the median over the timed steps; `share` is of
the envelope's median. A timed step is not a timing sample: the lean rows are
the price, the tree is where the time goes inside it."""
import json
import statistics
import sys
from pathlib import Path

ROOT = 'envelope.native_call'
TREE = {
    'envelope.native_call': [
        'step.bind_scalar_admission', 'step.upload_inputs', 'step.unpack_weights',
        'step.embedding_forward', 'envelope.blocks_forward', 'step.head_forward',
        'step.ce_refuse_scan', 'step.ce_forward', 'step.loss_download', 'step.ce_backward',
        'step.head_backward_da', 'step.head_backward_db', 'envelope.blocks_backward',
        'step.embedding_backward', 'step.pack_grads', 'step.validate_grads_scan',
        'step.shadow_copy', 'step.opt_refuse_scan', 'step.optimizer',
        'step.validate_after_scan', 'step.bind_final_sync', 'step.bind_publish'],
    'envelope.blocks_forward': ['block.norm1', 'block.attention_total', 'block.mlp_and_residuals'],
    'block.norm1': ['fwd.refuse_call', 'fwd.norm1'],
    'block.attention_total': ['attn.qkv_proj', 'attn.rope_and_cache', 'attn.core', 'attn.o_proj'],
    'attn.qkv_proj': ['fwd.q_proj', 'fwd.k_proj', 'fwd.v_proj'],
    'attn.o_proj': ['fwd.o_proj'],
    'block.mlp_and_residuals': [
        'fwd.residual1_add', 'fwd.norm2', 'fwd.gate_proj', 'fwd.up_proj', 'fwd.silu',
        'fwd.gate_mul', 'fwd.down_proj', 'fwd.residual2_add'],
    'envelope.blocks_backward': [
        'bwd.mlp_through_oproj', 'bwd.before_attention', 'bwd.attention', 'bwd.after_attention'],
    'bwd.mlp_through_oproj': [
        'grad.refuse_scan', 'grad.residual2_copy', 'grad.down_dA', 'grad.down_dB', 'grad.gate_mul',
        'grad.silu', 'grad.gate_dB', 'grad.up_dB', 'grad.gate_dA', 'grad.up_dA', 'grad.gateup_fanin',
        'grad.norm2_kernels', 'grad.norm2_dW', 'grad.residual1_add', 'grad.o_copy', 'grad.o_dA',
        'grad.o_dB'],
    'bwd.after_attention': [
        'grad.kv_slice', 'grad.rope', 'grad.q_dB', 'grad.k_dB', 'grad.v_dB', 'grad.q_dA', 'grad.k_dA',
        'grad.v_dA', 'grad.qkv_fanin', 'grad.norm1_kernels', 'grad.norm1_dW', 'grad.x_add'],
}
# Children found by prefix: the fused attention launchers' per-kernel lines
# (-D MOJOLEARN_ATTN_PHASE_TIMERS=1), whose names depend on the arm.
PREFIX_CHILDREN = {'attn.core': 'attn.fwd_', 'bwd.attention': 'attn.bwd_'}

GEMM_LEAVES = [
    'fwd.q_proj', 'fwd.k_proj', 'fwd.v_proj', 'fwd.o_proj', 'fwd.gate_proj', 'fwd.up_proj',
    'fwd.down_proj', 'step.head_forward', 'step.head_backward_da', 'step.head_backward_db',
    'grad.down_dA', 'grad.down_dB', 'grad.gate_dB', 'grad.up_dB', 'grad.gate_dA', 'grad.up_dA',
    'grad.o_dA', 'grad.o_dB', 'grad.q_dB', 'grad.k_dB', 'grad.v_dB', 'grad.q_dA', 'grad.k_dA',
    'grad.v_dA', 'grad.norm1_dW', 'grad.norm2_dW']
CATEGORIES = [
    ('gemm (every GEMM call, head and RMSNorm weight gradients included)', GEMM_LEAVES),
    ('embedding forward and backward', ['step.embedding_forward', 'step.embedding_backward']),
    ('rmsnorm forward', ['fwd.norm1', 'fwd.norm2']),
    ('rmsnorm backward kernels (weight gradient GEMM under gemm)', ['grad.norm1_kernels', 'grad.norm2_kernels']),
    ('swiglu activation forward (silu, gate product)', ['fwd.silu', 'fwd.gate_mul']),
    ('swiglu activation backward (gate product, silu)', ['grad.gate_mul', 'grad.silu']),
    ('residual adds, copies and fan-in adds', [
        'fwd.residual1_add', 'fwd.residual2_add', 'grad.residual2_copy', 'grad.residual1_add',
        'grad.o_copy', 'grad.gateup_fanin', 'grad.qkv_fanin', 'grad.x_add']),
    ('attention rope and kv cache', ['attn.rope_and_cache', 'grad.kv_slice', 'grad.rope']),
    ('attention kernels (fused launchers, regime scans, corner flags)', ['bwd.before_attention']),
    ('refusal and validation scans', [
        'fwd.refuse_call', 'grad.refuse_scan', 'step.ce_refuse_scan', 'step.opt_refuse_scan',
        'step.validate_grads_scan', 'step.validate_after_scan']),
    ('logits softmax and cross entropy, forward and backward, loss readback', [
        'step.ce_forward', 'step.ce_backward', 'step.loss_download']),
    ('adamw update and its shadow copy', ['step.optimizer', 'step.shadow_copy']),
    ('ids upload, weight unpack and gradient pack copies', [
        'step.upload_inputs', 'step.unpack_weights', 'step.pack_grads']),
    ('binding admission, final wait and publish', [
        'step.bind_scalar_admission', 'step.bind_final_sync', 'step.bind_publish']),
]
PREFIX_CATEGORY = {'attn.fwd_': 'attention kernels (fused launchers, regime scans, corner flags)',
                   'attn.bwd_': 'attention kernels (fused launchers, regime scans, corner flags)'}
# A parent that printed no children (a build without the timers) is a leaf.
PARENT_CATEGORY = {
    'block.norm1': 'rmsnorm forward',
    'block.mlp_and_residuals': 'unsplit: block.mlp_and_residuals',
    'block.attention_total': 'unsplit: block.attention_total',
    'bwd.after_attention': 'unsplit: bwd.after_attention',
}
REMAINDER_CATEGORY = 'remainders (timer prints, host work between ticks, untimed code)'
HEADER = ['corpus', 'kind', 'component', 'parent', 'depth', 'ms_per_step_median', 'share_of_envelope',
          'lines_per_step', 'launches_per_step', 'syncs_per_step', 'note']


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _fmt(value, digits=3):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _category(leaf):
    for name, members in CATEGORIES:
        if leaf in members:
            return name
    for prefix, name in PREFIX_CATEGORY.items():
        if leaf.startswith(prefix):
            return name
    return PARENT_CATEGORY.get(leaf, 'other: ' + leaf)


class Timing:
    def __init__(self, result):
        self.ms = result.get('component_timing_ms') or {}
        self.per = result.get('component_timing_ms_per_step') or {}
        self.counts = result.get('component_counts') or {}
        self.lines = result.get('component_line_counts') or {}
        self.n = int(result.get('component_timing_steps') or result.get('component_timing_steps_parsed') or 1)

    def steps(self, name):
        values = self.per.get(name)
        if values and len(values) == self.n:
            return list(values)
        return [self.ms[name]] * self.n

    def children(self, parent):
        kids = [k for k in TREE.get(parent, []) if k in self.ms]
        prefix = PREFIX_CHILDREN.get(parent)
        if prefix:
            kids += sorted(k for k in self.ms if k.startswith(prefix))
        return kids


def breakdown_rows(corpus, timing, lean):
    t = Timing(timing)
    rows = []
    if ROOT not in t.ms:
        return [[corpus, 'error', ROOT, '', '', '', '', '', '', '', 'no envelope line; nothing to break down']]
    env = t.ms[ROOT]
    env_steps = t.steps(ROOT)
    seen = set()
    leaves = []

    def add(kind, name, parent, depth, value, note=''):
        rows.append([corpus, kind, name, parent, depth, _fmt(value), _fmt(value / env if env else None, 4),
                     _fmt(t.lines.get(name)), _fmt(t.counts.get('launches.' + name), 1),
                     _fmt(t.counts.get('syncs.' + name), 1), note])

    def walk(node, parent, depth):
        seen.add(node)
        kids = t.children(node)
        add('parent' if kids else 'leaf', node, parent, depth, t.ms[node])
        if not kids:
            leaves.append((node, t.steps(node)))
            return
        for kid in kids:
            walk(kid, node, depth + 1)
        rem_steps = [p - sum(t.steps(k)[i] for k in kids) for i, p in enumerate(t.steps(node))]
        rem = statistics.median(rem_steps)
        add('remainder', 'remainder:' + node, node, depth + 1, rem,
            'parent minus its children per step, then the median')
        leaves.append(('remainder:' + node, rem_steps))

    walk(ROOT, '', 0)
    # Categories: leaves and remainders summed PER STEP, then the median.
    by_cat = {}
    for name, steps in leaves:
        cat = REMAINDER_CATEGORY if name.startswith('remainder:') else _category(name)
        acc = by_cat.setdefault(cat, [0.0] * t.n)
        for i, v in enumerate(steps):
            acc[i] += v
    for cat, steps in sorted(by_cat.items(), key=lambda kv: -statistics.median(kv[1])):
        value = statistics.median(steps)
        rows.append([corpus, 'category', cat, ROOT, 1, _fmt(value), _fmt(value / env if env else None, 4),
                     '', '', '', 'leaves summed per step, then the median'])
    # The GEMM call kinds (a cross-cut of leaves above).
    gemm_total = [0.0] * t.n
    for name in sorted(k for k in t.ms if k.startswith('gemm.')):
        steps = t.steps(name)
        for i, v in enumerate(steps):
            gemm_total[i] += v
        calls = t.lines.get(name)
        per_call = (t.ms[name] / calls) if calls else None
        add('gemm', name, '(cross-cut)', '', t.ms[name],
            'calls per step = lines_per_step; ms per call ' + _fmt(per_call))
    if any(gemm_total):
        value = statistics.median(gemm_total)
        rows.append([corpus, 'gemm', 'gemm.total', '(cross-cut)', '', _fmt(value),
                     _fmt(value / env if env else None, 4), '', '', '', 'every gemm.* line per step'])
    for name in sorted(k for k in t.counts if k.startswith('count.')):
        rows.append([corpus, 'count', name, '', '', '', '', '', _fmt(t.counts[name], 1), '', 'per native step'])
    # Names the tree does not reach (outside the envelope, or a tick this
    # table does not know: a new line shows up here instead of vanishing).
    for name in sorted(t.ms):
        if name in seen or name.startswith('gemm.'):
            continue
        add('untreed', name, '', '', t.ms[name], 'not a node of the tree (outside the envelope or unknown)')
    # The price: lean medians on the same pod, and the timer overhead.
    shipped = (lean.get('shipped') or {}).get('steady_median_seconds')
    timers_off = (lean.get('timers') or {}).get('steady_median_seconds')
    shipped2 = (lean.get('shipped2') or {}).get('steady_median_seconds')
    timing_wall = timing.get('component_timing_step_seconds')

    def lean_row(name, seconds, note):
        ms = seconds * 1000.0 if seconds is not None else None
        rows.append([corpus, 'lean', name, '', '', _fmt(ms), _fmt(ms / env if (ms is not None and env) else None, 4),
                     '', '', '', note])

    lean_row('lean.shipped_build', shipped, 'steady median, shipped build, untimed (THE price)')
    lean_row('lean.timers_build_switch_off', timers_off, 'steady median, timers build, switch off')
    lean_row('lean.timers_build_switch_on_wall', timing_wall, 'median wall of the timed steps (not a price)')
    lean_row('lean.shipped_build_again', shipped2, 'shipped build rerun last (pod drift)')
    if shipped is not None and timers_off is not None:
        lean_row('overhead.compiled_in_switch_off', timers_off - shipped, 'timers build switch off minus shipped')
    if shipped is not None and timing_wall is not None:
        lean_row('overhead.switch_on', timing_wall - shipped, 'switch-on wall minus shipped')
    rows.append([corpus, 'envelope', ROOT, '', 0, _fmt(env), '1.0000', '', '', '',
                 'median of ' + str(t.n) + ' timed steps: ' + ', '.join(_fmt(v) for v in env_steps)])
    return rows


def witness_rows(corpus, runs):
    def sequence(result, with_timing):
        if not result:
            return None
        seq = [w.get('sha256') for w in result.get('step_witnesses') or []]
        if with_timing:
            seq += [w.get('sha256') for w in result.get('component_timing_witnesses') or []]
        return seq

    seqs = {
        'shipped': sequence(runs.get('shipped'), False),
        'timers_off': sequence(runs.get('timers'), False),
        'timers_on': sequence(runs.get('timing'), True),
        'shipped_again': sequence(runs.get('shipped2'), False),
    }
    ref = seqs['shipped']
    rows = []
    ok = bool(ref)
    for name in ('timers_off', 'timers_on', 'shipped_again'):
        seq = seqs[name]
        if seq is None:
            continue
        for i, hashes in enumerate(seq):
            equal = ref is not None and i < len(ref) and hashes is not None and hashes == ref[i]
            ok = ok and equal
            rows.append([corpus, name, i + 1, equal])
        if not seq:
            ok = False
    return rows, ok


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: step_breakdown_summary.py OUT')
    out = Path(sys.argv[1])
    corpora = sorted({d.name.split('-', 1)[1] for d in out.glob('timing-*') if d.is_dir()})
    gpu = (out / 'nvidia_smi.txt').read_text().strip() if (out / 'nvidia_smi.txt').exists() else 'unrecorded'
    host = (out / 'host.txt').read_text().strip() if (out / 'host.txt').exists() else 'unrecorded'
    lines = ['# DEVIATION 2630 step breakdown; ms per step, median over the timed steps',
             '# gpu: ' + gpu.replace('\n', ' | '), '# host: ' + host.replace('\n', ' | '),
             '\t'.join(HEADER)]
    wit_lines = ['corpus\trun\tstep\twitnesses_equal_shipped']
    all_ok = bool(corpora)
    summary = []
    for corpus in corpora:
        runs = {
            'shipped': _load(out / f'lean-shipped-{corpus}' / 'result.json'),
            'timers': _load(out / f'lean-timers-{corpus}' / 'result.json'),
            'timing': _load(out / f'timing-{corpus}' / 'result.json'),
            'shipped2': _load(out / f'lean-shipped2-{corpus}' / 'result.json'),
        }
        if runs['timing'] is None:
            lines.append('\t'.join([corpus, 'error', 'timing', '', '', '', '', '', '', '', 'no result.json']))
            all_ok = False
            continue
        rows = breakdown_rows(corpus, runs['timing'], runs)
        lines += ['\t'.join(str(c) for c in row) for row in rows]
        wrows, ok = witness_rows(corpus, runs)
        wit_lines += ['\t'.join(str(c) for c in row) for row in wrows]
        wit_lines.append(f'{corpus}\tbits_identical\t\t{ok}')
        all_ok = all_ok and ok
        summary.append(f'== {corpus}: bits_identical={ok}')
        for row in rows:
            if row[1] in ('envelope', 'lean', 'category') or (row[1] == 'remainder' and row[3] == ROOT):
                summary.append(f'  {row[1]:9s} {row[5]:>10s} ms  {row[6]:>7s}  {row[2]}')
    wit_lines.append(f'all\tbits_identical\t\t{all_ok}')
    (out / 'breakdown.tsv').write_text('\n'.join(lines) + '\n')
    (out / 'witnesses.tsv').write_text('\n'.join(wit_lines) + '\n')
    print('\n'.join(summary))
    print(f'bits_identical_all={all_ok}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
