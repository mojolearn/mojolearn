"""On-pod fallback assertions: admitted selection, PTX-produced column, named refusals."""
import copy

import pytest

import nvidia_ptx_fallback_e2e as e2e

CONFIG = dict(device_name='NVIDIA A100 80GB PCIe', compute_capability=[8, 0],
              driver_version='580.159.04', cuda_driver_version=13000)
EXPECT = dict(source_commit='a' * 40, manifest_sha256='b' * 64, admission_sha256='c' * 64, configuration=CONFIG)
MANIFEST = dict(files=[dict(file='identical/_mojolearn.so', sha256='d' * 64)])
RECEIPT = dict(schema='mojolearn.ptx-baseline-selection.v1', requested='native-first', selected='ptx-baseline',
               native_fallback=False, code_format='ptx-baseline', identical_qualified=True,
               source_commit='a' * 40, manifest_sha256='b' * 64, admission_sha256='c' * 64,
               configuration=CONFIG, loaded_files=[dict(file='identical/_mojolearn.so', sha256='d' * 64)])
PLUGIN = dict(code_format='ptx-baseline', identical_qualified=True, distribution='mojolearn-nvidia')
STATE = dict(vendor='cuda', numeric_mode='identical', gpu_arch='sm_80')


def test_admitted_fallback_selection_passes():
    assert e2e.check_selection(RECEIPT, PLUGIN, STATE, EXPECT, MANIFEST) == dict(loaded_files=1)


@pytest.mark.parametrize('change', [
    lambda r, p, s: r.update(requested='ptx-baseline'),          # the forced experimental route
    lambda r, p, s: r.update(selected='native'),
    lambda r, p, s: r.update(native_fallback=True),
    lambda r, p, s: r.update(identical_qualified=False),
    lambda r, p, s: r.update(admission_sha256='0' * 64),
    lambda r, p, s: r.update(manifest_sha256='0' * 64),
    lambda r, p, s: r.update(source_commit='0' * 40),
    lambda r, p, s: r['configuration'].update(driver_version='581.0'),
    lambda r, p, s: r.update(loaded_files=[]),
    lambda r, p, s: r['loaded_files'][0].update(sha256='0' * 64),
    lambda r, p, s: p.update(code_format='native'),
    lambda r, p, s: p.update(identical_qualified=False),
    lambda r, p, s: s.update(vendor='cpu'),
    lambda r, p, s: s.update(gpu_arch='sm_89'),
    lambda r, p, s: s.update(numeric_mode='fast'),
])
def test_any_other_selection_fails(change):
    receipt, plugin, state = copy.deepcopy(RECEIPT), dict(PLUGIN), dict(STATE)
    change(receipt, plugin, state)
    with pytest.raises(ValueError):
        e2e.check_selection(receipt, plugin, state, EXPECT, MANIFEST)


def test_missing_receipt_means_the_fallback_was_not_taken():
    with pytest.raises(ValueError, match='fallback was not taken'):
        e2e.check_selection(None, PLUGIN, STATE, EXPECT, MANIFEST)


def column():
    return dict(complete=True, commit='a' * 40, mode='identical', skipped=[],
                fixtures=dict(base={}, denormal={}, odd={}), cells={'ridge/base': {}},
                package=dict(bindings=[dict(module='_mojolearn', sha256='d' * 64),
                                       dict(module='_mojolearn_host', sha256='e' * 64)]))


def test_column_must_come_from_the_bundled_ptx_payload():
    assert e2e.check_column(column(), EXPECT, MANIFEST) == dict(cells=1, gpu_bindings=1)
    for change in (lambda c: c['package']['bindings'][0].update(sha256='f' * 64),   # a native binding
                   lambda c: c['package'].update(bindings=c['package']['bindings'][1:]),  # host only
                   lambda c: c.update(complete=False), lambda c: c.update(commit='0' * 40),
                   lambda c: c.update(mode='fast'), lambda c: c.update(skipped=['ridge']),
                   lambda c: c['fixtures'].update(ties={}), lambda c: c['package'].update(bindings_error='x')):
        bad = column()
        change(bad)
        with pytest.raises(ValueError):
            e2e.check_column(bad, EXPECT, MANIFEST)


def refusal(case):
    text = 'mojolearn: no native set for sm_80\n' + e2e.REFUSED + ': ' + e2e.REASONS[case]
    return text, dict(returncode=1, stderr='mojolearn._backend.GpuPluginError: ' + text)


@pytest.mark.parametrize('case', sorted(e2e.REASONS))
def test_negative_install_refuses_by_name_and_never_selects_cpu(case):
    message, second = refusal(case)
    assert e2e.check_refusal(case, 'GpuPluginError', message, second)['case'] == case
    other = next(name for name in e2e.REASONS if name != case)
    for args in ((case, None, '', second),                      # the import succeeded
                 (case, 'ImportError', message, second),         # refused, but not by name
                 (case, 'GpuPluginError', refusal(other)[0], second),
                 (case, 'GpuPluginError', message, dict(returncode=0, stderr='')),
                 (case, 'GpuPluginError', message + ' cpu-only set selected', second),
                 ('unknown', 'GpuPluginError', message, second)):
        with pytest.raises(ValueError):
            e2e.check_refusal(*args)


def test_forcing_variables_fail_the_stage():
    e2e.unforced({'PATH': '/bin', 'MOJOLEARN_CUDA_PATH': ''})
    for name in e2e.FORCING:
        with pytest.raises(ValueError, match=name):
            e2e.unforced({name: '1'})
    assert {'MOJOLEARN_CUDA_PATH', 'MOJOLEARN_EXPERIMENTAL_PTX'} <= set(e2e.FORCING)
