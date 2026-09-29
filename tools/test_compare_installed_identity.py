import copy
import pytest

from compare_installed_identity import PARTS, VENDORS, compare


def records():
    selection = dict(package_source='a' * 40, lanes=['ridge'], fixtures=['base'], parts=list(PARTS))
    reports = {}
    for vendor in VENDORS:
        reports[vendor] = dict(format='mojolearn.verify-all-report.v1', exit=0, verdict='VERIFIED',
            device=dict(vendor=vendor, numeric_mode='identical', commit='a' * 40, commit_source='wheel COMMIT witness'),
            fixtures=['base'], repeats=1, lanes=['ridge'], models_checked=0,
            execution=dict(mode='fresh-process-per-cell', interrupted=None, completed_cells=1, total_cells=1),
            bindings=[dict(module='ridge', sha256='b' * 64, size=1)],
            harness=dict(sha256='c' * 64), table=dict(sha256='d' * 64),
            cells=[dict(lane='ridge', fixture='base', part=p, state='IDENTICAL', value='e' * 16,
                        reference='e' * 16) for p in PARTS])
    return reports, selection


def test_compares_original_reports_and_declares_limited_scope():
    reports, selection = records()
    original = copy.deepcopy(reports)
    result = compare(reports, selection)
    assert result['numeric_parts'] == 5 and result['status'] == 'AGREE'
    assert not result['release_qualified'] and not result['full_algorithm_coverage']
    assert reports == original


@pytest.mark.parametrize('mutation', [
    lambda r: r['cells'].pop(),
    lambda r: r['cells'].append(r['cells'][0]),
    lambda r: r['cells'][0].update(value='f' * 16),
    lambda r: r['cells'][0].update(value='f' * 16, reference='f' * 16),
    lambda r: r['cells'][0].update(state='OWED'),
    lambda r: r['cells'][0].update(state='N/A', value='n/a:UNDECLARED'),
    lambda r: r['device'].update(commit='f' * 40),
    lambda r: r['device'].update(vendor='cpu'),
    lambda r: r['table'].update(sha256='f' * 64),
    lambda r: r['execution'].update(completed_cells=0),
    lambda r: r.update(bindings=[]),
])
def test_incomplete_wrong_source_or_different_reports_fail(mutation):
    reports, selection = records()
    mutation(reports['cuda'])
    with pytest.raises(ValueError):
        compare(reports, selection)


def test_larger_nvidia_sweep_compares_shared_subset_without_discarding_failures():
    reports, selection = records()
    report = reports['cuda']
    report['lanes'].append('ols')
    report['execution'].update(completed_cells=2, total_cells=2)
    report['cells'].extend([dict(row, lane='ols') for row in report['cells']])
    assert compare(reports, selection)['numeric_parts'] == 5
    report['cells'][-1]['state'] = 'DIVERGENT'
    with pytest.raises(ValueError):
        compare(reports, selection)


def test_structural_na_is_counted_separately_and_missing_vendor_fails():
    reports, selection = records()
    for report in reports.values():
        report['cells'][-1].update(state='N/A', value='n/a:no-decode-state')
    assert compare(reports, selection)['structural_na_parts'] == 1
    del reports['hip']
    with pytest.raises(ValueError):
        compare(reports, selection)
