"""Invented metadata only: no estimator, tensor, compiler or measured receipts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from performance_measurement_board import build
from six_lane_compare_results import COLUMNS, SCHEMA, board_inputs, compare_cases, main


def captured(path, digest):
    return dict(status='CAPTURED', sha256=digest * 64, scope='fixture complete scope',
                encoding='fixture-encoding-v1', completeness='complete_declared_scope', missing_state=[],
                manifest=[dict(path=path, dtype='<f4', shape=[3], encoding='C-order raw bytes', bytes=12)])


class ComparisonMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = dict(defines=[], environment={}, runtime={})
        self.case = dict(id='invented-case', configuration_id='fixture:alternate',
                         implementation_ids=['fixture'], source_sha='1' * 40,
                         workload_id='fixture:no-estimator', mode='identical', columns={})
        self.expected = dict(dataset_sha256='2' * 64, dataset_version='fixture-v1', dataset_split='fixture-split',
                             seed=0, dimensions={'X': [3, 2]}, estimator_settings={'fixture': True},
                             harness_sha256='3' * 64, timed_boundary='fixture whole operation',
                             configurations={'A': dict(config, defines=['FIXTURE=1']), 'B': config},
                             capture_paths={'outputs': ['$.out'], 'model_state': ['$.state']})
        self.case['expected'] = self.expected
        self.manifest = dict(schema=SCHEMA, cases=[self.case])
        self.catalog = dict(entries=[dict(id='fixture', title='Metadata fixture', mode='identical',
                                         vendors=['nvidia', 'amd', 'apple', 'host'])])
        self.receipts = {}
        for column, vendor in COLUMNS.items():
            runs = []
            for arm in ('A', 'B'):
                data = {k: copy.deepcopy(v) for k, v in self.expected.items()
                        if k not in ('configurations', 'capture_paths')}
                data.update(schema='mojolearn.full-ab-result/1', status='PASS', source_sha=self.case['source_sha'],
                            workload_id=self.case['workload_id'], mode='identical', vendor=vendor, arm=arm,
                            phase='scored', sample_counts={'excluded_warmups': 0, 'scored': 1},
                            full_dataset_coverage=True, hashing_outside_timing=True, implementation_ids=['fixture'],
                            configuration=copy.deepcopy(self.expected['configurations'][arm]),
                            hardware={'machine': column}, compiler=['fixture-compiler'], resource_policy={'fixture': True},
                            thread_environment={},
                            loaded_artifacts={column + '.so': '4' * 64},
                            outputs=captured('$.out', 'a' if arm == 'A' else 'b'),
                            model_state=captured('$.state', 'c' if arm == 'A' else 'd'),
                            repeated_use=[],
                            output_sha256=('a' if arm == 'A' else 'b') * 64,
                            task_quality={'status': 'PENDING'}, timings={'full_operation_seconds': 1})
                runs.append(dict(phase='scored', arm=arm, excluded=False, returncode=0, result=data,
                                 log='fixture.log', output='fixture.json', result_sha256='5' * 64))
            artifact = dict(path=column + '.so', sha256='4' * 64, numerical_source_sha='6' * 40,
                            compiler='fixture-compiler', target=column, defines=[])
            receipt = dict(status='MEASURED_FULL', source_sha=self.case['source_sha'], mode='identical',
                           key='fixture-' + column, runs=runs, previous_receipt='older-retained-attempt.json',
                           workload=dict(master_selection={'id': self.case['configuration_id']},
                                         artifact_provenance={arm: [copy.deepcopy(artifact)] for arm in ('A', 'B')}))
            self.receipts[column] = receipt
            self.case['columns'][column] = column + '.json'
        self.save()

    def save(self):
        for name, value in self.receipts.items():
            (self.root / (name + '.json')).write_text(json.dumps(value))

    def result(self, column='amd', arm='A'):
        return next(r['result'] for r in self.receipts[column]['runs'] if r['arm'] == arm)

    def compare(self):
        self.save()
        return compare_cases(self.manifest, self.root)

    def pair(self, case, arm='A', left='nvidia-native', right='amd'):
        return next(p for p in case['arms'][arm]['pairs'] if p['left'] == left and p['right'] == right)

    def test_separate_arms_can_match_with_different_a_b_hashes(self):
        report = self.compare()
        self.assertEqual(report['counts']['MATCH'], 1)
        self.assertNotEqual(self.result()['outputs']['sha256'], self.result(arm='B')['outputs']['sha256'])
        self.assertFalse(report['accepted'])
        self.assertFalse(report['apple_timing_votes'])
        self.assertEqual(len(report['cases'][0]['arms']['A']['pairs']), 10)

    def test_native_and_amd_agreement_decides_identity(self):
        # Owner policy (2026-10-07): identity is required across the two GPU
        # vendors only. Apple, host and PTX columns are optional.
        del self.case['columns']['apple']
        del self.case['columns']['host']
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'MATCH')
        self.assertEqual(self.pair(case)['status'], 'MATCH')
        self.assertEqual(self.pair(case, left='nvidia-ptx')['status'], 'MATCH')
        self.assertEqual(case['missing_columns'], [])
        self.assertEqual(case['optional_columns_absent'], ['apple', 'host'])
        self.assertEqual(case['arms']['A']['decided_by'], ['nvidia-native', 'amd'])
        del self.case['columns']['amd']
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'INCOMPLETE')
        self.assertEqual(case['missing_columns'], ['amd'])

    def test_structured_saved_split_is_pinned_without_string_coercion(self):
        split = {'fit': [0, 3], 'evaluation_rows': 2}
        self.expected['dataset_split'] = split
        for column in COLUMNS:
            for arm in ('A', 'B'):
                self.result(column, arm)['dataset_split'] = copy.deepcopy(split)
        self.assertEqual(self.compare()['cases'][0]['status'], 'MATCH')
        self.result()['dataset_split']['evaluation_rows'] = 1
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'INCOMPLETE')
        self.assertIn('scope differs: dataset_split', case['columns']['amd']['arms']['A']['issues'])

    def test_output_match_decides_when_model_state_is_not_captured(self):
        # Owner policy (2026-10-07): the stored output hash decides identity
        # when a complete typed model-state export is absent; model state is
        # compared whenever both columns captured it completely.
        self.result()['model_state'] = dict(status='UNAVAILABLE', reason='no public complete state')
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'MATCH')
        self.assertEqual(case['arms']['B']['status'], 'MATCH')
        self.assertEqual(self.pair(case)['parts'], {'outputs': 'MATCH', 'model_state': 'NOT_CAPTURED'})
        self.result()['outputs']['sha256'] = self.result()['output_sha256'] = 'e' * 64
        self.assertEqual(self.compare()['cases'][0]['status'], 'MISMATCH')

    def test_model_hash_mismatch_is_preserved(self):
        self.result()['model_state']['sha256'] = 'e' * 64
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'MISMATCH')
        self.assertEqual(self.pair(case)['parts'], {'outputs': 'MATCH', 'model_state': 'MISMATCH'})

    def test_typed_metadata_is_compared_not_just_hashes(self):
        self.result()['outputs']['manifest'][0]['dtype'] = '<f8'
        case = self.compare()['cases'][0]
        self.assertEqual(self.pair(case)['parts']['outputs'], 'MISMATCH')

    def test_untyped_incomplete_or_wrong_paths_never_match(self):
        original = copy.deepcopy(self.result()['outputs'])
        changes = [dict(completeness='scope_not_qualified'), dict(missing_state=['$.missing']),
                   dict(manifest=[]), dict(sha256='bad'), dict(scope='')]
        for change in changes:
            with self.subTest(change=change):
                self.result()['outputs'] = dict(original, **change)
                self.assertEqual(self.compare()['cases'][0]['status'], 'INCOMPLETE')
        self.result()['outputs'] = copy.deepcopy(original)
        self.result()['outputs']['manifest'][0]['path'] = '$.not-the-declared-output'
        self.assertEqual(self.compare()['cases'][0]['status'], 'INCOMPLETE')
        self.result()['outputs'] = copy.deepcopy(original)
        # An incompletely captured model state is NOT_CAPTURED, never a match or a mismatch.
        state = copy.deepcopy(self.result()['model_state'])
        for change in (dict(completeness='scope_not_qualified'), dict(missing_state=['$.missing'])):
            with self.subTest(change=change):
                self.result()['model_state'] = dict(state, **change)
                case = self.compare()['cases'][0]
                self.assertEqual(case['status'], 'MATCH')
                self.assertEqual(self.pair(case)['parts']['model_state'], 'NOT_CAPTURED')
        self.result()['model_state'] = state
        self.result()['model_state']['manifest'][0]['path'] = '$.not-the-declared-model'
        self.assertEqual(self.compare()['cases'][0]['status'], 'INCOMPLETE')

    def test_scope_drift_is_incomplete_not_a_bit_mismatch(self):
        original = copy.deepcopy(self.result())
        changes = {'source_sha': '9' * 40, 'dataset_sha256': '9' * 64, 'dataset_version': 'other',
                   'dataset_split': 'other', 'seed': 1, 'mode': 'fast', 'dimensions': {'X': [4, 2]},
                   'estimator_settings': {'fixture': False}, 'workload_id': 'different',
                   'harness_sha256': '9' * 64, 'timed_boundary': 'other'}
        for key, value in changes.items():
            with self.subTest(key=key):
                self.result().clear()
                self.result().update(copy.deepcopy(original))
                self.result()[key] = value
                case = self.compare()['cases'][0]
                self.assertEqual(case['status'], 'INCOMPLETE')
                self.assertIn('scope differs', case['columns']['amd']['arms']['A']['issues'][0])

    def test_only_declared_transport_environment_is_ignored(self):
        env = {'MOJOLEARN_VENDOR': 'hip'}
        self.result()['configuration']['environment'] = env
        self.assertEqual(self.compare()['cases'][0]['status'], 'INCOMPLETE')
        self.case['columns']['amd'] = dict(receipt='amd.json', transport_environment={'A': env})
        self.assertEqual(self.compare()['cases'][0]['status'], 'MATCH')
        unsafe = {'MOJOLEARN_CANDIDATE': '1'}
        self.result()['configuration']['environment'] = unsafe
        self.case['columns']['amd']['transport_environment']['A'] = unsafe
        self.assertEqual(self.compare()['cases'][0]['status'], 'INCOMPLETE')

    def test_compiler_hardware_and_binary_hashes_can_differ(self):
        self.result()['compiler'] = ['different-compiler-target']
        self.result()['loaded_artifacts'] = {'different.so': 'f' * 64}
        self.receipts['amd']['workload']['artifact_provenance']['A'][0].update(path='different.so', sha256='f' * 64,
                                                                           compiler='different-compiler-target')
        self.assertEqual(self.compare()['cases'][0]['status'], 'MATCH')

    def test_same_receipt_cannot_fill_both_nvidia_routes(self):
        self.case['columns']['nvidia-ptx'] = 'nvidia-native.json'
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'INCOMPLETE')
        self.assertIn('same receipt reused', case['columns']['nvidia-native']['arms']['A']['issues'][0])

    def test_incomplete_or_inconsistent_artifact_provenance_is_not_a_match(self):
        self.receipts['amd']['workload']['artifact_provenance']['A'][0]['sha256'] = 'f' * 64
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'INCOMPLETE')
        self.assertIn('artifact declarations differ', case['columns']['amd']['arms']['A']['issues'][0])

    def test_failed_attempt_or_duplicate_scored_sample_is_not_admitted(self):
        self.receipts['amd']['status'] = 'MEASUREMENT_FAILED'
        self.receipts['amd']['error'] = 'retained original failure'
        report = self.compare()
        self.assertEqual(report['cases'][0]['status'], 'INCOMPLETE')
        self.assertEqual(report['cases'][0]['columns']['amd']['attempt']['error'], 'retained original failure')
        self.receipts['amd']['status'] = 'MEASURED_FULL'
        del self.receipts['amd']['error']
        self.receipts['amd']['runs'].append(copy.deepcopy(self.receipts['amd']['runs'][0]))
        self.assertEqual(self.compare()['cases'][0]['status'], 'INCOMPLETE')

    def test_warmup_hashes_are_never_compared(self):
        warmup = copy.deepcopy(self.receipts['amd']['runs'][0])
        warmup.update(phase='warmup', excluded=True)
        warmup['result']['outputs']['sha256'] = 'e' * 64
        self.receipts['amd']['runs'].append(warmup)
        self.assertEqual(self.compare()['cases'][0]['status'], 'MATCH')

    def test_declared_repeated_captures_are_compared_and_not_silently_omitted(self):
        for column in COLUMNS:
            for arm in ('A', 'B'):
                data = self.result(column, arm)
                data['repeated_use'] = [dict(index=0, outputs=copy.deepcopy(data['outputs']),
                                            model_state=copy.deepcopy(data['model_state']))]
        self.assertEqual(self.compare()['cases'][0]['status'], 'INCOMPLETE')
        self.expected['repeated_operations'] = 1
        self.assertEqual(self.compare()['cases'][0]['status'], 'MATCH')
        self.result()['repeated_use'][0]['model_state']['sha256'] = 'f' * 64
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'MISMATCH')
        self.assertEqual(self.pair(case)['parts']['repeated.0.model_state'], 'MISMATCH')

    def test_fast_missing_receipts_are_still_incomplete_coverage(self):
        self.case.update(mode='fast', columns={})
        report = self.compare()
        self.assertEqual(report['cases'][0]['status'], 'NOT_REQUIRED')
        self.assertEqual(report['cases'][0]['receipt_status'], 'INCOMPLETE')
        self.assertEqual(report['incomplete_receipt_cases'], 1)

    def test_retained_failure_history_is_not_current_evidence(self):
        (self.root / 'failed.json').write_text(json.dumps(dict(status='MEASUREMENT_FAILED', error='original failure')))
        self.case['columns']['amd'] = dict(receipt='amd.json', history=['failed.json', 'missing.json'])
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'MATCH')
        self.assertEqual(case['columns']['amd']['history'][0]['attempt']['error'], 'original failure')
        self.assertIn('error', case['columns']['amd']['history'][1])

    def test_board_inputs_preserve_history_and_never_admit_or_invent_ratios(self):
        old_cell = dict(id='fixture', vendor='amd', status='INTERRUPTED', evidence='old-original-log')
        old_index = dict(cells=[old_cell], notes=['historical note'], decisions=[{'decision': 'keep off'}])
        old_inventory = dict(campaign='retained-campaign', candidates=copy.deepcopy(self.catalog['entries']))
        report = self.compare()
        inventory, index = board_inputs(report, self.catalog, old_inventory, old_index)
        board = build(inventory, index)
        self.assertEqual(index['cells'][0], old_cell)
        self.assertEqual(old_index['cells'], [old_cell])
        self.assertEqual(index['decisions'], old_index['decisions'])
        self.assertTrue(all(c['status'] == 'PENDING_ADMISSION' for c in index['cells'][1:]))
        self.assertTrue(all('candidate_over_baseline' not in c for c in board['cards'][0]['cells']))
        self.assertFalse(board['promotion'])

    def test_fast_does_not_use_cross_vendor_bitwise_gate(self):
        self.case['mode'] = 'fast'
        self.case['columns'] = {'apple': 'apple.json'}
        self.receipts['apple']['mode'] = 'fast'
        for arm in ('A', 'B'):
            self.result('apple', arm)['mode'] = 'fast'
            self.result('apple', arm)['model_state'] = {'status': 'UNAVAILABLE'}
            self.result('apple', arm)['task_quality'] = {'status': 'FAIL', 'reason': 'fixture quality failure'}
        self.catalog['entries'][0].update(mode='fast', vendors=['apple'])
        report = self.compare()
        self.assertEqual(report['cases'][0]['status'], 'NOT_REQUIRED')
        inventory, index = board_inputs(report, self.catalog)
        self.assertEqual(index['cells'][0]['status'], 'QUALITY_FAILED')
        self.assertFalse(build(inventory, index)['promotion'])

    def test_missing_column_file_and_missing_capture_contract_are_explicit(self):
        self.case['columns']['amd'] = 'missing-receipt.json'
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'INCOMPLETE')
        self.assertIn('error', case['columns']['amd'])
        del self.expected['capture_paths']
        case = self.compare()['cases'][0]
        self.assertEqual(case['status'], 'INCOMPLETE')
        self.assertIn('pin nonempty', case['error'])

    def test_cli_emits_board_command_and_refuses_to_overwrite(self):
        manifest = self.root / 'input.json'
        manifest.write_text(json.dumps(self.manifest))
        catalog = self.root / 'catalog.json'
        catalog.write_text(json.dumps(self.catalog))
        out = self.root / 'report'
        argv = ['--manifest', str(manifest), '--catalog', str(catalog), '--out', str(out)]
        self.assertEqual(main(argv), 0)
        command = json.loads((out / 'future_command.json').read_text())
        self.assertEqual(command['execution'], 'NOT RUN')
        self.assertIn('performance_measurement_board.py', command['argv'][1])
        original = (out / 'report.json').read_bytes()
        with self.assertRaises(FileExistsError):
            main(argv)
        self.assertEqual((out / 'report.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
