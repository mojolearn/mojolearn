# SPDX-License-Identifier: Apache-2.0
"""Small policy tests for lane-only admission; no GPU or native imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_installed_lanes as gate


class InstalledLaneTests(unittest.TestCase):
    def statuses(self, directory, failure=('mamba', 'identical')):
        text = ''.join(f'{surface}\t{mode}\t{134 if (surface, mode) == failure else 0}\n'
                       for surface, mode in sorted(gate.expected_jobs({})))
        (directory / 'results.tsv').write_text(text)

    def test_unrelated_failure_retained_in_all_rows(self):
        # DEVIATION 2490: smoke in three tiers plus seven identical-only surfaces.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.statuses(directory)
            rows = gate.status_rows(directory, {})
            self.assertEqual(len(rows), 10)
            self.assertEqual([r for r in rows if r['exit_code']],
                             [{'surface': 'mamba', 'mode': 'identical', 'exit_code': 134}])

    def test_target_failure_or_missing_row_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.statuses(directory, ('ordered-rmse', 'identical'))
            with self.assertRaisesRegex(ValueError, 'Target lane failed'):
                gate.status_rows(directory, {})
            self.statuses(directory)
            path = directory / 'results.tsv'
            path.write_text('\n'.join(path.read_text().splitlines()[:-1]))
            with self.assertRaisesRegex(ValueError, 'every installed exit row'):
                gate.status_rows(directory, {})

    def fixtures(self):
        fields = ('training_input', 'query_input', 'training_embedding', 'query_embedding',
                  'parameters', 'transform_schedule', 'fitted_config', 'fitted_mode')
        quality = {'results': [dict(profile=fixture, **{field: [1, 2] for field in fields})
                               for fixture in gate.FIXTURES]}
        ordered = dict(model_text='trees 3\n', model_sha256='d' * 64,
                       **{field: [0] * count for field, count in gate.ARRAYS.items()})
        results = []
        for vendor in ('hip', 'cuda'):
            summary = {'vendor': vendor, 'source_sha256': 'a' * 64,
                       'qualification_sources_sha256': 'b' * 64,
                       'candidate_status': {'status': 'FAILED', 'exit_code': 1},
                       'qualification_status': {'status': 'FAILED'},
                       'installed_exit_rows': [{'surface': s, 'mode': m,
                                               'exit_code': 1 if s == 'mamba' else 0}
                                              for s, m in sorted(gate.expected_jobs({}))]}
            results.append((summary, copy.deepcopy(quality), copy.deepcopy(ordered)))
        return results

    def test_lane_match_cannot_approve_failed_candidates(self):
        fixtures = self.fixtures()
        with patch.object(gate, 'load', side_effect=fixtures):
            result = gate.compare('amd', 'nvidia')
        self.assertEqual(result['lane_status'], 'MATCH')
        self.assertIs(result['overall_release_eligible'], False)
        self.assertEqual([c['candidate_status']['status'] for c in result['candidates']],
                         ['FAILED', 'FAILED'])
        self.assertTrue(all(len(c['installed_exit_rows']) == 10 for c in result['candidates']))

    def test_source_umap_and_ordered_mismatches_refused(self):
        for mutation, needle in (
            (lambda item: item[0].update(source_sha256='c' * 64), 'Different source'),
            (lambda item: item[0].update(qualification_sources_sha256='c' * 64), 'Different source'),
            (lambda item: item[1]['results'][0].update(query_embedding=[9]), 'UMAP mismatch'),
            (lambda item: item[2].update(model_text='trees 2\n'), 'Ordered mismatch'),
            (lambda item: item[2].update(query_prediction_bits=[1] * 8), 'Ordered mismatch'),
        ):
            with self.subTest(needle=needle):
                fixtures = self.fixtures()
                mutation(fixtures[1])
                with patch.object(gate, 'load', side_effect=fixtures):
                    with self.assertRaisesRegex(ValueError, needle):
                        gate.compare('amd', 'nvidia')

    def test_installed_wrong_tier_path_refused_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = '/remote/gone/venv/lib/python3.12/site-packages/mojolearn'
            bindings = {name: {'path': base + '/hip/gfx942/identical/' + name + '.so',
                               'sha256': 'a' * 64, 'mode_code': 1}
                        for name in gate.BINDINGS}
            audit = {'qualification_vendor': 'hip', 'sha256': 'b' * 64,
                     'extension_hashes': {row['path'].removeprefix(base + '/'): row['sha256']
                                          for row in bindings.values()}}
            record = {'mode': 'identical', 'vendor': 'hip', 'wheel_sha256': 'b' * 64,
                      'package': base + '/__init__.py', 'installed_bindings': bindings}
            path = directory / 'ordered-rmse-identical.installed.json'
            path.write_text(json.dumps(record))
            gate.installed(directory, 'ordered-rmse', 'identical', audit)
            row = bindings['_mojolearn_gbdt']
            row['path'] = row['path'].replace('/identical/', '/deterministic/')
            audit['extension_hashes'][row['path'].removeprefix(base + '/')] = row['sha256']
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'path/tier differs'):
                gate.installed(directory, 'ordered-rmse', 'identical', audit)


if __name__ == '__main__':
    unittest.main()
