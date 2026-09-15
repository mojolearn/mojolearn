"""tools/release_wheel_content_audit.py on synthetic wheels: a clean wheel passes
every check, and each defect is refused by the check that names it."""
from pathlib import Path
import tempfile
import unittest
import zipfile

import release_wheel_content_audit as content

NOTICE = ('mojolearn\nCopyright 2026 Andrew Hendel\n\n'
          'Licensed under the Apache License, Version 2.0. See LICENSE.\n\n'
          'MAX (R) and Mojo (R) are trademarks of Modular, Inc., used under license.\n\n'
          'Modular components\n------------------\n'
          'licensed by Modular Inc under the Apache License v2.0 with LLVM Exceptions\n'
          'under the Modular MAX Community License\n')
HOST_SURFACE = 'def wheel_bindings():\n    return ["_mojolearn_core_host"]\n'


def make_wheel(root, members):
    path = Path(root) / 'mojolearn-9.9.9-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def clean_members():
    return {'mojolearn/__init__.py': '', 'mojolearn/host_surface.py': HOST_SURFACE,
            'mojolearn/tokenizer.py': '# loads a user-supplied vocabulary\n',
            'mojolearn/identical/_mojolearn.so': 'inert',
            'mojolearn-9.9.9.dist-info/licenses/NOTICE': NOTICE}


class ContentAudit(unittest.TestCase):
    def run_audit(self, members, notice=NOTICE):
        with tempfile.TemporaryDirectory() as tmp:
            notice_path = Path(tmp) / 'NOTICE'
            notice_path.write_text(notice)
            wheel = make_wheel(tmp, members)
            return {name: problems for name, problems in content.audit(wheel, notice_path)}

    def test_clean_wheel_passes_every_check(self):
        results = self.run_audit(clean_members())
        self.assertEqual(set(results), {'notice', 'no-gpt2-data', 'no-vendored-env', 'host-surface'})
        self.assertEqual({k: v for k, v in results.items() if v}, {})

    def test_each_defect_is_refused_by_its_check(self):
        defects = {
            'notice differs': ('notice', lambda m: m.update({'mojolearn-9.9.9.dist-info/licenses/NOTICE': NOTICE + 'x'})),
            'notice missing': ('notice', lambda m: m.pop('mojolearn-9.9.9.dist-info/licenses/NOTICE')),
            'gpt2 table': ('no-gpt2-data', lambda m: m.update({'mojolearn/data/gpt2_ranks.tsv': 'a 0'})),
            'gpt2 fixture': ('no-gpt2-data', lambda m: m.update({'mojolearn/checks/gpt2_reference.json': '{}'})),
            'vocab file': ('no-gpt2-data', lambda m: m.update({'mojolearn/data/merges.txt': ''})),
            'venv dir': ('no-vendored-env', lambda m: m.update({'mojolearn/.venv/lib/x.py': ''})),
            'site-packages': ('no-vendored-env', lambda m: m.update({'mojolearn/env/lib/site-packages/numpy/__init__.py': ''})),
            'pyvenv.cfg': ('no-vendored-env', lambda m: m.update({'mojolearn/env/pyvenv.cfg': ''})),
            'host surface missing': ('host-surface', lambda m: m.pop('mojolearn/host_surface.py')),
            'host surface broken': ('host-surface', lambda m: m.update({'mojolearn/host_surface.py': 'import nonexistent_module_xyz\n'})),
            'host surface empty': ('host-surface', lambda m: m.update({'mojolearn/host_surface.py': 'def wheel_bindings():\n    return []\n'})),
        }
        for label, (check, mutate) in defects.items():
            with self.subTest(defect=label):
                members = clean_members()
                mutate(members)
                results = self.run_audit(members)
                self.assertTrue(results[check], f'{check} did not refuse {label}')

    def test_notice_without_modular_components_is_refused_even_when_equal(self):
        trimmed = NOTICE.split('Modular components')[0]
        members = clean_members()
        members['mojolearn-9.9.9.dist-info/licenses/NOTICE'] = trimmed
        self.assertTrue(self.run_audit(members, notice=trimmed)['notice'])

    def test_tokenizer_module_name_is_not_data(self):
        members = clean_members()
        members['mojolearn/_tokenizer_synthetic.py'] = '# generates a synthetic vocabulary\n'
        self.assertEqual(self.run_audit(members)['no-gpt2-data'], [])


if __name__ == '__main__':
    unittest.main()
