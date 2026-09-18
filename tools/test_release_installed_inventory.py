"""Exercise the installed runner's actual tier-specific binary readback block."""
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from verify_linux_surface_qualification import expected_bindings

ROOT = Path(__file__).resolve().parents[1]


class InstalledInventoryTests(unittest.TestCase):
    def readback(self, mode, missing=None, changed=None, wrong_mode=False):
        script = (ROOT / 'tools/release_linux_surface_qualification.sh').read_text()
        block = script.split('# Older bindings expose vendor', 1)[1]
        block = '# Older bindings expose vendor' + block.split("record = {'package':", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory).resolve() / 'site-packages/mojolearn'
            package.mkdir(parents=True)
            hashes = {}
            modules = {}
            loaded = []
            for name in expected_bindings(mode, True):
                path = package / ('cuda/sm_90a/' + (mode+'/' if mode != 'fast' else '') + name+'.so')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                hashes[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
                code = {'fast': 0, 'deterministic': 2, 'identical': 1}[mode]
                class Binding:
                    __file__ = str(path)
                    def __getattr__(self, key):
                        if key.endswith('_numeric_mode'):
                            return lambda: -1 if wrong_mode else code
                        raise AttributeError(key)
                modules[name] = Binding()
            if changed:
                Path(modules[changed].__file__).write_bytes(b'changed')
            def binding(name):
                loaded.append(name)
                if name == missing:
                    raise ImportError(name)
                return modules[name]  # a forbidden tier binding is absent
            env = {'MOJOLEARN_REPO': str(ROOT), 'MOJOLEARN_EXPECT_VENDOR': 'cuda', 'MOJOLEARN_NUMERIC_MODE': mode}
            context = dict(hashlib=hashlib, pathlib=__import__('pathlib'), sys=__import__('sys'), os=os,
                           mojolearn=types.SimpleNamespace(numeric_mode=lambda: mode),
                           _backend=types.SimpleNamespace(binding=binding, read_vendor=lambda module: 'cuda'),
                           installed=package/'__init__.py', release_profile=True,
                           audit={'extension_hashes': hashes}, architecture={'selected_architecture': 'sm_90a'})
            with patch.dict(os.environ, env):
                exec(compile(block, 'installed-readback', 'exec'), context)
            return set(loaded), context['readback']

    def test_all_advertised_bindings_and_only_supported_tiers_load(self):
        for mode, count in [('fast', 3), ('deterministic', 3), ('identical', 23)]:
            with self.subTest(mode=mode):
                loaded, rows = self.readback(mode)
                self.assertEqual(loaded, expected_bindings(mode, True))
                self.assertEqual(len(rows), count)

    def test_new_binding_cannot_be_missing_or_changed(self):
        with self.assertRaises(ImportError):
            self.readback('identical', missing='_mojolearn_embedding')
        with self.assertRaisesRegex(AssertionError, 'Installed binary differs'):
            self.readback('identical', changed='_mojolearn_preprocessing')

    def test_wrong_compiled_mode_fails(self):
        with self.assertRaises(AssertionError):
            self.readback('fast', wrong_mode=True)


if __name__ == '__main__':
    unittest.main()
