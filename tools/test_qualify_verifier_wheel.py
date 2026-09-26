import copy
import unittest

from qualify_verifier_wheel import admit, expanded_checks


class VerifierAdmissionTests(unittest.TestCase):
    def models(self):
        models = [{"lane": "rf-clf", "fixture": "base"}]
        doc = dict(exit=0, verdict="VERIFIED", selection={"models_only": True},
                   cells=[dict(lane="portable:rf-clf", fixture="base", part=part, state="IDENTICAL")
                          for part in ("model", "batch")])
        return doc, models

    def test_models_require_every_manifest_entry_and_part(self):
        doc, models = self.models()
        admit("models", doc, models)
        for part in (0, 1):
            broken = copy.deepcopy(doc)
            broken["cells"].pop(part)
            with self.assertRaises(AssertionError):
                admit("models", broken, models)
        with self.assertRaises(AssertionError):
            admit("models", doc, models + [{"lane": "ols", "fixture": "base"}])

    def test_success_headline_cannot_hide_bad_cells(self):
        for state in ("REFUSED", "DIVERGENT", "OWED", "N/A"):
            doc, models = self.models()
            doc["cells"][0]["state"] = state
            with self.assertRaises(AssertionError):
                admit("models", doc, models)

    def test_duplicate_cells_refuse(self):
        doc, models = self.models()
        doc["cells"].append(doc["cells"][0])
        with self.assertRaises(AssertionError):
            admit("models", doc, models)

    def test_self_test_requires_clean_match_and_detected_perturbation(self):
        doc = dict(passed=True, clean={"state": "IDENTICAL"}, perturbed={"state": "DIVERGENT"})
        admit("self-test", doc, [])
        for part in ("clean", "perturbed"):
            broken = copy.deepcopy(doc)
            broken[part]["state"] = "REFUSED"
            with self.assertRaises(AssertionError):
                admit("self-test", broken, [])

    def test_extended_pending_reference_stays_incomplete_and_failures_refuse(self):
        parts = ("train", "infer", "batch", "batchgrad", "batchscale", "ragged", "rlpair")
        doc = dict(exit=5, verdict="INCOMPLETE", cells=[
            dict(lane="knn", fixture="base", part=part,
                 state="OWED" if part == "batchscale" else "IDENTICAL",
                 value="0123456789abcdef", error=None) for part in parts])
        admit("extended", doc, [])
        for state in ("REFUSED", "DIVERGENT"):
            broken = copy.deepcopy(doc)
            broken["cells"][4]["state"] = state
            with self.assertRaises(AssertionError):
                admit("extended", broken, [])
        for value in (None, "MOVED", "BATCH_MOVED:x"):
            broken = copy.deepcopy(doc)
            broken["cells"][4]["value"] = value
            with self.assertRaises((AssertionError, TypeError)):
                admit("extended", broken, [])
        doc.update(exit=0, verdict="VERIFIED")
        with self.assertRaises(AssertionError):
            admit("extended", doc, [])


class ExpandedWheelTests(unittest.TestCase):
    def run_scope(self, scope='expanded', vendor='cuda', devices=None, fail=None):
        from pathlib import Path
        calls = []
        def run(name, command, cwd, json_output=False):
            calls.append((name, command))
            if name == fail:
                raise RuntimeError('simulated installed failure')
            return {'native': 'checked'} if json_output else None
        result = expanded_checks(run, '/installed/python', Path('/external'), Path('/results'),
                                 vendor=vendor, scope=scope, devices=devices)
        return result, calls

    def test_default_requires_native_entries_and_loaded_cpu_gpu_proof(self):
        result, calls = self.run_scope()
        self.assertEqual([n for n, _ in calls], ['expanded-api', 'loaded-lm-cpu', 'loaded-lm-gpu',
                                               'loaded-lm-cpu-gpu-compare'])
        self.assertEqual(result['multi_gpu'], 'OWED')
        self.assertFalse(result['release_qualified'])
        self.assertIn('ivf_flat_partial_search', calls[0][1][-1])
        self.assertIn('ivf_finalize_distances', calls[0][1][-1])

    def test_cpu_scope_is_explicit_and_does_not_claim_gpu_proof(self):
        result, calls = self.run_scope(scope='cpu-only', vendor='cpu')
        self.assertEqual([n for n, _ in calls], ['loaded-lm-cpu'])
        self.assertEqual(result['scope'], 'cpu-only')
        self.assertFalse(result['release_qualified'])
        with self.assertRaises(RuntimeError): self.run_scope(vendor='cpu')
        with self.assertRaises(RuntimeError): self.run_scope(vendor='metal', devices=(0,1))

    def test_two_gpu_scope_requires_both_layer_orders_and_installed_distributed_gate(self):
        result, calls = self.run_scope(devices=(2,0))
        commands = dict(calls)
        self.assertIn('--require-installed', commands['distributed'])
        self.assertIn('--require-installed', commands['cross-validation'])
        self.assertIn('--require-backend', commands['cross-validation'])
        self.assertEqual(commands['distributed'][commands['distributed'].index('--devices')+1], '2,0')
        for name, expected in [('split',['2','0']), ('reversed',['0','2'])]:
            command = commands['loaded-lm-'+name]
            at = command.index('--layer-devices')
            self.assertEqual(command[at+1:at+3], expected)
            self.assertIn('loaded-lm-'+name+'-compare', commands)
        self.assertFalse(result['release_qualified'])
        self.assertEqual(result['physical_execution_trace'], 'OWED')

    def test_candidate_source_pin_mismatch_refuses_before_install(self):
        import json, tempfile, zipfile
        from pathlib import Path
        from unittest.mock import patch
        from qualify_verifier_wheel import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); wheel=root/'mojolearn-0.8.7-py3-none-any.whl'
            with zipfile.ZipFile(wheel,'w') as archive:
                archive.writestr('mojolearn/verify_reference/models/models.json',json.dumps({'models':[]}))
                archive.writestr('mojolearn/identity_columns/COMMIT','a'*40)
            argv=['qualifier',str(wheel),'--output',str(root/'out'),'--expected-source-commit','b'*40]
            with patch('sys.argv',argv), patch('subprocess.Popen') as spawn:
                with self.assertRaises(SystemExit): main()
                spawn.assert_not_called()

    def _split_wheels(self, root, plugin_version='0.8.7'):
        import json, zipfile
        core = root/'mojolearn-0.8.7-py3-none-manylinux_2_35_x86_64.whl'
        with zipfile.ZipFile(core,'w') as archive:
            archive.writestr('mojolearn/verify_reference/models/models.json',json.dumps({'models':[]}))
            archive.writestr('mojolearn/identity_columns/COMMIT','a'*40)
        plugin = root/f'mojolearn_nvidia-{plugin_version}-py3-none-manylinux_2_35_x86_64.whl'
        with zipfile.ZipFile(plugin,'w') as archive:
            archive.writestr('mojolearn/cuda/sm_89/_mojolearn_knn.so','inert')
        return core, plugin

    def test_split_plugin_is_installed_with_the_core_and_named_in_the_receipt(self):
        import hashlib, json, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from qualify_verifier_wheel import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); core, plugin = self._split_wheels(root)
            argv=['qualifier',str(core),'--plugin',str(plugin),'--output',str(root/'out')]
            commands=[]
            def spawn(command, **kw):
                commands.append(command)
                class P:
                    pid=0
                    def wait(self, timeout=None): return 0 if len(commands)==1 else 1
                return P()
            with patch('sys.argv',argv), patch('subprocess.Popen', side_effect=spawn):
                self.assertEqual(main(), 1)
            install = commands[1]
            self.assertIn(str(core.resolve()), install); self.assertIn(str(plugin.resolve()), install)
            receipt = json.loads((root/'out'/'results.json').read_text())
            self.assertEqual(receipt['plugins'], [dict(wheel=str(plugin.resolve()), distribution='mojolearn-nvidia',
                wheel_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest())])

    def test_split_plugin_of_another_version_refused_before_install(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from qualify_verifier_wheel import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); core, plugin = self._split_wheels(root, plugin_version='0.8.6')
            argv=['qualifier',str(core),'--plugin',str(plugin),'--output',str(root/'out')]
            with patch('sys.argv',argv), patch('subprocess.Popen') as spawn:
                with self.assertRaises(SystemExit): main()
                spawn.assert_not_called()

    def test_optimized_interpreter_cannot_disable_admission_checks(self):
        import os, subprocess, sys
        from pathlib import Path
        script = Path(__file__).with_name('qualify_verifier_wheel.py')
        result = subprocess.run([sys.executable, '-O', str(script), '--help'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python -O is refused', result.stderr)

    def test_every_expanded_stage_failure_propagates(self):
        _, calls = self.run_scope(devices=(0,1))
        for name, _ in calls:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):
                    self.run_scope(devices=(0,1), fail=name)


if __name__ == "__main__":
    unittest.main()
