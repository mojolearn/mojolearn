"""Host mocks only. Authored without execution; root/main runs tests."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nvidia_public_compare as compare


class ExternalModes(unittest.TestCase):
    def test_default_excludes_our_fast_for_both_external_profiles(self):
        for external_mode in ('fast', 'deterministic'):
            args = SimpleNamespace(external_mode=external_mode, worker='fast')
            self.assertEqual(compare.selected_arms(args), ('identical', 'external'))
            with patch.object(compare.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'excluded by --ours-mode'):
                    compare.worker_command(args, 'fast')
                with self.assertRaisesRegex(ValueError, 'explicit legacy'):
                    compare.worker(args)  # refuses before NumPy/Torch/native imports
                launch.assert_not_called()

    def test_only_explicit_legacy_option_selects_our_fast(self):
        args = SimpleNamespace(ours_mode='legacy-fast-and-identical')
        self.assertEqual(compare.selected_arms(args), ('fast', 'identical', 'external'))

    def test_worker_commands_propagate_identical_only_policy(self):
        args = SimpleNamespace(lane='nt', dim=8, rows=8, gram_rows=8,
            index=8, queries=2, k=1, knn_external='torch', umap_rows=32,
            umap_epochs=10, external_mode='deterministic')
        for arm in compare.selected_arms(args):
            command = compare.worker_command(args, arm)
            self.assertEqual(command[command.index('--worker') + 1], arm)
            self.assertEqual(command[command.index('--ours-mode') + 1], 'identical-only')
            self.assertEqual(command[command.index('--external-mode') + 1], 'deterministic')

    def test_unsupported_incumbents_refuse_before_import_or_worker(self):
        for lane, backend in [('gbdt', 'cuml'), ('umap', 'cuml'), ('knn', 'cuml')]:
            args = SimpleNamespace(lane=lane, knn_external=backend, external_mode='deterministic')
            with self.subTest(lane=lane), patch.object(compare.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'EXTERNAL_DETERMINISTIC_UNSUPPORTED'):
                    compare.main(args)
                launch.assert_not_called()

    def test_default_fast_and_torch_deterministic_admission(self):
        self.assertEqual(compare.validate_external_mode(SimpleNamespace(lane='gbdt')), 'fast')
        for lane in ('gemv', 'nt', 'gram', 'knn'):
            self.assertEqual(compare.validate_external_mode(SimpleNamespace(
                lane=lane, knn_external='torch', external_mode='deterministic')), 'deterministic')

    def test_workspace_is_worker_environment_before_torch_import(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':bad:'}):
            external = compare.worker_environment('external', 'nt', 'deterministic')
            native = compare.worker_environment('identical', 'nt', 'deterministic')
            self.assertEqual(external['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
            self.assertEqual(native['CUBLAS_WORKSPACE_CONFIG'], ':bad:')

    def test_strict_flags_and_fp32_for_both_external_profiles(self):
        for mode in ('fast', 'deterministic'):
            state = {'deterministic': False, 'warn_only': False}
            def use_deterministic(enabled, warn_only=False):
                state.update(deterministic=enabled, warn_only=warn_only)
            fake = SimpleNamespace(
                use_deterministic_algorithms=use_deterministic,
                are_deterministic_algorithms_enabled=lambda: state['deterministic'],
                is_deterministic_algorithms_warn_only_enabled=lambda: state['warn_only'],
                backends=SimpleNamespace(
                    cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                    cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False)))
            with self.subTest(mode=mode), patch.dict(os.environ, {}, clear=True):
                flags = compare.configure_external_torch(fake, mode)
                self.assertFalse(flags['cuda_matmul_allow_tf32'])
                self.assertFalse(flags['cudnn_allow_tf32'])
                self.assertEqual(flags['deterministic_algorithms'], mode == 'deterministic')
                self.assertFalse(flags['deterministic_warn_only'])
                if mode == 'deterministic':
                    self.assertEqual(flags['cublas_workspace_config'], ':4096:8')
                    self.assertFalse(flags['cudnn_benchmark'])
                    self.assertTrue(flags['cudnn_deterministic'])


if __name__ == '__main__':
    unittest.main()
