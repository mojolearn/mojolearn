"""Authored root-only mocked guards; no subprocess or GPU workload is launched."""
import argparse
import contextlib
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import amd_serial_guard as guard


class DetectionTests(unittest.TestCase):
    def device(self, root, name, vendor='0x1002', used='0', total='17179869184'):
        device = root / name / 'device'
        device.mkdir(parents=True)
        (device / 'vendor').write_text(vendor)
        (device / 'mem_info_vram_used').write_text(used)
        (device / 'mem_info_vram_total').write_text(total)
        return device

    def test_render_alias_and_duplicate_card_do_not_add_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = self.device(root, 'card0')
            (root / 'renderD128').symlink_to(root / 'card0', target_is_directory=True)
            (root / 'card1').mkdir()
            (root / 'card1' / 'device').symlink_to(device, target_is_directory=True)
            self.device(root, 'card2', vendor='0x10de')
            self.assertEqual(guard.amd_device(root), device.resolve())

    def test_multiple_amd_devices_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.device(root, 'card0')
            self.device(root, 'card1')
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                guard.amd_device(root)

    def test_missing_and_invalid_counters_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = self.device(root, 'card0', total='0')
            with self.assertRaisesRegex(RuntimeError, 'VRAM'):
                guard.amd_device(root)
            (device / 'mem_info_vram_total').unlink()
            with self.assertRaises(FileNotFoundError):
                guard.amd_device(root)


class GuardTests(unittest.TestCase):
    def test_non_linux_refuses_before_launch(self):
        with patch.object(guard.sys, 'platform', 'darwin'), patch.object(guard.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'Refusing local'):
                guard.run(argparse.Namespace(command=['forbidden'], seconds=10, rss_gib=12))
            launch.assert_not_called()

    def exercise(self, violation):
        handlers = {}
        proc = Mock(pid=4321, returncode=-15)
        proc.poll.return_value = None
        def mem(group):
            if group == -1:
                return 0, 8 * guard.GIB
            if violation == 'signal':
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            if violation == 'monitor':
                raise OSError('proc memory monitor unavailable')
            if violation == 'rss':
                return 13 * guard.GIB, 8 * guard.GIB
            return 0, (1 if violation == 'pressure' else 8) * guard.GIB
        with contextlib.ExitStack() as stack:
            locks = [stack.enter_context(tempfile.TemporaryFile()) for _ in guard.LOCK_PATHS]
            stack.enter_context(patch.object(guard.sys, 'platform', 'linux'))
            stack.enter_context(patch.object(guard.Path, 'is_dir', return_value=True))
            stack.enter_context(patch.object(guard.Path, 'exists', return_value=True))
            opened = stack.enter_context(patch('builtins.open', side_effect=locks))
            flock = stack.enter_context(patch.object(guard.fcntl, 'flock'))
            stack.enter_context(patch.object(guard, 'amd_device', return_value=Path('/fake/amd')))
            vram = [(0, 16 * guard.GIB), (15 * guard.GIB if violation == 'vram' else 0, 16 * guard.GIB)]
            if violation == 'occupied':
                vram[0] = (513 * 2**20, 16 * guard.GIB)
            stack.enter_context(patch.object(guard, 'gpu_memory', side_effect=vram))
            stack.enter_context(patch.object(guard, 'memory', side_effect=mem))
            stack.enter_context(patch.object(guard.os, 'sched_getaffinity', return_value={0, 1, 2}, create=True))
            stack.enter_context(patch.object(guard.signal, 'signal', side_effect=lambda n, h: handlers.update({n: h})))
            stack.enter_context(patch.object(guard.time, 'monotonic', side_effect=[0, 11 if violation == 'deadline' else 1]))
            launch = stack.enter_context(patch.object(guard.subprocess, 'Popen', return_value=proc))
            stop = stack.enter_context(patch.object(guard, 'stop_group'))
            args = argparse.Namespace(command=['fake-command'], seconds=10, rss_gib=12)
            if violation == 'lock':
                flock.side_effect = [None, BlockingIOError('NVIDIA job already active')]
                with self.assertRaises(BlockingIOError):
                    guard.run(args)
                launch.assert_not_called()
                stop.assert_not_called()
                return
            if violation == 'occupied':
                with self.assertRaisesRegex(RuntimeError, 'occupied'):
                    guard.run(args)
                launch.assert_not_called()
                stop.assert_not_called()
                return
            if violation in ('signal', 'monitor'):
                with self.assertRaises(InterruptedError if violation == 'signal' else OSError):
                    guard.run(args)
            else:
                self.assertEqual(guard.run(args), 124)
            self.assertEqual([c.args[0] for c in opened.call_args_list], list(guard.LOCK_PATHS))
            stop.assert_called_once_with(4321)
            proc.wait.assert_called_once()
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            self.assertEqual(launch.call_args.args[0][:3], ['taskset', '-c', '0,1'])
            self.assertEqual(launch.call_args.kwargs['env']['MAX_JOBS'], '2')

    def test_rss_stops_entire_group(self):
        self.exercise('rss')

    def test_memory_pressure_stops_entire_group(self):
        self.exercise('pressure')

    def test_vram_stops_entire_group(self):
        self.exercise('vram')

    def test_deadline_stops_entire_group(self):
        self.exercise('deadline')

    def test_termination_signal_cleans_descendants(self):
        self.exercise('signal')

    def test_monitor_failure_cleans_descendants(self):
        self.exercise('monitor')

    def test_occupied_gpu_refused_before_launch(self):
        self.exercise('occupied')

    def test_legacy_nvidia_lock_refuses_overlap(self):
        self.exercise('lock')


if __name__ == '__main__':
    unittest.main()
