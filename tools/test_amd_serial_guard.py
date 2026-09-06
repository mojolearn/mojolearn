"""Authored root-only mocked guards; no subprocess or GPU workload is launched."""
import argparse
import contextlib
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import amd_serial_guard as guard


class DetectionTests(unittest.TestCase):
    def setUp(self):
        # Regular fixture files stand in for Linux character nodes. Separate
        # render_number tests below cover actual device-type admission.
        mocked = patch.object(guard, 'render_number', side_effect=lambda node: '226:' + node.name[7:])
        mocked.start()
        self.addCleanup(mocked.stop)

    def device(self, root, name, vendor='0x1002', used='0', total='17179869184'):
        device = root / name / 'device'
        device.mkdir(parents=True)
        (device / 'vendor').write_text(vendor)
        (device / 'mem_info_vram_used').write_text(used)
        (device / 'mem_info_vram_total').write_text(total)
        render_name = 'renderD' + str(128 + int(name[4:]))
        render = root / render_name
        render.mkdir()
        (render / 'device').symlink_to(device, target_is_directory=True)
        (render / 'dev').write_text('226:' + render_name[7:])
        nodes = root / 'nodes'
        nodes.mkdir(exist_ok=True)
        (nodes / render_name).touch()
        return device

    def test_render_alias_and_duplicate_card_do_not_add_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = self.device(root, 'card0')
            (root / 'card1').mkdir()
            (root / 'card1' / 'device').symlink_to(device, target_is_directory=True)
            self.device(root, 'card2', vendor='0x10de')
            self.assertEqual(guard.amd_device(root, root / 'nodes'), device.resolve())

    def test_multiple_amd_devices_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.device(root, 'card0')
            self.device(root, 'card1')
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                guard.amd_device(root, root / 'nodes')

    def test_missing_and_invalid_counters_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = self.device(root, 'card0', total='0')
            with self.assertRaisesRegex(RuntimeError, 'VRAM'):
                guard.amd_device(root, root / 'nodes')
            (device / 'mem_info_vram_total').unlink()
            with self.assertRaises(FileNotFoundError):
                guard.amd_device(root, root / 'nodes')

    def test_unallocated_host_cards_do_not_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chosen = self.device(root, 'card0')
            for index in range(1, 8):
                self.device(root, 'card' + str(index))
                (root / 'nodes' / ('renderD' + str(128 + index))).unlink()
            self.assertEqual(guard.amd_device(root, root / 'nodes'), chosen.resolve())

    def test_visible_node_sysfs_number_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.device(root, 'card0')
            (root / 'renderD128' / 'dev').write_text('226:129')
            with self.assertRaisesRegex(RuntimeError, 'device number'):
                guard.amd_device(root, root / 'nodes')

    def test_visible_node_without_sysfs_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.device(root, 'card0')
            (root / 'nodes' / 'renderD200').touch()
            with self.assertRaises(FileNotFoundError):
                guard.amd_device(root, root / 'nodes')


class NodeTypeTests(unittest.TestCase):
    def test_regular_file_cannot_witness_a_render_device(self):
        with tempfile.NamedTemporaryFile() as fixture:
            with self.assertRaisesRegex(RuntimeError, 'character device'):
                guard.render_number(Path(fixture.name))

    def test_permission_denial_refused(self):
        node = Mock()
        node.stat.return_value = Mock(st_mode=guard.stat.S_IFCHR, st_rdev=0)
        with patch.object(guard.os, 'access', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'accessible'):
                guard.render_number(node)


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
            printed = stack.enter_context(patch('builtins.print'))
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
                terminal = json.loads(printed.call_args.args[0])
                data = terminal['telemetry']
                self.assertEqual(data['samples'], 1)
                self.assertEqual(data['initial_vram_bytes'], 0)
                self.assertEqual(data['initial_host_available_bytes'], 8 * guard.GIB)
                self.assertEqual(data['peak_vram_bytes'], 15 * guard.GIB if violation == 'vram' else 0)
                self.assertEqual(data['last_vram_bytes'], data['peak_vram_bytes'])
                self.assertEqual(data['initial_rss_bytes'], 13 * guard.GIB if violation == 'rss' else 0)
                self.assertEqual(data['last_rss_bytes'], data['initial_rss_bytes'])
                self.assertEqual(data['peak_rss_bytes'], data['initial_rss_bytes'])
                self.assertEqual(data['crossing_sample'], dict(
                    reason=terminal['reason'], elapsed_seconds=11 if violation == 'deadline' else 1,
                    vram_bytes=data['last_vram_bytes'], rss_bytes=data['last_rss_bytes'],
                    host_available_bytes=(1 if violation == 'pressure' else 8) * guard.GIB))
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
