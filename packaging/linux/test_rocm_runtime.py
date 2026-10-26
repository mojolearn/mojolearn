# SPDX-License-Identifier: Apache-2.0
"""One ROCm runtime per process (python/mojolearn/_rocm_runtime.py).

Pure Python: /proc/self/maps, the RTLD_NOLOAD answer and ctypes.CDLL are
replaced, so this runs on any box and loads no ROCm library.

    .pixi/envs/test/bin/python -m pytest -q packaging/linux/test_rocm_runtime.py
"""
import ctypes
import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
PKG_SRC = ROOT / "python" / "mojolearn"
ALIAS = "mojolearn_rocm_runtime_under_test"

TORCH_LIB = "/venv/lib/python3.12/site-packages/torch/lib"
HIP = TORCH_LIB + "/libamdhip64.so"
HSA = TORCH_LIB + "/libhsa-runtime64.so"


def fresh(name):
    for n in [n for n in sys.modules if n == ALIAS or n.startswith(ALIAS + ".")]:
        del sys.modules[n]
    pkg = types.ModuleType(ALIAS)
    pkg.__path__ = [str(PKG_SRC)]
    sys.modules[ALIAS] = pkg
    return importlib.import_module(ALIAS + "." + name)


def maps(*paths):
    return "".join("7f0000000000-7f0000001000 r-xp 00000000 08:01 %d                %s\n" % (i + 1, p)
                   for i, p in enumerate(paths))


class RocmRuntime(unittest.TestCase):
    def setUp(self):
        self.R = fresh("_rocm_runtime")
        self.loads = []
        p = mock.patch.object(ctypes, "CDLL", side_effect=lambda name, *a, **k: self.loads.append(name))
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop(self.R.ENV_SWITCH, None)

    def with_maps(self, text):
        return mock.patch("builtins.open", mock.mock_open(read_data=text))

    def linux(self):
        return mock.patch.object(sys, "platform", "linux")

    def test_loaded_copies_reads_maps(self):
        with self.with_maps(maps(HIP, HSA, "/usr/lib/libc.so.6", HIP)), \
                mock.patch.object(os.path, "realpath", side_effect=lambda p: p):
            got = self.R.loaded_copies()
        self.assertEqual(got, {"libamdhip64": [HIP], "libhsa-runtime64": [HSA]})

    def test_hashed_wheel_names_count(self):
        hashed = "/x/torch.libs/libamdhip64-1a2b3c4d.so.7"
        with self.with_maps(maps(hashed)), mock.patch.object(os.path, "realpath", side_effect=lambda p: p):
            self.assertEqual(self.R.loaded_copies()["libamdhip64"], [hashed])

    def test_nothing_loaded_preloads_the_names_max_opens(self):
        empty = {"libamdhip64": [], "libhsa-runtime64": []}
        with self.linux(), mock.patch.object(self.R, "loaded_copies", return_value=empty):
            self.R.prepare()
        self.assertEqual(self.loads, list(self.R.RUNTIME_NAMES))

    def test_loaded_copy_max_reuses_is_fine(self):
        have = {"libamdhip64": [HIP], "libhsa-runtime64": [HSA]}
        with self.linux(), mock.patch.object(self.R, "loaded_copies", return_value=have), \
                mock.patch.object(self.R, "_reaches_loaded", return_value=True):
            self.R.prepare()
        self.assertEqual(self.loads, [])

    def test_loaded_copy_max_would_duplicate_refuses(self):
        have = {"libamdhip64": [HIP], "libhsa-runtime64": [HSA]}
        with self.linux(), mock.patch.object(self.R, "loaded_copies", return_value=have), \
                mock.patch.object(self.R, "_reaches_loaded", return_value=False), \
                mock.patch.object(self.R, "locate", return_value="/opt/rocm-7.2.3/lib/libamdhip64.so.7.2"):
            with self.assertRaises(self.R.RocmRuntimeConflict) as cm:
                self.R.prepare()
        msg = str(cm.exception)
        self.assertIn(HIP, msg)
        self.assertIn("LD_LIBRARY_PATH=" + TORCH_LIB, msg)
        self.assertIn("import mojolearn before torch", msg)
        self.assertEqual(self.loads, [])

    def test_switch_off_does_nothing(self):
        os.environ[self.R.ENV_SWITCH] = "0"
        self.addCleanup(os.environ.pop, self.R.ENV_SWITCH, None)
        with self.linux(), mock.patch.object(self.R, "loaded_copies", side_effect=AssertionError("read")):
            self.R.prepare()
        self.assertEqual(self.loads, [])

    def test_not_linux_does_nothing(self):
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch.object(self.R, "loaded_copies", side_effect=AssertionError("read")):
            self.R.prepare()


class ProbeNeverLoadsHip(unittest.TestCase):
    def test_probe_looks_without_loading(self):
        B = fresh("_backend")
        loads = []
        with mock.patch.object(ctypes, "CDLL", side_effect=lambda name, *a, **k: loads.append(name)), \
                mock.patch.object(B._rocm_runtime, "locate", side_effect=lambda n: "/opt/rocm/lib/" + n):
            got = B._probe_box()
        self.assertFalse([n for n in loads if "amdhip64" in n or "hsa-runtime64" in n], loads)
        self.assertTrue(all(got["hip"]["libs"].values()))

    def test_conflict_is_a_plugin_error_never_the_cpu_set(self):
        B = fresh("_backend")
        with mock.patch.object(B._rocm_runtime, "prepare",
                               side_effect=B._rocm_runtime.RocmRuntimeConflict("two copies")):
            with self.assertRaises(B.GpuPluginError):
                B._rocm_prepare("hip")
            B._rocm_prepare("cuda")


if __name__ == "__main__":
    unittest.main()
