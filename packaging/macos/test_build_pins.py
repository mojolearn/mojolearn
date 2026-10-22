"""Linker pins remain checked independently of artifact-derived wheel tags."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("wheel_ci", Path(__file__).parents[1] / "wheel_ci.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class PinsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "bindings").mkdir()
        self.base = ('MACOS_FLOOR="11.0"\n--target-cpu apple-m1\n'
                     '--target-cpu "${MOJOLEARN_LINUX_CPU:-x86-64-v3}"\n')
        self.put("build.sh", self.base)

    def put(self, name, text):
        (self.root / "bindings" / name).write_text(text)

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return gate.pins([str(self.root)])

    def test_linker_floor_does_not_depend_on_a_setup_tag_constant(self):
        self.put("build_host.sh", self.base)
        self.assertEqual(self.check(), 0)

    def test_different_linker_floor_and_missing_cpu_pins_fail(self):
        for text in (self.base.replace('"11.0"', '"15.0"'), self.base.replace("apple-m1", "apple-m4"),
                     self.base.replace("x86-64-v3", "native")):
            with self.subTest(text=text):
                self.put("build_host.sh", text)
                self.assertEqual(self.check(), 1)

    def test_apple_only_tokenizer_requires_its_actual_platform_refusal(self):
        source = ('MACOS_FLOOR="11.0"\n--target-cpu apple-m1\n'
                  '[ "$(uname)" = Darwin ] || {\n echo "Apple only" >&2\n exit 2; }\n')
        self.put("build_tokenizer_fast.sh", source)
        self.assertEqual(self.check(), 0)
        self.put("build_tokenizer_fast.sh", source.replace("exit 2;", "echo allowed;"))
        self.assertEqual(self.check(), 1)


if __name__ == "__main__":
    unittest.main()
