"""Wheel installability includes the shipped Metal deployment contract."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("compatibility")
library = load("test_gpu_embedded").library


def macho(minimum="11.0", sdk="26.5", platform="1"):
    return ("payload:\nLoad command 0\n      cmd LC_BUILD_VERSION\n"
            f" platform {platform}\n    minos {minimum}\n      sdk {sdk}\n")


def metal(major=26, minor=0, update=0):
    return (f"Platform: MACOS\nPlatformMajor: {major}\n"
            f"PlatformMinor: {minor}\nPlatformUpdate: {update}\n").encode()


class MetadataTests(unittest.TestCase):
    def test_sdk_is_not_deployment_floor(self):
        self.assertEqual(gate.macho_floor(macho("11.0", "26.5")), (11, 0, 0))

    def test_all_native_load_commands_count(self):
        legacy = "\nLoad command 1\n cmd LC_VERSION_MIN_MACOSX\n version 15.2\n sdk 26.5\n"
        self.assertEqual(gate.macho_floor(macho() + legacy), (15, 2, 0))

    def test_missing_or_non_macos_native_metadata_is_rejected(self):
        for output in ("no commands", macho(platform="IOS"), macho().replace("minos", "other")):
            with self.subTest(output=output), self.assertRaises(ValueError):
                gate.macho_floor(output)

    def test_observed_apple_octet_fields_are_preserved(self):
        output = b"Platform: MACOS\nPlatformMajor: 26\nPlatformMinor: \x01\nPlatformUpdate: \x00\n"
        self.assertEqual(gate.metal_floor(output), (26, 1, 0))
        self.assertEqual(gate.metal_floor(metal(26, 2, 3)), (26, 2, 3))

    def test_unknown_metal_metadata_is_rejected(self):
        for output in (metal().replace(b"MACOS", b"IOS"), metal(0), metal().replace(b"PlatformUpdate", b"Unknown")):
            with self.subTest(output=output), self.assertRaises(ValueError):
                gate.metal_floor(output)

    def test_bounded_embedded_code_extraction(self):
        blob = library()
        self.assertEqual(list(gate.metallibs(b"prefix" + blob + b"suffix")), [(6, blob)])
        self.assertEqual(list(gate.metallibs(b"MTLB" + b"\0" * 140)), [])
        with self.assertRaisesRegex(ValueError, "unclassified"):
            list(gate.metallibs(blob.replace(b"BC\xc0\xde", b"NONE")))


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def tool(self, minimum="11.0", metal_output=None):
        def output(args, **kwargs):
            if args[0] == "otool":
                return macho(minimum)
            self.assertEqual(args[:4], ["xcrun", "metal-objdump", "--metallib", "--all-headers"])
            result = b""
            for name in args[4:]:
                self.assertTrue((Path(kwargs["cwd"]) / name).is_file())
                result += name.encode() + b":\n" + (metal() if metal_output is None else metal_output)
            return result
        return mock.patch.object(gate.subprocess, "check_output", side_effect=output)

    def test_embedded_gpu_floor_overrules_outer_native_header(self):
        self.put("identical/a.so", library())
        with self.tool():
            report = gate.inspect_package(self.root)
        self.assertEqual(report["tag"], "macosx_26_0_arm64")
        self.assertEqual(report["native_files"][0]["macho_minimum"], "11.0")
        self.assertEqual(report["native_files"][0]["metal_payloads"][0]["minimum"], "26.0")

    def test_supported_runtime_floor_applies_without_gpu_code(self):
        self.put("host/a.so", b"host code")
        with self.tool():
            self.assertEqual(gate.inspect_package(self.root)["tag"], "macosx_15_0_arm64")

    def test_higher_dylib_minimum_and_explicit_override_apply(self):
        self.put(".dylibs/runtime.dylib", b"host code")
        with self.tool("27.0"):
            self.assertEqual(gate.inspect_package(self.root)["tag"], "macosx_27_0_arm64")
            self.assertEqual(gate.inspect_package(self.root, "28.0")["tag"], "macosx_28_0_arm64")
            with self.assertRaisesRegex(ValueError, "below required"):
                gate.inspect_package(self.root, "26.0")

    def test_lower_override_cannot_hide_embedded_gpu_requirement(self):
        self.put("a.so", library())
        with self.tool(), self.assertRaisesRegex(ValueError, "below required.*26.0"):
            gate.inspect_package(self.root, "11.0")

    def test_patch_floor_is_not_silently_rounded_down(self):
        self.put("a.so", b"host code")
        with self.tool("26.0.1"), self.assertRaisesRegex(ValueError, "patch deployment"):
            gate.inspect_package(self.root)

    def test_only_byte_identical_metal_payloads_reuse_inspection(self):
        first = library()
        second = first.replace(b"tiny_kernel", b"next_kernel")
        self.put("a.so", first + second)
        self.put("identical/a.so", first)
        with self.tool() as call:
            report = gate.inspect_package(self.root)
        tool_calls = [args.args[0] for args in call.call_args_list if args.args[0][0] == "xcrun"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(len(tool_calls[0][4:]), 2)
        self.assertEqual(len(report["metal_headers"]), 2)

    def test_missing_or_duplicate_tool_payload_identity_is_rejected(self):
        self.put("a.so", library())
        for output in (metal(), b"0" * 64 + b".metallib:\n" + metal()):
            def tool(args, **kwargs):
                return macho() if args[0] == "otool" else output
            with mock.patch.object(gate.subprocess, "check_output", side_effect=tool):
                with self.assertRaisesRegex(ValueError, "payload"):
                    gate.inspect_package(self.root)

    def test_tagging_without_staged_native_payloads_fails(self):
        with self.assertRaisesRegex(ValueError, "staged native"):
            gate.inspect_package(self.root)

    def wheel(self, floor="26_0", metadata_floor=None, omit=None):
        path = self.root / f"mojolearn-0.0.0-py3-none-macosx_{floor}_arm64.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mojolearn/a.so", library())
            archive.writestr("mojolearn-0.0.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-macosx_"
                             + (metadata_floor or floor) + "_arm64\n")
            for member in ("mojolearn/CITATION.cff", "mojolearn-0.0.0.dist-info/licenses/LICENSE",
                           "mojolearn-0.0.0.dist-info/licenses/NOTICE"):
                if member != omit:
                    archive.writestr(member, "fixture attribution")
        return path

    def test_final_wheel_checks_shipped_bytes_and_metadata(self):
        with self.tool():
            self.assertEqual(gate.check_wheel(self.wheel())["selected_floor"], "26.0")
            with self.assertRaisesRegex(ValueError, "below required"):
                gate.check_wheel(self.wheel("11_0"))
            with self.assertRaisesRegex(ValueError, "tags disagree"):
                gate.check_wheel(self.wheel("26_0", "11_0"))

    def test_attribution_must_ship_in_the_final_wheel(self):
        for member in ("mojolearn/CITATION.cff", "mojolearn-0.0.0.dist-info/licenses/LICENSE",
                       "mojolearn-0.0.0.dist-info/licenses/NOTICE"):
            with self.subTest(member=member), self.assertRaisesRegex(ValueError, "required attribution"):
                gate.check_wheel(self.wheel(omit=member))


if __name__ == "__main__":
    unittest.main()
