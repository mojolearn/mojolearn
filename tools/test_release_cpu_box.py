"""The CPU-pod build route is opt-in (--build-backend cpu-box, 2026-09-25): one
tools/release_linux_build.sh per set, all three launched together, each
writing the release-build tree pack_wheel.py and proof_ok read; the GPU legs
stay available by name and a CPU leg never takes the NVIDIA GPU walk."""
import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location("release", Path(__file__).resolve().parent / "release.py")
rel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rel)
C = "a" * 40


def ptx_shipped():
    """Whether a release ships the NVIDIA PTX slot: gpu_plugins `ptx_shipped`
    (off since 0.8.39 while PyPI's 100 MiB mojolearn-nvidia limit stands)."""
    from verify_linux_surface_qualification import load_gpu_plugins
    return load_gpu_plugins().PLUGINS["cuda"].get("ptx_shipped", True)


#: The release's Linux legs in release_reuse.LINUX_SETS order, read from the registry.
RELEASE_LEGS = ["cuda-sm_89"] + (["cuda-sm_80"] if ptx_shipped() else []) + ["hip-gfx942"]


class CpuBox(unittest.TestCase):
    def setUp(self):
        self.ctx = argparse.Namespace(rel=Path(tempfile.mkdtemp()), commit=C)

    def test_default_backend_is_github_for_either_layout(self):
        # Andrew 2026-10-26: compiling never uses paid rentals; cpu-box stays selectable by name
        self.assertIs(rel.BUILD_BACKENDS["cpu-box"], rel.cpu_legs)
        self.assertIs(rel.BUILD_BACKENDS["gpu-legs"], rel.gpu_legs)
        src = (Path(__file__).resolve().parent / "release.py").read_text()
        self.assertIn('ap.add_argument("--build-backend", default=None', src)
        self.assertEqual(rel.default_build_backend(rel.git("rev-parse", "HEAD")), "github")
        # a commit without packaging/linux/merged_layout.py is per-binding: GitHub too
        self.assertEqual(rel.default_build_backend(C), "github")

    def test_merged_pod_and_per_binding_pod(self):
        per = argparse.Namespace(rel=self.ctx.rel, commit=C, linux_merged=False)
        merged = argparse.Namespace(rel=self.ctx.rel, commit=C, linux_merged=True)
        self.assertEqual(rel.cpu_box_pod(per), ["--flavors", rel.CPU_BOX_FLAVORS])
        self.assertEqual(rel.cpu_box_pod(per, amd_small=True)[:2], ["--vcpu", "16"])
        want = ["--vcpu", str(rel.CPU_BOX_MERGED_VCPU), "--flavors", rel.CPU_BOX_MERGED_FLAVORS]
        self.assertEqual(rel.cpu_box_pod(merged), want)
        self.assertEqual(rel.cpu_box_pod(merged, amd_small=True), want)
        for l in rel.cpu_legs(merged):
            i = l.command.index("--flavors")
            self.assertEqual(l.command[i + 1], rel.CPU_BOX_MERGED_FLAVORS)
            self.assertEqual(l.command[l.command.index("--vcpu") + 1], str(rel.CPU_BOX_MERGED_VCPU))
        # >= 64 GB at 4 GB per vCPU, no compute-optimized (2 GB) flavor
        self.assertGreaterEqual(rel.CPU_BOX_MERGED_VCPU * 4, 64)
        self.assertNotRegex(rel.CPU_BOX_MERGED_FLAVORS, r"c(,|$)")
        # tools/release_linux_build.sh has the same merged defaults
        sh = (Path(__file__).resolve().parent / "release_linux_build.sh").read_text()
        block = sh.split('if [ "$MERGED" = 1 ]; then', 1)[1].split("else", 1)[0]
        self.assertIn(f'[ -n "$VCPU" ] || VCPU={rel.CPU_BOX_MERGED_VCPU}', block)
        self.assertIn(f'[ -n "$FLAVORS" ] || FLAVORS="{rel.CPU_BOX_MERGED_FLAVORS}"', block)
        self.assertIn('[ -n "$JOBS" ] || JOBS=1', block)

    def test_merged_layout_reaches_the_box_build_cap(self):
        here = Path(__file__).resolve().parent
        box = (here / "release_linux_cpu_box.sh").read_text()
        self.assertIn("MERGED_RSS_GIB=64", box)
        self.assertIn('MOJOLEARN_RELEASE_BUILD_RSS_GIB="$build_rss_gib"', box)
        remote = (here / "release061_remote_build.sh").read_text()
        self.assertIn("BUILD_RSS_GIB=$MOJOLEARN_RELEASE_BUILD_RSS_GIB", remote)
        # the per-binding cap is unchanged (16 GiB per job) when the box sets none
        self.assertIn('[[ "$NO_DEVICE" = 1 ]] && BUILD_RSS_GIB=$((16 * BUILD_JOBS))', remote)
        # MOJOLEARN_RELEASE_* never reaches a binding cache key
        import bincache
        self.assertEqual(bincache.build_env({"MOJOLEARN_RELEASE_BUILD_RSS_GIB": "64",
                                             "MOJOLEARN_RELEASE_BUILD_CORES": "16"}), {})

    def test_github_and_cpu_box_build_without_renting_a_gpu(self):
        from unittest import mock
        import contextlib
        import io
        import os
        # The split merged groups now fit GitHub runners. Both layouts support
        # the identical-only release on GitHub and on CPU boxes.
        for merged in (False, True):
            with self.subTest(merged=merged), \
                    mock.patch.object(rel.Release, "go", lambda self: self.args), \
                    mock.patch.object(rel.release_github_build, "commit_merged", lambda c, env=None: merged), \
                    mock.patch.dict(os.environ, {"MOJOLEARN_RELEASE_MODES": "identical"}):
                self.assertEqual(rel.main(["0.8.14"]).build_backend, "github")
                for backend in ("github", "cpu-box"):
                    args = rel.main(["0.8.14", "--build-backend", backend])
                    self.assertEqual(args.build_backend, backend)
                    self.assertFalse(args.gpu_smoke)
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                    rel.main(["0.8.14", "--build-backend", "gpu-legs"])
                self.assertIn("selected release tiers require --build-backend github or cpu-box", err.getvalue())

    def test_three_legs_one_set_each(self):
        legs = rel.cpu_legs(self.ctx)
        # sm_80 is the PTX slot of mojolearn-nvidia, a leg like any arch (Andrew 2026-10-10),
        # while the release ships it
        self.assertEqual([l.name for l in legs], RELEASE_LEGS)
        for l in legs:
            self.assertEqual(l.command[:4], ["bash", "tools/release_linux_build.sh", C, "--rent"])
            self.assertEqual(l.command[l.command.index("--archs") + 1], l.arch)
            out = Path(l.command[l.command.index("--out") + 1])
            self.assertEqual(out, l.out_dir)
            self.assertEqual(l.release_build, out / l.name / "release-build")
            self.assertNotIn("--gpu", l.command)


if __name__ == "__main__":
    unittest.main()
