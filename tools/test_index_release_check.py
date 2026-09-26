"""tools/index_release_check.py without an index: the precheck against a
stand-in JSON API, and the verdict over pip --report documents of the shapes
pip writes (the dependency confusion guard included). Nothing here reaches
TestPyPI or PyPI."""
import unittest

from index_release_check import judge_report, precheck, requirement_pins

V = "0.9.0"
PREFIX = {"mojolearn": "mojolearn", "mojolearn-nvidia": "mojolearn_nvidia", "mojolearn-amd": "mojolearn_amd"}


def fake_index(missing=(), requires=None, linux=True, yanked=()):
    """(fetch, fetch_meta) of an index serving the split release at V."""
    requires = requires or {
        "mojolearn": ["numpy>=1.24", f"mojolearn-nvidia=={V}", f"mojolearn-amd=={V}"],
        "mojolearn-nvidia": [f"mojolearn=={V}"],
        "mojolearn-amd": [f"mojolearn=={V}"],
    }
    docs = {}
    for project, prefix in PREFIX.items():
        tag = "manylinux_2_35_x86_64" if linux else "macosx_11_0_arm64"
        name = f"{prefix}-{V}-py3-none-{tag}.whl"
        docs[f"https://idx/pypi/{project}/{V}/json"] = {
            "info": {"requires_dist": None},
            "urls": [{"filename": name, "packagetype": "bdist_wheel", "url": f"https://files/{name}",
                      "yanked": project in yanked}]}
        docs[f"https://files/{name}.metadata"] = "Metadata-Version: 2.1\nName: %s\n%s\n" % (
            project, "\n".join("Requires-Dist: " + r for r in requires[project]))

    def fetch(url):
        if any(f"/{m}/" in url for m in missing) or url not in docs:
            return 404, None
        return 200, docs[url]

    def fetch_meta(url):
        return docs.get(url)
    return fetch, fetch_meta


def run(**kw):
    fetch, meta = fake_index(**kw)
    return precheck("testpypi", V, api="https://idx/pypi", fetch=fetch, fetch_meta=meta)


class PrecheckTests(unittest.TestCase):
    def test_the_split_release_passes(self):
        lines, problems = run()
        self.assertEqual(problems, [])
        self.assertEqual(len(lines), 3)

    def test_a_missing_plugin_is_refused_by_name(self):
        _, problems = run(missing=("mojolearn-amd",))
        self.assertEqual(len(problems), 1)
        self.assertIn(f"mojolearn-amd=={V} is not on testpypi (HTTP 404)", problems[0])

    def test_a_combined_core_is_refused(self):
        req = {"mojolearn": ["numpy>=1.24"], "mojolearn-nvidia": [f"mojolearn=={V}"],
               "mojolearn-amd": [f"mojolearn=={V}"]}
        _, problems = run(requires=req)
        self.assertTrue(any("not the split core" in p for p in problems), problems)

    def test_loose_or_marked_pins_are_refused(self):
        for core in ([f"mojolearn-nvidia>={V}", f"mojolearn-amd=={V}"],
                     [f"mojolearn-nvidia=={V}; sys_platform == 'linux'", f"mojolearn-amd=={V}"],
                     [f"mojolearn-nvidia[x]=={V}", f"mojolearn-amd=={V}"]):
            with self.subTest(core=core):
                req = {"mojolearn": core, "mojolearn-nvidia": [f"mojolearn=={V}"], "mojolearn-amd": [f"mojolearn=={V}"]}
                _, problems = run(requires=req)
                self.assertTrue(any("not the split core" in p for p in problems), problems)
        req = {"mojolearn": [f"mojolearn-nvidia=={V}", f"mojolearn-amd=={V}"],
               "mojolearn-nvidia": ["mojolearn==0.8.0"], "mojolearn-amd": [f"mojolearn=={V}"]}
        _, problems = run(requires=req)
        self.assertEqual(len(problems), 1)
        self.assertIn(f"mojolearn-nvidia=={V} on testpypi does not require mojolearn=={V}", problems[0])

    def test_no_linux_wheel_and_yanked_are_refused(self):
        _, problems = run(linux=False)
        self.assertEqual(sum("no manylinux x86_64 wheel" in p for p in problems), 3)
        _, problems = run(yanked=("mojolearn",))
        self.assertTrue(any("yanked" in p for p in problems), problems)

    def test_pins_parser(self):
        self.assertEqual(requirement_pins(["a==1", "B_c == 2", "d>=3", "e[x]==4", "f==5; python_version>'3'"]),
                         {"a": "1", "b-c": "2"})  # a marker or an extra is not an unconditional pin


def item(name, version, host, direct=False):
    return {"metadata": {"name": name, "version": version}, "is_direct": direct,
            "download_info": {"url": f"https://{host}/packages/xx/{name}-{version}.whl",
                              "archive_info": {"hashes": {"sha256": "0" * 64}}}}


def report(index="testpypi", **over):
    ours = "test-files.pythonhosted.org" if index == "testpypi" else "files.pythonhosted.org"
    rows = {"mojolearn": item("mojolearn", V, ours), "mojolearn-nvidia": item("mojolearn-nvidia", V, ours),
            "mojolearn_amd": item("mojolearn_amd", V, ours), "numpy": item("numpy", "2.1.0", "files.pythonhosted.org")}
    rows.update(over)
    return {"install": [r for r in rows.values() if r is not None]}


class ReportTests(unittest.TestCase):
    def test_testpypi_and_pypi_pass(self):
        for index in ("testpypi", "pypi"):
            with self.subTest(index=index):
                rows, problems = judge_report(report(index), index, V)
                self.assertEqual(problems, [])
                self.assertEqual({r["name"] for r in rows}, {"mojolearn", "mojolearn-nvidia", "mojolearn-amd", "numpy"})

    def test_ours_from_pypi_during_a_testpypi_check_fails(self):
        _, problems = judge_report(report(mojolearn=item("mojolearn", V, "files.pythonhosted.org")), "testpypi", V)
        self.assertEqual(len(problems), 1)
        self.assertIn("mojolearn 0.9.0 came from files.pythonhosted.org, not test-files.pythonhosted.org", problems[0])

    def test_dependency_confusion_guard(self):
        _, problems = judge_report(report(numpy=item("numpy", "2.1.0", "test-files.pythonhosted.org")), "testpypi", V)
        self.assertEqual(len(problems), 1)
        self.assertIn("DEPENDENCY CONFUSION GUARD", problems[0])
        _, problems = judge_report(report(evil=item("mojolearn-extra", "1.0", "test-files.pythonhosted.org")),
                                   "testpypi", V)
        self.assertTrue(any("unexpected mojolearn-named" in p for p in problems), problems)

    def test_missing_wrong_version_and_direct_fail(self):
        _, problems = judge_report(report(mojolearn_amd=None), "pypi", V)
        self.assertTrue(any("mojolearn-amd is not in pip's report" in p for p in problems), problems)
        _, problems = judge_report(report("pypi", **{"mojolearn-nvidia": item("mojolearn-nvidia", "0.8.0",
                                                                              "files.pythonhosted.org")}), "pypi", V)
        self.assertTrue(any("pip installed mojolearn-nvidia 0.8.0, not 0.9.0" in p for p in problems), problems)
        _, problems = judge_report(report("pypi", mojolearn=item("mojolearn", V, "files.pythonhosted.org", True)),
                                   "pypi", V)
        self.assertTrue(any("direct URL" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
