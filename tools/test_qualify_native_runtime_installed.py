# SPDX-License-Identifier: Apache-2.0
"""Artifact admission negative controls; no compiler/product runtime involved."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import subprocess
import shutil
import tarfile
import types
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("installed_native_audit", Path(__file__).with_name("qualify_native_runtime_installed.py"))
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def wheel(path, *, tamper=False, extra=False):
    files = {"mojolearn/__init__.py": b"source\n", "mojolearn/host/example.so": b"native-bytes"}
    rows = []
    for name, raw in files.items():
        sha = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        rows.append((name, "sha256=" + sha, str(len(raw))))
    record = "mojolearn_core-0.dist-info/RECORD"
    out = io.StringIO();csv.writer(out).writerows(rows + [(record, "", "")])
    if tamper:files["mojolearn/host/example.so"] = b"tampered"
    if extra:files["mojolearn/unchecked.py"] = b"unrecorded"
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():archive.writestr(name, raw)
        archive.writestr(record, out.getvalue())


class InstalledNativeAudit(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_every_wheel_file_and_native_byte_is_checked(self):
        path = self.root / "test.whl";wheel(path)
        payload, records = q.wheel_inventory([path])
        self.assertIn("mojolearn/host/example.so", payload)
        self.assertEqual(records[0]["sha256"], q.digest(path))
        wheel(path, tamper=True)
        with self.assertRaisesRegex(ValueError, "RECORD mismatch"):
            q.wheel_inventory([path])
        wheel(path, extra=True)
        with self.assertRaisesRegex(ValueError, "every wheel file"):
            q.wheel_inventory([path])

    def test_duplicate_split_ownership_refuses(self):
        one, two = self.root / "one.whl", self.root / "two.whl"
        wheel(one);wheel(two)
        with self.assertRaisesRegex(ValueError, "two wheels own"):
            q.wheel_inventory([one, two])

    def modules(self):
        prefix = self.root / "venv";site = prefix / "lib/python3.14/site-packages"
        package = site / "mojolearn";package.mkdir(parents=True)
        init = package / "__init__.py";init.write_text("source\n")
        native = package / "host/example.so";native.parent.mkdir();native.write_bytes(b"native-bytes")
        base = types.ModuleType("mojolearn");base.__file__ = str(init)
        ext = types.ModuleType("_mojolearn_example");ext.__file__ = str(native)
        base.runtime = ext  # Also inspect cached extensions absent from sys.modules.
        payload = {str(p.relative_to(site)):dict(sha256=q.digest(p)) for p in [init, native]}
        tests = self.root / "frozen-tests";tests.mkdir()
        return prefix, payload, tests, {"mojolearn":base}

    def test_only_installed_product_paths_and_hashes_are_admitted(self):
        prefix, payload, tests, modules = self.modules()
        found = q.installed_module_inventory(modules, prefix, payload, tests)
        self.assertTrue(found["mojolearn/host/example.so"]["native"])
        Path(modules["mojolearn"].runtime.__file__).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            q.installed_module_inventory(modules, prefix, payload, tests)

    def test_native_aliases_hash_once_per_observation_but_never_reuse_across_tests(self):
        prefix, payload, tests, modules = self.modules()
        native = Path(modules["mojolearn"].runtime.__file__)
        for index in range(32):
            setattr(modules["mojolearn"], "alias" + str(index), modules["mojolearn"].runtime)
        witness = dict(module="mojolearn._host._mojolearn_example", file=str(native),
                       sha256=q.digest(native), size=native.stat().st_size)
        original = q.digest
        calls = []
        def counted(path):
            calls.append(Path(path).resolve())
            return original(path)
        with patch.object(q, "digest", side_effect=counted):
            first = q.installed_module_inventory(modules, prefix, payload, tests, [witness])
            self.assertEqual(len(calls), 2)  # one Python file and one native file
            second = q.installed_module_inventory(modules, prefix, payload, tests, [witness])
            self.assertEqual(len(calls), 4)
        self.assertEqual(first, second)
        native.write_bytes(b"X" * native.stat().st_size)
        with self.assertRaisesRegex(ValueError, "witness bytes changed"):
            q.installed_module_inventory(modules, prefix, payload, tests, [witness])

    def test_native_mutation_with_restored_mtime_during_snapshot_refuses(self):
        import os
        prefix, payload, tests, modules = self.modules()
        native = Path(modules["mojolearn"].runtime.__file__)
        stat = native.stat()
        witness = dict(module="mojolearn._host._mojolearn_example", file=str(native),
                       sha256=q.digest(native), size=stat.st_size)
        del modules["mojolearn"].runtime
        original = q.digest
        def mutate_after_native_witness(path):
            value = original(path)
            if Path(path).name == "__init__.py":
                native.write_bytes(b"X" * stat.st_size)
                os.utime(native, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            return value
        with patch.object(q, "digest", side_effect=mutate_after_native_witness):
            with self.assertRaisesRegex(ValueError, "changed during inventory"):
                q.installed_module_inventory(modules, prefix, payload, tests, [witness])

    def test_artifact_mutated_while_hashing_refuses(self):
        prefix, payload, tests, modules = self.modules()
        original = q.digest
        def mutate(path):
            value = original(path)
            Path(path).write_bytes(b"changed")
            return value
        with patch.object(q, "digest", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                q.installed_module_inventory(modules, prefix, payload, tests)

    def test_product_source_shadow_and_unshipped_native_module_refuse(self):
        prefix, payload, tests, modules = self.modules()
        source = self.root / "source.py";source.write_text("source\n")
        modules["mojolearn"].__file__ = str(source)
        with self.assertRaisesRegex(ValueError, "outside installed"):
            q.installed_module_inventory(modules, prefix, payload, tests)
        modules["mojolearn"].__file__ = str(prefix / "lib/python3.14/site-packages/mojolearn/__init__.py")
        del payload["mojolearn/host/example.so"]
        with self.assertRaisesRegex(ValueError, "absent from wheel"):
            q.installed_module_inventory(modules, prefix, payload, tests)

    def test_only_frozen_test_namespace_can_live_outside_install(self):
        prefix, payload, tests, modules = self.modules()
        helper = tests / "test_helper.py";helper.write_text("oracle\n")
        test = types.ModuleType("mojolearn.tests.test_helper");test.__file__ = str(helper)
        modules[test.__name__] = test
        q.installed_module_inventory(modules, prefix, payload, tests)
        test.__file__ = str(self.root / "unchecked-test.py")
        with self.assertRaisesRegex(ValueError, "test namespace escaped"):
            q.installed_module_inventory(modules, prefix, payload, tests)

    def test_archived_freeze_uses_real_git_objects_and_refuses_changed_source(self):
        repo = self.root / "repo";repo.mkdir()
        source_file = repo / "python/mojolearn/__init__.py"
        source_file.parent.mkdir(parents=True);source_file.write_text("source\n")
        for args in (["init", "-q"], ["add", "."],
                     ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "fixture"]):
            subprocess.run(["git", "-C", str(repo)] + list(args), check=True, capture_output=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        frozen = self.root / "frozen";shutil.copytree(repo / "python", frozen / "python")
        artifact = self.root / "witness.whl";witness = "mojolearn/identity_columns/COMMIT"
        with zipfile.ZipFile(artifact, "w") as archive:archive.writestr(witness, commit + "\n")
        payload = {witness:dict(wheel=str(artifact)),
                   "mojolearn/__init__.py":dict(sha256=q.digest(source_file))}
        self.assertEqual(len(q.source_match(frozen, commit, [artifact], payload, repo)), 1)
        (frozen / "python/mojolearn/__init__.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "wheel/source Python mismatch"):
            q.source_match(frozen, commit, [artifact], payload, repo)

    def test_original_failure_and_skip_are_kept_distinct(self):
        path = self.root / "tests.xml"
        path.write_text('<testsuites><testsuite><testcase name="ok"/><testcase name="bad"><failure>original numerical mismatch</failure></testcase><testcase name="missing"><skipped message="no physical GPU"/></testcase></testsuite></testsuites>')
        report = q.junit_summary(path)
        self.assertEqual((report["total"], report["passed"]), (3, 1))
        self.assertIn("original numerical mismatch", report["failures"][0]["reason"])
        self.assertEqual(report["skipped"][0]["reason"], "no physical GPU")

    def test_origin_only_cached_extension_is_checked_against_installed_wheel(self):
        prefix, payload, tests, modules = self.modules()
        extension = modules["mojolearn"].runtime
        path = extension.__file__
        del extension.__file__
        extension.__spec__ = types.SimpleNamespace(origin=path)
        found = q.installed_module_inventory(modules, prefix, payload, tests)
        row = found["mojolearn/host/example.so"]
        self.assertTrue(row["native"])
        self.assertEqual(row["path_source"], "__spec__.origin")
        Path(path).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            q.installed_module_inventory(modules, prefix, payload, tests)

    def test_missing_conflicting_or_external_native_origin_refuses(self):
        prefix, payload, tests, modules = self.modules()
        extension = modules["mojolearn"].runtime
        original = extension.__file__
        extension.__spec__ = types.SimpleNamespace(origin=str(self.root / "outside.so"))
        with self.assertRaisesRegex(ValueError, "conflicting loaded product paths"):
            q.installed_module_inventory(modules, prefix, payload, tests)
        del extension.__file__
        with self.assertRaisesRegex(ValueError, "outside installed"):
            q.installed_module_inventory(modules, prefix, payload, tests)
        extension.__spec__ = None
        with self.assertRaisesRegex(ValueError, "no artifact path"):
            q.installed_module_inventory(modules, prefix, payload, tests)
        extension.__spec__ = types.SimpleNamespace(origin=original)
        q.installed_module_inventory(modules, prefix, payload, tests)

    def test_shared_native_witness_keeps_installed_path_and_hash_constraints(self):
        prefix, payload, tests, modules = self.modules()
        path = Path(modules["mojolearn"].runtime.__file__)
        witness = dict(module="mojolearn._host._mojolearn_example", file=str(path),
                       sha256=q.digest(path), size=path.stat().st_size, path_source="__spec__.origin")
        del modules["mojolearn"].runtime
        found = q.installed_module_inventory(modules, prefix, payload, tests, [witness])
        self.assertTrue(found["mojolearn/host/example.so"]["native"])
        with self.assertRaisesRegex(ValueError, "witness bytes changed"):
            q.installed_module_inventory(modules, prefix, payload, tests,
                                         [dict(witness, sha256="0" * 64)])
        external = self.root / "outside.so";external.write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "outside installed"):
            q.installed_module_inventory(modules, prefix, payload, tests,
                                         [dict(witness, file=str(external))])

    def test_exact_backend_placeholder_is_not_a_missing_native_artifact(self):
        prefix, payload, tests, modules = self.modules()
        class Placeholder(types.ModuleType):
            pass
        backend = types.ModuleType("mojolearn._backend")
        backend._HostBinding = Placeholder
        modules[backend.__name__] = backend
        modules["mojolearn._mojolearn_placeholder"] = Placeholder("_mojolearn_placeholder")
        q.installed_module_inventory(modules, prefix, payload, tests)
        class Unknown(Placeholder):
            pass
        modules["mojolearn._mojolearn_placeholder"] = Unknown("_mojolearn_placeholder")
        with self.assertRaisesRegex(ValueError, "no artifact path"):
            q.installed_module_inventory(modules, prefix, payload, tests)

    def test_fitted_error_lifetime_scope_preserves_interpreter_qualification_limit(self):
        name = "test_not_fitted_native_lifetime.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        self.assertNotIn("scikit-learn", q.test_dependencies([name]))
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
            self.assertIn("CPython 3.11", scope["coverage_limit"])
            self.assertIn("pickle is caller-owned", scope["coverage_limit"])
            self.assertIn("no GPU kernel", scope["execution_route"])

    def test_exception_bridge_selection_retains_sklearn_dependency_and_scope(self):
        name = "test_classical_exceptions_native_owner.py"
        nodes = [name + "::test_native_error_bridge_pickle_and_repeated_refusals_keep_type_identity[" + cls + "]"
                 for cls in ("LinearRegression", "Ridge")]
        self.assertEqual(q.selected_files(test_nodes=nodes), [name])
        self.assertIn("scikit-learn", q.test_dependencies([name]))
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
            self.assertIn("pickle is caller-owned", scope["coverage_limit"])
            self.assertIn("Direct native entry checks", scope["coverage_limit"])
            self.assertIn("CPython 3.11", scope["coverage_limit"])

    def test_cache_lookup_scope_excludes_mutation_and_context_lifecycle(self):
        name = "test_cache_get_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("Only DeviceCache.get", scope["coverage_limit"])
            self.assertIn("no device slots", scope["execution_route"])

    def test_iforest_depth_scope_is_public_explicit_host_on_each_package(self):
        name = "test_iforest_depth_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("explicit CPU host", scope["execution_route"])
            self.assertIn("depth-state arithmetic", scope["coverage_limit"])
            self.assertIn("separate execution evidence", scope["coverage_limit"])
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
        self.assertIn("numpy", q.test_dependencies([name]))
        self.assertTrue({"pandas", "scipy", "torch"}.isdisjoint(q.test_dependencies([name])))

    def test_cache_statistics_scope_does_not_claim_get_or_lifecycle(self):
        name = "test_cache_stats_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("Only DeviceCache.__len__ and resident.stats", scope["coverage_limit"])
            self.assertIn("get, construction/upload/release", scope["coverage_limit"])
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
            self.assertIn("numpy", q.test_dependencies([name]))

    def test_device_rows_materialize_scope_is_public_and_backend_applicable(self):
        name = "test_device_rows_materialize_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("DeviceRows.materialize", scope["coverage_limit"])
            self.assertIn("remain separate", scope["coverage_limit"])
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
            self.assertIn("numpy", q.test_dependencies([name]))

    def test_owner_builtin_identity_scope_is_public_on_every_backend(self):
        name = "test_native_owner_builtin_identity.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("reinitialization", scope["coverage_limit"])
            self.assertIn("remain separate", scope["coverage_limit"])
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
            self.assertIn("numpy", q.test_dependencies([name]))

    def test_public_archive_and_buffer_scopes_keep_private_pca_excluded(self):
        names = ("test_classical_archive_writer_native.py",
                 "test_standard_scaler_native_lifecycle.py",
                 "test_minmax_scaler_native_lifecycle.py",
                 "test_sequential_native_buffers.py")
        self.assertNotIn("test_pca_exact_native_owner.py", q.PUBLIC_TESTS)
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scopes = q.test_applicability(names, vendor)
            for name in names:
                self.assertIn(name, q.PUBLIC_TESTS)
                self.assertTrue((q.ROOT / "python/mojolearn/tests" / name).is_file())
                self.assertEqual(scopes[name]["cpu_not_applicable_cases"], {})
                self.assertIn("numpy", q.test_dependencies([name]))
            self.assertIn("private PCA", scopes[names[0]]["coverage_limit"])
            self.assertIn("remain separate", scopes[names[1]]["coverage_limit"])
            self.assertIn("test oracles", scopes[names[3]]["coverage_limit"])

    def test_public_reader_and_density_scopes_preserve_unqualified_boundaries(self):
        reader = "test_classical_archive_reader_native.py"
        density = "test_host_density_save_native.py"
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scopes = q.test_applicability([reader, density], vendor)
            for name in (reader, density):
                self.assertIn(name, q.PUBLIC_TESTS)
                self.assertEqual(scopes[name]["cpu_not_applicable_cases"], {})
                self.assertIn("native host", scopes[name]["execution_route"])
                self.assertIn("remain separate", scopes[name]["coverage_limit"])
                self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest"])
            self.assertIn("codec-hook", scopes[reader]["coverage_limit"])
            self.assertIn("fit/inference", scopes[density]["coverage_limit"])
        self.assertNotIn("test_classical_archive_native_owner.py", q.PUBLIC_TESTS)

    def test_public_reader_and_density_exact_selection_remains_bounded(self):
        nodes = ["test_classical_archive_reader_native.py::test_public_load_without_numpy_preserves_native_owned_arrays[None]",
                 "test_host_density_save_native.py::test_density_save_without_numpy_preserves_frozen_archive[kde]"]
        self.assertEqual(q.selected_files(test_nodes=nodes),
                         [node.partition("::")[0] for node in nodes])
        with self.assertRaises(ValueError):
            q.selected_files(test_nodes=nodes + nodes[:1])

    def test_public_density_hash_scope_preserves_unqualified_boundaries(self):
        name = "test_host_density_hash_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
            self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest"])
            self.assertIn("explicit CPU host", scope["execution_route"])
            self.assertIn("Other estimators' hash methods", scope["coverage_limit"])
            self.assertIn("archive-file digests", scope["coverage_limit"])
            self.assertIn("remain separate", scope["coverage_limit"])

    def test_public_density_hash_exact_selection_remains_bounded(self):
        node = "test_host_density_hash_native.py::test_density_hash_without_numpy_has_exact_digest[kde]"
        self.assertEqual(q.selected_files(test_nodes=[node]),
                         ["test_host_density_hash_native.py"])
        with self.assertRaises(ValueError):
            q.selected_files(test_nodes=[node, node])

    def test_public_cache_supports_scope_is_metadata_only_on_every_target(self):
        name = "test_cache_supports_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest"])
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("DeviceCache.supports", scope["execution_route"])
            self.assertIn("no device allocation", scope["execution_route"])
            self.assertIn("constructor, release, close", scope["coverage_limit"])
            self.assertEqual(scope["cpu_not_applicable_cases"], {})

    def test_public_cache_supports_exact_node_remains_bounded(self):
        name = "test_cache_supports_native.py"
        node = name + "::test_cache_supports_preserves_unicode_prefix_names[surrogate]"
        self.assertEqual(q.selected_files(test_nodes=[node]), [name])
        with self.assertRaises(ValueError):
            q.selected_files(test_nodes=[node, node])

    def test_public_cache_live_scope_retains_unfinished_lifecycle(self):
        name = "test_cache_lifecycle_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest"])
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("DeviceCache.live", scope["execution_route"])
            self.assertIn("constructor, supports, release, close", scope["coverage_limit"])
            self.assertEqual(len(scope["cpu_not_applicable_cases"]), 8 if vendor == "cpu" else 0)

    def test_public_cache_live_cpu_exclusions_are_exact_real_store_cases(self):
        name = "test_cache_lifecycle_native.py"
        cases = set()
        for prefix in ("x_prep", "x_metrics"):
            for slots in (0, 1, 2):
                cases.add(f"test_cache_live_matches_real_store_count_after_consumed_allocations[{slots}-{prefix}]")
            cases.add(f"test_cache_live_uses_captured_native_entry_and_ignores_mutable_int[{prefix}]")
        self.assertEqual(set(q.CPU_NOT_APPLICABLE[name]), cases)
        node = name + "::test_cache_live_without_optional_store_counter_is_none[normal]"
        self.assertEqual(q.selected_files(test_nodes=[node]), [name])

    def test_public_cache_release_scope_and_cpu_exclusions(self):
        name = "test_cache_release_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest", "pyarrow"])
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("resident scope", scope["execution_route"])
            self.assertEqual(len(scope["cpu_not_applicable_cases"]), 24 if vendor == "cpu" else 0)
        host = "test_empty_lifecycle_calls_are_native_on_every_package"
        self.assertNotIn(host, q.CPU_NOT_APPLICABLE[name])
        self.assertEqual(q.selected_files(test_nodes=[name + "::" + host]), [name])

    def test_public_resident_lifecycle_scope_requires_arrow_dependencies(self):
        name = "test_resident_lifecycle_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest", "pandas", "pyarrow"])
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
            self.assertIn("Arrow-backed pandas", scope["coverage_limit"])

    def test_public_cache_upload_scope_preserves_pending_lifecycle(self):
        name = "test_cache_upload_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest"])
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("DeviceCache.id_of", scope["execution_route"])
            self.assertIn("x_prep/x_metrics", scope["execution_route"])
            self.assertIn("constructor, release, close", scope["coverage_limit"])
            self.assertIn("compound cleanup failure injection", scope["coverage_limit"])
            self.assertEqual(len(scope["cpu_not_applicable_cases"]), 20 if vendor == "cpu" else 0)

    def test_public_cache_upload_cpu_exclusions_are_exact_real_store_cases(self):
        name = "test_cache_upload_native.py"
        cases = set()
        for prefix in ("x_prep", "x_metrics"):
            cases.add("test_cache_id_of_real_upload_hit_alias_and_owned_publication[" + prefix + "]")
            cases.add("test_cache_id_of_real_upload_uses_connected_store_entries[" + prefix + "]")
            for failure in ("null", "address_overflow", "size_overflow"):
                cases.add("test_cache_id_of_real_native_upload_failure_preserves_empty_state[" + failure + "-" + prefix + "]")
            for change in ("replace_dictionary", "publish_competitor", "change_route", "delete_counter_slot", "foreign_dictionary"):
                cases.add("test_cache_id_of_real_concurrent_publication_cleanup[" + change + "-" + prefix + "]")
        self.assertEqual(set(q.CPU_NOT_APPLICABLE[name]), cases)
        self.assertEqual(set(q.CPU_NOT_APPLICABLE[name].values()),
                         {"real word-store upload requires a GPU package; explicitly excluded from CPU selection"})
        host = "test_cache_id_of_preserves_hit_owners_and_native_metadata[tuple-array]"
        self.assertNotIn(host, q.CPU_NOT_APPLICABLE[name])
        self.assertEqual(q.selected_files(test_nodes=[name + "::" + host]), [name])

    def test_public_training_scalar_buffer_scope_is_version_and_operation_bounded(self):
        name = "test_training_scalar_buffer_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        self.assertTrue((q.ROOT / "python/mojolearn/tests" / name).is_file())
        self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest"])
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertIn("public RMS, embedding and cross entropy", scope["execution_route"])
            self.assertIn("installed NumPy version", scope["coverage_limit"])
            self.assertIn("Text, void, object and StringDType", scope["coverage_limit"])
            self.assertIn("cross-vendor identity/fault", scope["coverage_limit"])
            self.assertEqual(scope["operation_mode"], "identical (explicit fixture configuration)")
            self.assertEqual(len(scope["cpu_not_applicable_cases"]), 4 if vendor == "cpu" else 0)

    def test_public_training_scalar_buffer_cpu_exclusions_are_exact_device_cases(self):
        name = "test_training_scalar_buffer_native.py"
        prefix = "test_device_loss_scalar_conversion_and_failure_remain_native"
        cases = {prefix + "[" + case + "]" for case in (
            "label_smoothing-g-0.125-swapped", "ignore_index-c8-(-100+1j)-scalar",
            "num_items-g-3.75-swapped", "num_items-g-nan-scalar")}
        self.assertEqual(set(q.CPU_NOT_APPLICABLE[name]), cases)
        host = "test_loss_integer_range_preserves_downstream_error_or_consumed_result[u8-18446744073709551615-scalar]"
        self.assertNotIn(host, q.CPU_NOT_APPLICABLE[name])
        self.assertEqual(q.selected_files(test_nodes=[name + "::" + host]), [name])
        with self.assertRaises(ValueError):
            q.selected_files(test_nodes=[name + "::" + host] * 2)

    def test_public_archive_unicode_scope_preserves_warning_limitation(self):
        name = "test_archive_unicode_native.py"
        self.assertIn(name, q.PUBLIC_TESTS)
        for vendor in ("cpu", "metal", "cuda", "hip"):
            scope = q.test_applicability([name], vendor)[name]
            self.assertEqual(scope["cpu_not_applicable_cases"], {})
            self.assertEqual(q.test_dependencies([name]), ["numpy", "pytest"])
            self.assertIn("strict ZIP UTF-8", scope["coverage_limit"])
            self.assertIn("may still reach Python warning display", scope["coverage_limit"])
            self.assertIn("complete archive callback closure remains unqualified", scope["coverage_limit"])

    def test_public_owner_and_host_scopes_exclude_private_array_candidate(self):
        names = ("test_native_owner_qualification.py", "test_sequence_shutdown_native.py", "test_training_rms_public_native.py", "test_optimizer_native_step_errors.py",
                 "test_optimizer_native_array_ownership.py", "test_device_rows_metadata_native.py", "test_host_kde_fit_native.py", "test_host_iforest_fit_native.py",
                 "test_training_embedding_public_native.py",
                 "test_training_cross_entropy_public_native.py", "test_training_device_native_allocator.py",
                 "test_optimizer_native_container_admission.py", "test_optimizer_native_config_order.py",
                 "test_optimizer_native_scalar_constructors.py",
                 "test_host_density_constructors_native.py",
                 "test_training_device_native_finalizer.py", "test_training_device_native_rollback.py",
                 "test_robust_scaler_native_lifecycle.py", "test_scaler_rows_snapshot_native.py", "test_scaler_complex_warning_native.py",
                 "test_robust_scaler_native_transport.py",
                 "test_training_linear_public_native.py", "test_maxabs_scaler_native_transport.py",
                 "test_host_inference_native.py", "test_host_kernel_inference_native.py", "test_host_projection_native.py",
                 "test_host_logistic_native.py", "test_host_svr_native.py", "test_host_svc_native.py", "test_host_kde_native.py", "test_host_iforest_native.py")
        for name in names:
            self.assertIn(name, q.PUBLIC_TESTS)
            self.assertTrue((q.ROOT / "python/mojolearn/tests" / name).is_file())
        for name in ("test_sequence_array_native_candidate.py", "test_sequence_array_native_guards.py", "test_scaler_native_descriptor_lifetime.py"):
            self.assertNotIn(name, q.PUBLIC_TESTS)
        scope = q.test_applicability(names, "cuda")
        self.assertIn("explicit CPU host", scope["test_host_inference_native.py"]["execution_route"])
        self.assertIn("explicit CPU host", scope["test_host_svr_native.py"]["execution_route"])
        self.assertIn("score_samples/score", scope["test_host_kde_native.py"]["coverage_limit"])
        self.assertIn("explicit CPU host", scope["test_host_kde_fit_native.py"]["execution_route"])
        self.assertIn("state publication", scope["test_host_kde_fit_native.py"]["coverage_limit"])
        self.assertIn("explicit CPU host", scope["test_host_iforest_fit_native.py"]["execution_route"])
        self.assertIn("seeded training", scope["test_host_iforest_fit_native.py"]["coverage_limit"])
        self.assertIn("materialize", scope["test_device_rows_metadata_native.py"]["coverage_limit"])
        self.assertIn("public __array__", scope["test_optimizer_native_array_ownership.py"]["coverage_limit"])
        self.assertIn("installed version", scope["test_training_embedding_public_native.py"]["coverage_limit"])
        self.assertIn("subclass registration", scope["test_host_density_constructors_native.py"]["coverage_limit"])
        self.assertIn("retained numerical kernels", scope["test_training_cross_entropy_public_native.py"]["coverage_limit"])
        self.assertEqual(len(q.CPU_NOT_APPLICABLE["test_training_cross_entropy_public_native.py"]), 9)
        self.assertEqual(len(q.CPU_NOT_APPLICABLE["test_training_device_native_allocator.py"]), 2)
        self.assertEqual(len(q.CPU_NOT_APPLICABLE["test_optimizer_native_scalar_constructors.py"]), 2)
        self.assertEqual(len(q.CPU_NOT_APPLICABLE["test_optimizer_native_array_ownership.py"]), 6)
        self.assertEqual(len(q.CPU_NOT_APPLICABLE["test_training_device_native_finalizer.py"]), 7)
        self.assertEqual(len(q.CPU_NOT_APPLICABLE["test_training_device_native_rollback.py"]), 6)
        self.assertIn("identical", scope["test_sequence_shutdown_native.py"]["operation_mode"])
        self.assertEqual(q.test_dependencies(names), ["numpy", "pytest", "pandas", "scipy", "torch"])
        self.assertEqual(q.test_dependencies([names[0]]), ["numpy", "pytest"])

    def scope_fixture(self, file, *, passed=1, reason=q.CPU_RESIDENT_SKIP):
        cases = [dict(name=name, classname="mojolearn.tests." + Path(file).stem,
                      outcome="skipped", skip_reason=reason)
                 for name in q.CPU_NOT_APPLICABLE[file]]
        cases += [dict(name="applicable_" + str(i), classname="mojolearn.tests." + Path(file).stem,
                       outcome="passed", skip_reason="") for i in range(passed)]
        return dict(cases=cases, failures=[])

    def test_cpu_resident_skips_remain_explicit_without_discarding_applicable_passes(self):
        name = "test_native_owner_qualification.py"
        summary = self.scope_fixture(name, passed=7)
        scope = q.test_scope_summary(summary, [name], "cpu")
        self.assertEqual((scope["applicable_passed"], scope["not_applicable_count"]), (7, 2))
        self.assertEqual(scope["scopes"][name]["status"], "passed")
        self.assertEqual(q.selected_test_status(0, summary, scope, True), "selected-tests-passed")
        self.assertEqual(len(summary["cases"]), 9)  # original skip cases are retained

    def test_all_cpu_shutdown_cases_are_na_and_never_a_passing_qualification(self):
        name = "test_sequence_shutdown_native.py"
        summary = self.scope_fixture(name, passed=0)
        scope = q.test_scope_summary(summary, [name], "cpu")
        self.assertEqual((scope["applicable_passed"], scope["not_applicable_count"]), (0, 5))
        self.assertEqual(q.selected_test_status(0, summary, scope, True), "selected-tests-not-applicable")

    def test_gpu_skips_and_unrecognized_cpu_skips_remain_incomplete(self):
        name = "test_native_owner_qualification.py"
        for vendor, reason in (("metal", q.CPU_RESIDENT_SKIP), ("cpu", "no physical GPU"),
                               ("cpu", "could not import scipy")):
            summary = self.scope_fixture(name, reason=reason)
            scope = q.test_scope_summary(summary, [name], vendor)
            self.assertEqual(q.selected_test_status(0, summary, scope, True), "failed-or-incomplete")
        summary = self.scope_fixture(name)
        summary["cases"][0]["name"] = "new_unreviewed_resident_test"
        self.assertFalse(q.test_scope_summary(summary, [name], "cpu")["complete"])

    def test_missing_selected_scope_failure_or_missing_receipt_never_qualifies(self):
        name = "test_native_owner_qualification.py"
        summary = self.scope_fixture(name)
        scope = q.test_scope_summary(summary, [name, "test_host_inference_native.py"], "cpu")
        self.assertFalse(scope["complete"])
        scope = q.test_scope_summary(summary, [name], "cpu")
        for code, receipt in ((1, True), (0, False)):
            self.assertEqual(q.selected_test_status(code, summary, scope, receipt), "failed-or-incomplete")
        summary["failures"] = [dict(reason="original failure")]
        self.assertEqual(q.selected_test_status(0, summary, scope, True), "failed-or-incomplete")

    def test_junit_case_identity_distinguishes_a_declared_cpu_skip(self):
        path = self.root / "tests.xml"
        path.write_text('<testsuites><testsuite><testcase classname="mojolearn.tests.test_native_owner_qualification" name="test_failed_native_sequence_transfer_releases_allocated_handle[seq_tensor_copy]"><skipped message="the CPU binding has no resident tensor pool"/></testcase></testsuite></testsuites>')
        summary = q.junit_summary(path)
        scope = q.test_scope_summary(summary, ["test_native_owner_qualification.py"], "cpu")
        self.assertEqual(scope["not_applicable_count"], 1)
        self.assertEqual(summary["skipped"][0]["classname"], "mojolearn.tests.test_native_owner_qualification")

    def test_exact_selection_rejects_duplicates_paths_empty_and_mixed_scopes(self):
        name = "test_host_kde_fit_native.py"
        node = name + "::test_case[1]"
        self.assertEqual(q.selected_files(test_nodes=[node]), [name])
        for nodes in ([], [node, node], ["../" + node], [name], [name + "::"],
                      ["unregistered.py::test_case"], [node + "\n"]):
            with self.assertRaises(ValueError):
                q.selected_files(test_nodes=nodes)
        with self.assertRaises(ValueError):
            q.selected_files([name], [node])

    def test_exact_receipts_reject_missing_extra_duplicate_and_unidentified_cases(self):
        name = "test_host_kde_fit_native.py"
        node = name + "::test_case[1]"
        case = dict(name="test_case[1]", classname="mojolearn.tests." + Path(name).stem,
                    outcome="passed", skip_reason="", exact_node=node)
        self.assertTrue(q.test_scope_summary(dict(cases=[case]), [name], "cpu", [node])["complete"])
        for cases in ([], [case, case], [dict(case, exact_node="")],
                      [case, dict(case, exact_node=name + "::test_case[2]")]):
            self.assertFalse(q.test_scope_summary(dict(cases=cases), [name], "cpu", [node])["complete"])

    def test_exact_pytest_guard_runs_one_parameter_and_refuses_expanded_selection_before_calls(self):
        name = "test_host_kde_fit_native.py"
        node = name + "::test_case[1]"
        (self.root / name).write_text(
            "import pytest\nfrom pathlib import Path\n"
            "@pytest.mark.parametrize('value', [1, 2])\n"
            "def test_case(value):\n    Path('executed').write_text(str(value))\n")
        (self.root / "expected.json").write_text(json.dumps([node]))
        (self.root / "conftest.py").write_text(
            "import importlib.util, json\nfrom pathlib import Path\n"
            "spec = importlib.util.spec_from_file_location('q', " + repr(q.__file__) + ")\n"
            "q = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(q)\n"
            "guard = q.ExactNodeGuard(Path(__file__).parent, json.loads(Path('expected.json').read_text()))\n"
            "pytest_collection_finish = guard.pytest_collection_finish\n")
        env = dict(q.os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        for selector, expected, label in ((node, 0, "exact"), (name + "::test_case", 3, "expanded")):
            marker = self.root / "executed"
            marker.unlink(missing_ok=True)
            with (self.root / (label + ".log")).open("wb") as log:
                run = subprocess.run([q.sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                      "--junitxml=" + str(self.root / (label + ".xml")), selector],
                                     cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.assertEqual(run.returncode, expected, (self.root / (label + ".log")).read_text()[-1600:])
            if label == "exact":
                self.assertEqual(marker.read_text(), "1")
                summary = q.junit_summary(self.root / "exact.xml")
                self.assertEqual(summary["cases"][0]["exact_node"], node)
                self.assertTrue(q.test_scope_summary(summary, [name], "cpu", [node])["complete"])
            else:
                self.assertFalse(marker.exists())

    def worker_fixture(self, *, final_error=None, never_load=False):
        prefix, payload, tests, modules = self.modules()
        package = modules["mojolearn"]
        path = Path(package.runtime.__file__)
        del package.runtime  # Initial import is genuinely lazy in this fixture.
        package.vendor = lambda: "cpu"
        backend = types.ModuleType("mojolearn._backend")
        backend.default_mode = lambda: "identical"
        package._backend = backend
        verification = types.ModuleType("mojolearn._verify")
        runner = types.ModuleType("pytest")
        calls, state = [], dict(ran=False)

        def artifacts(*, require_loaded=True):
            calls.append(dict(require_loaded=require_loaded, after_tests=state["ran"]))
            if require_loaded and final_error:
                raise RuntimeError(final_error)
            if not state["ran"] or never_load:
                return []
            return [dict(module="mojolearn._host._mojolearn_example", file=str(path),
                         sha256=q.digest(path), size=path.stat().st_size, path_source="__spec__.origin")]

        def run(*args, plugins):
            state["ran"] = True
            plugins[0].pytest_runtest_teardown(None, None)
            return 0

        verification.binding_artifacts, runner.main = artifacts, run
        modules.update({"mojolearn._backend": backend, "mojolearn._verify": verification, "pytest": runner})
        out = self.root / "run"; (out / "work").mkdir(parents=True)
        request = out / "request.json"
        request.write_text(json.dumps(dict(out=str(out), tests=str(tests), payload=payload,
                                          test_files=["test_native_owner_qualification.py"],
                                          vendor="cpu", numeric_mode="identical", generation_device="cpu")))
        with patch.dict(q.sys.modules, modules), patch.object(q.sys, "prefix", str(prefix)), \
             patch.object(q.Path, "cwd", return_value=out / "work"), \
             patch.dict(q.os.environ, {"MOJOLEARN_GENERATION_DEVICE": "cpu"}):
            # A pytest process that already imported the source package (the
            # Mac suites run this file beside mojolearn's own tests) holds real
            # mojolearn.* modules the fixture's installed tree does not own;
            # hide them so the inventory reads only the fixture's modules.
            # patch.dict restores them on exit.
            for name in [k for k in q.sys.modules
                         if (k == "mojolearn" or k.startswith("mojolearn.")) and k not in modules]:
                del q.sys.modules[name]
            code = q.worker(request)
        return code, json.loads((out / "installed-modules.json").read_text()), calls

    def test_lazy_initial_import_requires_native_witnesses_only_after_tests(self):
        code, receipt, calls = self.worker_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(receipt["pytest_exit"], 0)
        self.assertEqual(receipt["inventory_errors"], [])
        self.assertEqual(len(receipt["bindings"]), 1)
        self.assertEqual(calls[0], dict(require_loaded=False, after_tests=False))
        self.assertEqual(calls[-1], dict(require_loaded=True, after_tests=True))

    def test_final_origin_failure_keeps_original_test_exit_and_error_receipt(self):
        code, receipt, _ = self.worker_fixture(final_error="loaded extension origin is unresolved")
        self.assertEqual(code, 1)
        self.assertEqual(receipt["pytest_exit"], 0)
        self.assertEqual(receipt["inventory_errors"][0]["phase"], "final")
        self.assertIn("origin is unresolved", receipt["inventory_errors"][0]["error"])
        self.assertEqual(len(receipt["bindings"]), 1)  # Earlier valid observation remains retained.

    def test_empty_final_inventory_cannot_qualify_a_metadata_only_run(self):
        code, receipt, _ = self.worker_fixture(never_load=True)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["pytest_exit"], 0)
        self.assertEqual(receipt["bindings"], [])
        self.assertIn("no loaded native binding witnesses", receipt["inventory_errors"][0]["error"])

    def archive_fixture(self, *, exclude_product=False):
        repo = self.root / "archive-repo"; repo.mkdir()
        files = {
            ".gitattributes": "/bench/results/* export-ignore\n/bench/results/kept -export-ignore\n",
            "bench/results/old/snapshot.mojo": "historical source\n",
            "bench/results/kept/reference.mojo": "retained reference\n",
            "autograd/runtime.mojo": "product source\n",
            "pixi.toml": "toolchain\n",
        }
        if exclude_product:
            files[".gitattributes"] += "/autograd/runtime.mojo export-ignore\n"
        for rel, content in files.items():
            path = repo / rel; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        for args in (["init", "-q"], ["add", "."],
                     ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "archive fixture"]):
            subprocess.run(["git", "-C", str(repo)] + list(args), check=True, capture_output=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "-z", commit])
        committed = {row.split(b"\t", 1)[1].decode(): row.split(b"\t", 1)[0].split()[2].decode()
                     for row in tree.split(b"\0") if row}
        frozen = self.root / "archive-frozen"; frozen.mkdir()
        archive = subprocess.check_output(["git", "-C", str(repo), "archive", commit])
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(frozen, filter="data")
        return repo, commit, committed, frozen

    def test_archive_omissions_are_explicit_and_use_committed_parent_policy(self):
        repo, commit, committed, frozen = self.archive_fixture()
        # Uncommitted rules must not decide what the archived commit omitted.
        (repo / ".gitattributes").write_text("/autograd/* export-ignore\n")
        hashes, omitted = q.native_source_inventory(frozen, commit, committed, repo)
        self.assertEqual(set(omitted), {"bench/results/old/snapshot.mojo"})
        self.assertEqual(omitted["bench/results/old/snapshot.mojo"]["export_ignore_path"], "bench/results/old")
        self.assertEqual(omitted["bench/results/old/snapshot.mojo"]["git_blob"], committed["bench/results/old/snapshot.mojo"])
        self.assertEqual(hashes["autograd/runtime.mojo"], q.digest(frozen / "autograd/runtime.mojo"))
        self.assertIn("bench/results/kept/reference.mojo", hashes)
        (frozen / "bench/results/kept/reference.mojo").unlink()
        with self.assertRaisesRegex(ValueError, "missing non-export-ignored native source"):
            q.native_source_inventory(frozen, commit, committed, repo)

    def test_missing_or_changed_product_source_is_never_an_archive_exemption(self):
        repo, commit, committed, frozen = self.archive_fixture()
        product = frozen / "autograd/runtime.mojo"
        product.write_text("tampered product\n")
        with self.assertRaisesRegex(ValueError, "uncommitted requested native source"):
            q.native_source_inventory(frozen, commit, committed, repo)
        product.unlink()
        with self.assertRaisesRegex(ValueError, "missing requested native source"):
            q.native_source_inventory(frozen, commit, committed, repo)

    def test_export_ignored_product_source_still_refuses(self):
        repo, commit, committed, frozen = self.archive_fixture(exclude_product=True)
        self.assertFalse((frozen / "autograd/runtime.mojo").exists())
        with self.assertRaisesRegex(ValueError, "missing requested native source"):
            q.native_source_inventory(frozen, commit, committed, repo)

    def test_fixture_device_matches_requested_vendor_and_cannot_inherit_cpu_override(self):
        parent = dict(PATH="/bin", PYTHONPATH="/unchecked/source", PYTHONHOME="/unchecked/python",
                      MOJOLEARN_GENERATION_DEVICE="cpu", MOJOLEARN_VENDOR="cpu",
                      MOJOLEARN_UNCHECKED_BINARY="/stale/native.so")
        for vendor, expected in (("cpu", "cpu"), ("metal", "gpu"), ("cuda", "gpu"), ("hip", "gpu")):
            with self.subTest(vendor=vendor):
                env = q.test_environment(parent, vendor, "identical")
                self.assertEqual(env["MOJOLEARN_GENERATION_DEVICE"], expected)
                self.assertEqual(env["MOJOLEARN_VENDOR"], vendor)
                self.assertEqual(env["MOJOLEARN_NUMERIC_MODE"], "identical")
                self.assertEqual(env["PATH"], "/bin")
                self.assertNotIn("MOJOLEARN_UNCHECKED_BINARY", env)
                self.assertNotIn("PYTHONPATH", env)
                self.assertNotIn("PYTHONHOME", env)

    def test_generation_fixture_device_is_observed_and_wrong_route_refuses(self):
        module = types.ModuleType("frozen.test_generation_native")
        module.__file__ = "/frozen/test_generation_native.py"
        module.DEVICE = "gpu"
        self.assertEqual(q.generation_test_devices({module.__name__: module}, "gpu"),
                         {module.__name__: "gpu"})
        module.DEVICE = "cpu"
        with self.assertRaisesRegex(ValueError, "generation fixture selected a different device"):
            q.generation_test_devices({module.__name__: module}, "gpu")


if __name__ == "__main__":
    unittest.main()
