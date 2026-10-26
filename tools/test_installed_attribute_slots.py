"""Source-only witness validation. No product imports or native execution."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_installed_attribute_slots as collector
import development_build as build
import installed_attribute_slots as slots
import merged_bindings as merged
import public_runtime_inventory as inventory

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return slots.reference(path)


class BuildProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = {p: hashlib.sha256(build.recipe_text((ROOT / p).read_text()).encode()).hexdigest()
                   for p in build.BUILD_INPUTS}
        graph = build.SourceGraph(ROOT)
        cls.fields_by_name = {}
        for name, member in slots.owner_members(ROOT).items():
            stem, flags, members = next(r for r in merged.plan("host", "darwin", True, "development")
                                        if member in r[2])
            cls.fields_by_name[name] = {
                "schema": build.KEY_SCHEMA, "mode": "host", "platform": "darwin", "stem": stem,
                "flags_class": flags, "members": members, "source": graph.closure(members),
                "builders": build.compile_recipe(scripts, "host", stem, flags),
                "root_sha256": hashlib.sha256(merged.root_text(stem, members).encode()).hexdigest(),
                "toolchain": {"available": True, "compiler_sha256": "a" * 64},
                "environment": {"MOJOLEARN_NUMERIC_MODE": "identical", "MOJOLEARN_PACKAGE_BYTE_LM": "1"}}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.receipts, self.binaries = {}, {}
        self.proof = {"owners": {}}
        for name, member in slots.owner_members(ROOT).items():
            fields = self.fields_by_name[name]
            binary = self.home / (fields["stem"] + ".so")
            binary.write_bytes(b"inert source-test fixture; not executable or runtime evidence")
            self.binaries[name] = binary
            self.receipts[name] = {"state": "built", "exit_code": 0, "fields": copy.deepcopy(fields),
                                   "mode": "host", "stem": fields["stem"], "sha256": slots.sha256(binary)}
            self.proof["owners"][name] = {"member": member, "binary": slots.reference(binary),
                "origin": {"name": name, "path": str(binary), "offset": 1}}
        self.refresh()

    def refresh(self):
        for name, receipt in self.receipts.items():
            receipt["key"] = build.digest(receipt["fields"])
            self.proof["owners"][name]["receipt"] = write(self.home / (name + ".receipt.json"), receipt)
        self.proof["index"] = write(self.home / "_mojolearn_merged.json", {
            "format": merged.INDEX_FORMAT, "layout": "development",
            "files": {self.binaries[n].name: r["fields"]["members"] for n, r in self.receipts.items()},
            "provenance": {r["fields"]["stem"]: r["key"] for r in self.receipts.values()}})

    def test_receipted_split_owners_are_accepted_without_build(self):
        self.assertNotEqual(self.binaries[slots.ENTRY], self.binaries[slots.FACTORY])
        self.assertEqual(slots.validate_build(ROOT, self.proof),
                         {n: r["key"] for n, r in self.receipts.items()})

    def test_mutations_of_either_owner_fail_with_refreshed_hashes(self):
        changes = {
            "failed compiler": lambda r: r.update(exit_code=1),
            "wrong output": lambda r: r.update(sha256="b" * 64),
            "missing compiler": lambda r: r["fields"]["toolchain"].pop("compiler_sha256"),
            "malformed compiler": lambda r: r["fields"]["toolchain"].update(compiler_sha256="name-only"),
            "wrong result identity": lambda r: r.update(stem="other"),
            "wrong source": lambda r: r["fields"]["source"]["files"].update({"bindings/public_attributes_native.mojo": "b" * 64}),
            "wrong recipe": lambda r: r["fields"]["builders"].update(command=[]),
            "wrong root": lambda r: r["fields"].update(root_sha256="b" * 64),
            "wrong members": lambda r: r["fields"].update(members=[]),
            "wrong mode": lambda r: r["fields"]["environment"].update(MOJOLEARN_NUMERIC_MODE="fast"),
            "unsupported define": lambda r: r["fields"]["environment"].update(MOJOLEARN_EXTRA_DEFINES="x"),
        }
        original = copy.deepcopy(self.receipts)
        for owner in self.receipts:
            for name, change in changes.items():
                with self.subTest(owner=owner, mutation=name):
                    self.receipts = copy.deepcopy(original)
                    change(self.receipts[owner])
                    self.refresh()
                    with self.assertRaises((slots.WitnessError, build.BuildError)):
                        slots.validate_build(ROOT, self.proof)

    def test_missing_or_changed_evidence_refuses_for_either_owner(self):
        for owner in self.receipts:
            for field in ("binary", "receipt"):
                with self.subTest(owner=owner, field=field):
                    proof = copy.deepcopy(self.proof)
                    proof["owners"][owner][field]["sha256"] = "c" * 64
                    with self.assertRaisesRegex(slots.WitnessError, "digest mismatch"):
                        slots.validate_build(ROOT, proof)
        proof = copy.deepcopy(self.proof)
        proof["index"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(slots.WitnessError, "digest mismatch"):
            slots.validate_build(ROOT, proof)
        self.binaries[slots.FACTORY].unlink()
        with self.assertRaisesRegex(slots.WitnessError, "restore retained bytes"):
            slots.validate_build(ROOT, self.proof)

    def test_wrong_loaded_origin_and_unactivated_binary_refuse(self):
        for owner in self.receipts:
            proof = copy.deepcopy(self.proof)
            proof["owners"][owner]["origin"]["path"] = str(self.home / "other.so")
            with self.subTest(owner=owner), self.assertRaisesRegex(slots.WitnessError, "pointer"):
                slots.validate_build(ROOT, proof)
        self.proof["index"] = write(self.home / "index.json", {
            "format": 1, "layout": "development", "files": {},
            "provenance": {r["fields"]["stem"]: r["key"] for r in self.receipts.values()}})
        with self.assertRaisesRegex(slots.WitnessError, "activated"):
            slots.validate_build(ROOT, self.proof)

    def test_entry_receipt_cannot_substitute_for_factory_receipt(self):
        self.proof["owners"][slots.FACTORY] = copy.deepcopy(self.proof["owners"][slots.ENTRY])
        with self.assertRaisesRegex(slots.WitnessError, "wrong declared owner"):
            slots.validate_build(ROOT, self.proof)

    def test_missing_factory_proof_refuses(self):
        del self.proof["owners"][slots.FACTORY]
        with self.assertRaisesRegex(slots.WitnessError, "owner proofs missing"):
            slots.validate_build(ROOT, self.proof)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (*slots.POLICY, "python/mojolearn/__init__.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# committed source fixture\n")
        def git(*args):
            subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        self.git = git

    def test_exact_source_and_policy_bytes(self):
        stamp = slots.source_snapshot(self.root)
        self.assertEqual(stamp["files_sha256"], slots.digest(stamp["files"]))
        (self.root / slots.POLICY[0]).write_text("# changed admission policy\n")
        with self.assertRaisesRegex(slots.WitnessError, "uncommitted"):
            slots.source_snapshot(self.root)

    def test_untracked_import_source_refuses(self):
        (self.root / "python/mojolearn/shadow.py").write_text("pass\n")
        with self.assertRaisesRegex(slots.WitnessError, "untracked"):
            slots.source_snapshot(self.root)

    def test_new_commit_is_distinct_even_when_package_source_unchanged(self):
        old = slots.source_snapshot(self.root)
        (self.root / "note.txt").write_text("new freeze\n")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "next freeze")
        self.assertNotEqual(old, slots.source_snapshot(self.root))


class LiveIdentityTests(unittest.TestCase):
    def setup_model(self):
        class Model:
            def fit(self):
                pass
        context = {"plans": {}, "pending": {}, "missing": object()}
        descriptor = SimpleNamespace(__func__=MethodType(len, context))
        Model.__getattribute__ = descriptor
        fields = vars(Model)
        context["plans"][Model] = {"mro": Model.__mro__, "fields_descriptor": fields["__dict__"],
                                   "bindings": {name: (fields[name], None) for name in ("__getattribute__", "fit")},
                                   "properties": {}, "availability": {}}
        return Model, context, descriptor

    def test_live_identities_not_boolean_claims(self):
        cls, context, descriptor = self.setup_model()
        self.assertEqual(collector.inspect_registered(cls, context, descriptor, len, SimpleNamespace), list(slots.CHECKS))

    def test_monkeypatched_slot_entry_context_and_plan_refuse(self):
        for name in ("slot", "entry", "context", "binding", "pending", "getattr", "mro"):
            with self.subTest(name=name):
                cls, context, descriptor = self.setup_model()
                if name == "slot":
                    cls.__getattribute__ = object.__getattribute__
                elif name == "entry":
                    descriptor.__func__ = MethodType(abs, context)
                elif name == "context":
                    descriptor.__func__ = MethodType(len, {})
                elif name == "binding":
                    cls.fit = property(lambda self: None)
                elif name == "pending":
                    context["pending"][cls] = "unsupported"
                elif name == "getattr":
                    cls.__getattr__ = lambda self, name: None
                else:
                    context["plans"][cls]["mro"] = (object,)
                with self.assertRaises(slots.WitnessError):
                    collector.inspect_registered(cls, context, descriptor, len, SimpleNamespace)

    def test_python_replacement_never_enters_pointer_inspection(self):
        with self.assertRaisesRegex(slots.WitnessError, "replaced"):
            collector.native_origin(lambda: None, slots.ENTRY)

    def setup_later_owner(self):
        cls = type("LogisticRegression", (), {"__module__": "mojolearn.linear_model"})
        context = {}
        descriptor = SimpleNamespace(__func__=MethodType(len, context))
        cls.__getattribute__ = descriptor
        context.update(classical_attribute_descriptor=descriptor,
            classical_public_layouts={cls: {"mro": cls.__mro__, "bindings": {"__getattribute__": (descriptor, None)}}})
        module = ModuleType("mojolearn.linear_model")
        module.LogisticRegression, module._LR_CONTEXT = cls, context
        proof = {"owners": {slots.FACTORY: {"binary": {"path": "/receipted/factory.so"}}}}
        origin = {"name": "standard_scaler_attribute_call", "path": "/receipted/factory.so", "offset": 1}
        return cls, context, descriptor, module, proof, origin

    def test_known_later_owner_is_evidence_only_and_not_a_getter_admission(self):
        cls, context, descriptor, module, proof, origin = self.setup_later_owner()
        with mock.patch.dict(sys.modules, {module.__name__: module}), \
             mock.patch.object(collector, "native_origin", return_value=origin):
            result = collector.later_owner(cls, len, SimpleNamespace, proof)
        self.assertIs(result["runtime_qualified"], False)
        self.assertEqual(result["origin"], origin)
        self.assertIn("_native_logistic_bridge.py:connect_public", result["installer"])
        self.assertNotIn("checks", result)

    def test_later_owner_replacements_refuse_instead_of_becoming_unmodeled(self):
        for mutation in ("slot", "context", "entry", "layout", "binary", "class"):
            cls, context, descriptor, module, proof, origin = self.setup_later_owner()
            if mutation == "slot":
                cls.__getattribute__ = object.__getattribute__
            elif mutation == "context":
                descriptor.__func__ = MethodType(len, {})
            elif mutation == "entry":
                descriptor.__func__ = MethodType(abs, context)
            elif mutation == "layout":
                context["classical_public_layouts"][cls]["bindings"] = {}
            elif mutation == "binary":
                origin["path"] = "/unreceipted/other.so"
            else:
                module.LogisticRegression = type("Replacement", (), {})
            with self.subTest(mutation=mutation), \
                 mock.patch.dict(sys.modules, {module.__name__: module}), \
                 mock.patch.object(collector, "native_origin", return_value=origin), \
                 self.assertRaises(slots.WitnessError):
                collector.later_owner(cls, len, SimpleNamespace, proof)

    def test_unrecognized_replacement_cannot_use_the_later_owner_exception(self):
        cls, _, _ = self.setup_model()
        self.assertIsNone(collector.later_owner(cls, len, SimpleNamespace, {}))

    def test_later_plain_function_binding_does_not_execute_or_qualify_method(self):
        cls, context, descriptor = self.setup_model()
        def replacement(self):
            raise AssertionError("method body must not run during slot collection")
        cls.fit = replacement
        self.assertEqual(collector.inspect_registered(cls, context, descriptor, len, SimpleNamespace), list(slots.CHECKS))

    def test_availability_identity_is_checked_without_running_predicate(self):
        cls, context, descriptor = self.setup_model()
        class Available:
            def __init__(self):
                self.check = lambda instance: (_ for _ in ()).throw(AssertionError("predicate must not run"))
                self.fn = lambda instance: None
                self.attribute_name = "transform"
            def __get__(self, instance, owner):
                raise AssertionError("descriptor must not run")
        member = Available()
        cls.fit_transform = member
        hooks = []
        for name in ("__get__", "__set__", "__delete__", "__getattribute__", "__getattr__"):
            found = next((vars(base)[name] for base in Available.__mro__ if name in vars(base)), context["missing"])
            hooks.append((name, found))
        context["plans"][cls]["availability"]["fit_transform"] = {
            "member": member, "check": member.check, "code": member.check.__code__, "function": member.fn,
            "attribute_name": member.attribute_name, "mro": Available.__mro__, "hooks": tuple(hooks)}
        collector.inspect_registered(cls, context, descriptor, len, SimpleNamespace)
        member.check = lambda instance: True
        with self.assertRaisesRegex(slots.WitnessError, "data changed"):
            collector.inspect_registered(cls, context, descriptor, len, SimpleNamespace)

    def test_old_observations_are_preserved_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "witness.json"
            path.write_text("retained earlier result")
            with self.assertRaisesRegex(slots.WitnessError, "new evidence path"):
                collector.collect(Path(temp), Path(temp), path)
            self.assertEqual(path.read_text(), "retained earlier result")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stamp = {"revision": "frozen", "files": {"connection.py": "sha"}}
        self.data = {"schema": slots.OBSERVATIONS, "source": self.stamp, "owner": slots.OWNER,
                     "runtime_qualified": False, "provenance": {"owners": {n: {"binary": {"sha256": n}} for n in (slots.ENTRY, slots.FACTORY)}},
                     "rows": [{"class": "mojolearn.est.Model", "exports": ["mojolearn.Model"],
                               "method": "__getattribute__", "status": "witnessed", "checks": list(slots.CHECKS)}]}
        self.path = self.root / "witness.json"
        self.refresh()

    def refresh(self):
        write(self.path, {"schema": slots.SCHEMA,
                         "observations": write(self.root / "observations.json", self.data)})

    def validate(self):
        # Only producer/source setup is synthetic here. Real build/source validation
        # is exercised above; no fixture exemption exists in production admission.
        with mock.patch.object(slots, "source_snapshot", return_value=self.stamp), \
             mock.patch.object(slots, "validate_build", return_value={n: n + "-key" for n in (slots.ENTRY, slots.FACTORY)}):
            return slots.validate(self.path, self.root)

    def test_scope_is_getter_only_and_never_claims_runtime_pass(self):
        witnessed = self.validate()
        for method in ("__getattribute__", "__init__", "__init_subclass__", "get_params"):
            with self.subTest(method=method):
                row = {"method": method, "aliases": ["mojolearn.Model"], "static_verdict": "python",
                       "runtime_status": "prohibited_source", "qualification_debt": ["old"],
                       "evidence": {"runtime": {"status": "unreviewed", "records": []}}}
                slots.apply(row, "mojolearn.est.Model", witnessed)
                self.assertEqual(row["runtime_status"], "native_boundary" if method == "__getattribute__" else "prohibited_source")
                self.assertEqual(row["evidence"]["runtime"], {"status": "unreviewed", "records": []})
                if method == "__getattribute__":
                    self.assertEqual(row["superseded_source"]["runtime_status"], "prohibited_source")
                    self.assertIn("unqualified", row["qualification_debt"][0])

    def test_stale_tampered_scope_and_runtime_claim_refuse(self):
        original = copy.deepcopy(self.data)
        for name in ("stale", "constructor", "checks", "duplicate", "runtime"):
            with self.subTest(name=name):
                self.data = copy.deepcopy(original)
                if name == "stale":
                    self.data["source"]["revision"] = "old"
                elif name == "constructor":
                    self.data["rows"][0]["method"] = "__init__"
                elif name == "checks":
                    self.data["rows"][0]["checks"].pop()
                elif name == "duplicate":
                    self.data["rows"].append(copy.deepcopy(self.data["rows"][0]))
                else:
                    self.data["runtime_qualified"] = True
                self.refresh()
                with self.assertRaises(slots.WitnessError):
                    self.validate()

    def test_pending_and_other_native_owners_remain_unqualified(self):
        for status in ("pending", "unmodeled"):
            self.data["rows"][0]["status"] = status
            self.refresh()
            self.assertEqual(self.validate(), {})

    def test_raw_observation_tampering_refuses(self):
        (self.root / "observations.json").write_text("{}")
        with self.assertRaisesRegex(slots.WitnessError, "digest mismatch"):
            self.validate()

    def test_inventory_uses_validated_witness_only_for_matched_owner(self):
        package = self.root / "python/mojolearn"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("from .est import Model\n__all__ = ['Model']\n")
        (package / "est.py").write_text("class Model:\n    def __init__(self):\n        self.x = 1\n"
                                       "    def __getattribute__(self, name):\n        return object.__getattribute__(self, name)\n")
        witnessed = self.validate()
        with mock.patch.object(slots, "validate", return_value=witnessed):
            report = inventory.endpoint_inventory(self.root, installed_slots=self.path)
        rows = {row["method"]: row for row in report["endpoints"]}
        self.assertEqual(rows["__getattribute__"]["runtime_status"], "native_boundary")
        self.assertNotEqual(rows["__init__"]["runtime_status"], "native_boundary")
        self.assertEqual(report["summary"]["installed_attribute_slots"], 1)
        witnessed["mojolearn.est.Other"] = witnessed.pop("mojolearn.est.Model")
        with mock.patch.object(slots, "validate", return_value=witnessed), self.assertRaisesRegex(ValueError, "do not resolve"):
            inventory.endpoint_inventory(self.root, installed_slots=self.path)


if __name__ == "__main__":
    unittest.main()
