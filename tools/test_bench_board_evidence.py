"""CPU-only evidence retention and anti-stale-import regressions."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = load("bench_board_evidence")
P = load("bench_board_params")


class Evidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_raw_parameters_survive_worker_and_alias_filter(self):
        obj = {"__library__":"sklearn", "alpha": 1, "random_state":7, "n_jobs":32,
               "future_parameter":{"nested": [True, 3]}}
        record = P.arm_record(obj)
        report = P.check("ridge", {"ours":record, "opponent":record})
        raw = report["arms"]["opponent"]["raw_params"]
        self.assertEqual(raw["n_jobs"], 32)
        self.assertEqual(raw["future_parameter"], {"nested":[True,3]})

    def test_snapshot_contains_actual_working_source_and_external_driver(self):
        repo = self.root / "repo"; repo.mkdir()
        p = repo / "driver.py"; p.write_text("old\n")
        subprocess.run(["git","init","-q",str(repo)],check=True,capture_output=True)
        subprocess.run(["git","-C",str(repo),"add","."],check=True,capture_output=True)
        subprocess.run(["git","-C",str(repo),"-c","user.name=test","-c","user.email=test@example.invalid",
                        "commit","-qm","source"],check=True,capture_output=True)
        p.write_text("actual working bytes\n")
        other=self.root/"external.py";other.write_text("external\n")
        ref, inventory=E.source_snapshot(self.root/"out",repo,[other])
        with tarfile.open(ref["local_path"]) as t:
            self.assertEqual(t.extractfile("driver.py").read(),p.read_bytes())
            self.assertTrue(any(n.startswith("external/") for n in t.getnames()))
        self.assertEqual(inventory[str(p.resolve())],E.file_hash(p))

    def test_readback_is_required_and_failure_preserves_local_bytes(self):
        p=self.root/"run.tar.gz";ref=E.archive(p,{"log.txt":b"full evidence"})
        objects={}
        def request(method,key,data=None):
            if method=="PUT": objects[key]=data;return 200,b""
            return 200,objects[key]
        done=E.upload(ref,request)
        self.assertEqual(done["status"],"verified")
        self.assertEqual(done["sha256"],E.digest(objects[done["r2_key"]]))
        with self.assertRaisesRegex(ValueError,"read-back"):
            E.upload(ref,lambda *args: (200,b"wrong bytes"))
        self.assertTrue(p.exists())

    def test_fetch_verifies_before_overwriting_destination(self):
        p=self.root/"download";p.write_bytes(b"keep")
        ref=dict(r2_key="key",sha256=E.digest(b"retained"))
        with self.assertRaisesRegex(ValueError,"checksum"):
            E.fetch(ref,p,lambda key:(200,b"corrupt"))
        self.assertEqual(p.read_bytes(),b"keep")
        E.fetch(ref,p,lambda key:(200,b"retained"))
        self.assertEqual(p.read_bytes(),b"retained")

    def test_seal_preserves_stored_opponent_provenance(self):
        d=self.root/"evidence/runs/r1";(d/"workers").mkdir(parents=True)
        E.write(d/"recipe.json",{})
        src=self.root/"source.py";src.write_text("code")
        ctx=dict(out=str(self.root),measurement_inventory={str(src):E.file_hash(src)},
                 measurement_source={"status":"local"},evidence_r2=False)
        recipe=dict(source={"status":"local"},dataset={"identity":"hash"},hardware={})
        own=dict(arm="ours",library="mojolearn",params={"params":{"alpha":1}},library_version="1",
                 comparability={"span":{"input_home":"host"}},times_ms=[2],median_ms=2)
        old=dict(arm="lib",stored={"commit":"old"},measurement={"id":"accepted-old"},median_ms=4)
        rec=dict(cells=[own,old],stored_arms=["lib"])
        previous=copy.deepcopy(old)
        E.seal(ctx,d,recipe,rec)
        self.assertEqual(old,previous)
        self.assertIn("worker receipt",own["measurement"]["missing"])
        self.assertEqual(own["median_ms"],2)

    def test_sync_retries_bytes_without_modifying_times(self):
        d=self.root/"evidence/runs/r1";d.mkdir(parents=True)
        ref=E.archive(d/"run.tar.gz",{"log":b"original"})
        E.write(d/"upload.json",dict(source=ref,run=ref))
        b=dict(races={"r":dict(evidence_directory="evidence/runs/r1",evidence_failure="R2 retention failed; retry sync without rerunning the measurement",
                 cells=[dict(arm="ours",times_ms=[3,4],median_ms=3.5,measurement=dict(artifacts=E.public_ref(ref)))])})
        p=self.root/"board.json";E.write(p,b)
        with patch.object(E,"upload",side_effect=lambda r: dict(r,status="verified",r2_key="key")):
            E.sync(p)
        rr=json.loads(p.read_text())["races"]["r"]
        self.assertEqual(rr["cells"][0]["times_ms"],[3,4])
        self.assertEqual(rr["cells"][0]["median_ms"],3.5)
        self.assertNotIn("evidence_failure",rr)

    def test_preflight_failure_never_enters_timed_driver(self):
        from types import SimpleNamespace
        B=load("bench_board")
        directory=self.root/"run";directory.mkdir()
        evidence=SimpleNamespace(begin=lambda *a: (directory, {}), write=E.write, seal=lambda *a: None)
        original=B._load_tool
        race=dict(id="r",family="algos",lane="ridge",dataset="synthetic",rows=None,
                  arms=["ours","lib"],opponents=["lib"],our_arms={"ours":"identical"})
        ctx=dict(capture_measurements=True,out=str(self.root),box={},with_opponents=True)
        with patch.object(B,"_load_tool",side_effect=lambda n: evidence if n=="bench_board_evidence" else original(n)), \
             patch.object(B,"race_data_files",return_value=[]), patch.object(B,"race_data_sha",return_value="input"), \
             patch.object(B,"race_settings",return_value={}), patch.object(B,"stored_opponents",return_value={}), \
             patch.object(B,"unsupported_our_arms",return_value={}), patch.object(B,"params_probe",return_value={}), \
             patch.object(B,"_run_race") as timed, patch.object(B,"store_opponents",return_value=0):
            rec=B.run_race(ctx,race)
        timed.assert_not_called()
        self.assertEqual(rec["params_check"],"REFUSED")
        self.assertTrue((directory/"preflight.json").exists())

    def test_failed_probe_keeps_parameters_but_never_admits(self):
        B=load("bench_board")
        record=P.check("ridge", {"ours":{"__library__":"library","alpha":1,"seed":7}})["arms"]["ours"]
        record["implementation"]={"module":"library.ridge","name":"Ridge"}
        probes={a:dict(record=record,version="1",probe_rc=2,hardware={"status":"verified"})
                for a in ("ours","torch-gpu")}
        race=dict(family="classical",lane="ridge",our_arms={"ours":"identical"})
        with patch.object(B,"params_probe",return_value=probes):
            rep=B.preflight_check(dict(vendor="amd"),race,list(probes))
        self.assertEqual(rep["verdict"],"REFUSED")
        self.assertEqual(rep["arms"]["torch-gpu"]["raw_params"]["alpha"],1)
        self.assertTrue(any("probe failed" in s for s in rep["problems"]))

    def sealed_cell(self, receipt_arm="lib-gpu", worker_version="1", worker_hardware=None,
                    changed_params=False):
        d=self.root/"evidence/runs/r1";(d/"workers").mkdir(parents=True)
        E.write(d/"recipe.json", {})
        hardware=dict(status="verified", gpu="physical-123", driver="580", os="Linux")
        record=P.check("ridge", {"ours":{"__library__":"library", "alpha":1, "seed":7},
                                "lib-gpu":{"__library__":"library", "alpha":1, "seed":7}})["arms"]["lib-gpu"]
        record["implementation"]={"module":"library.ridge", "name":"Ridge"}
        pre=dict(verdict="MATCHED",arms={"lib-gpu":copy.deepcopy(record)},
                 probe_workers={"lib-gpu":dict(version="1",hardware=hardware,environment={})})
        if changed_params:
            record["raw_params"]["alpha"]=2
        cell=dict(arm="lib-gpu",library="library",mode="opponent",device="gpu",status="ok",
                  library_version="1",shape="100x10",times_ms=[2],median_ms=2,
                  comparability={"span":{"input_home":"host"}})
        rec=dict(arms=["lib-gpu"],cells=[cell],preflight=pre,
                 params=dict(schema="mojolearn.parameter-parity.v2",arms={"lib-gpu":record}),
                 command=["python", "driver.py", "--seed", "7"])
        E.write(d/"workers/1.json",dict(status="recorded",arms={receipt_arm:"library"},environment={},
                versions={receipt_arm:worker_version},hardware=hardware if worker_hardware is None else worker_hardware))
        E.seal(dict(out=str(self.root),measurement_inventory={},measurement_source={"status":"local"},
                    evidence_r2=False), d, dict(source={},dataset={"identity":"abc"},hardware={}),rec)
        return rec, cell

    def test_complete_record_retains_call_settings_seed_and_final_receipt(self):
        rec,cell=self.sealed_cell()
        m=cell["measurement"]
        self.assertEqual(m["completeness"],"complete",m["missing"])
        self.assertEqual(m["seed"],7)
        self.assertEqual(m["invocation"]["effective_arguments"]["alpha"],1)
        with tarfile.open(self.root/"evidence/runs/r1/receipt.tar.gz") as t:
            saved=json.load(t.extractfile("result.json"))
        self.assertEqual(saved["cells"][0]["measurement"],m)

    def test_same_library_different_arm_cannot_supply_receipt(self):
        rec,cell=self.sealed_cell(receipt_arm="lib-other")
        self.assertIn("worker receipt",cell["measurement"]["missing"])
        self.assertIn("evidence_failure",rec)

    def test_changed_worker_version_refuses_admission(self):
        rec,cell=self.sealed_cell(worker_version="2")
        self.assertIn("evidence_failure",rec)
        self.assertTrue(any("version changed" in s for s in cell["measurement"]["missing"]))

    def test_changed_worker_gpu_refuses_admission(self):
        rec,cell=self.sealed_cell(worker_hardware=dict(status="verified",gpu="different"))
        self.assertIn("evidence_failure",rec)
        self.assertTrue(any("GPU/driver/OS" in s for s in cell["measurement"]["missing"]))

    def test_changed_raw_parameters_refuses_admission(self):
        rec,cell=self.sealed_cell(changed_params=True)
        self.assertIn("evidence_failure",rec)
        self.assertTrue(any("changed after preflight" in s for s in cell["measurement"]["missing"]))

    def test_runtime_flags_are_retained_without_credentials_or_receipt_paths(self):
        got=E.runtime_environment(dict(MOJOLEARN_IDN_ALL_OFF="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
            MOJOLEARN_R2_CREDS="private", MODULAR_AUTH_TOKEN="secret", MOJOLEARN_BOARD_SESSION="run",
            MOJOLEARN_MEASUREMENT_RECEIPTS="changing-directory", UNRELATED="value"))
        self.assertEqual(got,dict(MOJOLEARN_IDN_ALL_OFF="1",CUBLAS_WORKSPACE_CONFIG=":4096:8"))

    def test_main_board_retains_exact_worker_cell_evidence(self):
        M=load("main_board_ingest")
        BB=M.load_bench_board()
        original=dict(measurement={"id":"frozen-worker","software":{"version":"old"}},
                      params={"params":{"alpha":2}},settings={"rows_cap":999},
                      times_ms=[2,4],warmup_ms=7,rounds=2,shape="999x8")
        ob=dict(status="ok",median_ms=3,sha="a"*40,box="nvidia-l40s",job="n0001",ct=100,
                kind="json",family="algos",lane="ridge",dataset="taxi",vendor="nvidia",measured_cell=original)
        cell=M.our_cell(BB,ob,{"other_column":"amd","status":"MATCH"})
        for k,v in original.items():
            self.assertEqual(cell[k],v)
        cell["params"]["params"]["alpha"]=99
        self.assertEqual(original["params"]["params"]["alpha"],2)


if __name__ == "__main__":
    unittest.main()
