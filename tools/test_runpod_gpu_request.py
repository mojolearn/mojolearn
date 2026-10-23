"""Inert RunPod request tests: no network, pod creation, SSH or credentials."""
import json

import pytest

from runpod_gpu_request import OVERRIDE, allowed_cuda_versions, build_request, main


def catalog(*versions):
    return {"gpus": [{"id": "NVIDIA L40S", "manufacturer": "NVIDIA", "cudaVersions": [
        {"version": version, "available": available} for version, available in versions]}]}


def request(*, env=None, data=None, amd=False, image="image", bootstrap=""):
    return build_request("owned-pod", image, "NVIDIA L40S", "80", amd, "2",
                         env={} if env is None else env, catalog=data, bootstrap=bootstrap)


def test_live_catalog_floor_includes_newer_available_hosts_and_numeric_order():
    data = catalog(("13.0", False), ("13.2", True), ("13.10", False), ("12.8", True))
    req = request(data=data)
    assert req == {"name": "owned-pod", "imageName": "image", "gpuTypeIds": ["NVIDIA L40S"],
                   "gpuCount": 2, "cloudType": "SECURE", "containerDiskInGb": 80,
                   "volumeInGb": 0, "ports": ["22/tcp"], "supportPublicIp": True,
                   "interruptible": False, "allowedCudaVersions": ["13.0", "13.2", "13.10"]}
    assert "minCudaVersion" not in req  # Existing v1 create API; no mixed constraints.


def test_override_preserves_placement_and_does_not_require_catalog():
    req = request(env={OVERRIDE: " 13.2,13.0,13.2 ",
                       "MOJOLEARN_DEVPOD_CLOUD_TYPE": "COMMUNITY", "MOJOLEARN_DEVPOD_MIN_VCPU": "8"})
    assert req["allowedCudaVersions"] == ["13.0", "13.2"]
    assert req["cloudType"] == "COMMUNITY"
    assert req["minVCPUPerGPU"] == 8
    assert req["gpuCount"] == 2
    assert req["interruptible"] is False


@pytest.mark.parametrize("override", ["", "13", "13.0,", ",13.2", "12.9", "13.2,12.10",
                                       "13.2.1", "013.2", "13.02", "nan", "-13.0"])
def test_invalid_or_below_floor_override_refuses(override):
    with pytest.raises(ValueError):
        request(env={OVERRIDE: override}, data=catalog(("13.2", True)))


@pytest.mark.parametrize("data", [None, {}, {"gpus": []}, {"gpus": [None]},
                                  {"gpus": [{"id": "NVIDIA L40S", "manufacturer": "NVIDIA"}]},
                                  catalog(("12.8", True))])
def test_absent_or_incomplete_catalog_fails_closed(data):
    with pytest.raises(ValueError):
        request(data=data)


def test_only_selected_gpu_versions_are_considered():
    data = catalog(("13.2", True))
    data["gpus"].append({"id": "NVIDIA other", "manufacturer": "NVIDIA",
                         "cudaVersions": [{"version": "15.0", "available": True}]})
    assert request(data=data)["allowedCudaVersions"] == ["13.2"]
    with pytest.raises(ValueError, match="no NVIDIA metadata"):
        allowed_cuda_versions(["NVIDIA missing"], data)


def test_every_selected_gpu_has_valid_metadata_and_versions_are_unioned():
    data = catalog(("13.2", True))
    data["gpus"].append({"id": "NVIDIA other", "manufacturer": "NVIDIA",
                         "cudaVersions": [{"version": "13.4", "available": False}]})
    assert allowed_cuda_versions(["NVIDIA L40S", "NVIDIA other"], data) == ["13.2", "13.4"]
    data["gpus"][1]["cudaVersions"][0]["available"] = "yes"
    with pytest.raises(ValueError, match="malformed version metadata"):
        allowed_cuda_versions(["NVIDIA L40S", "NVIDIA other"], data)


def test_amd_keeps_original_bootstrap_and_ignores_cuda_override():
    req = build_request("owned-amd", "rocm/image", "AMD Instinct MI300X OAM", 80, True, 1,
                        env={OVERRIDE: "12.0"}, bootstrap="existing bootstrap")
    assert "allowedCudaVersions" not in req
    assert req["dockerEntrypoint"] == ["/bin/bash", "-lc"]
    assert req["dockerStartCmd"] == ["existing bootstrap"]
    assert req["gpuTypeIds"] == ["AMD Instinct MI300X OAM"]
    assert req["gpuCount"] == 1
    assert req["ports"] == ["22/tcp"]


def test_cli_invalid_catalog_cannot_write_create_payload(tmp_path, monkeypatch):
    monkeypatch.delenv(OVERRIDE, raising=False)
    bad = tmp_path / "catalog.json"
    bad.write_text("not JSON")
    out = tmp_path / "create.json"
    with pytest.raises(SystemExit, match="refused before create"):
        main([str(out), "pod", "image", "NVIDIA L40S", "80", "0", "unused", "1", str(bad)])
    assert not out.exists()


def test_cli_catalog_snapshot_builds_validated_payload(tmp_path, monkeypatch):
    monkeypatch.delenv(OVERRIDE, raising=False)
    monkeypatch.delenv("MOJOLEARN_DEVPOD_CLOUD_TYPE", raising=False)
    monkeypatch.delenv("MOJOLEARN_DEVPOD_MIN_VCPU", raising=False)
    data = tmp_path / "catalog.json"
    data.write_text(json.dumps(catalog(("13.0", False), ("13.2", True))))
    out = tmp_path / "create.json"
    assert main([str(out), "pod", "image", "NVIDIA L40S", "80", "0", "unused", "1", str(data)]) == 0
    assert json.loads(out.read_text())["allowedCudaVersions"] == ["13.0", "13.2"]
