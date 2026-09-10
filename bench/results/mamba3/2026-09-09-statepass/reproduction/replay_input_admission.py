import json
from pathlib import Path

def read(path):
    expected = {}
    for line in path.read_text().splitlines():
        if line.startswith("FSPEED-WEIGHTS "):
            fields = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
            key = (fields["lane"], fields["shape"], fields["tensor"])
            expected[key] = (fields["n"], fields["hash"])
    return expected

root = Path(__file__).resolve().parent.parent / "opponent-admission"
expected = read(root / "seq.mamba3.ours.log")
actual = read(root / "seq.mamba3.torch.log")
assert expected and all(actual.get(k) == v for k, v in expected.items())
print("FSPEED-INPUT-GATE passed=true tensors=%d" % len(expected))
