"""Write the "Verified models" page (user_guide/verified_models.md) from the documented release's model
catalog (mojolearn.models.catalog(), the shipped catalog.json that tools/model_catalog.py builds).

Only entries whose status is 'verified' are listed (broken or unverified = not documented). With no
verified entry the page is not written and an old one is removed; user_guide/index.md lists it through a
glob, so the guide then simply has no such page.
Usage: python tools/gen_models.py   (from docs-site/, with the documented mojolearn installed)
"""
# cpu-route: documentation build tooling, never imported by the mojolearn runtime.
import pathlib

import mojolearn

VENDORS = (("metal", "Apple"), ("cuda", "NVIDIA"), ("hip", "AMD"))
CONTRACT = "In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs."


def _version_key(v):
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in str(v).replace("+", ".").replace("-", ".").split(".")]


def _newest(entry):
    vs = sorted((v for v, modes in entry.get("versions", {}).items() if "identical" in modes), key=_version_key)
    return (vs[-1], entry["versions"][vs[-1]]["identical"]) if vs else (None, None)


def _cell(text):
    return str(text).replace("|", "\\|")


def _source(e):
    s = e.get("source") or {}
    if e.get("kind") == "torchvision":
        return f"torchvision `{s.get('weights')}`"
    if e.get("kind") == "hf-causal-lm":
        return f"Hugging Face `{s.get('repo')}` at `{str(s.get('revision'))[:12]}`"
    name = (s.get("url") or "").rsplit("/", 1)[-1]
    return f"[{name}]({s.get('url')}), SHA-256 `{str(s.get('sha256'))[:12]}`"


def _pct(v):
    return "-" if v is None else f"{100.0 * v:.1f}%"


def _num(v, fmt):
    return "-" if v is None else fmt % v


def page(entries):
    rows = [e for e in entries if e.get("status") == "verified"]
    if not rows:
        return None
    slices = sorted({((_newest(e)[1] or {}).get("accuracy") or {}).get("slice") for e in rows} - {None})
    out = ["# Verified models", "",
           "These published pretrained models import with {func}`mojolearn.import_model` and give the same "
           "output on every GPU listed: their output digests on a fixed, seeded check input are equal on "
           "each vendor. " + CONTRACT, "",
           "| Model | Source | Input | Top-1, published | Top-1, mojolearn | Output digest | Checked on |",
           "|---|---|---|---|---|---|---|"]
    for e in rows:
        version, cell = _newest(e)
        acc = e.get("accuracy") or {}
        pub = (acc.get("published") or {}).get("value")
        ours = (cell.get("accuracy") or {}).get("value")
        shape = " x ".join(str(d) for d in (e.get("inputs") or [{}])[0].get("shape", []))
        digests = cell.get("digests") or {}
        on = ", ".join(name for v, name in VENDORS if v in digests)
        digest = next(iter(digests.values()), "-")
        out.append("| " + " | ".join(_cell(x) for x in (
            f"`{e['name']}`", _source(e), shape, _pct(pub), _pct(ours), f"`{digest}` ({version})", on)) + " |")
    out += [""]
    if slices:
        out += ["Top-1, published, is the number published with the weights. Top-1, mojolearn, is the imported "
                "model's accuracy on " + "; ".join(slices) + ".", ""]
    out += ["## Latency and memory", "",
            "One run on the check input, after one warm-up run. The time includes copying the input to the GPU "
            "and the output back. Memory is the process's GPU memory after the run, where the vendor "
            "reports it.", "",
            "| Model | " + " | ".join(f"{n} ms" for _, n in VENDORS) + " | "
            + " | ".join(f"{n} MB" for _, n in VENDORS[1:]) + " |",
            "|---|" + "---|" * (len(VENDORS) + len(VENDORS) - 1)]
    for e in rows:
        _, cell = _newest(e)
        vend = cell.get("vendors") or {}
        ms = [_num((vend.get(v) or {}).get("latency_ms"), "%.2f") for v, _ in VENDORS]
        mb = [_num((vend.get(v) or {}).get("device_memory_mb"), "%.0f") for v, _ in VENDORS[1:]]
        out.append("| " + " | ".join([f"`{e['name']}`"] + ms + mb) + " |")
    out += ["", "## Check a model on your GPU", "",
            "{func}`mojolearn.models.verify_catalog_entry` downloads the model from its source, imports it, "
            "runs the check input and compares the output digest with the one listed here:", "",
            "```python",
            "import mojolearn",
            "",
            "report = mojolearn.models.verify_catalog_entry(\"%s\")" % rows[0]["name"],
            "print(report[\"vendor\"], report[\"digest\"], report[\"match\"])",
            "```", "",
            "{func}`mojolearn.models.catalog` returns every entry, with its pinned source, input shapes and "
            "the numbers above.", ""]
    return "\n".join(out)


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    target = root / "user_guide" / "verified_models.md"
    models = getattr(mojolearn, "models", None)
    entries = models.catalog() if models is not None and hasattr(models, "catalog") else []
    text = page(entries)
    if text is None:
        if target.exists():
            target.unlink()
        print("gen_models: no verified model; no page")
        return
    target.write_text(text)
    print("gen_models: %d verified models" % sum(e.get("status") == "verified" for e in entries))


if __name__ == "__main__":
    main()
