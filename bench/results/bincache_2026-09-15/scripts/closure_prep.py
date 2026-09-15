# Build a scratch tree where every .mojo OUTSIDE a binding's closure is garbage.
import sys, shutil, os
from pathlib import Path
W = Path(sys.argv[1]); dst = Path(sys.argv[2]); script = sys.argv[3]; break_inside = sys.argv[4] if len(sys.argv) > 4 else ""
sys.path.insert(0, str(W / "tools"))
import bincache as b
src, rels = b.source_digest(W, W / script, [])
closure = set(rels)
if dst.exists(): shutil.rmtree(dst)
n_bad = 0
for p in b.tree_sources(W):
    rel = os.path.relpath(p, W)
    out = dst / rel; out.parent.mkdir(parents=True, exist_ok=True)
    if rel in closure and rel != break_inside:
        shutil.copyfile(p, out)
    else:
        out.write_text("@@@ this is not mojo, the closure says the compiler never reads %s\n" % rel); n_bad += 1
for rel in ("pixi.toml", "pixi.lock"):
    shutil.copyfile(W / rel, dst / rel)
print("scope", src["scope"], "closure", len(closure), "garbage files", n_bad, "broken inside:", break_inside or "-")
