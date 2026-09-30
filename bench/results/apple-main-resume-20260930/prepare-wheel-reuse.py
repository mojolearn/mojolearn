import json, pathlib, sys, zipfile
sys.path.insert(0, 'tools')
import release_reuse as reuse

root = pathlib.Path.cwd()
old = '10523ede164135c2293d16f857b49c157eab71aa'
new = '3d0da1acf5303ebe7463222544b5791d78c93c09'
assert reuse.git('rev-parse', 'HEAD', root=root).strip() == new
wheels = list((root / 'python/dist').glob('*.whl'))
assert len(wheels) == 1, wheels
wheel = wheels[0]
with zipfile.ZipFile(wheel) as z:
    assert z.read('mojolearn/identity_columns/COMMIT').decode().strip() == old
cache = pathlib.Path.home() / 'mojolearn-evidence/apple-main-reuse-identities'
before = reuse.facts_for_commit(old, cache, root=root)
after = reuse.facts_for_commit(new, cache, root=root)
toolchain = reuse.darwin_toolchain()
rows = []
for binding in reuse.bindings(reuse.MACOS):
    a = reuse.identity(binding, before, toolchain)
    b = reuse.identity(binding, after, toolchain)
    decision, reason = reuse.compare(b, a)
    rows.append(dict(binding.row(), decision=decision, reason=reason,
                     identity=b, identity_digest=reuse.digest_of(b)))
plan = {'previous': {'version': '0.8.31-candidate', 'source_commit': old},
        'commit': new, 'rows': rows}
dest = pathlib.Path.home() / 'mojolearn-evidence/apple-main-reuse-3d0da1acf'
reuse.assemble_macos(plan, wheel, dest)
(dest / 'source-comparison.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({'reused': sum(r['decision']=='REUSE' for r in rows),
                  'rebuild': [r['key'] for r in rows if r['decision']=='BUILD']}))
