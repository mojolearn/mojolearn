import sys, os
src=open(sys.argv[1]).read(); base=sys.argv[2]
arms={
 'no_sha_check': ('if len(data) != f.get("size") or sha256_bytes(data) != f.get("sha256"):', 'if False:'),
 'image_not_keyed': ('        image=image,\n', '        image="",\n'),
 'no_sabotage_refusal': ('            return "sabotage:" + k', '            pass'),
 'jobs_not_keyed': ('NON_BUILD_ENV = ("MOJOLEARN_COMMIT",', 'NON_BUILD_ENV = ("MOJOLEARN_COMPILE_JOBS", "MOJOLEARN_COMMIT",'),
}
for name,(a,b) in arms.items():
    assert src.count(a)==1, name
    os.makedirs(os.path.join(base,name), exist_ok=True)
    open(os.path.join(base,name,'bincache.py'),'w').write(src.replace(a,b))
print("arms regenerated from", sys.argv[1])
