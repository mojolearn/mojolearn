import glob, json, os, re
H = os.path.expanduser("~")
for d in sorted(glob.glob(H + "/mq/out/race-ident-*")):
    tag = os.path.basename(d)[5:]
    t = os.path.join(d, "A-1", "race.txt")
    if not os.path.exists(t):
        print("AFB family=? lane=? ds=? status=missing tag=%s" % tag); continue
    fam, line, dig = None, None, None
    for ln in open(t, errors="replace"):
        m = re.search(r'"family": "(\w+)"', ln) if ln.startswith("BOARD-PARAMS") else None
        if m: fam = m.group(1)
        if re.match(r"^[A-Z]+ lane=\S+ dataset=\S+ arm=ours .*status=", ln): line = ln.strip()
        m2 = re.search(r"arm=ours round=\d+ .*digest=(\S+)", ln)
        if m2: dig = m2.group(1)
    if not line:
        print("AFB family=%s lane=? ds=? status=noresult tag=%s" % (fam, tag)); continue
    kv = dict(re.findall(r"(\w+)=(\{[^}]*\}|\S+)", line))
    fam = fam or line.split()[0].lower()
    q = kv.get("quality", "-")
    try: q = ",".join("%s=%.6g" % (k, v) for k, v in json.loads(q).items() if isinstance(v, (int, float)))
    except Exception: q = "-"
    print("AFB family=%s lane=%s ds=%s status=%s median_ms=%s q=%s digest=%s runs=[%s] tag=%s head=cf94a6be6" % (
        fam, kv["lane"], kv["dataset"], kv["status"], kv.get("median_ms"), q or "-", dig, kv.get("median_ms"), tag))
